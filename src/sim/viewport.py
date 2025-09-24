from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple, Any

import os
import importlib.util
import pygame


ORANGE = (255, 165, 0)


def _load_insect_params() -> Dict[str, Any]:
    here = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    cfg_path = os.path.join(repo_root, "config", "insect_config.py")
    if os.path.exists(cfg_path):
        try:
            spec = importlib.util.spec_from_file_location("insect_config", cfg_path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "INSECT_PARAMS"):
                return getattr(mod, "INSECT_PARAMS")
        except Exception:
            pass
    try:
        from config.config_structure import get_insect_base_config  # type: ignore
        return get_insect_base_config()
    except Exception:
        return {}


class Viewport:
    def __init__(self, cfg: Dict, layout_idx: List[Dict]):
        self.cfg = cfg
        self.idx = layout_idx
        self.total_leds = len(layout_idx)
        self.speed = float(cfg["wave"]["speed_mps"])  # m/s
        self.tail_m = float(cfg["wave"]["tail_m"])  # m
        self.leds_per_m = int(cfg["leds_per_meter"])  # leds/m
        self.tail_leds = max(1, int(round(self.tail_m * self.leds_per_m)))

        # Simulation area (meters -> pixels scale handled here)
        # View size and scale autoset to fit content
        self.view_w = int(900)
        self.view_h = int(700)
        self.margin = 40
        # dynamic scale so the U-shape fits vertically and horizontally
        m_left = float(self.cfg["segments_m"]["left"])
        m_bottom = float(self.cfg["segments_m"]["bottom"])
        scale_h = (self.view_h - 2 * self.margin) / max(1e-6, m_left)
        scale_w = (self.view_w - 2 * self.margin) / max(1e-6, m_bottom)
        self.px_per_m = int(max(1.0, min(scale_h, scale_w)))

        # Load insect params (colors, patterns, sound files)
        self.insect_params = _load_insect_params()

        # Audio setup (pygame mixer)
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.set_num_channels(64)
        except Exception:
            pass

        # Preload species sounds once
        self.species_sounds: Dict[str, pygame.mixer.Sound] = {}
        audio_cfg = cfg.get("audio", {})
        master_db = float(audio_cfg.get("gain_master", -6.0))
        self.master_amp = 10 ** (master_db / 20.0)
        for sp_key, v in self.insect_params.items():
            p = v.get("sound_files", {}).get("default")
            if p and os.path.exists(p):
                try:
                    self.species_sounds[sp_key] = pygame.mixer.Sound(p)
                except Exception:
                    pass

    def _pos_for_index(self, entry: Dict) -> Tuple[int, int]:
        # Map into U-shape rectangle in screen space
        left = float(self.cfg["segments_m"]["left"]) * self.px_per_m
        bottom = float(self.cfg["segments_m"]["bottom"]) * self.px_per_m
        right = float(self.cfg["segments_m"]["right"]) * self.px_per_m
        total_h = int(left)  # same as right; use left
        total_w = int(bottom)
        # Place in a nice frame
        ox = self.margin
        oy = self.margin
        name = entry["seg"]
        s = float(entry["s"])  # 0..1 along segment
        if name == "left_down":
            x = ox
            y = oy + int(s * total_h)
        elif name == "bottom_right":
            x = ox + int(s * total_w)
            y = oy + total_h
        else:  # right_up
            x = ox + total_w
            y = oy + (total_h - int(s * total_h))
        return x, y

    def run_simulation(self):
        pygame.init()
        screen = pygame.display.set_mode((self.view_w, self.view_h))
        clock = pygame.time.Clock()

        # Precompute pixel positions
        pts = [self._pos_for_index(e) for e in self.idx]

        # Group LEDs into 100 insects (12 LEDs each)
        group_size = 12
        self.insects = []
        species_keys = list(self.insect_params.keys()) or [
            "aomatsumushi",
            "kutsuwa",
            "matsumushi",
            "umaoi",
            "koorogi",
            "kirigirisu",
            "suzumushi",
        ]
        for g in range(self.total_leds // group_size):
            start = g * group_size
            end = start + group_size
            sp_key = species_keys[g % len(species_keys)]
            colors = self.insect_params.get(sp_key, {}).get("colors", {})
            base = tuple(int(x) for x in colors.get("base", [200, 200, 200]))
            accent = tuple(int(x) for x in colors.get("accent", [255, 255, 255]))
            pattern = self.insect_params.get(sp_key, {}).get("chirp_pattern", {}).get("default", [])
            duration = 0.0
            if pattern:
                duration = float(pattern[-1][0])
            else:
                pattern = [(0.0, 1.0), (0.2, 0.0)]
                duration = 0.2
            # Chirp interval (more frequent by default)
            sim_cfg = self.cfg.get("sim", {})
            chirp_min = float(sim_cfg.get("chirp_interval_min_s", 3.0))
            chirp_max = float(sim_cfg.get("chirp_interval_max_s", 12.0))
            self.insects.append({
                "indices": list(range(start, end)),
                "sp_key": sp_key,
                "base": base,
                "accent": accent,
                "pattern": pattern,
                "period": duration,
                "active": False,
                "start_t": 0.0,
                "next_t": time.time() + random.uniform(chirp_min, chirp_max),
                "chirp_min": chirp_min,
                "chirp_max": chirp_max,
            })

        # Wave state
        wave_running = False
        wave_t0 = 0.0
        wave_duration = sum(self.cfg["segments_m"].values()) / max(1e-6, self.speed)

        # Audio/light state machine (visualized only)
        state = "IDLE"  # IDLE -> SILENCE -> WAVE -> RESUME -> IDLE
        silence_until = 0.0
        resume_t0 = 0.0
        resume_time = 2.0  # seconds to fade back lights

        # Kakon schedule
        mean_kakon = float(self.cfg.get("sim", {}).get("kakon_mean_s", 8.0))
        std_kakon = float(self.cfg.get("sim", {}).get("kakon_std_s", 2.0))
        def schedule_kakon(now: float) -> float:
            return now + max(3.0, random.gauss(mean_kakon, std_kakon))
        next_kakon_at = schedule_kakon(time.time())

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Manual trigger with SPACE
                    if event.key == pygame.K_SPACE and state == "IDLE" and not wave_running:
                        state = "SILENCE"
                        silence_until = time.time() + max(0.1, float(self.cfg["wave"].get("pause_ms", 900)) / 1000.0)

            now = time.time()
            # Schedule kakon -> SILENCE
            if state == "IDLE" and not wave_running and now >= next_kakon_at:
                state = "SILENCE"
                silence_until = now + max(0.1, float(self.cfg["wave"].get("pause_ms", 900)) / 1000.0)
                next_kakon_at = schedule_kakon(now)

            # Transition SILENCE -> WAVE
            if state == "SILENCE" and now >= silence_until and not wave_running:
                wave_running = True
                wave_t0 = now
                state = "WAVE"

            screen.fill((10, 10, 10))

            # Draw LEDs
            if wave_running:
                t = now - wave_t0
                pos01 = min(1.0, t / wave_duration)
                # Two-front wave: from both ends toward center
                half_span = (self.total_leds - 1) / 2.0
                front_offset = pos01 * half_span
                c1 = front_offset
                c2 = (self.total_leds - 1) - front_offset
                for i, (x, y) in enumerate(pts):
                    d1 = abs(i - c1)
                    d2 = abs(i - c2)
                    d = min(d1, d2)
                    w = max(0.0, 1.0 - d / float(self.tail_leds))
                    if w > 0:
                        col = (
                            int(ORANGE[0] * w),
                            int(ORANGE[1] * w),
                            int(ORANGE[2] * w),
                        )
                        pygame.draw.circle(screen, col, (x, y), 3)
                if pos01 >= 1.0:
                    wave_running = False
                    state = "RESUME"
                    resume_t0 = now
            else:
                # Insect-driven blinking (no background when idle). Off in SILENCE/WAVE.
                if state == "RESUME":
                    amp_gate = max(0.0, min(1.0, (now - resume_t0) / resume_time))
                    if amp_gate >= 1.0:
                        state = "IDLE"
                elif state == "SILENCE":
                    amp_gate = 0.0
                else:
                    amp_gate = 1.0

                for insect in self.insects:
                    # Schedule chirps (occasional)
                    if state == "IDLE" and not insect["active"] and now >= insect["next_t"]:
                        insect["active"] = True
                        insect["start_t"] = now
                        period = max(0.1, insect["period"]) if insect["period"] else 0.2
                        insect["next_t"] = now + random.uniform(insect["chirp_min"], insect["chirp_max"]) + period
                        # trigger short sound
                        snd = self.species_sounds.get(insect["sp_key"]) if self.species_sounds else None
                        if snd:
                            ch = pygame.mixer.find_channel(True)
                            if ch:
                                ch.set_volume(min(1.0, self.master_amp))
                                ch.play(snd, maxtime=int(period * 1000))

                    if insect["active"]:
                        t_local = now - insect["start_t"]
                        pat = insect["pattern"]
                        period = max(0.1, insect["period"]) if insect["period"] else 0.2
                        # sample level from pattern
                        level = 0.0
                        for i_seg in range(len(pat) - 1):
                            t0, v0 = pat[i_seg]
                            t1, v1 = pat[i_seg + 1]
                            if t_local < t0:
                                break
                            if t0 <= t_local <= t1:
                                if t1 > t0:
                                    a = (t_local - t0) / (t1 - t0)
                                    level = v0 + (v1 - v0) * a
                                else:
                                    level = v1
                                break
                            level = v1
                        if t_local >= period:
                            insect["active"] = False
                            level = 0.0

                        if amp_gate > 0.0 and level > 0.0:
                            lv = max(0.0, min(1.0, float(level))) * amp_gate
                            r = int(insect["base"][0] * (1 - lv) + insect["accent"][0] * lv)
                            g = int(insect["base"][1] * (1 - lv) + insect["accent"][1] * lv)
                            b = int(insect["base"][2] * (1 - lv) + insect["accent"][2] * lv)
                            col = (r, g, b)
                            for led_i in insect["indices"]:
                                x, y = pts[led_i]
                                pygame.draw.circle(screen, col, (x, y), 3)

            # Sounds are per-chirp; no global fade needed

            # HUD
            # We avoid font init for simplicity; draw a simple progress bar for next kakon
            rem = max(0.0, next_kakon_at - now)
            bar_w = int(self.view_w * max(0.0, min(1.0, 1.0 - rem / 40.0)))
            pygame.draw.rect(screen, (255, 120, 0), (0, 0, bar_w, 6))

            # simple state indicator light
            state_col = {"IDLE": (50, 200, 50), "SILENCE": (200, 50, 50), "WAVE": (255, 165, 0), "RESUME": (80, 160, 255)}.get(state, (200, 200, 200))
            pygame.draw.circle(screen, state_col, (self.view_w - 20, 20), 8)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
