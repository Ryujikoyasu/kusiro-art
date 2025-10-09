from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple, Any

import os
import importlib.util
import pygame
import numpy as np


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


from ..serial_handler import SerialWriterThread


class Viewport:
    def __init__(self, cfg: Dict, layout_idx: List[Dict], mirror_to_device: bool = False):
        self.cfg = cfg
        self.idx = layout_idx
        self.total_leds = len(layout_idx)

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
                # 44.1kHz, 16-bit, mono to match generated kakon sound
                pygame.mixer.init(frequency=44100, size=-16, channels=1)
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

        # Optional hardware mirroring (MAGIC 0x7E frames)
        self.mirror = mirror_to_device
        self.serial_thread: SerialWriterThread | None = None
        if self.mirror:
            try:
                port = cfg.get('serial_port')
                baud = int(cfg.get('baud', 115200))
                num_pixels = (self.total_leds + 2) // 3
                self.serial_thread = SerialWriterThread(port, baud, 0x7E, num_pixels)
                self.serial_thread.start()
            except Exception:
                self.serial_thread = None

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

    # Wave/calm blue simulation and kakon logic removed.

    def run_ambient_simulation(self,
                               species_count: int = 2,
                               change_interval: float = 60.0,
                               wave_seconds: float = 20.0,
                               max_total_override: int | None = None,
                               max_per_species_override: int | None = None,
                               interval_scale: float = 1.0):
        import math
        pygame.init()
        screen = pygame.display.set_mode((self.view_w, self.view_h))
        clock = pygame.time.Clock()

        # Precompute pixel positions
        pts = [self._pos_for_index(e) for e in self.idx]

        # Build insects (LED groups) same as run_simulation
        group_size = 12
        insects = []
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
            base = tuple(int(x) for x in (colors.get("color") or colors.get("base", [200, 200, 200])))
            pattern = self.insect_params.get(sp_key, {}).get("chirp_pattern", {}).get("default", [])
            duration = 0.0
            if pattern:
                duration = float(pattern[-1][0])
            else:
                pattern = [(0.0, 1.0), (0.2, 0.0)]
                duration = 0.2
            sim_cfg = self.cfg.get("sim", {})
            chirp_min = float(sim_cfg.get("chirp_interval_min_s", 3.0)) * max(0.05, float(interval_scale))
            chirp_max = float(sim_cfg.get("chirp_interval_max_s", 12.0)) * max(0.05, float(interval_scale))
            insects.append({
                "indices": list(range(start, end)),
                "sp_key": sp_key,
                "color": base,
                "pattern": pattern,
                "period": duration or 0.2,
                "active": False,
                "start_t": 0.0,
                "next_t": time.time() + random.uniform(chirp_min, chirp_max),
                "chirp_min": chirp_min,
                "chirp_max": chirp_max,
            })

        # Concurrency limits from config
        sim_cfg2 = self.cfg.get("sim", {})
        if "max_concurrent_chirps" in sim_cfg2:
            max_concurrent_total = int(sim_cfg2.get("max_concurrent_chirps", 12))
        else:
            max_concurrent_total = int(sim_cfg2.get("max_concurrent_total", 24))
        max_concurrent_per_species = int(sim_cfg2.get("max_concurrent_per_species", 12))
        if isinstance(max_total_override, int) and max_total_override > 0:
            max_concurrent_total = max_total_override
        if isinstance(max_per_species_override, int) and max_per_species_override > 0:
            max_concurrent_per_species = max_per_species_override

        # Active species selection and rotation
        species_count = max(1, int(species_count))
        species_count = min(species_count, len(species_keys))
        current_species_set = set(random.sample(species_keys, species_count))
        next_change_at = time.time() + max(1.0, float(change_interval))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time()
            # Rotate species set
            if now >= next_change_at:
                current_species_set = set(random.sample(species_keys, species_count))
                next_change_at = now + max(1.0, float(change_interval))

            # LFO for 10s volume/concurrency wave
            phase = (now % max(0.1, float(wave_seconds))) / max(0.1, float(wave_seconds))
            amp = 0.2 + 0.8 * (0.5 * (1.0 + math.sin(2 * math.pi * phase)))  # 0.2..1.0

            screen.fill((10, 10, 10))
            frame = np.zeros((self.total_leds, 3), dtype=np.uint8)

            # Compute allowed totals
            total_active = sum(1 for ins in insects if ins["active"])
            allowed_total = max(1, int(max_concurrent_total * amp))
            counts_by_sp: Dict[str, int] = {}
            for ins in insects:
                if ins["active"]:
                    counts_by_sp[ins["sp_key"]] = counts_by_sp.get(ins["sp_key"], 0) + 1
            allowed_per_sp = max(1, int(max_concurrent_per_species * amp))

            # Schedule chirps
            started_this_tick = 0
            for ins in insects:
                if not ins["active"] and now >= ins["next_t"] and (ins["sp_key"] in current_species_set):
                    sp = ins["sp_key"]
                    if (total_active + started_this_tick) >= allowed_total or counts_by_sp.get(sp, 0) >= allowed_per_sp:
                        ins["next_t"] = now + random.uniform(ins["chirp_min"], ins["chirp_max"]) * 0.3
                    else:
                        ins["active"] = True
                        ins["start_t"] = now
                        period = max(0.1, ins["period"]) if ins["period"] else 0.2
                        ins["next_t"] = now + random.uniform(ins["chirp_min"], ins["chirp_max"]) + period
                        started_this_tick += 1
                        counts_by_sp[sp] = counts_by_sp.get(sp, 0) + 1
                        # Sound
                        snd = self.species_sounds.get(sp) if self.species_sounds else None
                        if snd:
                            ch = pygame.mixer.find_channel(True)
                            if ch:
                                density = min(1.0, (total_active / float(max(1, max_concurrent_total))))
                                vol = max(0.0, min(1.0, self.master_amp * (0.2 + 0.8 * (0.5 * amp + 0.5 * density))))
                                ch.set_volume(vol)
                                ch.play(snd, maxtime=int(period * 1000))

            # Draw active insects
            for ins in insects:
                if ins["active"]:
                    t_local = now - ins["start_t"]
                    pat = ins["pattern"]
                    period = max(0.1, ins["period"]) if ins["period"] else 0.2
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
                        ins["active"] = False
                        level = 0.0

                    if level > 0.0:
                        lv = max(0.0, min(1.0, float(level) * amp))
                        col = (
                            int(ins["color"][0] * lv),
                            int(ins["color"][1] * lv),
                            int(ins["color"][2] * lv),
                        )
                        for led_i in ins["indices"]:
                            x, y = pts[led_i]
                            pygame.draw.circle(screen, col, (x, y), 3)
                            frame[led_i] = col

            # Mirror to device
            if self.serial_thread is not None:
                try:
                    num_pixels = (self.total_leds + 2) // 3
                    pixel_buf = frame[:num_pixels]
                    self.serial_thread.send(pixel_buf)
                except Exception:
                    pass

            # HUD: species rotation progress and LFO
            rem = max(0.0, next_change_at - now)
            bar_w = int(self.view_w * max(0.0, min(1.0, 1.0 - rem / max(1.0, float(change_interval)))))
            pygame.draw.rect(screen, (80, 160, 255), (0, 0, bar_w, 6))
            # Indicator of amp
            amp_w = int(self.view_w * amp)
            pygame.draw.rect(screen, (255, 165, 0), (0, 8, amp_w, 4))

            pygame.display.flip()
            clock.tick(60)

        # On exit: send BLACK and close serial
        if self.serial_thread is not None:
            try:
                self.serial_thread.send(np.zeros(((self.total_leds + 2)//3, 3), dtype=np.uint8))
                time.sleep(0.1)
                self.serial_thread.close()
            except Exception:
                pass
        pygame.quit()
