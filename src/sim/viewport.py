from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple, Any

import os
import importlib.util
import pygame
import numpy as np


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


from ..serial_handler import SerialWriterThread


class Viewport:
    def __init__(self, cfg: Dict, layout_idx: List[Dict], mirror_to_device: bool = False):
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
            # support new single color or legacy base
            base = tuple(int(x) for x in (colors.get("color") or colors.get("base", [200, 200, 200])))
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
                "color": base,
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
        self.wave_base_duration = sum(self.cfg["segments_m"].values()) / max(1e-6, self.speed)
        self.kakon_wave_speed_factor = float(self.cfg.get("sim", {}).get("kakon_wave_speed_factor", 3.0))
        self.effect_version = int(self.cfg.get("sim", {}).get("effect_version", 1))
        calm = self.cfg.get("sim", {}).get("calm_blue_rgb", [80, 140, 255])
        self.calm_blue = (int(calm[0]), int(calm[1]), int(calm[2]))
        self.calm_hold_s = float(self.cfg.get("sim", {}).get("calm_hold_s", 5.0))
        self.disable_calm_blue = bool(self.cfg.get("sim", {}).get("disable_calm_blue", False))

        # Audio/light state machine (visualized only)
        state = "IDLE"  # IDLE -> SILENCE -> WAVE -> RESUME -> IDLE
        silence_until = 0.0
        resume_t0 = 0.0
        resume_time = float(self.cfg.get("sim", {}).get("resume_seconds", 6.0))
        resume_min_gate = float(self.cfg.get("sim", {}).get("resume_min_gate", 0.15))
        insect_wave_hz = float(self.cfg.get("sim", {}).get("insect_wave_hz", 0.08))
        insect_wave_phase = float(self.cfg.get("sim", {}).get("insect_wave_phase", 0.0))

        # Kakon schedule
        mean_kakon = float(self.cfg.get("sim", {}).get("kakon_mean_s", 8.0))
        std_kakon = float(self.cfg.get("sim", {}).get("kakon_std_s", 2.0))
        def schedule_kakon(now: float) -> float:
            return now + max(3.0, random.gauss(mean_kakon, std_kakon))
        next_kakon_at = schedule_kakon(time.time())

        # Prepare simple shishi-odoshi sound (procedural)
        self.kakon_sound = None
        try:
            self.kakon_sound = self._make_kakon_sound()
        except Exception:
            self.kakon_sound = None

        # Track active chirp audio channels to stop on SILENCE
        self.active_channels: List[pygame.mixer.Channel] = []
        # Concurrency and species gating after kakon
        sim_cfg2 = self.cfg.get("sim", {})
        # Back-compat: if old key exists, use it as total
        if "max_concurrent_chirps" in sim_cfg2:
            self.max_concurrent_total = int(sim_cfg2.get("max_concurrent_chirps", 12))
        else:
            self.max_concurrent_total = int(sim_cfg2.get("max_concurrent_total", 24))
        self.max_concurrent_per_species = int(sim_cfg2.get("max_concurrent_per_species", 12))
        self.species_keys = list(self.insect_params.keys()) or [
            "aomatsumushi", "kutsuwa", "matsumushi", "umaoi", "koorogi", "kirigirisu", "suzumushi",
        ]
        # Only N species active at a time (many individuals allowed). Reset each kakon.
        self.active_species_count = int(self.cfg.get("sim", {}).get("active_species_count", 2))
        self.current_species_set = set(random.sample(self.species_keys, min(self.active_species_count, len(self.species_keys))))

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
                # play kakon sound and stop any active chirp sounds for immediate silence
                if self.kakon_sound:
                    try:
                        ch = pygame.mixer.find_channel(True)
                        if ch:
                            ch.set_volume(0.9)
                            ch.play(self.kakon_sound)
                    except Exception:
                        pass
                if self.active_channels:
                    for ch in self.active_channels:
                        try:
                            ch.stop()
                        except Exception:
                            pass
                    self.active_channels.clear()
                # Choose N species for this session
                self.current_species_set = set(random.sample(self.species_keys, min(self.active_species_count, len(self.species_keys))))

            # Transition SILENCE -> WAVE
            if state == "SILENCE" and now >= silence_until and not wave_running:
                wave_running = True
                wave_t0 = now
                state = "WAVE"

            screen.fill((10, 10, 10))

            # Draw LEDs
            # Prepare hardware frame buffer (per-LED RGB)
            frame = np.zeros((self.total_leds, 3), dtype=np.uint8)
            if wave_running:
                t = now - wave_t0
                # For effect 1 use wave duration, for effect 2 use calm_hold_s
                if self.effect_version == 1:
                    wave_duration = max(1e-6, self.wave_base_duration / max(0.1, self.kakon_wave_speed_factor))
                else:
                    wave_duration = max(1e-6, self.calm_hold_s)
                pos01 = min(1.0, t / wave_duration)
                if self.effect_version == 1:
                    # Two-front orange wave
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
                else:
                    # Calm blue glow: hold blue for calm_hold_s (optional)
                    if not self.disable_calm_blue:
                        a = 1.0
                        blue = self.calm_blue
                        col = (int(blue[0] * a), int(blue[1] * a), int(blue[2] * a))
                        for i, (x, y) in enumerate(pts):
                            pygame.draw.circle(screen, col, (x, y), 3)
                            frame[i] = col
                if pos01 >= 1.0:
                    wave_running = False
                    state = "RESUME"
                    resume_t0 = now
                    # Re-seed next chirp times near-future so it doesn't stay silent
                    for insect in self.insects:
                        insect["next_t"] = now + random.uniform(0.1, 1.0)
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

                # Only 3 species are allowed throughout (until next kakon)
                if state == "SILENCE":
                    allowed_species = set()
                else:
                    allowed_species = set(self.current_species_set)

                # Limit concurrent chirps overall and per species; during RESUME grow gradually
                total_active = sum(1 for ins in self.insects if ins["active"])
                # Wave-like modulation of allowed totals
                wave_mod = 1.0
                if insect_wave_hz > 0.0:
                    wave_mod = 0.7 + 0.3 * math.sin(2 * math.pi * insect_wave_hz * (now if state == "IDLE" else (now - resume_t0)) + insect_wave_phase)
                    wave_mod = max(0.4, min(1.0, wave_mod))
                if state == "IDLE":
                    allowed_total = max(1, int(self.max_concurrent_total * wave_mod))
                else:
                    allowed_total = max(1, int(self.max_concurrent_total * max(resume_min_gate, amp_gate) * wave_mod))
                # per-species counts
                counts_by_sp: Dict[str, int] = {}
                for ins in self.insects:
                    if ins["active"]:
                        counts_by_sp[ins["sp_key"]] = counts_by_sp.get(ins["sp_key"], 0) + 1
                if state == "IDLE":
                    allowed_per_sp = max(1, int(self.max_concurrent_per_species * wave_mod))
                else:
                    allowed_per_sp = max(1, int(self.max_concurrent_per_species * max(resume_min_gate, amp_gate) * wave_mod))

                for insect in self.insects:
                    # Schedule chirps (occasional)
                    if (state in ("IDLE", "RESUME")) and (not insect["active"]) and (now >= insect["next_t"]):
                        sp = insect["sp_key"]
                        if (total_active >= allowed_total) or (sp not in allowed_species) or (counts_by_sp.get(sp, 0) >= allowed_per_sp):
                            # Retry a bit later
                            insect["next_t"] = now + random.uniform(insect["chirp_min"], insect["chirp_max"]) * 0.3
                        else:
                            insect["active"] = True
                            insect["start_t"] = now
                            period = max(0.1, insect["period"]) if insect["period"] else 0.2
                            insect["next_t"] = now + random.uniform(insect["chirp_min"], insect["chirp_max"]) + period
                            total_active += 1
                            counts_by_sp[sp] = counts_by_sp.get(sp, 0) + 1
                            # trigger short sound
                            snd = self.species_sounds.get(insect["sp_key"]) if self.species_sounds else None
                            if snd:
                                ch = pygame.mixer.find_channel(True)
                                if ch:
                                    # Volume increases with number of active insects; quieter during RESUME
                                    vol_factor = 0.3 + 0.7 * (total_active / float(max(1, self.max_concurrent_total)))
                                    vol = max(0.0, min(1.0, self.master_amp * (vol_factor * (amp_gate if state == "RESUME" else 1.0))))
                                    ch.set_volume(vol)
                                    ch.play(snd, maxtime=int(period * 1000))
                                    self.active_channels.append(ch)

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
                            col = (
                                int(insect["color"][0] * lv),
                                int(insect["color"][1] * lv),
                                int(insect["color"][2] * lv),
                            )
                            for led_i in insect["indices"]:
                                x, y = pts[led_i]
                                pygame.draw.circle(screen, col, (x, y), 3)
                                frame[led_i] = col

            # Sounds are per-chirp; no global fade needed

            # Mirror to device: convert per-LED frame to pixel buffer and send
            if self.serial_thread is not None:
                try:
                    num_pixels = (self.total_leds + 2) // 3
                    # Map LED i -> pixel i (Arduino expands each pixel to 3 LEDs)
                    pixel_buf = frame[:num_pixels]
                    self.serial_thread.send(pixel_buf)
                except Exception:
                    pass

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

        # On exit: send BLACK and close serial
        if self.serial_thread is not None:
            try:
                self.serial_thread.send(np.zeros(((self.total_leds + 2)//3, 3), dtype=np.uint8))
                time.sleep(0.1)
                self.serial_thread.close()
            except Exception:
                pass
        pygame.quit()

    def _make_kakon_sound(self) -> pygame.mixer.Sound:
        import numpy as np
        import pygame.sndarray as sndarray
        sr = 44100
        # helper envelopes and tones
        def env_decay(n, tau):
            t = np.arange(n, dtype=np.float32)
            return np.exp(-t / float(tau)).astype(np.float32)
        def tone(freq, dur_s, amp=0.5):
            n = int(sr * dur_s)
            t = np.arange(n, dtype=np.float32) / sr
            w = np.sin(2 * np.pi * freq * t).astype(np.float32)
            return (amp * w)
        # compose 'ka' then 'kon'
        x1 = tone(220, 0.05, 0.9) * env_decay(int(sr * 0.05), 900)
        gap = np.zeros(int(sr * 0.06), dtype=np.float32)
        x2 = tone(1800, 0.03, 0.6) * env_decay(int(sr * 0.03), 300)
        x = np.concatenate([x1, gap, x2]).astype(np.float32)
        # click transient
        if len(x) > 6:
            x[:6] += np.array([1.0, -0.9, 0.7, -0.5, 0.3, -0.2], dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        xi16 = (x * 32767).astype(np.int16)
        return sndarray.make_sound(xi16)

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
