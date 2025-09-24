from __future__ import annotations

import os
import time
import random
from typing import Dict, List

import pygame

from .util.config import load_config
from .led.mapper import build_u_shape_idx
from .led.serial_link import SerialLink


def _load_insect_params() -> Dict:
    from pathlib import Path
    import importlib.util
    here = Path(__file__).resolve()
    repo = here.parents[1]
    cfg_path = repo / "config" / "insect_config.py"
    if cfg_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("insect_config", str(cfg_path))
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


def _make_kakon_sound() -> pygame.mixer.Sound | None:
    try:
        import numpy as np
        import pygame.sndarray as sndarray
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        sr = 44100
        def env_decay(n, tau):
            import numpy as np
            t = np.arange(n, dtype=np.float32)
            return np.exp(-t / float(tau)).astype(np.float32)
        def tone(freq, dur_s, amp=0.5):
            import numpy as np
            n = int(sr * dur_s)
            t = np.arange(n, dtype=np.float32) / sr
            w = np.sin(2 * np.pi * freq * t).astype(np.float32)
            return (amp * w)
        x1 = tone(220, 0.05, 0.9) * env_decay(int(sr * 0.05), 900)
        gap = (0.0 * tone(100, 0.06)).astype("float32")
        x2 = tone(1800, 0.03, 0.6) * env_decay(int(sr * 0.03), 300)
        import numpy as np
        x = np.concatenate([x1, gap, x2]).astype(np.float32)
        if len(x) > 6:
            x[:6] += np.array([1.0, -0.9, 0.7, -0.5, 0.3, -0.2], dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        xi16 = (x * 32767).astype(np.int16)
        return sndarray.make_sound(xi16)
    except Exception:
        return None


def run(effect_version: int | None = None):
    cfg = load_config()
    idx = build_u_shape_idx(cfg)
    total = len(idx)
    link = SerialLink(cfg)

    # LED wave timings
    speed = float(cfg["wave"]["speed_mps"])  # m/s
    total_m = sum(cfg["segments_m"].values())
    base_duration = total_m / max(1e-6, speed)
    tail_m = float(cfg["wave"]["tail_m"])  # m
    speed_factor = float(cfg.get("sim", {}).get("kakon_wave_speed_factor", 3.0))
    if effect_version is None:
        effect_version = int(cfg.get("sim", {}).get("effect_version", 1))

    # Audio: insect orchestration using simple mixer
    insect_params = _load_insect_params()
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        pygame.mixer.set_num_channels(64)
    except Exception:
        pass
    species_sounds: Dict[str, pygame.mixer.Sound] = {}
    for sp, v in insect_params.items():
        p = v.get("sound_files", {}).get("default")
        if p and os.path.exists(p):
            try:
                species_sounds[sp] = pygame.mixer.Sound(p)
            except Exception:
                pass
    master_db = float(cfg.get("audio", {}).get("gain_master", -6.0))
    master_amp = 10 ** (master_db / 20.0)

    # Parameters
    simcfg = cfg.get("sim", {})
    chirp_min = float(simcfg.get("chirp_interval_min_s", 3.0))
    chirp_max = float(simcfg.get("chirp_interval_max_s", 12.0))
    max_total = int(simcfg.get("max_concurrent_total", 24))
    max_per_sp = int(simcfg.get("max_concurrent_per_species", 12))
    mean_kakon = float(simcfg.get("kakon_mean_s", 8.0))
    std_kakon = float(simcfg.get("kakon_std_s", 2.0))

    species_keys = list(insect_params.keys()) or [
        "aomatsumushi", "kutsuwa", "matsumushi", "umaoi", "koorogi", "kirigirisu", "suzumushi",
    ]
    current_species = set(random.sample(species_keys, min(3, len(species_keys))))

    # Prepare kakon sound
    kakon_sound = _make_kakon_sound()

    def schedule_kakon(now: float) -> float:
        return now + max(3.0, random.gauss(mean_kakon, std_kakon))

    # Keep per-individual timers: simulate many individuals by many slots per species
    slots: List[Dict] = []
    per_species_slots = max_per_sp  # approximate
    for sp in species_keys:
        for _ in range(per_species_slots):
            slots.append({
                "sp": sp,
                "next": time.time() + random.uniform(chirp_min, chirp_max),
                "period": float(insect_params.get(sp, {}).get("chirp_pattern", {}).get("default", [[0, 0], [0.2, 0]])[-1][0]) if insect_params else 0.2,
            })

    with link:
        link.send_conf(total=total)
        print("real-run: connected. Ctrl+C to stop. Auto-scheduling kakon and chirps.")
        next_kakon = schedule_kakon(time.time())
        state = "IDLE"
        silence_until = 0.0
        try:
            while True:
                now = time.time()
                # Handle kakon
                if state == "IDLE" and now >= next_kakon:
                    state = "SILENCE"
                    silence_until = now + max(0.1, float(cfg["wave"].get("pause_ms", 900)) / 1000.0)
                    next_kakon = schedule_kakon(now)
                    # play kakon and stop all current sounds
                    if kakon_sound:
                        try:
                            ch = pygame.mixer.find_channel(True)
                            if ch:
                                ch.set_volume(1.0)
                                ch.play(kakon_sound)
                        except Exception:
                            pass
                    try:
                        pygame.mixer.stop()
                    except Exception:
                        pass
                if state == "SILENCE" and now >= silence_until:
                    t0 = time.time()
                    duration = max(1e-6, base_duration / max(0.1, speed_factor))
                    if effect_version == 1:
                        # Wave (two-front computed on Arduino side as brightness envelope)
                        while True:
                            t = time.time() - t0
                            pos = min(1.0, t / duration)
                            link.send_wave(pos=pos, tail_m=tail_m, bright=220)
                            if pos >= 1.0:
                                break
                            time.sleep(1 / 60.0)
                    else:
                        # Calm blue glow: fill blue, hold, then black
                        blue = (60, 120, 255)
                        # Set once (1200 SET lines). Acceptable per-event.
                        for i in range(total):
                            link.set_pixel(i, blue[0], blue[1], blue[2])
                        hold = duration
                        time.sleep(hold)
                        link.black()
                    # pick new 3 species for next session
                    current_species = set(random.sample(species_keys, min(3, len(species_keys))))
                    state = "RESUME"
                    resume_t0 = time.time()
                # During RESUME/IDLE: schedule chirps
                if state in ("RESUME", "IDLE"):
                    # ramp concurrent limits during RESUME
                    if state == "RESUME":
                        gate = min(1.0, (now - resume_t0) / 2.0)
                    else:
                        gate = 1.0
                    allowed_total = max(1, int(max_total * max(0.1, gate)))
                    allowed_per_sp = max(1, int(max_per_sp * max(0.1, gate)))

                    # count current active by species
                    counts = {k: 0 for k in species_keys}
                    try:
                        # pygame doesn't expose active count easily; approximate by tracking recent starts
                        # We'll just enforce starts against allowed counts.
                        pass
                    except Exception:
                        pass
                    # schedule starts
                    started_this_tick = 0
                    for s in slots:
                        if s["sp"] not in current_species:
                            continue
                        if now < s["next"]:
                            continue
                        if started_this_tick > 8:  # throttle starts per frame
                            break
                        # evaluate current totals by scanning channels (approx: do nothing, rely on allowed_total)
                        if started_this_tick >= allowed_total:
                            break
                        if counts.get(s["sp"], 0) >= allowed_per_sp:
                            s["next"] = now + random.uniform(chirp_min, chirp_max)
                            continue
                        # start chirp
                        snd = species_sounds.get(s["sp"]) if species_sounds else None
                        if snd:
                            ch = pygame.mixer.find_channel(True)
                            if ch:
                                ch.set_volume(min(1.0, master_amp))
                                ch.play(snd, maxtime=int(max(0.1, s["period"]) * 1000))
                                counts[s["sp"]] = counts.get(s["sp"], 0) + 1
                                started_this_tick += 1
                        s["next"] = now + random.uniform(chirp_min, chirp_max)
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    run()
