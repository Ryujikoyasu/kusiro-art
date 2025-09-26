from __future__ import annotations

import os
import time
import random
from typing import Dict, List

import pygame

from .util.config import load_config


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


def run_ambient(
    species_count: int = 2,
    change_interval_s: float = 60.0,
    wave_seconds: float = 20.0,
    duration: float | None = None,
):
    """Ambient insect loop: 10s volume waves, rotating species every minute.

    - No kakon detection or LED control.
    - Picks N species and schedules short chirps with a slow LFO controlling
      allowed concurrency and per-chirp volume.
    """
    cfg = load_config()
    insect_params = _load_insect_params()

    # Audio setup
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        pygame.mixer.set_num_channels(64)
    except Exception:
        pass

    # Load species sounds
    species_sounds: Dict[str, pygame.mixer.Sound] = {}
    for sp, v in insect_params.items():
        p = v.get("sound_files", {}).get("default")
        if p and os.path.exists(p):
            try:
                species_sounds[sp] = pygame.mixer.Sound(p)
            except Exception:
                pass

    audio_cfg = cfg.get("audio", {})
    master_db = float(audio_cfg.get("gain_master", -6.0))
    master_amp = 10 ** (master_db / 20.0)

    # Scheduling parameters
    simcfg = cfg.get("sim", {})
    chirp_min = float(simcfg.get("chirp_interval_min_s", 3.0))
    chirp_max = float(simcfg.get("chirp_interval_max_s", 12.0))
    max_total = int(simcfg.get("max_concurrent_total", 24))
    max_per_sp = int(simcfg.get("max_concurrent_per_species", 12))

    # Individuals per species (virtual slots)
    slots: List[Dict] = []
    species_keys = list(insect_params.keys()) or [
        "aomatsumushi", "kutsuwa", "matsumushi", "umaoi", "koorogi", "kirigirisu", "suzumushi",
    ]
    for sp in species_keys:
        # derive typical chirp period from pattern if present
        period = float(insect_params.get(sp, {}).get("chirp_pattern", {}).get("default", [[0, 0], [0.2, 0]])[-1][0]) if insect_params else 0.2
        for _ in range(max_per_sp):
            slots.append({
                "sp": sp,
                "next": time.time() + random.uniform(chirp_min, chirp_max),
                "period": max(0.1, period),
            })

    # Active set of species (rotates)
    if species_count < 1:
        species_count = 1
    if species_count > len(species_keys):
        species_count = len(species_keys)
    current_species = set(random.sample(species_keys, species_count))
    next_change = time.time() + change_interval_s

    # Active chirps (for volume scaling)
    active_chirps: List[Dict] = []  # {sp, end}

    start_t = time.time()
    try:
        while True:
            now = time.time()
            if duration and (now - start_t) >= duration:
                break

            # retire finished chirps
            if active_chirps:
                active_chirps = [c for c in active_chirps if c.get("end", 0.0) > now]

            # rotate species set every change_interval_s
            if now >= next_change:
                current_species = set(random.sample(species_keys, species_count))
                next_change = now + change_interval_s

            # wave LFO over wave_seconds period
            import math
            phase = (now % max(0.1, wave_seconds)) / max(0.1, wave_seconds)
            # 0.2..1.0 amplitude for both gating and volume flavor
            amp = 0.2 + 0.8 * (0.5 * (1.0 + math.sin(2 * math.pi * phase)))

            # allowed concurrent totals modulated by amp
            allowed_total = max(1, int(max_total * amp))
            allowed_per_sp = max(1, int(max_per_sp * amp))

            # count per species currently active
            counts = {k: 0 for k in species_keys}
            for c in active_chirps:
                sp = c.get("sp")
                if sp in counts:
                    counts[sp] += 1

            # schedule starts
            started_this_tick = 0
            for s in slots:
                if s["sp"] not in current_species:
                    continue
                if now < s["next"]:
                    continue
                if started_this_tick > 8:
                    break
                total_now = len(active_chirps) + started_this_tick
                if total_now >= allowed_total:
                    break
                if counts.get(s["sp"], 0) >= allowed_per_sp:
                    s["next"] = now + random.uniform(chirp_min, chirp_max)
                    continue
                snd = species_sounds.get(s["sp"]) if species_sounds else None
                if snd:
                    ch = pygame.mixer.find_channel(True)
                    if ch:
                        # volume grows with active chirps and the LFO amp
                        density = min(1.0, (len(active_chirps) / float(max(1, max_total))))
                        vol = master_amp * (0.2 + 0.8 * max(0.0, min(1.0, 0.5 * amp + 0.5 * density)))
                        ch.set_volume(max(0.0, min(1.0, vol)))
                        dur = max(0.1, s["period"])  # conservative
                        ch.play(snd, maxtime=int(dur * 1000))
                        counts[s["sp"]] = counts.get(s["sp"], 0) + 1
                        started_this_tick += 1
                        active_chirps.append({"sp": s["sp"], "end": now + dur})
                s["next"] = now + random.uniform(chirp_min, chirp_max)

            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    try:
        pygame.mixer.stop()
    except Exception:
        pass
