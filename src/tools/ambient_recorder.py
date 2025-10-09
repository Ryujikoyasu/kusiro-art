from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import librosa

from ..util.config import load_config


def _load_insect_params() -> Dict:
    from ..ambient import _load_insect_params as _lp  # reuse same loader
    return _lp()


def run_record(
    out_path: Path,
    seconds: float = 60.0,
    species_count: int = 3,
    change_interval_s: float = 60.0,
    wave_seconds: float = 20.0,
    density: float = 1.0,
):
    """Headless ambient scheduler that mixes insect sounds into a WAV file.

    - Uses the same species selection and LFO gating as the ambient simulation.
    - Loads per-species sound files and mixes chirps into a mono buffer.
    """
    cfg = load_config()
    insect_params = _load_insect_params()

    sr = 44100
    total_samples = int(max(1.0, float(seconds)) * sr)
    mix = np.zeros(total_samples, dtype=np.float32)

    # Preload species audio (mono, sr=44100)
    audio_cache: Dict[str, np.ndarray] = {}
    for sp, v in insect_params.items():
        p = v.get("sound_files", {}).get("default")
        if p and os.path.exists(p):
            try:
                y, _ = librosa.load(p, sr=sr, mono=True)
                audio_cache[sp] = y.astype(np.float32)
            except Exception:
                pass

    # Gains
    master_db = float(cfg.get("audio", {}).get("gain_master", -6.0))
    master_amp = 10 ** (master_db / 20.0)

    # Behavior parameters (scaled by density, similar to viewport.ambient)
    simcfg = cfg.get("sim", {})
    base_total = int(simcfg.get("max_concurrent_total", 24))
    base_per = int(simcfg.get("max_concurrent_per_species", 12))
    max_total = max(1, int(round(base_total * max(0.1, density))))
    max_per_sp = max(1, int(round(base_per * max(0.1, density))))
    chirp_min = float(simcfg.get("chirp_interval_min_s", 3.0)) / max(0.1, density)
    chirp_max = float(simcfg.get("chirp_interval_max_s", 12.0)) / max(0.1, density)

    # Individuals (virtual slots) per species
    species_keys = list(insect_params.keys()) or [
        "aomatsumushi", "kutsuwa", "matsumushi", "umaoi", "koorogi", "kirigirisu", "suzumushi",
    ]
    if species_count < 1:
        species_count = 1
    species_count = min(species_count, len(species_keys))
    # Pick random initial species set (not deterministic order)
    current_species = set(random.sample(species_keys, species_count))
    next_change = time.time() + float(change_interval_s)

    # Per-species pattern period (to bound chirp length)
    period_by_sp: Dict[str, float] = {}
    for sp in species_keys:
        pat = insect_params.get(sp, {}).get("chirp_pattern", {}).get("default", [])
        if pat:
            period_by_sp[sp] = float(pat[-1][0])
        else:
            period_by_sp[sp] = 0.2

    # Virtual slots per species
    slots: List[Dict] = []
    for sp in species_keys:
        for _ in range(max_per_sp):
            slots.append({
                "sp": sp,
                "next": time.time() + np.random.uniform(chirp_min, chirp_max),
            })

    t0 = time.time()
    now = t0
    active_chirps: List[Tuple[str, int]] = []  # (sp, end_sample)

    def seconds_to_sample(ts: float) -> int:
        return int((ts - t0) * sr)

    next_log = 0.0
    try:
        while (now - t0) < float(seconds):
            now = time.time()
            # progress log every ~5s
            elapsed = now - t0
            if elapsed >= next_log:
                pct = min(100, int(100.0 * elapsed / max(1e-6, float(seconds))))
                print(f"[ambient_record] progress: {pct}% ({int(elapsed)}/{int(seconds)}s)")
                next_log = elapsed + 5.0
            if now >= next_change:
                # Rotate to a new random set of species
                if len(species_keys) >= species_count:
                    current_species = set(random.sample(species_keys, species_count))
                next_change = now + float(change_interval_s)

            # LFO for gating
            phase = (now % max(0.1, float(wave_seconds))) / max(0.1, float(wave_seconds))
            amp = 0.2 + 0.8 * (0.5 * (1.0 + math.sin(2 * math.pi * phase)))

            # Compute allowed totals
            # Track counts by species using active_chirps and those we will start this tick
            counts_by_sp: Dict[str, int] = {}
            for sp, end_idx in active_chirps:
                if seconds_to_sample(now) < end_idx:
                    counts_by_sp[sp] = counts_by_sp.get(sp, 0) + 1
            allowed_total = max(1, int(max_total * amp))
            allowed_per_sp = max(1, int(max_per_sp * amp))

            # Schedule starts (limit per tick for stability)
            started = 0
            total_active_now = sum(counts_by_sp.values())
            for s in slots:
                if s["sp"] not in current_species:
                    continue
                if now < s["next"]:
                    continue
                if (total_active_now + started) >= allowed_total:
                    break
                if counts_by_sp.get(s["sp"], 0) >= allowed_per_sp:
                    s["next"] = now + np.random.uniform(chirp_min, chirp_max)
                    continue

                snd = audio_cache.get(s["sp"])  # preloaded audio
                if snd is None or len(snd) == 0:
                    s["next"] = now + np.random.uniform(chirp_min, chirp_max)
                    continue

                # Determine playback length (bounded by pattern period)
                max_len_s = max(0.1, float(period_by_sp.get(s["sp"], 0.2)))
                max_len_samples = int(max_len_s * sr)
                seg = snd[:max_len_samples]

                start_idx = seconds_to_sample(now)
                end_idx = min(total_samples, start_idx + len(seg))
                if start_idx < total_samples:
                    seg_use = seg[: end_idx - start_idx]
                    # Simple volume model similar to simulation
                    density_now = min(1.0, (total_active_now / float(max(1, max_total))))
                    vol = float(master_amp) * (0.2 + 0.8 * (0.5 * float(amp) + 0.5 * float(density_now)))
                    mix[start_idx:end_idx] += np.asarray(seg_use, dtype=np.float32) * float(vol)
                    counts_by_sp[s["sp"]] = counts_by_sp.get(s["sp"], 0) + 1
                    started += 1
                    active_chirps.append((s["sp"], end_idx))

                s["next"] = now + np.random.uniform(chirp_min, chirp_max)
                if started >= 8:
                    break

            # Drop ended chirps from book-keeping
            cur_idx = seconds_to_sample(now)
            if active_chirps:
                active_chirps = [(sp, e) for (sp, e) in active_chirps if e > cur_idx]

            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    # Write WAV (int16)
    mix = np.clip(mix, -1.0, 1.0)
    pcm16 = (mix * 32767.0).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import wave
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    return str(out_path)
