from __future__ import annotations

import csv as _csv
import sys
import time
from typing import Dict, Optional

import numpy as np

from ..detect.kakon_mic import _cfg_to_params


def _meter(db: float, lo: float, hi: float, width: int = 40) -> str:
    clamped = max(lo, min(hi, db))
    frac = (clamped - lo) / max(1e-6, (hi - lo))
    n = int(frac * width)
    return f"[{('#' * n).ljust(width)}] {db:6.1f} dB"


def run_tuner(cfg: Dict, duration: float = 0.0, csv_path: Optional[str] = None) -> None:
    try:
        import sounddevice as sd
    except Exception as e:
        print("sounddevice is required: pip install sounddevice", file=sys.stderr)
        raise

    p = _cfg_to_params(cfg)

    lo = p.release_db - 20.0
    hi = p.threshold_db + 6.0

    writer = None
    f = None
    if csv_path:
        f = open(csv_path, "w", newline="", encoding="utf-8")
        writer = _csv.writer(f)
        writer.writerow(["time_s", "db_high_band"])  # header

    def block_db(x: np.ndarray) -> float:
        n = x.shape[0]
        if n == 0:
            return -120.0
        w = np.hanning(n).astype(np.float32)
        X = np.fft.rfft(x * w)
        mag = np.abs(X).astype(np.float32)
        freqs = np.fft.rfftfreq(n, d=1.0 / p.samplerate)
        m = (freqs >= p.band_hz[0]) & (freqs <= p.band_hz[1])
        e = float(np.mean((mag[m] ** 2))) if np.any(m) else 0.0
        return 10.0 * np.log10(max(e, 1e-12))

    print("Mic tuner: Ctrl+C to stop. Adjust detect.mic.* in config.yaml")
    print(f"Device={p.device or 'default'} SR={p.samplerate} block={p.blocksize} band={p.band_hz} ")
    print(f"Threshold={p.threshold_db} dB, Release={p.release_db} dB, MinInterval={p.min_interval_ms} ms")

    t_end = time.time() + duration if duration > 0 else float("inf")
    last_print = 0.0

    with sd.InputStream(samplerate=p.samplerate, channels=1, dtype="float32", blocksize=p.blocksize, device=p.device) as stream:
        try:
            while time.time() < t_end:
                data, _ = stream.read(p.blocksize)
                x = np.squeeze(data).astype(np.float32)
                db = block_db(x)
                now = time.time()
                if writer:
                    writer.writerow([f"{now:.3f}", f"{db:.2f}"])
                # Limit console rate
                if now - last_print >= 0.03:
                    last_print = now
                    bar = _meter(db, lo, hi)
                    thr = f" | TH={p.threshold_db:.1f} REL={p.release_db:.1f}"
                    print("\r" + bar + thr, end="", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            if f:
                f.close()
            print()

