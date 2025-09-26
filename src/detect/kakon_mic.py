from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple

import numpy as np


@dataclass
class MicParams:
    samplerate: int = 22050
    blocksize: int = 1024
    band_hz: Tuple[float, float] = (1500.0, 8000.0)
    threshold_db: float = -12.0
    release_db: float = -16.0
    min_interval_ms: float = 1600.0
    device: Optional[str] = None  # None means default


def _cfg_to_params(cfg: Dict) -> MicParams:
    c = cfg.get("detect", {}).get("mic", {})
    # Backward compat: original config had only threshold_db/min_interval_ms
    sr = int(c.get("samplerate", 22050))
    bs = int(c.get("blocksize", 1024))
    band = tuple(float(x) for x in c.get("band_hz", [1500.0, 8000.0]))  # type: ignore
    th = float(c.get("threshold_db", -12.0))
    rel = float(c.get("release_db", th - 4.0))
    mi = float(c.get("min_interval_ms", 1600.0))
    dev = c.get("device")
    return MicParams(sr, bs, (band[0], band[1]), th, rel, mi, dev)  # type: ignore[arg-type]


class KakonMicDetector:
    """Detects bell-like 'shaka shaka' by monitoring high-frequency energy from mic.

    Implementation details:
      - Captures mono float32 blocks using sounddevice
      - Applies Hann window and computes magnitude FFT
      - Aggregates RMS power within band_hz
      - Converts to dBFS and applies hysteresis (threshold_db/release_db)
      - Enforces min_interval_ms between fires
    """

    def __init__(self, cfg: Dict):
        try:
            import sounddevice as sd  # noqa: F401
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "sounddevice is required for mic detection. Install with `pip install sounddevice`"
            )
        self.params = _cfg_to_params(cfg)
        self._armed = True
        self._last_fire = 0.0

    def _stream_iter(self):
        import sounddevice as sd

        p = self.params
        # 1 channel, float32 in [-1, 1]
        with sd.InputStream(
            samplerate=p.samplerate,
            channels=1,
            dtype="float32",
            blocksize=p.blocksize,
            device=p.device,
        ) as stream:
            while True:
                data, _ = stream.read(p.blocksize)
                yield np.squeeze(data).astype(np.float32)

    def _block_db(self, x: np.ndarray) -> float:
        p = self.params
        n = x.shape[0]
        if n == 0:
            return -120.0
        # Window + FFT
        w = np.hanning(n).astype(np.float32)
        X = np.fft.rfft(x * w)
        mag = np.abs(X).astype(np.float32)
        freqs = np.fft.rfftfreq(n, d=1.0 / p.samplerate)
        # Band mask
        lo, hi = p.band_hz
        m = (freqs >= lo) & (freqs <= hi)
        band_energy = float(np.mean((mag[m] ** 2))) if np.any(m) else 0.0
        # Convert to dBFS (reference 1.0 RMS)
        db = 10.0 * np.log10(max(band_energy, 1e-12))
        return db

    def watch(self) -> Generator[None, None, None]:
        p = self.params
        min_interval_s = p.min_interval_ms / 1000.0
        for block in self._stream_iter():
            db = self._block_db(block)
            now = time.time()
            if self._armed and db >= p.threshold_db and (now - self._last_fire) >= min_interval_s:
                self._last_fire = now
                self._armed = False
                yield None
            elif (not self._armed) and db <= p.release_db:
                self._armed = True

