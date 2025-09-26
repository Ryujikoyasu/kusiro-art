from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np

from ..led.serial_link import SerialLink
from ..led.mapper import build_u_shape_idx
from ..util.config import load_config
from ..detect.kakon_mic import _cfg_to_params as _mic_params


def _meter(db: float, lo: float, hi: float, width: int = 40) -> str:
    clamped = max(lo, min(hi, db))
    frac = (clamped - lo) / max(1e-6, (hi - lo))
    n = int(frac * width)
    return f"[{('#' * n).ljust(width)}] {db:6.1f} dB"


def _flash_led(link: SerialLink, total: int, rgb=(255, 40, 40), count: int = 24, hold_s: float = 0.15):
    n = min(total, max(1, count))
    for i in range(n):
        link.set_pixel(i, rgb[0], rgb[1], rgb[2])
    time.sleep(hold_s)
    link.black()


def run_detect_tune(mode: Optional[str] = None, with_led: bool = False):
    """Tune detector parameters with immediate visual feedback.

    - Mic mode: shows live high-band dB meter, triggers on threshold, flashes LEDs.
    - Cam mode: listens for events and flashes LEDs on detection.
    - Config hot-reload: edits to config.yaml are picked up automatically.
    """
    cfg = load_config()
    if mode is None:
        mode = str(cfg.get("detect", {}).get("mode", "mic")).lower()
    total = 0
    link: Optional[SerialLink] = None
    if with_led:
        try:
            idx = build_u_shape_idx(cfg)
            total = len(idx)
            link = SerialLink(cfg)
            link.__enter__()
            link.send_conf(total=total)
        except Exception as e:
            print(f"LED flashing disabled (serial unavailable): {e}")
            link = None
            total = 0
    try:
        if mode == "mic":
            _mic_tune_loop(cfg, link if with_led else None, total)
        elif mode in ("cam", "auto"):
            _cam_tune_loop(cfg, link if with_led else None, total)
        else:
            print(f"Unsupported mode for tuning: {mode}. Set detect.mode to 'mic' or 'cam'.")
    finally:
        if link is not None:
            link.__exit__(None, None, None)


def _mic_tune_loop(cfg0: Dict, link: Optional[SerialLink], total: int):
    try:
        import sounddevice as sd
    except Exception:
        raise RuntimeError("Mic tuning requires sounddevice. pip install sounddevice")

    cfg = dict(cfg0)
    params = _mic_params(cfg)
    print("Mic detect tune: edit config.yaml (detect.mic.*) to adjust. Hot-reloading...")
    print(f"Device={params.device or 'default'} SR={params.samplerate} block={params.blocksize} band={params.band_hz}")
    print("Press Ctrl+C to stop.")

    cfg_mtime = _mtime_config()
    armed = True
    last_fire = 0.0

    lo = params.release_db - 20.0
    hi = params.threshold_db + 6.0

    def reload_if_changed():
        nonlocal cfg, params, lo, hi
        m2 = _mtime_config()
        if m2 != cfg_mtime:
            # Note: we don't update outer cfg_mtime to keep reload frequent; that's fine for tuning
            cfg = load_config()
            params = _mic_params(cfg)
            lo = params.release_db - 20.0
            hi = params.threshold_db + 6.0
            print(f"\nReloaded config: TH={params.threshold_db} REL={params.release_db} band={params.band_hz} minInt={params.min_interval_ms}ms")

    def block_db(x: np.ndarray) -> float:
        n = x.shape[0]
        if n == 0:
            return -120.0
        w = np.hanning(n).astype(np.float32)
        X = np.fft.rfft(x * w)
        mag = np.abs(X).astype(np.float32)
        freqs = np.fft.rfftfreq(n, d=1.0 / params.samplerate)
        m = (freqs >= params.band_hz[0]) & (freqs <= params.band_hz[1])
        e = float(np.mean((mag[m] ** 2))) if np.any(m) else 0.0
        return 10.0 * np.log10(max(e, 1e-12))

    min_interval_s = params.min_interval_ms / 1000.0
    with sd.InputStream(samplerate=params.samplerate, channels=1, dtype="float32", blocksize=params.blocksize, device=params.device) as stream:
        try:
            last_print = 0.0
            while True:
                data, _ = stream.read(params.blocksize)
                x = np.squeeze(data).astype(np.float32)
                db = block_db(x)
                now = time.time()
                # hot reload occasionally
                if now - last_print > 0.2:
                    reload_if_changed()
                # hysteresis
                if armed and db >= params.threshold_db and (now - last_fire) >= min_interval_s:
                    last_fire = now
                    armed = False
                    print("\nDETECTED (mic)")
                    if link:
                        _flash_led(link, total)
                elif (not armed) and db <= params.release_db:
                    armed = True

                # meter
                if now - last_print >= 0.03:
                    last_print = now
                    bar = _meter(db, lo, hi)
                    thr = f" | TH={params.threshold_db:.1f} REL={params.release_db:.1f}"
                    print("\r" + bar + thr, end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")


def _cam_tune_loop(cfg0: Dict, link: Optional[SerialLink], total: int):
    # For camera, reuse existing detector generator and flash LED on events.
    from ..detect.kakon_cam import KakomCameraDetector
    cfg = dict(cfg0)
    det = KakomCameraDetector(cfg)
    print("Cam detect tune: edit config.yaml (detect.cam.*) to adjust. Press Ctrl+C to stop.")
    try:
        for _ in det.watch():
            print("DETECTED (cam)")
            if link:
                _flash_led(link, total, rgb=(40, 120, 255))
    except KeyboardInterrupt:
        print("\nStopped.")


def _mtime_config() -> float:
    try:
        return os.path.getmtime("config.yaml")
    except Exception:
        return 0.0
