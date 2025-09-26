from __future__ import annotations

"""
Standalone bell tester (no serial required).

Usage:
  python -m src.tools.bell_tester           # 丸いインジケータが光る（GUI）。コンソールにもメーター表示。
  python -m src.tools.bell_tester --no-gui  # GUIを使わず、コンソールのみ。
  python -m src.tools.bell_tester --no-beep # 検出時のビープを無効化。

Edits to config.yaml (detect.mic.*) are hot-reloaded while running.
"""

import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np

from ..util.config import load_config
from ..detect.kakon_mic import _cfg_to_params


def _mtime_config() -> float:
    try:
        return os.path.getmtime("config.yaml")
    except Exception:
        return 0.0


def _make_beep() -> Tuple[callable, callable]:
    """Return (init_audio, play_beep) using pygame if available; otherwise no-op."""
    def _noop_init():
        pass

    def _noop_play():
        pass

    try:
        import pygame
        import numpy as np
        import pygame.sndarray as sndarray

        def _init():
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1)
            # Precompute a short beep
            sr = 44100
            t = np.arange(int(sr * 0.06), dtype=np.float32) / sr
            w = (np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 80)).astype(np.float32)
            arr = (w * 0.7 * 32767).astype(np.int16)
            snd = sndarray.make_sound(arr)

            def _play():
                ch = pygame.mixer.find_channel(True)
                if ch:
                    ch.set_volume(1.0)
                    ch.play(snd)

            return _play

        play = _init()
        return (lambda: None, play)
    except Exception:
        return (_noop_init, _noop_play)


def _meter(db: float, lo: float, hi: float, width: int = 40) -> str:
    clamped = max(lo, min(hi, db))
    frac = (clamped - lo) / max(1e-6, (hi - lo))
    n = int(frac * width)
    return f"[{('#' * n).ljust(width)}] {db:6.1f} dB"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bell tester (mic-based)")
    ap.add_argument("--gui", dest="gui", action="store_true", help="Show a round flashing indicator window")
    ap.add_argument("--no-gui", dest="gui", action="store_false", help="Disable GUI window")
    ap.set_defaults(gui=True)
    ap.add_argument("--no-beep", action="store_true", help="Disable detection beep")
    args = ap.parse_args(argv)

    try:
        import sounddevice as sd
    except Exception:
        print("sounddevice is required. pip install sounddevice", file=sys.stderr)
        return 2

    cfg = load_config()
    p = _cfg_to_params(cfg)
    lo = p.release_db - 20.0
    hi = p.threshold_db + 6.0

    init_beep, play_beep = _make_beep()
    if not args.no_beep:
        init_beep()

    # Optional GUI
    win = None
    if args.gui:
        try:
            import pygame
            pygame.init()
            win = pygame.display.set_mode((360, 360))
            pygame.display.set_caption("Bell Tester")
        except Exception:
            win = None

    print("Bell tester: Ctrl+C to stop. Edit detect.mic.* in config.yaml to tune.")
    print(f"Device={p.device or 'default'} SR={p.samplerate} block={p.blocksize} band={p.band_hz} ")
    print(f"TH={p.threshold_db} dB, REL={p.release_db} dB, MinInterval={p.min_interval_ms} ms")

    cfg_mtime = _mtime_config()
    armed = True
    last_fire = 0.0
    min_interval_s = p.min_interval_ms / 1000.0

    def reload_if_changed():
        nonlocal cfg, p, lo, hi, min_interval_s
        m2 = _mtime_config()
        if m2 != cfg_mtime:
            cfg = load_config()
            p2 = _cfg_to_params(cfg)
            p = p2
            lo = p.release_db - 20.0
            hi = p.threshold_db + 6.0
            min_interval_s = p.min_interval_ms / 1000.0
            print(f"\nReloaded: TH={p.threshold_db} REL={p.release_db} band={p.band_hz} minInt={p.min_interval_ms}ms")

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

    last_print = 0.0
    flash_until = 0.0
    with sd.InputStream(samplerate=p.samplerate, channels=1, dtype="float32", blocksize=p.blocksize, device=p.device) as stream:
        try:
            while True:
                # GUI pump
                if win is not None:
                    import pygame
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            raise KeyboardInterrupt
                data, _ = stream.read(p.blocksize)
                x = np.squeeze(data).astype(np.float32)
                db = block_db(x)
                now = time.time()
                if now - last_print > 0.2:
                    reload_if_changed()
                # detection logic
                if armed and db >= p.threshold_db and (now - last_fire) >= min_interval_s:
                    last_fire = now
                    armed = False
                    if not args.no_beep:
                        play_beep()
                    print("\nDETECTED")
                    flash_until = now + 0.15
                elif (not armed) and db <= p.release_db:
                    armed = True

                # meter
                if now - last_print >= 0.03:
                    last_print = now
                    bar = _meter(db, lo, hi)
                    thr = f" | TH={p.threshold_db:.1f} REL={p.release_db:.1f}"
                    print("\r" + bar + thr, end="", flush=True)

                # GUI: draw round indicator
                if win is not None:
                    import pygame
                    w, h = win.get_size()
                    cx, cy = w // 2, h // 2
                    win.fill((12, 12, 16))
                    # Normalize db into 0..1 between release and threshold
                    if p.threshold_db == p.release_db:
                        norm = 0.0
                    else:
                        norm = (db - p.release_db) / max(1e-6, (p.threshold_db - p.release_db))
                        norm = max(0.0, min(1.0, norm))
                    # Base radius and color
                    r = int(60 + 180 * norm)
                    # Color gradient: below release -> gray, near threshold -> orange/red
                    base = np.array([80, 80, 90], dtype=np.float32)
                    hot = np.array([255, 80, 60], dtype=np.float32)
                    col = (base * (1 - norm) + hot * norm).astype(int)
                    # Flash on detection
                    if now < flash_until:
                        col = np.minimum(col + 50, 255)
                    pygame.draw.circle(win, col.tolist(), (cx, cy), r)
                    # Draw ring for thresholds
                    thr_r = int(60 + 180 * 1.0)  # at threshold
                    rel_r = int(60 + 180 * 0.0)  # at release
                    pygame.draw.circle(win, (120, 120, 160), (cx, cy), thr_r, width=2)
                    pygame.draw.circle(win, (60, 60, 90), (cx, cy), rel_r, width=1)
                    # Show numeric dB
                    try:
                        font = pygame.font.SysFont(None, 28)
                        txt = font.render(f"{db:.1f} dB", True, (230, 230, 240))
                        win.blit(txt, (cx - txt.get_width() // 2, cy - 14))
                    except Exception:
                        pass
                    pygame.display.flip()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    raise SystemExit(main())
