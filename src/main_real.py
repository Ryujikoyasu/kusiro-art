from __future__ import annotations

import time

from .util.config import load_config
from .led.mapper import build_u_shape_idx
from .led.serial_link import SerialLink
from .audio.engine import AudioEngine
from .detect.kakon_cam import KakomCameraDetector


def run():
    cfg = load_config()
    idx = build_u_shape_idx(cfg)
    audio = AudioEngine(cfg)
    det = KakomCameraDetector(cfg)
    link = SerialLink(cfg)

    total = len(idx)
    speed = float(cfg["wave"]["speed_mps"])  # m/s
    total_m = sum(cfg["segments_m"].values())
    duration = total_m / max(1e-6, speed)
    tail_m = float(cfg["wave"]["tail_m"])  # m

    with link:
        link.send_conf(total=total)
        print("main_real: running; Ctrl+C to stop")
        try:
            for _ in det.watch():
                # SILENCE
                audio.fade_out_all(int(cfg["wave"]["pause_ms"]))
                # WAVE
                t0 = time.time()
                while True:
                    t = time.time() - t0
                    pos = min(1.0, t / duration)
                    link.send_wave(pos=pos, tail_m=tail_m, bright=220)
                    if pos >= 1.0:
                        break
                    time.sleep(1 / 60.0)
                # RESUME
                audio.resume_randomized()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    run()

