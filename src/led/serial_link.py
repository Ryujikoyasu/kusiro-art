from __future__ import annotations

import time
from typing import Optional

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover
    serial = None  # type: ignore


class SerialLink:
    def __init__(self, cfg: dict):
        self.port = cfg.get("serial_port")
        self.baud = int(cfg.get("baud", 115200))
        self._ser: Optional["serial.Serial"] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def open(self):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self._ser = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(0.2)

    def close(self):
        if self._ser:
            self._ser.close()
            self._ser = None

    def _writeln(self, s: str):
        if not self._ser:
            raise RuntimeError("Serial not open")
        self._ser.write((s + "\n").encode("utf-8"))

    def _readline(self) -> str:
        if not self._ser:
            raise RuntimeError("Serial not open")
        line = self._ser.readline().decode("utf-8", errors="ignore").strip()
        return line

    def ping(self) -> str:
        self._writeln("PING")
        return self._readline() or "(no response)"

    def send_conf(self, total: int, fps: int = 120):
        self._writeln(f"CONF N={total} FPS={fps}")

    def send_wave(self, pos: float, tail_m: float, bright: int = 180):
        pos = max(0.0, min(1.0, float(pos)))
        bright = max(0, min(255, int(bright)))
        self._writeln(f"WAVE POS={pos:.4f} TAIL={tail_m:.3f} BRIGHT={bright}")

    def set_pixel(self, index: int, r: int, g: int, b: int):
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        self._writeln(f"SET i={index} r={r} g={g} b={b}")

    def black(self):
        self._writeln("BLACK")

