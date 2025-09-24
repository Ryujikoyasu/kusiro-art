from __future__ import annotations

import time
from typing import Dict, Generator, Tuple

import cv2  # type: ignore
import numpy as np


def _angle_of_longest_line(edges: np.ndarray) -> float | None:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=50, maxLineGap=10)
    if lines is None:
        return None
    longest = None
    longest_len = 0.0
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = l
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length > longest_len:
            longest_len = length
            longest = (dx, dy)
    if not longest:
        return None
    dx, dy = longest
    angle = np.degrees(np.arctan2(dy, dx))  # -180..180, 0 is horizontal right
    return float(angle)


class KakomCameraDetector:
    def __init__(self, cfg: Dict, cam_index: int = 0):
        self.cfg = cfg
        c = cfg.get("detect", {}).get("cam", {})
        self.roi = tuple(float(x) for x in c.get("roi", [0.5, 0.1, 0.4, 0.8]))  # x,y,w,h in 0..1
        self.th_fire = float(c.get("angle_fire_deg", -25))
        self.th_release = float(c.get("angle_release_deg", -15))
        self.min_interval = float(c.get("min_interval_ms", 1500)) / 1000.0
        self.cap = cv2.VideoCapture(cam_index)
        self.last_fire = 0.0
        self.armed = True

    def __del__(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def watch(self) -> Generator[None, None, None]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            h, w, _ = frame.shape
            x, y, rw, rh = self.roi
            rx1, ry1 = int(x * w), int(y * h)
            rx2, ry2 = int((x + rw) * w), int((y + rh) * h)
            roi = frame[ry1:ry2, rx1:rx2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            angle = _angle_of_longest_line(edges)

            now = time.time()
            if angle is not None:
                # fire when crossing below fire threshold; release when above release threshold
                if self.armed and angle <= self.th_fire and (now - self.last_fire) >= self.min_interval:
                    self.last_fire = now
                    self.armed = False
                    yield None
                elif not self.armed and angle >= self.th_release:
                    self.armed = True

