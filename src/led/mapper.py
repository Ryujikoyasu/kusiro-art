from __future__ import annotations

from typing import Dict, List


def build_u_shape_idx(cfg: Dict) -> List[Dict]:
    l = float(cfg["segments_m"]["left"])  # meters
    b = float(cfg["segments_m"]["bottom"])  # meters
    r = float(cfg["segments_m"]["right"])  # meters
    d = int(cfg["leds_per_meter"])  # leds/m
    segs = [
        ("left_down", int(round(l * d))),
        ("bottom_right", int(round(b * d))),
        ("right_up", int(round(r * d))),
    ]
    idx = []
    base = 0
    for name, count in segs:
        if count <= 0:
            continue
        for i in range(count):
            idx.append({"i": base + i, "seg": name, "s": (i / max(1, count - 1))})
        base += count
    return idx

