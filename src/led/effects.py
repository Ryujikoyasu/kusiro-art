from __future__ import annotations

import math
import random
from typing import List, Tuple


def ease_quad_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return -t * (t - 2.0)


def wave_positions(length: int, pos01: float, tail_leds: int) -> List[Tuple[int, float]]:
    """Return list of (index, weight 0..1) for a 1D wave center at pos01 with tail decay.

    weight decays linearly to 0 over tail_leds.
    """
    center = pos01 * (length - 1)
    out: List[Tuple[int, float]] = []
    if length <= 0 or tail_leds <= 0:
        return out
    for i in range(length):
        d = abs(i - center)
        w = max(0.0, 1.0 - d / float(tail_leds))
        if w > 0:
            out.append((i, w))
    return out


def sparkle_color(base: Tuple[int, int, int], amount: float = 0.25) -> Tuple[int, int, int]:
    r, g, b = base
    jitter = lambda v: max(0, min(255, int(v * (1.0 + random.uniform(-amount, amount)))))
    return jitter(r), jitter(g), jitter(b)

