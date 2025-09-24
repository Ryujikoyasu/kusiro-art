from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple

import pygame

from ..audio.engine import INSECT_COLORS


ORANGE = (255, 165, 0)


class Viewport:
    def __init__(self, cfg: Dict, layout_idx: List[Dict]):
        self.cfg = cfg
        self.idx = layout_idx
        self.total_leds = len(layout_idx)
        self.speed = float(cfg["wave"]["speed_mps"])  # m/s
        self.tail_m = float(cfg["wave"]["tail_m"])  # m
        self.leds_per_m = int(cfg["leds_per_meter"])  # leds/m
        self.tail_leds = max(1, int(round(self.tail_m * self.leds_per_m)))

        # Simulation area (meters -> pixels scale handled here)
        self.px_per_m = 60
        w = int((self.cfg["segments_m"]["left"] + self.cfg["segments_m"]["right"]) * 0.4 + self.cfg["segments_m"]["bottom"])  # not to scale, just a pleasant wide view
        self.view_w = int(1200)
        self.view_h = int(400)
        self.margin = 40

        # Colors for idle sparkle by species
        self.species_colors = list(INSECT_COLORS.values())

    def _pos_for_index(self, entry: Dict) -> Tuple[int, int]:
        # Map into U-shape rectangle in screen space
        left = float(self.cfg["segments_m"]["left"]) * self.px_per_m
        bottom = float(self.cfg["segments_m"]["bottom"]) * self.px_per_m
        right = float(self.cfg["segments_m"]["right"]) * self.px_per_m
        total_h = int(left)  # same as right; use left
        total_w = int(bottom)
        # Place in a nice frame
        ox = self.margin
        oy = self.margin
        name = entry["seg"]
        s = float(entry["s"])  # 0..1 along segment
        if name == "left_down":
            x = ox
            y = oy + int(s * total_h)
        elif name == "bottom_right":
            x = ox + int(s * total_w)
            y = oy + total_h
        else:  # right_up
            x = ox + total_w
            y = oy + (total_h - int(s * total_h))
        return x, y

    def run_simulation(self):
        pygame.init()
        screen = pygame.display.set_mode((self.view_w, self.view_h))
        clock = pygame.time.Clock()

        # Precompute pixel positions
        pts = [self._pos_for_index(e) for e in self.idx]

        # Idle sparkle state
        idle_colors = [random.choice(self.species_colors) for _ in range(self.total_leds)]
        idle_phases = [random.random() * math.tau for _ in range(self.total_leds)]

        # Wave state
        wave_running = False
        wave_t0 = 0.0
        wave_duration = sum(self.cfg["segments_m"].values()) / max(1e-6, self.speed)

        # Kakon schedule
        next_kakon_at = time.time() + max(5.0, random.gauss(30.0, 5.0))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time()
            if not wave_running and now >= next_kakon_at:
                wave_running = True
                wave_t0 = now
                next_kakon_at = now + max(10.0, random.gauss(30.0, 5.0))

            screen.fill((10, 10, 10))

            # Draw LEDs
            if wave_running:
                t = now - wave_t0
                pos01 = min(1.0, t / wave_duration)
                center = pos01 * (self.total_leds - 1)
                for i, (x, y) in enumerate(pts):
                    d = abs(i - center)
                    w = max(0.0, 1.0 - d / float(self.tail_leds))
                    if w > 0:
                        col = (
                            int(ORANGE[0] * w),
                            int(ORANGE[1] * w),
                            int(ORANGE[2] * w),
                        )
                    else:
                        # faint idle when wave passes
                        base = idle_colors[i]
                        a = 0.1
                        col = (int(base[0] * a), int(base[1] * a), int(base[2] * a))
                    pygame.draw.circle(screen, col, (x, y), 3)
                if pos01 >= 1.0:
                    wave_running = False
            else:
                # idle sparkle
                for i, (x, y) in enumerate(pts):
                    base = idle_colors[i]
                    # flicker via slow sin + jitter
                    idle_phases[i] += 0.02 + random.uniform(-0.005, 0.005)
                    s = (math.sin(idle_phases[i]) * 0.5 + 0.5) * 0.4 + 0.1
                    col = (int(base[0] * s), int(base[1] * s), int(base[2] * s))
                    pygame.draw.circle(screen, col, (x, y), 3)

            # HUD
            # We avoid font init for simplicity; draw a simple progress bar for next kakon
            rem = max(0.0, next_kakon_at - now)
            bar_w = int(self.view_w * max(0.0, min(1.0, 1.0 - rem / 40.0)))
            pygame.draw.rect(screen, (255, 120, 0), (0, 0, bar_w, 6))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

