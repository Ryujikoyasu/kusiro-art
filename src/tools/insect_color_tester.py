from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import pygame
import numpy as np

from ..util.config import load_config
from ..serial_handler import SerialWriterThread


NAME_MAP = {
    "aomatsumushi": "アオマツムシ",
    "kutsuwa": "クツワムシ",
    "matsumushi": "マツムシ",
    "umaoi": "ウマオイ",
    "koorogi": "コオロギ",
    "kirigirisu": "キリギリス",
    "suzumushi": "スズムシ",
}


def load_insect_params() -> Dict:
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    here = Path(__file__).resolve()
    repo = here.parents[2]
    cfg_path = repo / "config" / "insect_config.py"
    if cfg_path.exists():
        try:
            spec = spec_from_file_location("insect_config", str(cfg_path))
            assert spec and spec.loader
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "INSECT_PARAMS"):
                return getattr(mod, "INSECT_PARAMS")
        except Exception:
            pass
    from config.config_structure import get_insect_base_config  # type: ignore
    return get_insect_base_config()


def run():
    cfg = load_config()
    SERIAL_PORT = cfg.get('serial_port')
    BAUD = int(cfg.get('baud', 115200))
    leds_per_m = int(cfg.get('leds_per_meter', 60))
    seg = cfg.get('segments_m', {"left": 9.5, "bottom": 1.0, "right": 9.5})
    total_leds = int(round((float(seg['left']) + float(seg['bottom']) + float(seg['right'])) * leds_per_m))
    NUM_PIXELS = (total_leds + 2) // 3
    MAGIC = 0x7E

    params = load_insect_params()
    species = list(params.keys())
    if not species:
        print("No insect species found.")
        return

    pygame.init()
    screen = pygame.display.set_mode((900, 500))
    pygame.display.set_caption("Insect Color Tester (MAGIC 0x7E)")
    font = pygame.font.Font(None, 48)
    small = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    ser = SerialWriterThread(SERIAL_PORT, BAUD, MAGIC, NUM_PIXELS)
    ser.start()

    idx = 0
    running = True

    def get_color(sp_key: str) -> Tuple[int, int, int]:
        cols = params.get(sp_key, {}).get('colors', {})
        col = cols.get('color') or cols.get('base', [200, 200, 200])
        r, g, b = (int(col[0]), int(col[1]), int(col[2]))
        return (r, g, b)

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in (pygame.K_RIGHT, pygame.K_d):
                    idx = (idx + 1) % len(species)
                elif e.key in (pygame.K_LEFT, pygame.K_a):
                    idx = (idx - 1) % len(species)
                # no accent toggle anymore

        sp_key = species[idx]
        jp = NAME_MAP.get(sp_key, sp_key)
        col = get_color(sp_key)

        # send to device
        data = np.tile(np.array(col, dtype=np.uint8), (NUM_PIXELS, 1))
        ser.send(data)

        # draw UI
        screen.fill((20, 20, 24))
        pygame.draw.rect(screen, col, (80, 120, 740, 220), border_radius=16)
        title = font.render(f"{jp} ({sp_key})", True, (230, 230, 240))
        screen.blit(title, (80, 60))
        hint = small.render("←/→ 種切替 | Esc: 終了", True, (180, 180, 190))
        screen.blit(hint, (80, 360))
        rgb = small.render(f"RGB: {col}", True, (200, 200, 210))
        screen.blit(rgb, (80, 400))

        pygame.display.flip()
        clock.tick(30)

    # blackout and close
    ser.send(np.zeros((NUM_PIXELS, 3), dtype=np.uint8))
    time.sleep(0.1)
    ser.close()
    pygame.quit()


if __name__ == "__main__":
    run()
