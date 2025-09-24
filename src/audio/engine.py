from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


INSECT_COLORS = {
    "アオマツムシ": (50, 200, 180),
    "クツワムシ": (255, 80, 20),
    "マツムシ": (180, 255, 80),
    "ウマオイ": (50, 255, 100),
    "コオロギ": (160, 100, 255),
    "キリギリス": (255, 200, 40),
    "スズムシ": (200, 220, 255),
}


@dataclass
class Species:
    id: str
    file: str
    gain: float
    pan: float


class AudioEngine:
    """Minimal audio engine facade.

    In this repo we provide non-realtime stubs so the simulation can run
    without hard audio dependencies. The CLI "audio check" prints asset info.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.species: List[Species] = []
        for sp in cfg.get("audio", {}).get("species", []):
            self.species.append(
                Species(id=sp["id"], file=sp["file"], gain=float(sp.get("gain", 0)), pan=float(sp.get("pan", 0)))
            )
        self.master_gain = float(cfg.get("audio", {}).get("gain_master", 0.0))
        self.state: str = "IDLE"

    def print_assets(self):
        print("Audio assets:")
        for sp in self.species:
            ok = os.path.exists(sp.file)
            col = INSECT_COLORS.get(sp.id)
            print(f"- {sp.id}: file={sp.file} exists={ok} gain={sp.gain} pan={sp.pan} color={col}")
        print(f"master_gain={self.master_gain}")

    # Stub control methods used by main loops
    def fade_out_all(self, ms: int = 500):
        self.state = "SILENCE"

    def resume_randomized(self):
        self.state = "IDLE"

