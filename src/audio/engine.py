from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import importlib.util


def _load_insect_params() -> Dict:
    # Preferred: load from explicit file path under repo/config/insect_config.py
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # src/audio/engine.py -> repo_root
    cfg_path = repo_root / "config" / "insect_config.py"
    if cfg_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("insect_config", str(cfg_path))
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "INSECT_PARAMS"):
                return getattr(mod, "INSECT_PARAMS")
        except Exception:
            pass
    # Fallback to base structure
    try:
        from config.config_structure import get_insect_base_config  # type: ignore
        return get_insect_base_config()
    except Exception:
        return {}


INSECT_PARAMS = _load_insect_params()


def _insect_colors_from_params(params: Dict) -> Dict[str, tuple[int, int, int]]:
    out: Dict[str, tuple[int, int, int]] = {}
    name_map = {
        "aomatsumushi": "アオマツムシ",
        "kutsuwa": "クツワムシ",
        "matsumushi": "マツムシ",
        "umaoi": "ウマオイ",
        "koorogi": "コオロギ",
        "kirigirisu": "キリギリス",
        "suzumushi": "スズムシ",
    }
    for key, v in params.items():
        jp = name_map.get(key, key)
        col = v.get("colors", {}).get("color")
        if col is None:
            # legacy fallback to base
            col = v.get("colors", {}).get("base", [200, 200, 200])
        base = tuple(int(x) for x in col)
        out[jp] = base
    return out


INSECT_COLORS = _insect_colors_from_params(INSECT_PARAMS)


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
        # Build species from INSECT_PARAMS (single default file per species)
        name_map = {
            "aomatsumushi": "アオマツムシ",
            "kutsuwa": "クツワムシ",
            "matsumushi": "マツムシ",
            "umaoi": "ウマオイ",
            "koorogi": "コオロギ",
            "kirigirisu": "キリギリス",
            "suzumushi": "スズムシ",
        }
        for key, v in INSECT_PARAMS.items():
            jp = name_map.get(key, key)
            f = v.get("sound_files", {}).get("default")
            if not f:
                continue
            # Gain/pan are not in insect_params; keep defaults or read hints from cfg
            # Look up optional overrides in cfg.audio.overrides
            ov = (cfg.get("audio", {}).get("overrides", {}) or {}).get(jp, {})
            gain = float(ov.get("gain", 0))
            pan = float(ov.get("pan", 0))
            self.species.append(Species(id=jp, file=f, gain=gain, pan=pan))
        self.master_gain = float(cfg.get("audio", {}).get("gain_master", 0.0))
        self.state: str = "IDLE"

    def print_assets(self):
        print("Audio assets (from INSECT_PARAMS):")
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
