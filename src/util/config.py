from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


CFG_PATH = Path("config.yaml")


def load_config() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Missing {CFG_PATH}. Please create it.")
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: Dict[str, Any]) -> None:
    with CFG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

