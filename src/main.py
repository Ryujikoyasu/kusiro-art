from __future__ import annotations

from .util.config import load_config
from .led.mapper import build_u_shape_idx
from .sim.viewport import Viewport


def run(mirror: bool = False):
    cfg = load_config()
    idx = build_u_shape_idx(cfg)
    vp = Viewport(cfg, idx, mirror_to_device=mirror)
    vp.run_simulation()


if __name__ == "__main__":
    run()
