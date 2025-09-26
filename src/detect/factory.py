from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Generator


class KakonDetector(Protocol):
    def watch(self) -> Generator[None, None, None]:
        ...


def build_detector(cfg: Dict[str, Any]) -> Optional[KakonDetector]:
    """Build a detector based on config.detect.mode.

    Modes:
      - 'mic': microphone-based bell/jingle detection
      - 'cam': camera-based angle/line detection
      - 'timer' or anything else: no detector (returns None)
      - 'auto': prefer mic, fallback to cam; if neither, None
    """
    mode = str(cfg.get("detect", {}).get("mode", "timer")).lower()

    if mode == "mic":
        try:
            from .kakon_mic import KakonMicDetector  # type: ignore
            return KakonMicDetector(cfg)
        except Exception as e:
            raise RuntimeError(f"Mic detector init failed: {e}")
    if mode == "cam":
        try:
            from .kakon_cam import KakomCameraDetector  # type: ignore
            return KakomCameraDetector(cfg)
        except Exception as e:
            raise RuntimeError(f"Camera detector init failed: {e}")
    if mode == "ai":
        try:
            from .kakon_ai import KakonAIDetector  # type: ignore
            return KakonAIDetector(cfg)
        except Exception as e:
            raise RuntimeError(f"AI detector init failed: {e}")
    if mode == "auto":
        # Best-effort: try mic -> ai -> cam
        try:
            from .kakon_mic import KakonMicDetector  # type: ignore
            return KakonMicDetector(cfg)
        except Exception:
            try:
                from .kakon_ai import KakonAIDetector  # type: ignore
                return KakonAIDetector(cfg)
            except Exception:
                try:
                    from .kakon_cam import KakomCameraDetector  # type: ignore
                    return KakomCameraDetector(cfg)
                except Exception:
                    return None
    # timer: no detector, caller should fallback to internal scheduler if desired
    return None
