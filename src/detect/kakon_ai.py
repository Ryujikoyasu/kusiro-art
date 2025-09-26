from __future__ import annotations

"""
AI-based bell detector using a local YAMNet TFLite model.

Config (config.yaml):

detect:
  mode: "ai"
  ai:
    model_path: "assets/models/yamnet.tflite"
    label_map_csv: "assets/models/yamnet_class_map.csv"
    target_labels: ["bell", "bells", "jingle", "sleigh"]
    threshold: 0.6
    release: 0.4
    min_interval_ms: 1800
    samplerate: 16000
    blocksize: 1024
    device: null

This detector listens to mic audio, accumulates ~0.96s, runs YAMNet, and fires
when the maximum score among target_labels exceeds threshold (with hysteresis).
"""

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

import numpy as np


@dataclass
class AIParams:
    model_path: str
    label_map_csv: Optional[str]
    target_labels: List[str]
    threshold: float
    release: float
    min_interval_ms: float
    samplerate: int
    blocksize: int
    device: Optional[str]


def _cfg_to_params(cfg: Dict) -> AIParams:
    a = cfg.get("detect", {}).get("ai", {})
    model_path = str(a.get("model_path", "assets/models/yamnet.tflite"))
    label_map_csv = a.get("label_map_csv")
    if label_map_csv is not None:
        label_map_csv = str(label_map_csv)
    target_labels = [str(x).lower() for x in a.get("target_labels", ["bell", "bells", "jingle", "sleigh"])]
    threshold = float(a.get("threshold", 0.6))
    release = float(a.get("release", 0.4))
    min_interval_ms = float(a.get("min_interval_ms", 1800))
    samplerate = int(a.get("samplerate", 16000))
    blocksize = int(a.get("blocksize", 1024))
    device = a.get("device")
    return AIParams(model_path, label_map_csv, target_labels, threshold, release, min_interval_ms, samplerate, blocksize, device)  # type: ignore[arg-type]


def _load_labels(csv_path: Optional[str]) -> Optional[List[str]]:
    if not csv_path:
        return None
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label map CSV not found: {csv_path}")
    names: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect a column 'display_name'
        for row in reader:
            names.append(row.get("display_name") or row.get("name") or "")
    return names


class KakonAIDetector:
    def __init__(self, cfg: Dict):
        try:
            import sounddevice as sd  # noqa: F401
        except Exception:
            raise RuntimeError("sounddevice is required for AI detection. Install with `pip install sounddevice`. ")
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except Exception as e:
            raise RuntimeError("tflite-runtime is required. Install with `pip install tflite-runtime`. ")

        self.params = _cfg_to_params(cfg)
        self.labels = _load_labels(self.params.label_map_csv)
        self.target_indices: Optional[np.ndarray] = None

        # Load TFLite model
        if not os.path.exists(self.params.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.params.model_path}. Place YAMNet TFLite at this path."
            )
        self._interpreter = Interpreter(model_path=self.params.model_path)
        self._interpreter.allocate_tensors()
        self._in = self._interpreter.get_input_details()[0]
        self._outs = self._interpreter.get_output_details()
        # Determine target indices if labels are available
        if self.labels:
            tl = [t.lower() for t in self.params.target_labels]
            idxs = [i for i, name in enumerate(self.labels) if any(t in (name or "").lower() for t in tl)]
            self.target_indices = np.array(idxs, dtype=np.int64) if idxs else None

        # Audio state
        self._armed = True
        self._last_fire = 0.0
        # Buffer for ~1s of audio at model SR
        self._buf = np.zeros(0, dtype=np.float32)

    def _infer_scores(self, waveform: np.ndarray) -> np.ndarray:
        # YAMNet TFLite commonly accepts [N] float32; sometimes requires [1, N]
        x = waveform.astype(np.float32)
        # Normalize to [-1, 1] if not already
        x = np.clip(x, -1.0, 1.0)
        # Resize input if needed
        in_info = self._in
        in_shape = list(in_info["shape"])  # e.g., [1, 15600] or [15600]
        tensor_index = in_info["index"]
        need_batch = (len(in_shape) == 2)
        if need_batch:
            # Resize to [1, N]
            self._interpreter.resize_tensor_input(tensor_index, [1, x.shape[0]], strict=False)
            self._interpreter.allocate_tensors()
            tensor_index = self._interpreter.get_input_details()[0]["index"]
            self._interpreter.set_tensor(tensor_index, x.reshape(1, -1))
        else:
            self._interpreter.resize_tensor_input(tensor_index, [x.shape[0]], strict=False)
            self._interpreter.allocate_tensors()
            tensor_index = self._interpreter.get_input_details()[0]["index"]
            self._interpreter.set_tensor(tensor_index, x)
        # Invoke
        self._interpreter.invoke()
        # Assume output 0 is scores: [frames, 521]
        out0 = self._interpreter.get_output_details()[0]
        scores = self._interpreter.get_tensor(out0["index"])  # type: ignore
        # scores shape may be [1, frames, 521]; squeeze
        scores = np.array(scores)
        scores = np.squeeze(scores)
        return scores

    def _scores_to_prob(self, scores: np.ndarray) -> float:
        # Pick target class indices; if not provided, try a heuristic on label names if available
        if self.target_indices is not None and self.target_indices.size > 0:
            s = scores[:, self.target_indices]
            v = float(np.max(s))
            return v
        # Fallback: max across all classes (least ideal)
        return float(np.max(scores))

    def watch(self) -> Generator[None, None, None]:
        import sounddevice as sd
        p = self.params
        model_sr = p.samplerate
        min_len = int(model_sr * 0.96)
        hop_len = int(model_sr * 0.48)

        def stream_iter():
            with sd.InputStream(samplerate=p.samplerate, channels=1, dtype="float32", blocksize=p.blocksize, device=p.device) as stream:
                while True:
                    data, _ = stream.read(p.blocksize)
                    yield np.squeeze(data).astype(np.float32)

        for block in stream_iter():
            # If input SR differs, resample to model_sr
            if p.samplerate != model_sr:
                try:
                    import librosa  # noqa: F401
                    x = librosa.resample(block, orig_sr=p.samplerate, target_sr=model_sr)  # type: ignore
                except Exception:
                    # Naive fallback: drop/linear interpolate
                    ratio = model_sr / float(p.samplerate)
                    idx = (np.arange(int(len(block) * ratio)) / ratio).astype(np.float32)
                    x = np.interp(idx, np.arange(len(block), dtype=np.float32), block).astype(np.float32)
            else:
                x = block
            self._buf = np.concatenate([self._buf, x])
            if self._buf.shape[0] < min_len:
                continue
            # Take the last 0.96s
            window = self._buf[-min_len:]
            scores = self._infer_scores(window)
            prob = self._scores_to_prob(scores)
            now = time.time()
            if self._armed and prob >= p.threshold and (now - self._last_fire) >= (p.min_interval_ms / 1000.0):
                self._last_fire = now
                self._armed = False
                yield None
            elif (not self._armed) and prob <= p.release:
                self._armed = True
            # Advance hop
            if self._buf.shape[0] > hop_len:
                self._buf = self._buf[-hop_len:]

