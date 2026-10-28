"""Vehicle detector: YOLOX-s (COCO) running through ONNX Runtime on CPU."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from . import config


class VehicleDetector:
    """Detect road users in a BGR image and return per-class counts."""

    def __init__(self, model_path=None, size: int = config.DET_INPUT_SIZE,
                 score_thr: float = config.DET_SCORE_THRESHOLD,
                 nms_thr: float = config.DET_NMS_THRESHOLD):
        self.model_path = self.resolve_model_path(model_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=self._providers())
        self.device = "GPU" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU"
        self.input_name = self.session.get_inputs()[0].name
        self.size, self.score_thr, self.nms_thr = size, score_thr, nms_thr
        self._grid, self._strides = self._make_grid(size)

    @staticmethod
    def _providers() -> list[str]:
        """CUDA when onnxruntime-gpu is installed (override with CCTV_DEVICE=cpu), else CPU."""
        if os.environ.get("CCTV_DEVICE", "auto").lower() != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def resolve_model_path(model_path=None) -> Path:
        """Chosen model if present, else any .onnx in models/ (the online demo only ships the small one)."""
        if model_path is not None:
            return Path(model_path)
        if config.MODEL_PATH.exists():
            return config.MODEL_PATH
        fallbacks = sorted(config.MODEL_PATH.parent.glob("*.onnx"))
        if not fallbacks:
            raise FileNotFoundError(f"No detector weights found. Run: python scripts/00_download_model.py ({config.MODEL_PATH})")
        return fallbacks[0]

    @staticmethod
    def _make_grid(size: int):
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = size // stride
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def _preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        return canvas.transpose(2, 0, 1)[None].astype(np.float32), r

    def detect(self, img: np.ndarray):
        """Return list of (class_id, score, x1, y1, x2, y2) in original image pixels."""
        blob, ratio = self._preprocess(img)
        out = self.session.run(None, {self.input_name: blob})[0][0]
        out[:, :2] = (out[:, :2] + self._grid) * self._strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self._strides
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf > self.score_thr
        if not keep.any():
            return []
        boxes, cls, conf = out[keep, :4], cls[keep], conf[keep]
        xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1) / ratio
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
            conf.tolist(), self.score_thr, self.nms_thr)
        idx = np.array(idx).reshape(-1)
        return [(int(cls[i]), float(conf[i]), *map(float, xyxy[i])) for i in idx]

    def count_from_detections(self, dets) -> dict:
        """Counts per vehicle type + persons + passenger-car-equivalent load."""
        counts = {name: 0 for name in config.VEHICLE_CLASSES.values()}
        persons = 0
        for c, *_ in dets:
            if c in config.VEHICLE_CLASSES:
                counts[config.VEHICLE_CLASSES[c]] += 1
            elif c == config.PERSON_CLASS:
                persons += 1
        counts["person"] = persons
        counts["vehicle_total"] = sum(counts[n] for n in config.VEHICLE_CLASSES.values())
        counts["pce_load"] = round(sum(counts[n] * w for n, w in config.PCE_WEIGHTS.items()), 2)
        return counts

    def count(self, img: np.ndarray) -> dict:
        return self.count_from_detections(self.detect(img))
