"""Detector factory + the fine-tuned (Ultralytics-exported) detector.

``CCTV_MODEL=yolox_x``  -> the stock COCO YOLOX (cctv.detector.VehicleDetector)
``CCTV_MODEL=cctv_ft``  -> models/cctv_ft.onnx trained by scripts/11_train_detector.py on our own labelled CCTV frames
Both expose the same API: ``detect(img)`` -> [(coco_class_id, score, x1, y1, x2, y2)], ``count(img)``, ``device``.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .detector import VehicleDetector

# class order of our own dataset; mapped back to COCO ids so counting/PCE code stays unchanged
FT_CLASSES = ["car", "motorcycle", "bus", "truck", "bicycle", "person"]
_NAME_TO_COCO = {"bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "person": 0}
FT_TO_COCO = [_NAME_TO_COCO[n] for n in FT_CLASSES]
COCO_TO_FT = {c: i for i, c in enumerate(FT_TO_COCO)}


class FinetunedDetector(VehicleDetector):
    """Ultralytics-style ONNX head: output (1, 4 + n_classes, N), boxes are cx,cy,w,h in input pixels, no objectness."""

    def _preprocess(self, img: np.ndarray):
        shape = self.session.get_inputs()[0].shape
        if isinstance(shape[2], int):          # fixed-size export: follow the model, not the default
            self.size = shape[2]
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        rgb = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb), r

    def detect(self, img: np.ndarray):
        blob, ratio = self._preprocess(img)
        out = self.session.run(None, {self.input_name: blob})[0][0].T          # (N, 4 + nc)
        scores = out[:, 4:]
        cls, conf = scores.argmax(1), scores.max(1)
        keep = conf > self.score_thr
        if not keep.any():
            return []
        box, cls, conf = out[keep, :4], cls[keep], conf[keep]
        xyxy = np.concatenate([box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2], 1) / ratio
        idx = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
                               conf.tolist(), self.score_thr, self.nms_thr)
        idx = np.array(idx).reshape(-1)
        return [(FT_TO_COCO[int(cls[i])], float(conf[i]), *map(float, xyxy[i])) for i in idx]


def _single(model_path=None, **kw) -> VehicleDetector:
    path = VehicleDetector.resolve_model_path(model_path)
    cls = FinetunedDetector if path.stem.startswith("cctv_") else VehicleDetector
    return cls(model_path=path, **kw)


def make_detector(model_path=None, **kw) -> VehicleDetector:
    """Pick the right decoder from the model name (``cctv_*`` = fine-tuned, otherwise YOLOX).

    A spec such as ``yolox_x@tile`` or ``yolox_x@tile+cctv_ft`` (see ensemble.py) builds a tiled or fused detector;
    it can be passed here or set through CCTV_MODEL."""
    spec = str(model_path.stem if hasattr(model_path, "stem") else model_path) if model_path is not None else config.MODEL_VARIANT
    if "@" in spec or "+" in spec:
        from .ensemble import build
        score_thr = kw.pop("score_thr", None)
        det = build(spec, lambda name: _single(config.ROOT / "models" / f"{name}.onnx", **kw))
        if score_thr is not None:
            det.score_thr = score_thr
        return det
    if isinstance(model_path, str) and not model_path.endswith(".onnx") and "/" not in model_path and "\\" not in model_path:
        model_path = config.ROOT / "models" / f"{model_path}.onnx"        # a bare model name such as "yolox_x"
    return _single(model_path, **kw)
