"""Helpers for the detector fine-tuning workflow: YOLO label files and box matching."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .detectors import COCO_TO_FT, FT_CLASSES, FT_TO_COCO

LABEL_DIR = config.DATA / "labeling"
IMAGES = LABEL_DIR / "images"
LABELS = LABEL_DIR / "labels"            # one YOLO .txt per image: "class cx cy w h" (normalised)
VEHICLE_FT = {i for i, c in enumerate(FT_TO_COCO) if c in config.VEHICLE_CLASSES}


def write_labels(path: Path, dets, w: int, h: int) -> None:
    """dets: [(coco_id, score, x1, y1, x2, y2)] in pixels -> YOLO txt using our class order."""
    lines = []
    for coco, _, x1, y1, x2, y2 in dets:
        if coco not in COCO_TO_FT:
            continue
        x1, x2, y1, y2 = max(x1, 0), min(x2, w), max(y1, 0), min(y2, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{COCO_TO_FT[coco]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_labels(path: Path, w: int, h: int) -> list[tuple]:
    """YOLO txt -> [(ft_class, x1, y1, x2, y2)] in pixels."""
    out = []
    if path.exists():
        for line in path.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt: list[tuple], pred: list[tuple], thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one matching by IoU within the same class. Returns (tp, fp, fn).

    gt: [(cls, x1, y1, x2, y2)]; pred: [(cls, score, x1, y1, x2, y2)] with cls in OUR class order.
    """
    used, tp = set(), 0
    for p in sorted(pred, key=lambda d: -d[1]):
        best, best_i = 0.0, None
        for i, g in enumerate(gt):
            if i in used or g[0] != p[0]:
                continue
            v = iou(g[1:], p[2:])
            if v > best:
                best, best_i = v, i
        if best_i is not None and best >= thr:
            used.add(best_i)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def dets_to_ft(dets) -> list[tuple]:
    """Detector output (COCO ids) -> our class order, dropping classes we do not model."""
    return [(COCO_TO_FT[c], s, *box) for c, s, *box in dets if c in COCO_TO_FT]
