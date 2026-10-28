"""Ways to get more out of the same detectors: slice the frame into tiles (small far vehicles get bigger) and fuse several models.

A *spec* names a detector in one string, so any script can select it with ``CCTV_MODEL``:

    yolox_x                  one model
    yolox_x@tile             that model run on the whole frame + four overlapping, enlarged tiles
    yolox_x@tile+cctv_ft     weighted-box-fusion of two models (each term may carry @tile)

Detections are ``(coco_class_id, score, x1, y1, x2, y2)`` like every other detector here.
"""
from __future__ import annotations

import numpy as np

from . import config
from .detector import VehicleDetector

LOW_THR = 0.10          # inner detectors run permissively; the wrapper applies the real threshold at the end


def iou_one(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nms(dets, iou_thr: float = 0.5):
    """Class-wise greedy NMS; a box mostly inside a higher-scoring same-class box is also dropped (tile duplicates)."""
    keep = []
    for d in sorted(dets, key=lambda t: -t[1]):
        dup = False
        for k in keep:
            if k[0] != d[0]:
                continue
            ov = iou_one(d[2:], k[2:])
            iw = min(d[4], k[4]) - max(d[2], k[2])
            ih = min(d[5], k[5]) - max(d[3], k[3])
            inside = iw > 0 and ih > 0 and iw * ih / max((d[4] - d[2]) * (d[5] - d[3]), 1e-6) > 0.8
            if ov > iou_thr or inside:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


class _Wrapper(VehicleDetector):
    """Shares count()/count_from_detections() with the plain detector; subclasses implement detect()."""

    def __init__(self, score_thr: float = config.DET_SCORE_THRESHOLD):
        self.score_thr = score_thr
        self.device = "CPU"


class TiledDetector(_Wrapper):
    """Full frame + ``grid x grid`` overlapping tiles, each enlarged to the model's input size."""

    def __init__(self, base: VehicleDetector, grid: int = 2, overlap: float = 0.25, score_thr: float = config.DET_SCORE_THRESHOLD):
        super().__init__(score_thr)
        self.base, self.grid, self.overlap = base, grid, overlap
        self.base.score_thr = min(base.score_thr, LOW_THR)
        self.device = getattr(base, "device", "CPU")

    def detect(self, img):
        h, w = img.shape[:2]
        out = list(self.base.detect(img))
        tw, th = w / self.grid * (1 + self.overlap), h / self.grid * (1 + self.overlap)
        for gy in range(self.grid):
            for gx in range(self.grid):
                x0 = int(round(min(max(gx * w / self.grid - (tw - w / self.grid) / 2, 0), w - tw)))
                y0 = int(round(min(max(gy * h / self.grid - (th - h / self.grid) / 2, 0), h - th)))
                crop = img[y0: y0 + int(th), x0: x0 + int(tw)]
                for c, s, a, b, e, f in self.base.detect(crop):
                    # a box cut by an inner tile edge is only part of a vehicle; the full-frame pass covers it
                    cx1, cy1, cx2, cy2 = a + x0, b + y0, e + x0, f + y0
                    cut = ((a < 2 and x0 > 0) or (b < 2 and y0 > 0) or (e > crop.shape[1] - 2 and x0 + tw < w - 1)
                           or (f > crop.shape[0] - 2 and y0 + th < h - 1))
                    if not cut:
                        out.append((c, s, cx1, cy1, cx2, cy2))
        return [d for d in nms(out) if d[1] >= self.score_thr]


class EnsembleDetector(_Wrapper):
    """Weighted boxes fusion: boxes of the same class that overlap are merged; a box seen by one model only keeps a share of its score."""

    def __init__(self, members: list[VehicleDetector], weights: list[float] | None = None, iou_thr: float = 0.55,
                 score_thr: float = config.DET_SCORE_THRESHOLD):
        super().__init__(score_thr)
        self.members, self.iou_thr = members, iou_thr
        self.weights = weights or [1.0] * len(members)
        for m in members:
            m.score_thr = min(m.score_thr, LOW_THR)

    def detect(self, img):
        pool = []
        for mi, m in enumerate(self.members):
            pool += [(mi, *d) for d in m.detect(img)]
        clusters: list[list] = []
        for d in sorted(pool, key=lambda t: -t[2]):
            for cl in clusters:
                if cl[0][1] == d[1] and iou_one(d[3:], cl[0][3:]) > self.iou_thr and all(x[0] != d[0] for x in cl):
                    cl.append(d)
                    break
            else:
                clusters.append([d])
        total_w, out = sum(self.weights), []
        for cl in clusters:
            ws = np.array([self.weights[m[0]] * m[2] for m in cl])
            box = (np.array([m[3:] for m in cl]) * ws[:, None]).sum(0) / ws.sum()
            score = float(sum(self.weights[m[0]] * m[2] for m in cl) / total_w)
            out.append((cl[0][1], score, *map(float, box)))
        return [d for d in nms(out, 0.6) if d[1] >= self.score_thr]


def build(spec: str, factory) -> VehicleDetector:
    """``factory(name)`` returns a plain detector for a model file stem; ``spec`` is described in the module docstring."""
    def one(term: str):
        name, _, mode = term.partition("@")
        det = factory(name)
        return TiledDetector(det) if mode == "tile" else det
    terms = [t for t in spec.split("+") if t]
    members = [one(t) for t in terms]
    return members[0] if len(members) == 1 else EnsembleDetector(members)
