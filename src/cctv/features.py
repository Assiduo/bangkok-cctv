"""Step 2 – per-image feature extraction (quality metrics + vehicle counts)."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import VehicleDetector


def dhash(gray: np.ndarray, size: int = 8) -> str:
    """64-bit difference hash as hex; identical frames => identical hash (frozen-camera check)."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def quality_features(img: np.ndarray) -> dict:
    """Cheap photometric features that describe lighting and image quality."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float((cv2.Canny(gray, 100, 200) > 0).mean()),
        "saturation": float(hsv[..., 1].mean()),
        "glare_frac": float((gray > 240).mean()),
        "dark_frac": float((gray < 30).mean()),
        "img_hash": dhash(gray),
    }


def extract_features(img: np.ndarray, detector: VehicleDetector) -> dict:
    """All features for one decoded BGR image."""
    return {**quality_features(img), **detector.count(img)}
