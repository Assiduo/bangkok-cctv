"""Image loading that also works for non-ASCII (Thai) file names on Windows."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: str | Path) -> np.ndarray | None:
    """Decode a JPEG to a BGR array; None if the file is missing or corrupt.

    ``cv2.imread`` cannot open paths containing Thai characters on Windows, so decode from bytes instead.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
