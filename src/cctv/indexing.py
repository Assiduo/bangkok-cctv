"""Step 1 – turn a folder tree of raw JPEGs into a tidy image index table."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import config

# <camera_id>_<thai name>_<YYYYMMDD>_<HHMMSS>.jpg
_FILENAME = re.compile(r"^(?P<camera_id>\d+)_.*_(?P<date>\d{8})_(?P<time>\d{6})\.jpe?g$", re.I)


def parse_filename(name: str) -> dict | None:
    """Extract camera id and capture time from a raw filename; None if it does not match."""
    m = _FILENAME.match(name)
    if not m:
        return None
    ts = pd.to_datetime(m["date"] + m["time"], format="%Y%m%d%H%M%S", errors="coerce")
    if pd.isna(ts):
        return None
    return {"camera_id": int(m["camera_id"]), "timestamp": ts}


def build_image_index(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Scan ``raw_dir`` recursively and return one row per image file."""
    rows, skipped = [], 0
    for path in sorted(Path(raw_dir).resolve().rglob("*.jp*g")):
        meta = parse_filename(path.name)
        if meta is None:
            skipped += 1
            continue
        rows.append({"image_path": path.relative_to(Path(raw_dir).resolve()).as_posix(), **meta})
    if skipped:
        print(f"[index] skipped {skipped} files with unparseable names")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["camera_id", "timestamp"]).reset_index(drop=True)


def load_camera_info(path: Path = config.CAMERA_INFO_CSV) -> pd.DataFrame:
    """Load the camera metadata CSV with a cleaned English display name."""
    cams = pd.read_csv(path)
    cams["camera_name"] = cams["location_en"].fillna(cams["location"]).str.strip()
    cams["camera_name"] = cams["camera_name"].where(
        ~cams["camera_name"].str.contains("connect BMAcctv", case=False, na=False), cams["location"])
    cams["camera_label"] = cams["camera_id"].astype(str) + " · " + cams["camera_name"].str.slice(0, 40)
    return cams
