"""Central configuration: paths and constants shared by every pipeline step."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CCTV_DATA_DIR", ROOT / "data"))   # override for tests
def _find_raw_dir() -> Path:
    """CCTV_RAW_DIR wins; otherwise the first existing of data/raw and the usual cctv_images locations (no env var needed)."""
    env = os.environ.get("CCTV_RAW_DIR")
    if env:
        return Path(env)
    home = Path.home()
    for cand in (DATA / "raw", ROOT.parent / "cctv_images", ROOT.parent.parent / "cctv_images", home / "Downloads" / "cctv_images",
                 home / "cctv_images", Path("C:/cctv_images")):
        if cand.exists() and any(cand.iterdir()):
            return cand
    return DATA / "raw"


RAW_DIR = _find_raw_dir()   # folder containing the camera folders (11, 112, ...)
EXTERNAL_DIR = DATA / "external"        # scraped data (weather, air quality, ...)
PROCESSED_DIR = DATA / "processed"
# Detector size: yolox_s | yolox_m | yolox_l | yolox_x  (bigger = more accurate but slower). Set with the env var CCTV_MODEL.
# scripts/20_improve_all.py writes models/best_detector.json with the winner; CCTV_MODEL / CCTV_DET_THRESHOLD override it.
try:
    import json as _json
    _BEST = _json.loads((ROOT / "models" / "best_detector.json").read_text())
except Exception:        # no file (or unreadable): fall back to the defaults below
    _BEST = {}
MODEL_VARIANT = os.environ.get("CCTV_MODEL", _BEST.get("model", "yolox_m"))
MODEL_NAME = f"{MODEL_VARIANT.replace('@', '-').replace('+', '_')}_640"          # tags feature shards so results of different models never get mixed
MODEL_PATH = ROOT / "models" / f"{MODEL_VARIANT}.onnx"
SAMPLES_DIR = DATA / "samples"          # small set of frames shipped for the online demo
CAMERA_INFO_CSV = DATA / "cctv_info.csv"

INDEX_PARQUET = PROCESSED_DIR / "image_index.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "image_features.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "images_clean.parquet"
CAMERA_PARQUET = PROCESSED_DIR / "camera_summary.parquet"
HOURLY_PARQUET = PROCESSED_DIR / "camera_hourly.parquet"

TIMEZONE = "Asia/Bangkok"

# COCO class ids used for traffic counting (YOLOX is COCO-trained)
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0
# Passenger-car-equivalent weights -> a single congestion-load number
PCE_WEIGHTS = {"bicycle": 0.3, "motorcycle": 0.5, "car": 1.0, "bus": 2.5, "truck": 2.0}

DET_INPUT_SIZE = 640
DET_SCORE_THRESHOLD = float(os.environ.get("CCTV_DET_THRESHOLD", _BEST.get("threshold", 0.30)))
DET_NMS_THRESHOLD = 0.45

# Image quality / cleaning rules
BLANK_CONTRAST = 3.0        # grayscale std below this -> blank / white / black frame
DEGRADED_SHARPNESS_RATIO = 0.25   # sharpness < 25% of the camera's median -> blurred / obstructed
FROZEN_RUN_LENGTH = 3       # >= 3 identical consecutive frames -> camera feed is frozen
CAP_PERCENTILE = 0.995      # winsorise counts per camera at this percentile
GAP_MINUTES = 30            # capture gap longer than this is reported as missing data
ANOMALY_Z = 3.0             # |z| above this -> unusual camera-hour
EVENING_DARK_BRIGHTNESS = 80  # mean gray below this -> "low light" bucket

MODEL_TEST_DAYS = 2         # last N days are held out when evaluating the forecaster
RANDOM_STATE = 42
