"""Run the detector + quality metrics over every indexed image (parallel, resumable, sharded).

Usage:  python scripts/02_extract_features.py [--workers N] [--shard-size 1000] [--limit N]
Re-running skips shards that already exist, so an interrupted run can simply be restarted.
Shards are stored per model (config.MODEL_NAME), so changing the detector never reuses old results.
"""
import argparse
import os
import time
from multiprocessing import Pool

import pandas as pd

from cctv import config
from cctv.detector import VehicleDetector
from cctv.detectors import make_detector
from cctv.features import extract_features
from cctv.imageio import read_bgr

SHARD_DIR = config.PROCESSED_DIR / f"feature_shards_{config.MODEL_NAME}"
_detector: VehicleDetector | None = None


def _init_worker() -> None:
    global _detector
    _detector = make_detector()          # one single-threaded ONNX session per worker process


def _process_one(path: str) -> dict:
    img = read_bgr(config.RAW_DIR / path)
    if img is None:  # corrupt / truncated file -> keep a row so cleaning can report it
        return {"image_path": path, "decode_ok": False}
    return {"image_path": path, "decode_ok": True, **extract_features(img, _detector)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None, help="default: all CPU cores (3 when a GPU is used)")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="only process first N images (debug)")
    args = ap.parse_args()

    index = pd.read_parquet(config.INDEX_PARQUET)
    if args.limit:
        index = index.head(args.limit)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = -(-len(index) // args.shard_size)
    probe = make_detector()
    device = probe.device
    del probe
    workers = args.workers or (min(3, os.cpu_count() or 1) if device == "GPU" else os.cpu_count() or 1)
    print(f"[features] model={config.MODEL_NAME} device={device} workers={workers} shards={n_shards}", flush=True)
    started, done_now = time.time(), 0
    with Pool(workers, initializer=_init_worker) as pool:
        for i in range(n_shards):
            shard_path = SHARD_DIR / f"part_{i:04d}.parquet"
            if shard_path.exists():
                continue
            chunk = index.iloc[i * args.shard_size:(i + 1) * args.shard_size]
            rows = pool.map(_process_one, chunk["image_path"].tolist(), chunksize=8)
            pd.DataFrame(rows).to_parquet(shard_path, index=False)
            done_now += len(chunk)
            rate = done_now / (time.time() - started)
            eta = (len(index) - (i + 1) * args.shard_size) / max(rate, 1e-9) / 60
            print(f"[features] shard {i + 1}/{n_shards}  {rate:.1f} img/s  ETA {max(eta, 0):.0f} min", flush=True)

    shards = sorted(SHARD_DIR.glob("part_*.parquet"))
    features = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    features.to_parquet(config.FEATURES_PARQUET, index=False)
    print(f"[features] wrote {len(features):,} rows -> {config.FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
