"""Which detector threshold produced the counts in data/processed/image_features.parquet?

Re-detects N random frames at several thresholds and reports how many frames match the stored vehicle_total exactly.
    python scripts/21_check_threshold.py            (30 frames, ~1 min)
The threshold with the highest match rate is the one the stored results were made with.
"""
import argparse

import pandas as pd

from cctv import config
from cctv.detectors import make_detector
from cctv.imageio import read_bgr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.4])
    a = ap.parse_args()
    feats = pd.read_parquet(config.FEATURES_PARQUET)
    feats = feats[feats["decode_ok"]].sample(a.n, random_state=1)
    det = make_detector()
    print(f"detector in config: {config.MODEL_NAME}, config threshold {config.DET_SCORE_THRESHOLD}")
    imgs = [(read_bgr(config.RAW_DIR / r.image_path), r.vehicle_total) for r in feats.itertuples()]
    imgs = [(im, t) for im, t in imgs if im is not None]
    for thr in a.thresholds:
        det.score_thr = thr
        hit = sum(int(det.count(im)["vehicle_total"]) == int(t) for im, t in imgs)
        print(f"threshold {thr:.2f}: {hit}/{len(imgs)} frames match the stored counts")
    print("-> the threshold with the most matches is what the stored results used")


if __name__ == "__main__":
    main()
