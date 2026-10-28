"""Pick diverse frames and pre-label them with the current detector, so humans only have to CORRECT boxes.

    python scripts/10_make_label_set.py --n 300 --thr 0.15

Output: data/labeling/images/*.jpg, data/labeling/labels/*.txt (YOLO format), data/labeling/classes.txt, manifest.csv
Frames are spread over every camera and five parts of the day (evening/dusk frames are the hard ones, so they are included).
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd

from cctv import config
from cctv.detectors import FT_CLASSES, make_detector
from cctv.imageio import read_bgr
from cctv.labeling import IMAGES, LABEL_DIR, LABELS, write_labels

BUCKETS = [(6, 9), (9, 12), (12, 16), (16, 19), (19, 22)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of frames to label (150 is a workable minimum)")
    ap.add_argument("--thr", type=float, default=0.15, help="pre-label score threshold (low = more boxes to delete, fewer to add)")
    args = ap.parse_args()

    idx = pd.read_parquet(config.INDEX_PARQUET)
    idx["bucket"] = idx["timestamp"].dt.hour.map(lambda h: next((i for i, (a, b) in enumerate(BUCKETS) if a <= h < b), 4))
    rng = np.random.default_rng(config.RANDOM_STATE)
    cams = idx["camera_id"].unique()
    picks, slot = [], 0
    args.n = min(args.n, len(idx))
    while len(picks) < args.n:                       # round-robin over cameras, rotating through the day buckets
        before = len(picks)
        for cam in rng.permutation(cams):
            sub = idx[(idx.camera_id == cam) & (idx.bucket == slot % len(BUCKETS))]
            sub = sub[~sub["image_path"].isin([p for p in picks])]
            if len(sub):
                picks.append(sub.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]["image_path"])
            slot += 1
            if len(picks) >= args.n:
                break
        if len(picks) == before and slot > 50 * len(BUCKETS) * len(cams):
            break
    chosen = idx[idx["image_path"].isin(picks)].copy()

    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    det = make_detector()
    det.score_thr = args.thr
    print(f"[label-set] pre-labelling {len(chosen)} frames with {det.model_path.name} (thr {args.thr}, {det.device})")
    names, cands = [], {}
    for r in chosen.itertuples():
        img = read_bgr(config.RAW_DIR / r.image_path)
        if img is None:
            continue
        name = f"{r.camera_id}_{r.timestamp:%Y%m%d_%H%M%S}"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (IMAGES / f"{name}.jpg").write_bytes(buf.tobytes())
        dets = det.detect(img)
        write_labels(LABELS / f"{name}.txt", dets, img.shape[1], img.shape[0])
        cands[name] = [[c, round(sc, 3), *map(lambda v: round(v, 1), box)] for c, sc, *box in dets]
        names.append(name)
    (LABEL_DIR / "candidates.json").write_text(json.dumps(cands))   # detector boxes WITH scores, for review tools
    chosen["name"] = names if len(names) == len(chosen) else None
    chosen.drop(columns=["bucket"]).to_csv(LABEL_DIR / "manifest.csv", index=False)
    (LABEL_DIR / "classes.txt").write_text("\n".join(FT_CLASSES) + "\n")
    print(f"[label-set] done: {len(names)} images in {IMAGES}\n[label-set] next: open them in a labelling tool, fix the boxes, "
          f"and save YOLO txt files back into {LABELS}")


if __name__ == "__main__":
    main()
