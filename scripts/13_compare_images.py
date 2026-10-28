"""Draw side-by-side detections (left = baseline, right = fine-tuned) so you can SEE which model is right.

    python scripts/13_compare_images.py                 # 10 held-out frames -> outputs/compare/
    python scripts/13_compare_images.py --n 20 --thr 0.25
    python scripts/13_compare_images.py --models yolox_x.onnx yolox_m.onnx cctv_ft.onnx

Frames come from the validation cameras (never used for training). Box colour = vehicle type, label = class.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from cctv import config
from cctv.detectors import make_detector
from cctv.imageio import read_bgr
from cctv.labeling import IMAGES, LABEL_DIR, LABELS

NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLOURS = {2: (0, 200, 0), 3: (0, 170, 255), 5: (255, 80, 0), 7: (0, 0, 255), 1: (255, 0, 255)}


def draw(img, dets, title, scale=2):
    im = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    n = 0
    for c, _, x1, y1, x2, y2 in dets:
        if c not in NAMES:
            continue
        n += 1
        p1 = (int(x1 * scale), int(y1 * scale))
        cv2.rectangle(im, p1, (int(x2 * scale), int(y2 * scale)), COLOURS[c], 1)
        cv2.putText(im, NAMES[c][:3], (p1[0], max(p1[1] - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOURS[c], 1, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (260, 20), (0, 0, 0), -1)
    cv2.putText(im, f"{title}: {n} vehicles", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolox_x.onnx", "cctv_ft.onnx"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--thr", type=float, default=0.30)
    ap.add_argument("--images", help="folder of your own .jpg frames to use instead of the validation frames")
    args = ap.parse_args()

    if args.images:
        paths = sorted(Path(args.images).glob("*.jpg"))
    else:
        val = LABEL_DIR / "val_stems.txt"
        stems = sorted(s for s in val.read_text().split() if s) if val.exists() else sorted(p.stem for p in IMAGES.glob("*.jpg"))
        stems = [s for i, s in enumerate(stems) if i % max(len(paths) // args.n, 1) == 0][: args.n]   # evenly spread over cameras/times
        paths = [IMAGES / f"{s}.jpg" for s in stems]
    dets = [make_detector(config.ROOT / "models" / m, score_thr=args.thr) for m in args.models]

    out = config.ROOT / "outputs" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        img = read_bgr(path)
        h, w = img.shape[:2]
        if (w, h) != (352, 288):
            img = cv2.resize(img, (352, 288))
        panel = np.hstack([draw(img, d.detect(img), m.replace(".onnx", "")) for d, m in zip(dets, args.models)])
        cv2.imwrite(str(out / f"{path.stem}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[compare] wrote {len(paths)} images to {out}")


if __name__ == "__main__":
    main()
