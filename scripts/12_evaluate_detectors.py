"""Compare detectors on the held-out, human-corrected frames (the report table for the presentation).

    python scripts/12_evaluate_detectors.py --models yolox_x.onnx cctv_ft.onnx

For each model and score threshold: precision / recall / F1 at IoU 0.5 for vehicles, and the error of the per-frame vehicle COUNT
(mean absolute error and bias), which is what the congestion index is built from.
"""
import argparse

import pandas as pd

from cctv import config
from cctv.detectors import make_detector
from cctv.imageio import read_bgr
from cctv.labeling import IMAGES, LABEL_DIR, LABELS, VEHICLE_FT, dets_to_ft, match, read_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolox_x.onnx", "cctv_ft.onnx"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.25, 0.30, 0.40])
    ap.add_argument("--all", action="store_true", help="evaluate on every labelled image, not just the validation cameras")
    args = ap.parse_args()

    val = LABEL_DIR / "val_stems.txt"
    stems = ([s for s in val.read_text().split() if s] if val.exists() and not args.all
             else [p.stem for p in IMAGES.glob("*.jpg")])
    stems = [s for s in stems if (LABELS / f"{s}.txt").exists()]
    print(f"[eval] {len(stems)} labelled frames ({'validation cameras only' if val.exists() and not args.all else 'all'})")
    imgs = {s: read_bgr(IMAGES / f"{s}.jpg") for s in stems}
    gts = {s: [g for g in read_labels(LABELS / f"{s}.txt", imgs[s].shape[1], imgs[s].shape[0]) if g[0] in VEHICLE_FT] for s in stems}

    rows = []
    for m in args.models:
        det = make_detector(config.ROOT / "models" / m)
        for thr in args.thresholds:
            det.score_thr = thr
            tp = fp = fn = 0
            err = []
            for s in stems:
                pred = [p for p in dets_to_ft(det.detect(imgs[s])) if p[0] in VEHICLE_FT]
                a, b, c = match(gts[s], pred)
                tp, fp, fn = tp + a, fp + b, fn + c
                err.append(len(pred) - len(gts[s]))
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            rows.append({"model": m, "threshold": thr, "precision": prec, "recall": rec,
                         "F1": 2 * prec * rec / max(prec + rec, 1e-9),
                         "count_MAE": sum(abs(e) for e in err) / len(err), "count_bias": sum(err) / len(err)})
            print(f"[eval] {m:16s} thr={thr:.2f}  P={prec:.2f} R={rec:.2f} F1={rows[-1]['F1']:.2f}  MAE={rows[-1]['count_MAE']:.1f}  bias={rows[-1]['count_bias']:+.1f}")
    out = pd.DataFrame(rows).round(3)
    dest = config.ROOT / "outputs"
    dest.mkdir(exist_ok=True)
    out.to_csv(dest / "detector_eval.csv", index=False)
    best = out.loc[out["F1"].idxmax()]
    print(f"\n[eval] best F1: {best['model']} at threshold {best['threshold']} (F1 {best['F1']:.2f}). Saved outputs/detector_eval.csv")
    print(f"[eval] use it:  set CCTV_MODEL={best['model'].replace('.onnx', '')} and DET_SCORE_THRESHOLD={best['threshold']} in src/cctv/config.py")


if __name__ == "__main__":
    main()
