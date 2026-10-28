"""One command: train the stronger detector, compare every way of detecting, show the pictures, and pick the winner.

    python scripts/20_improve_all.py                    # train yolo11l @ 960, then evaluate everything (~1 h on a GPU)
    python scripts/20_improve_all.py --skip-train       # evaluate only (models already in models/)

Contenders (each is a CCTV_MODEL spec, see src/cctv/ensemble.py):
    yolox_x                    the original detector
    yolox_x@tile               same model on the whole frame + 4 enlarged tiles (finds small, far vehicles)
    cctv_ft / cctv_l960        fine-tuned on our labels (whatever exists in models/)
    yolox_x@tile+cctv_l960     fusion of the two (keeps YOLOX's big vehicles and the fine-tuned model's small ones)

Outputs (outputs/improvement/): report.csv, report_by_daypart.csv, pictures of the winner vs the original, and
models/best_detector.json, which every pipeline script then picks up automatically (CCTV_MODEL still overrides it).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cctv import config
from cctv.detectors import make_detector
from cctv.imageio import read_bgr
from cctv.labeling import IMAGES, LABEL_DIR, LABELS, VEHICLE_FT, dets_to_ft, match, read_labels

OUT = config.ROOT / "outputs" / "improvement"
LOW = 0.10
NAMES = {1: "bic", 2: "car", 3: "mot", 5: "bus", 7: "truck"}
COLOURS = {2: (0, 200, 0), 3: (0, 170, 255), 5: (255, 80, 0), 7: (0, 0, 255), 1: (255, 0, 255)}


def part(stem: str) -> str:
    h = int(stem.split("_")[2][:2])
    return "day (06-18h)" if 6 <= h < 18 else "night (18-06h)"


def draw(img, dets, title, thr, k=2):
    im = cv2.resize(img, (704, 576), interpolation=cv2.INTER_CUBIC)
    sx, sy = 704 / img.shape[1], 576 / img.shape[0]
    n = 0
    for c, s, x1, y1, x2, y2 in dets:
        if c not in NAMES or s < thr:
            continue
        n += 1
        p = (int(x1 * sx), int(y1 * sy))
        cv2.rectangle(im, p, (int(x2 * sx), int(y2 * sy)), COLOURS[c], 1)
        cv2.putText(im, NAMES[c], (p[0], max(p[1] - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOURS[c], 1, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (330, 20), (0, 0, 0), -1)
    cv2.putText(im, f"{title}: {n} vehicles", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yolo11l.pt")
    ap.add_argument("--name", default="cctv_l960")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--pipeline", action="store_true", help="afterwards re-run 02 / 04 / 05 on all frames with the winner (slow if the winner is a tile/ensemble mode)")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    args = ap.parse_args()
    models = config.ROOT / "models"
    OUT.mkdir(parents=True, exist_ok=True)

    n_lab = len(list(LABELS.glob("*.txt")))
    print(f"[1/4] labelled images: {n_lab}")
    if not args.skip_train:
        print(f"[2/4] training {args.base} at {args.imgsz}px for up to {args.epochs} epochs (a progress table will scroll by)")
        env = {**__import__("os").environ, "PYTHONPATH": str(config.ROOT / "src")}
        for batch in dict.fromkeys([args.batch, 4, 2]):          # out of GPU memory -> retry with a smaller batch
            r = subprocess.run([sys.executable, str(config.ROOT / "scripts" / "11_train_detector.py"), "--base", args.base, "--epochs", str(args.epochs),
                                "--imgsz", str(args.imgsz), "--batch", str(batch), "--name", args.name, "--patience", "30"], env=env)
            if r.returncode == 0:
                break
            print(f"training failed with batch={batch}; trying a smaller batch ...")
        else:
            raise SystemExit("training failed for every batch size - see the error above")
    else:
        print("[2/4] training skipped")

    have = {p.stem for p in models.glob("cctv_*.onnx")}
    tuned = [m for m in (args.name, "cctv_ft") if m in have]
    specs = ["yolox_x", "yolox_x@tile"] + tuned + [f"yolox_x@tile+{m}" for m in tuned] + [f"yolox_x+{m}" for m in tuned]
    specs = [s for s in specs if all((models / f"{t.partition('@')[0]}.onnx").exists() for t in s.split("+"))]

    val = LABEL_DIR / "val_stems.txt"
    stems = [s for s in val.read_text().split() if s] if val.exists() else [p.stem for p in IMAGES.glob("*.jpg")]
    stems = [s for s in stems if (LABELS / f"{s}.txt").exists()]
    imgs = {s: read_bgr(IMAGES / f"{s}.jpg") for s in stems}
    gts = {s: [g for g in read_labels(LABELS / f"{s}.txt", imgs[s].shape[1], imgs[s].shape[0]) if g[0] in VEHICLE_FT] for s in stems}
    print(f"[3/4] evaluating {len(specs)} detectors on {len(stems)} frames from cameras the models never trained on")

    rows, byp, cache = [], [], {}
    for spec in specs:
        det = make_detector(spec)
        det.score_thr = LOW
        cache[spec] = {s: det.detect(imgs[s]) for s in stems}      # detect once, filter by threshold afterwards
        for thr in args.thresholds:
            tp = fp = fn = 0
            err, per = [], {}
            for s in stems:
                pred = [p for p in dets_to_ft([d for d in cache[spec][s] if d[1] >= thr]) if p[0] in VEHICLE_FT]
                a, b, c = match(gts[s], pred)
                tp, fp, fn = tp + a, fp + b, fn + c
                e = len(pred) - len(gts[s])
                err.append(e)
                per.setdefault(part(s), []).append(e)
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            rows.append({"detector": spec, "threshold": thr, "precision": prec, "recall": rec,
                         "F1": 2 * prec * rec / max(prec + rec, 1e-9),
                         "count_MAE": float(np.mean(np.abs(err))), "count_bias": float(np.mean(err))})
            for dp, es in per.items():
                byp.append({"detector": spec, "threshold": thr, "daypart": dp, "frames": len(es),
                            "count_MAE": float(np.mean(np.abs(es))), "count_bias": float(np.mean(es))})
        best = max((r for r in rows if r["detector"] == spec), key=lambda r: r["F1"])
        print(f"   {spec:28s} best F1 {best['F1']:.3f} at thr {best['threshold']:.2f}  (P {best['precision']:.2f} R {best['recall']:.2f}  MAE {best['count_MAE']:.2f})")

    rep = pd.DataFrame(rows).round(3)
    rep.to_csv(OUT / "report.csv", index=False)
    pd.DataFrame(byp).round(3).to_csv(OUT / "report_by_daypart.csv", index=False)
    # winner = highest F1; among near-ties (within 0.01) prefer the one whose counts are least biased
    top = rep[rep["F1"] >= rep["F1"].max() - 0.01]
    win = top.loc[top["count_bias"].abs().idxmin()]
    base = rep[(rep["detector"] == "yolox_x") & (rep["threshold"] == config.DET_SCORE_THRESHOLD)]
    print("\n[4/4] RESULT")
    print(rep.sort_values("F1", ascending=False).head(8).to_string(index=False))
    if not base.empty:
        b = base.iloc[0]
        print(f"\noriginal (yolox_x @ {b['threshold']}):  F1 {b['F1']:.3f}  count error {b['count_MAE']:.2f}  bias {b['count_bias']:+.2f}")
    print(f"WINNER: {win['detector']} @ threshold {win['threshold']}:  F1 {win['F1']:.3f}  count error {win['count_MAE']:.2f}  bias {win['count_bias']:+.2f}")
    dp = pd.DataFrame(byp)
    dp = dp[(dp.detector == win["detector"]) & (dp.threshold == win["threshold"])]
    print("error by time of day for the winner:", {r.daypart: round(r.count_MAE, 2) for r in dp.itertuples()})

    step = max(len(stems) // 12, 1)
    for i, s in enumerate(stems[::step][:12]):
        left, right = draw(imgs[s], cache["yolox_x"][s], "yolox_x (original)", config.DET_SCORE_THRESHOLD), \
                      draw(imgs[s], cache[win["detector"]][s], f"winner: {win['detector']}", win["threshold"])
        cv2.imwrite(str(OUT / f"pic_{i + 1:02d}_{s}.jpg"), np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (models / "best_detector.json").write_text(json.dumps({"model": win["detector"], "threshold": float(win["threshold"])}, indent=1))
    print(f"\nPictures: {OUT}\\pic_*.jpg   Tables: {OUT}\\report.csv")
    print(f"Saved models/best_detector.json -> the pipeline now uses '{win['detector']}' at {win['threshold']}.")
    print("To apply it to all frames run:  python scripts/02_extract_features.py  then 04, 05 (see docs/FINETUNING.md).")
    print("Delete models/best_detector.json to go back to the original detector.")
    if args.pipeline:
        for step in ("02_extract_features.py", "04_clean_data.py", "05_analyze.py"):
            print(f"\n=== {step} ===", flush=True)
            subprocess.run([sys.executable, str(config.ROOT / "scripts" / step)], check=True, env={**__import__("os").environ, "PYTHONPATH": str(config.ROOT / "src")})


if __name__ == "__main__":
    main()
