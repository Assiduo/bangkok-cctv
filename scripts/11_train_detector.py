"""Fine-tune a COCO-pretrained detector on our corrected labels and export it as models/cctv_ft.onnx.

    pip install ultralytics            (once; installs PyTorch)
    python scripts/11_train_detector.py --epochs 80

The split is by CAMERA (20% of cameras held out for validation), so the score reflects cameras the model never saw.
Then: CCTV_MODEL=cctv_ft python scripts/02_extract_features.py ... (or the whole pipeline) to count with the new model.
"""
import argparse
import random
import shutil

from cctv import config
from cctv.detectors import FT_CLASSES
from cctv.labeling import IMAGES, LABEL_DIR, LABELS

DATASET = LABEL_DIR / "dataset"


def build_dataset(val_frac: float, seed: int) -> tuple[int, int]:
    stems = sorted(p.stem for p in IMAGES.glob("*.jpg") if (LABELS / f"{p.stem}.txt").exists())
    cams = sorted({s.split("_")[0] for s in stems})
    random.Random(seed).shuffle(cams)
    val_cams = set(cams[: max(1, int(len(cams) * val_frac))])
    if DATASET.exists():
        shutil.rmtree(DATASET)
    counts = {"train": 0, "val": 0}
    for s in stems:
        split = "val" if s.split("_")[0] in val_cams else "train"
        for kind, src, ext in (("images", IMAGES, "jpg"), ("labels", LABELS, "txt")):
            dst = DATASET / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / f"{s}.{ext}", dst / f"{s}.{ext}")
        counts[split] += 1
    (DATASET / "data.yaml").write_text(
        f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(FT_CLASSES)))
    (LABEL_DIR / "val_stems.txt").write_text("\n".join(s for s in stems if s.split("_")[0] in val_cams) + "\n")
    return counts["train"], counts["val"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yolo11m.pt", help="pretrained weights (yolo11s.pt is faster, yolo11l.pt stronger)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640, help="frames are only 352x288; 640 upsamples them so small cars are bigger")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--name", default="cctv_ft", help="output model name; must start with cctv_ (e.g. cctv_l960)")
    ap.add_argument("--freeze", type=int, default=10, help="backbone layers kept fixed (0 = train everything)")
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()

    from ultralytics import YOLO

    n_train, n_val = build_dataset(args.val_frac, config.RANDOM_STATE)
    print(f"[train] {n_train} train / {n_val} validation images (split by camera)")
    if n_train < 50:
        print("[train] WARNING: very few labelled images; results will be weak. Label at least 150.")
    assert args.name.startswith('cctv_'), '--name must start with cctv_'
    model = YOLO(args.base)
    model.train(data=str(DATASET / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                freeze=args.freeze, cos_lr=True, hsv_v=0.5, degrees=0.0, fliplr=0.5, mosaic=1.0, close_mosaic=10, workers=2,
                project=str(config.ROOT / "outputs" / "train"), name=args.name, exist_ok=True)
    best = config.ROOT / "outputs" / "train" / args.name / "weights" / "best.pt"
    exported = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    shutil.copy(exported, config.ROOT / "models" / f"{args.name}.onnx")
    print(f"[train] saved models/{args.name}.onnx.  Evaluate:  python scripts/12_evaluate_detectors.py")


if __name__ == "__main__":
    main()
