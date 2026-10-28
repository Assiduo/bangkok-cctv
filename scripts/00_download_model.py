"""Download the YOLOX ONNX weights chosen by CCTV_MODEL (default yolox_m) if missing.

    CCTV_MODEL=yolox_x python scripts/00_download_model.py                 # s | m | l | x
    CCTV_MODEL=yolox_x@tile+cctv_ft python scripts/00_download_model.py    # fused spec: fetches the YOLOX parts it needs
"""
import urllib.request

from cctv import config

URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{name}.onnx"
VALID = ("yolox_s", "yolox_m", "yolox_l", "yolox_x")


def main() -> None:
    names = sorted({t.partition("@")[0] for t in config.MODEL_VARIANT.split("+") if t})
    for name in names:
        path = config.ROOT / "models" / f"{name}.onnx"
        if name.startswith("cctv_"):
            print(f"[model] custom fine-tuned model {path.name}: nothing to download (see docs/FINETUNING.md)"
                  + ("" if path.exists() else "  -- but the file is MISSING, run scripts/11_train_detector.py first"))
            continue
        if name not in VALID:
            raise SystemExit(f"CCTV_MODEL parts must be one of {VALID} or cctv_*, got {name!r}")
        if path.exists():
            print(f"[model] already present: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = URL.format(name=name)
        print(f"[model] downloading {url}")
        urllib.request.urlretrieve(url, path)
        print(f"[model] saved to {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
