"""Extract cctv1..cctv4.zip into a short, ASCII-only folder layout:  <out>/<camera_id>/<camera_id>_cam_<YYYYMMDD>_<HHMMSS>.jpg

Why not use Windows "Extract All"?  The zips store Thai file names as UTF-8 without the flag Windows looks for,
so names turn into garbage and paths exceed Windows' 260-character limit.  This script repairs the names and writes
short ASCII ones (camera names live in cctv_info.csv anyway; the pipeline only needs camera id + timestamp).

Usage (Windows, keep the output path SHORT):
    python scripts/extract_raw_zips.py --out C:\\cctv_images C:\\path\\cctv1.zip C:\\path\\cctv2.zip C:\\path\\cctv3.zip C:\\path\\cctv4.zip
Then:  set CCTV_RAW_DIR to that folder.  Safe to re-run: existing files are skipped.
"""
import argparse
import zipfile
from pathlib import Path

from cctv import config
from cctv.indexing import parse_filename


def fix_name(name: str, flag_bits: int) -> str:
    """Repair names that are UTF-8 bytes mis-decoded as cp437 (zip entries without the UTF-8 flag)."""
    if flag_bits & 0x800:
        return name
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def extract_zip(zip_path: Path, out_dir: Path) -> tuple[int, int]:
    done = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = fix_name(info.filename, info.flag_bits)
            if info.is_dir() or name.startswith("__MACOSX") or "/." in name or not name.lower().endswith((".jpg", ".jpeg")):
                continue
            meta = parse_filename(name.rsplit("/", 1)[-1])
            if meta is None:
                skipped += 1
                continue
            cam, ts = meta["camera_id"], meta["timestamp"]
            target = out_dir / str(cam) / f"{cam}_cam_{ts:%Y%m%d_%H%M%S}.jpg"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))
            done += 1
    return done, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zips", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=config.RAW_DIR, help="short destination folder, e.g. C:\\cctv_images")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for z in args.zips:
        done, skipped = extract_zip(z, args.out)
        total += done
        print(f"[unzip] {z.name}: {done:,} images" + (f" ({skipped} unparseable names skipped)" if skipped else ""), flush=True)
    print(f"[unzip] total {total:,} images -> {args.out}\nNext: set CCTV_RAW_DIR to {args.out}")


if __name__ == "__main__":
    main()
