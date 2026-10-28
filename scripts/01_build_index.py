"""Scan data/raw and write data/processed/image_index.parquet."""
from cctv import config
from cctv.indexing import build_image_index


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index = build_image_index()
    index.to_parquet(config.INDEX_PARQUET, index=False)
    span = f"{index.timestamp.min()} -> {index.timestamp.max()}"
    print(f"[index] {len(index):,} images, {index.camera_id.nunique()} cameras, {span}")


if __name__ == "__main__":
    main()
