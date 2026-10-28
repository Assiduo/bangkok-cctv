"""Merge index + features + camera metadata, flag bad frames, engineer features."""
import json

import pandas as pd

from cctv import config
from cctv.cleaning import clean, gap_report
from cctv.indexing import load_camera_info


def main() -> None:
    index = pd.read_parquet(config.INDEX_PARQUET)
    features = pd.read_parquet(config.FEATURES_PARQUET)
    df, report = clean(index, features, load_camera_info())
    df.to_parquet(config.CLEAN_PARQUET, index=False)
    gap_report(df).to_parquet(config.PROCESSED_DIR / "capture_gaps.parquet", index=False)
    (config.PROCESSED_DIR / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
