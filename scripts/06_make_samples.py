"""Copy a small, representative set of raw frames to data/samples for the online demo.

The public dashboard cannot ship the 1.7 GB image set, so the Image explorer falls back to this folder:
3 weekday frames per camera (08h, 13h, 18h) plus 2 frames for each of the top anomalies.
Re-run after 05_analyze.py whenever the anomalies change.   Usage: python scripts/06_make_samples.py
"""
import shutil

import pandas as pd

from cctv import config

HOURS = (8, 13, 18)
TOP_ANOMALIES = 60


def main() -> None:
    df = pd.read_parquet(config.CLEAN_PARQUET, columns=["camera_id", "timestamp", "image_path", "date", "hour"])
    weekdays = df[df["date"].dt.dayofweek < 5]
    sample_day = weekdays.groupby("date").size().idxmax()
    picks = [g.iloc[len(g) // 2] for _, g in weekdays[(weekdays["date"] == sample_day) & weekdays["hour"].isin(HOURS)]
             .groupby(["camera_id", "hour"])]
    paths = {r["image_path"] for r in picks}

    anomalies = pd.read_parquet(config.PROCESSED_DIR / "anomalies.parquet").head(TOP_ANOMALIES)
    for a in anomalies.itertuples():
        frames = df[(df.camera_id == a.camera_id) & (df.date == a.date) & (df.hour == a.hour)].sort_values("timestamp")
        paths.update(frames["image_path"].iloc[[0, len(frames) // 2]] if len(frames) > 1 else frames["image_path"])

    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for rel in sorted(paths):
        target = config.SAMPLES_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.RAW_DIR / rel, target)
    size = sum(p.stat().st_size for p in config.SAMPLES_DIR.rglob("*.jpg")) / 1e6
    print(f"[samples] {len(paths)} frames ({size:.1f} MB) -> {config.SAMPLES_DIR}  (weekday used: {sample_day:%a %d %b})")


if __name__ == "__main__":
    main()
