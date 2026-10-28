"""Step 4 – data cleaning, quality flags and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

DAY_PARTS = [(6, 7, "early morning"), (7, 10, "morning peak"), (10, 16, "midday"),
             (16, 20, "evening peak"), (20, 24, "night")]
DAY_PART_ORDER = [p[2] for p in DAY_PARTS]


def day_part(hour: int) -> str:
    for lo, hi, name in DAY_PARTS:
        if lo <= hour < hi:
            return name
    return "night"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df = df.assign(
        date=ts.dt.normalize(), hour=ts.dt.hour, weekday=ts.dt.dayofweek,
        weekday_name=ts.dt.day_name(), is_weekend=ts.dt.dayofweek >= 5)
    df["day_part"] = pd.Categorical(df["hour"].map(day_part), categories=DAY_PART_ORDER, ordered=True)
    df["day_type"] = np.where(df["is_weekend"], "Weekend", "Weekday")
    return df


def flag_frozen(df: pd.DataFrame, run_length: int = config.FROZEN_RUN_LENGTH) -> pd.Series:
    """True for frames inside a run of >= ``run_length`` pixel-identical consecutive frames."""
    key = df["img_hash"].astype(str) + "|" + df["brightness"].round(4).astype(str) + "|" + df["sharpness"].round(2).astype(str)
    df = df.assign(_key=key).sort_values(["camera_id", "timestamp"])
    new_run = (df["_key"] != df.groupby("camera_id")["_key"].shift()) | (df["camera_id"] != df["camera_id"].shift())
    run_id = new_run.cumsum()
    run_size = df.groupby(run_id)["_key"].transform("size")
    return (run_size >= run_length).reindex(df.index).sort_index()


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_blank"] = df["contrast"] < config.BLANK_CONTRAST
    cam_median = df.groupby("camera_id")["sharpness"].transform("median")
    df["is_degraded"] = df["sharpness"] < config.DEGRADED_SHARPNESS_RATIO * cam_median
    df["is_frozen"] = flag_frozen(df)
    df["is_low_light"] = df["brightness"] < config.EVENING_DARK_BRIGHTNESS
    df["usable"] = ~(df["is_blank"] | df["is_degraded"] | df["is_frozen"])
    return df


def add_normalised_load(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise per camera and scale to a 0-100 *relative* congestion index.

    Cameras differ in angle/zoom, so raw counts are not comparable across cameras.
    The index expresses how busy a camera is relative to its own 95th percentile.
    """
    df = df.copy()
    usable = df[df["usable"]]
    cap = usable.groupby("camera_id")["pce_load"].quantile(config.CAP_PERCENTILE).rename("_cap")
    df = df.join(cap, on="camera_id")
    df["pce_capped"] = df["pce_load"].clip(upper=df["_cap"])
    p95 = usable.groupby("camera_id")["pce_load"].quantile(0.95).clip(lower=1.0).rename("_p95")
    df = df.join(p95, on="camera_id")
    df["congestion_idx"] = (100 * df["pce_capped"] / df["_p95"]).clip(upper=100).round(2)
    return df.drop(columns=["_cap", "_p95"])


def gap_report(df: pd.DataFrame) -> pd.DataFrame:
    """Capture gaps longer than GAP_MINUTES per camera (missing-data diagnostics)."""
    d = df.sort_values(["camera_id", "timestamp"])
    gap = d.groupby("camera_id")["timestamp"].diff().dt.total_seconds().div(60)
    same_day = d.groupby("camera_id")["date"].diff().dt.days.fillna(0) == 0
    out = d.assign(gap_minutes=gap)[same_day & (gap > config.GAP_MINUTES)]
    return out[["camera_id", "timestamp", "gap_minutes"]].reset_index(drop=True)


def clean(index: pd.DataFrame, features: pd.DataFrame, cameras: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge index + features + camera metadata and return (clean_df, quality_report)."""
    report = {"n_indexed": int(len(index)), "n_features": int(len(features))}
    df = index.merge(features, on="image_path", how="left", validate="1:1")
    report["n_undecodable"] = int((~df["decode_ok"].fillna(False).astype(bool)).sum())
    df = df[df["decode_ok"].fillna(False).astype(bool)].drop(columns="decode_ok")
    before = len(df)
    df = df.drop_duplicates(["camera_id", "timestamp"])
    report["n_duplicate_timestamps"] = int(before - len(df))
    unknown = ~df["camera_id"].isin(cameras["camera_id"])
    report["n_unknown_camera"] = int(unknown.sum())
    df = df[~unknown]

    df = add_time_features(df)
    df = add_quality_flags(df)
    df = add_normalised_load(df)
    df = df.merge(cameras[["camera_id", "camera_name", "camera_label", "latitude", "longitude"]],
                  on="camera_id", how="left")
    for col in ("is_blank", "is_degraded", "is_frozen", "is_low_light", "usable"):
        report[f"n_{col}"] = int(df[col].sum())
    report["n_clean"] = int(len(df))
    report["n_usable"] = int(df["usable"].sum())
    report["n_gaps_over_threshold"] = int(len(gap_report(df)))
    return df.sort_values(["camera_id", "timestamp"]).reset_index(drop=True), report
