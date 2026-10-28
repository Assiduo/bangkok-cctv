"""Cached loaders and sidebar filters for the dashboard."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cctv import config  # noqa: E402


@dataclass
class Data:
    hourly: pd.DataFrame
    cameras: pd.DataFrame
    anomalies: pd.DataFrame
    images: pd.DataFrame
    predictions: pd.DataFrame
    importance: pd.DataFrame
    model_meta: dict
    quality: dict
    gaps: pd.DataFrame
    insights: list[str] = field(default_factory=list)

    @property
    def has_weather(self) -> bool:
        return "precipitation" in self.hourly.columns and self.hourly["precipitation"].notna().any()

    @property
    def has_air(self) -> bool:
        return "pm2_5" in self.hourly.columns and self.hourly["pm2_5"].notna().any()

    @property
    def has_pois(self) -> bool:
        return "poi_total" in self.cameras.columns

    @property
    def has_districts(self) -> bool:
        return "district" in self.cameras.columns and self.cameras["district"].notna().any()


def _read_json(path: Path):
    """UTF-8 first; older Windows runs wrote cp1252, so fall back instead of crashing."""
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))


def pipeline_ready() -> bool:
    return all(p.exists() for p in (config.HOURLY_PARQUET, config.CAMERA_PARQUET, config.CLEAN_PARQUET))


def data_version() -> tuple:
    """Cache key that changes whenever the pipeline rewrites its outputs."""
    files = sorted(config.PROCESSED_DIR.glob("*"))
    return (str(config.PROCESSED_DIR), tuple(round(p.stat().st_mtime, 3) for p in files if p.is_file()))


@st.cache_resource(show_spinner="Loading processed data ...")  # no per-rerun copy: pages never mutate it
def load_data(version: tuple) -> Data:
    """Load every artefact written by scripts 04/05. ``version`` only serves as the cache key."""
    p = config.PROCESSED_DIR
    read = lambda name: pd.read_parquet(p / name)  # noqa: E731
    # image-level table: keep only columns the dashboard needs (keeps memory small)
    cols = ["camera_id", "timestamp", "image_path", "date", "hour", "day_type", "vehicle_total", "pce_capped",
            "congestion_idx", "brightness", "usable", "is_blank", "is_degraded", "is_frozen", "is_low_light"]
    images = pd.read_parquet(config.CLEAN_PARQUET, columns=cols)
    hourly = pd.read_parquet(config.HOURLY_PARQUET)
    cameras, insights = _describe_clusters(pd.read_parquet(config.CAMERA_PARQUET), hourly, _read_json(p / "insights.json"))
    return Data(
        hourly=hourly, cameras=cameras,
        anomalies=read("anomalies.parquet"), images=images, predictions=read("model_predictions.parquet"),
        importance=pd.read_csv(p / "model_importance.csv"), model_meta=_read_json(p / "model_metrics.json"),
        quality=_read_json(p / "quality_report.json"), gaps=read("capture_gaps.parquet"),
        insights=insights)


def _describe_clusters(cameras: pd.DataFrame, hourly: pd.DataFrame, insights: list) -> tuple:
    """Name the camera groups from the shape of each group's average day (same rule as the pipeline's name_clusters).

    Done at load time so older processed files get the same clear names without re-running the pipeline."""
    try:
        from cctv import analysis
        labels = cameras.set_index("camera_id")["cluster_id"]
        names = analysis.name_clusters(labels, analysis._hour_profile(hourly))
    except Exception:
        return _unique_cluster_names(cameras), insights
    cams = cameras.copy()
    cams["cluster"] = cams["cluster_id"].map(names)
    sizes = cams["cluster"].value_counts()
    line = "Cameras fall into behavioural groups: " + ", ".join(f"{n} ({c} cams)" for n, c in sizes.items()) + "."
    insights = [line if str(t).startswith("Cameras fall into behavioural groups") else t for t in insights]
    return cams, insights


def _unique_cluster_names(cameras: pd.DataFrame) -> pd.DataFrame:
    """'Morning-peak' / 'Morning-peak #2' become 'Morning-peak · 08:00' / 'Morning-peak · 09:00' (mean peak hour)."""
    if "peak_hour" not in cameras.columns:
        return cameras
    cams = cameras.copy()
    base = cams["cluster"].astype(str).str.replace(r" #\d+$", "", regex=True)
    peak = cams.groupby("cluster")["peak_hour"].mean().round().astype(int)
    dup = base.groupby(cams["cluster"]).first().value_counts()
    dup = set(dup[dup > 1].index)
    rename = {c: f"{b} · {peak[c]:02d}:00" for c, b in base.groupby(cams["cluster"]).first().items() if b in dup}
    cams["cluster"] = cams["cluster"].replace(rename)
    return cams


def sidebar_filters(data: Data, ui=None) -> dict:
    """Global filters; every page receives the same selection. ``ui`` is where the widgets are drawn (default: sidebar)."""
    ui = ui if ui is not None else st.sidebar
    dates = sorted(data.hourly["date"].dt.date.unique())
    d_range = ui.date_input("Date range", (dates[0], dates[-1]), min_value=dates[0], max_value=dates[-1])
    d0, d1 = (d_range if isinstance(d_range, tuple) and len(d_range) == 2 else (dates[0], dates[-1]))
    h0, h1 = ui.slider("Hour of day", 6, 21, (6, 21))
    day_type = ui.radio("Day type", ["All", "Weekday", "Weekend"], horizontal=True)
    clusters = sorted(data.cameras["cluster"].dropna().unique())
    chosen = ui.multiselect("Camera behaviour group", clusters, default=clusters)
    return {"d0": pd.Timestamp(d0), "d1": pd.Timestamp(d1), "h0": h0, "h1": h1, "day_type": day_type, "clusters": chosen,
            "n_clusters": len(clusters)}


def filter_cameras(data: Data, f: dict) -> pd.DataFrame:
    return data.cameras[data.cameras["cluster"].isin(f["clusters"])]


def filter_hourly(data: Data, f: dict) -> pd.DataFrame:
    h = data.hourly
    cams = filter_cameras(data, f)["camera_id"]
    m = h["camera_id"].isin(cams) & h["date"].between(f["d0"], f["d1"]) & h["hour"].between(f["h0"], f["h1"])
    if f["day_type"] != "All":
        m &= h["is_weekend"] == (f["day_type"] == "Weekend")
    return h[m]
