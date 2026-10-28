"""Step 5 – aggregation, camera clustering, anomaly detection and insight text."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

MIX_COLS = ["car", "motorcycle", "bus", "truck", "bicycle"]


def hourly_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per camera x date x hour using usable frames only."""
    u = df[df["usable"]]
    g = u.groupby(["camera_id", "date", "hour"], observed=True)
    out = g.agg(n_images=("image_path", "size"), vehicle_total=("vehicle_total", "mean"),
                pce_load=("pce_capped", "mean"), congestion_idx=("congestion_idx", "mean"),
                person=("person", "mean"), brightness=("brightness", "mean"),
                **{c: (c, "mean") for c in MIX_COLS}).reset_index()
    out["timestamp"] = out["date"] + pd.to_timedelta(out["hour"], unit="h")
    out["weekday"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["weekday"] >= 5
    return out


def _hour_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    """camera x hour matrix of mean congestion index (rows = cameras)."""
    prof = hourly.pivot_table(index="camera_id", columns="hour", values="congestion_idx", aggfunc="mean")
    return prof.apply(lambda r: r.interpolate(limit_direction="both"), axis=1)


def cluster_cameras(hourly: pd.DataFrame, k_range=range(3, 7)) -> tuple[pd.Series, pd.DataFrame, int]:
    """KMeans on the z-scored 24h profile shape; k picked by silhouette. Returns (labels, profiles, k)."""
    prof = _hour_profile(hourly)
    shape = StandardScaler().fit_transform(prof.sub(prof.mean(axis=1), axis=0).div(prof.std(axis=1).replace(0, 1), axis=0).T).T
    best = (-1.0, None, None)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(shape)
        score = silhouette_score(shape, km.labels_)
        if score > best[0]:
            best = (score, k, km.labels_)
    _, k, labels = best
    return pd.Series(labels, index=prof.index, name="cluster"), prof, k


def name_clusters(labels: pd.Series, prof: pd.DataFrame) -> dict[int, str]:
    """Plain-language names from the shape of each group's average 24-hour profile.

    * Daytime plateau: roughly equally busy for most of 07:00-17:00, then drops.
    * Morning-peak / Midday-peak / Evening-peak: the single highest hour falls before 11, 11-15 or from 16.
    * Twin-peak: morning and evening both reach 90% of the maximum with a clear midday dip.
    * "busy late" is added when 20:00-21:00 is still above 55% of the peak.
    """
    names, used = {}, {}
    for c in sorted(labels.unique()):
        m = prof.loc[labels[labels == c].index].mean()
        peak_h, top = int(m.idxmax()), m.max()
        am, mid, pm = m.loc[7:10].mean(), m.loc[11:15].mean(), m.loc[16:19].mean()
        day = m.loc[7:17]
        if m.std() / max(m.mean(), 1e-9) < 0.12:
            base = "All-day steady"
        elif (day >= 0.9 * top).mean() >= 0.75:
            base = "Daytime plateau"
        elif am >= 0.9 * top and pm >= 0.9 * top and mid <= 0.92 * min(am, pm):
            base = "Twin-peak (AM+PM)"
        elif peak_h < 11:
            base = "Morning-peak"
        elif peak_h >= 16:
            base = "Evening-peak"
        else:
            base = "Midday-peak"
        if m.loc[20:21].mean() >= 0.55 * top:
            base += " · busy late"
        used[base] = used.get(base, 0) + 1
        names[int(c)] = base if used[base] == 1 else f"{base} #{used[base]}"
    return names


def camera_summary(df: pd.DataFrame, hourly: pd.DataFrame, cameras: pd.DataFrame) -> pd.DataFrame:
    u = df[df["usable"]]
    base = u.groupby("camera_id").agg(
        n_images=("image_path", "size"), vehicle_mean=("vehicle_total", "mean"),
        pce_mean=("pce_capped", "mean"), congestion_mean=("congestion_idx", "mean"),
        congestion_p90=("congestion_idx", lambda s: s.quantile(0.9)),
        person_mean=("person", "mean"), brightness_mean=("brightness", "mean"),
        **{f"{c}_mean": (c, "mean") for c in MIX_COLS})
    tot = base[[f"{c}_mean" for c in MIX_COLS]].sum(axis=1).replace(0, np.nan)
    for c in MIX_COLS:
        base[f"{c}_share"] = base[f"{c}_mean"] / tot
    wk = u.groupby(["camera_id", "day_type"])["congestion_idx"].mean().unstack()
    base["weekend_vs_weekday_pct"] = 100 * (wk["Weekend"] / wk["Weekday"] - 1)
    prof = _hour_profile(hourly)
    base["peak_hour"] = prof.idxmax(axis=1)
    base["quietest_hour"] = prof.idxmin(axis=1)
    base["am_peak"] = prof.loc[:, 7:9].mean(axis=1)
    base["pm_peak"] = prof.loc[:, 16:18].mean(axis=1)
    labels, _, _ = cluster_cameras(hourly)
    names = name_clusters(labels, prof)
    base["cluster_id"] = labels
    base["cluster"] = labels.map(names)
    quality = df.groupby("camera_id").agg(usable_pct=("usable", lambda s: 100 * s.mean()),
                                          frozen_pct=("is_frozen", lambda s: 100 * s.mean()))
    out = cameras.set_index("camera_id").join(base).join(quality).reset_index()
    return out.drop(columns=[c for c in ("ip_address",) if c in out.columns])


def detect_anomalies(hourly: pd.DataFrame, z_thr: float = config.ANOMALY_Z) -> pd.DataFrame:
    """Flag camera-hours that deviate strongly from that camera's normal for the same hour & day type."""
    h = hourly[hourly["n_images"] >= 3].copy()
    h["expected"] = h.groupby(["camera_id", "hour", "is_weekend"])["congestion_idx"].transform("mean")
    h["resid"] = h["congestion_idx"] - h["expected"]
    h["z"] = h["resid"] / h.groupby("camera_id")["resid"].transform("std").replace(0, np.nan)
    out = h[h["z"].abs() >= z_thr].copy()
    out["kind"] = np.where(out["z"] > 0, "unusually busy", "unusually quiet")
    return out.sort_values("z", key=np.abs, ascending=False).reset_index(drop=True)


def city_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly.groupby(["date", "hour"], as_index=False).agg(
        congestion_idx=("congestion_idx", "mean"), vehicle_total=("vehicle_total", "mean"))


def build_insights(df: pd.DataFrame, hourly: pd.DataFrame, cams: pd.DataFrame, anomalies: pd.DataFrame) -> list[str]:
    """Plain-English findings computed from the data (shown in the dashboard and insights.md)."""
    out = []
    city_hour = hourly.groupby("hour")["congestion_idx"].mean()
    out.append(f"Citywide congestion peaks at {city_hour.idxmax():02d}:00 "
               f"(index {city_hour.max():.0f}) and is lowest at {city_hour.idxmin():02d}:00 ({city_hour.min():.0f}).")
    day = hourly.groupby(["date"])["congestion_idx"].mean()
    wk = hourly.groupby("is_weekend")["congestion_idx"].mean()
    if True in wk.index and False in wk.index:
        out.append(f"Weekend traffic is {100 * (wk[True] / wk[False] - 1):+.0f}% vs weekdays on the congestion index.")
    out.append(f"Busiest day: {day.idxmax():%a %d %b} (index {day.max():.0f}); quietest: {day.idxmin():%a %d %b} ({day.min():.0f}).")
    top = cams.nlargest(3, "congestion_mean")
    out.append("Most congested cameras (relative index): " + "; ".join(
        f"#{r.camera_id} {r.camera_name[:30]} ({r.congestion_mean:.0f})" for r in top.itertuples()))
    mix = cams[[f"{c}_share" for c in MIX_COLS]].mean() * 100
    out.append("Detected vehicle mix – " + ", ".join(f"{c} {mix[f'{c}_share']:.0f}%" for c in MIX_COLS)
               + " (motorcycle share is under-estimated: small objects are the detector's weak spot).")
    sizes = cams["cluster"].value_counts()
    out.append("Cameras fall into behavioural groups: " + ", ".join(f"{n} ({c} cams)" for n, c in sizes.items()) + ".")
    m = cams["weekend_vs_weekday_pct"]
    out.append(f"Largest weekend drop: #{cams.loc[m.idxmin(), 'camera_id']} ({m.min():.0f}%); "
               f"largest weekend rise: #{cams.loc[m.idxmax(), 'camera_id']} ({m.max():+.0f}%).")
    if len(anomalies):
        busy = (anomalies["kind"] == "unusually busy").sum()
        worst_day = anomalies["date"].value_counts().idxmax()
        out.append(f"{len(anomalies)} anomalous camera-hours detected ({busy} unusually busy); "
                   f"most occur on {worst_day:%a %d %b}.")
    usable = 100 * df["usable"].mean()
    out.append(f"{usable:.1f}% of {len(df):,} frames passed quality checks and were used for analysis.")
    return out
