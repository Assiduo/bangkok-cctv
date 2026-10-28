"""Synthetic data builders shared by the tests (no real images needed)."""
import numpy as np
import pandas as pd


def make_cameras(n=12):
    return pd.DataFrame({
        "camera_id": range(1, n + 1), "location": [f"loc{i}" for i in range(n)],
        "location_en": [f"Camera {i}" for i in range(n)], "latitude": 13.7 + np.arange(n) * 0.01,
        "longitude": 100.5 + np.arange(n) * 0.01, "ip_address": "0.0.0.0", "image_count": 100,
        "camera_name": [f"Camera {i}" for i in range(n)], "camera_label": [f"{i} Camera" for i in range(n)]})


def make_clean_frames(n_cams=12, days=7, seed=0):
    """Image-level frame shaped like the output of cleaning.clean()."""
    rng = np.random.default_rng(seed)
    rows = []
    for cam in range(1, n_cams + 1):
        peak = 8 if cam % 2 else 17                      # two behavioural groups
        for d in range(days):
            for hour in range(6, 21):
                for minute in (5, 15, 25, 35, 45, 55):
                    base = 10 + 20 * np.exp(-((hour - peak) ** 2) / 6)
                    n = max(0, rng.normal(base, 2))
                    rows.append({"camera_id": cam,
                                 "timestamp": pd.Timestamp("2026-08-25") + pd.Timedelta(days=d, hours=hour, minutes=minute),
                                 "image_path": f"c{cam}/{d}_{hour}_{minute}.jpg", "vehicle_total": n, "pce_load": n * 1.2,
                                 "person": 1.0, "brightness": 120.0, "car": n * .6, "motorcycle": n * .2,
                                 "bus": n * .1, "truck": n * .08, "bicycle": n * .02})
    df = pd.DataFrame(rows)
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["day_type"] = np.where(df["is_weekend"], "Weekend", "Weekday")
    df["usable"] = True
    df["pce_capped"] = df["pce_load"]
    df["congestion_idx"] = (100 * df["pce_load"] / df.groupby("camera_id")["pce_load"].transform(lambda s: s.quantile(.95))).clip(upper=100)
    return df


def make_external(cams):
    """Synthetic stand-ins for the four scraped sources (test fixtures only)."""
    rng = np.random.default_rng(1)
    stamps = pd.date_range("2026-08-25 06:00", "2026-08-31 21:00", freq="h")
    grid = pd.MultiIndex.from_product([cams.camera_id, stamps], names=["camera_id", "timestamp"]).to_frame(index=False)
    weather = grid.assign(temperature_2m=rng.normal(30, 2, len(grid)), relative_humidity_2m=rng.uniform(50, 90, len(grid)),
                          precipitation=rng.choice([0.0, 0.0, 1.5], len(grid)), cloud_cover=rng.uniform(0, 100, len(grid)),
                          wind_speed_10m=rng.uniform(0, 15, len(grid)))
    air = grid.assign(pm2_5=rng.uniform(10, 60, len(grid)), pm10=rng.uniform(20, 90, len(grid)))
    pois = pd.DataFrame({"camera_id": cams.camera_id, "poi_school": 1, "poi_total": rng.integers(0, 20, len(cams))})
    districts = pd.DataFrame({"camera_id": cams.camera_id, "district": [f"Khet {i % 4}" for i in range(len(cams))]})
    return {"weather": weather, "air": air, "pois": pois, "districts": districts}


def build_processed_dir(out_dir, with_external=False):
    """Write a complete synthetic data/processed folder (what scripts 04+05 would produce)."""
    import json

    from cctv import analysis, pipeline

    df = make_clean_frames()
    df["is_blank"], df["is_degraded"], df["is_frozen"], df["is_low_light"] = False, False, False, False
    df["decode_ok"] = True
    cams = make_cameras()
    df = df.merge(cams[["camera_id", "camera_name", "camera_label", "latitude", "longitude"]], on="camera_id")
    df.to_parquet(out_dir / "images_clean.parquet", index=False)
    pipeline.run_analysis(df, cams, make_external(cams) if with_external else {}, out_dir)
    report = {"n_indexed": len(df), "n_features": len(df), "n_undecodable": 0, "n_duplicate_timestamps": 0,
              "n_unknown_camera": 0, "n_is_blank": 0, "n_is_degraded": 0, "n_is_frozen": 0, "n_is_low_light": 0,
              "n_usable": len(df), "n_clean": len(df), "n_gaps_over_threshold": 0}
    (out_dir / "quality_report.json").write_text(json.dumps(report))
    pd.DataFrame({"camera_id": [1], "timestamp": [pd.Timestamp("2026-08-27 10:00")], "gap_minutes": [45.0]}) \
        .to_parquet(out_dir / "capture_gaps.parquet", index=False)
    return df
