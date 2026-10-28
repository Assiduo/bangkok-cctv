"""Short-horizon forecasting: predict a camera's congestion index for the next hour."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

from . import config

WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m",
                "pm2_5", "pm10"]


def build_model_frame(hourly: pd.DataFrame, cameras: pd.DataFrame) -> pd.DataFrame:
    """Add lag features and static camera attributes to the camera-hour table."""
    df = hourly.sort_values(["camera_id", "date", "hour"]).copy()
    grp = df.groupby(["camera_id", "date"])
    df["lag1"] = grp["congestion_idx"].shift(1)
    df["lag2"] = grp["congestion_idx"].shift(2)
    prev_hour = grp["hour"].shift(1)
    df.loc[df["hour"] - prev_hour != 1, ["lag1", "lag2"]] = np.nan   # only use truly consecutive hours
    df = df.merge(cameras[["camera_id", "latitude", "longitude"]], on="camera_id", how="left")
    return df.dropna(subset=["lag1"])


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    base = ["hour", "weekday", "is_weekend", "lag1", "lag2", "latitude", "longitude", "camera_mean", "camera_hour_mean"]
    return base + [c for c in WEATHER_COLS if c in frame.columns and frame[c].notna().any()]


def train_and_evaluate(frame: pd.DataFrame, test_days: int = config.MODEL_TEST_DAYS) -> dict:
    """Time-based split (last ``test_days`` days held out). Returns metrics, predictions and importances."""
    days = sorted(frame["date"].unique())
    test_set = set(days[-test_days:])
    is_test = frame["date"].isin(test_set)
    train, test = frame[~is_test].copy(), frame[is_test].copy()

    # target encodings computed on TRAIN ONLY (no leakage)
    cam_mean = train.groupby("camera_id")["congestion_idx"].mean().rename("camera_mean")
    cam_hour = train.groupby(["camera_id", "hour", "is_weekend"])["congestion_idx"].mean().rename("camera_hour_mean")
    for part in (train, test):
        part["camera_mean"] = part["camera_id"].map(cam_mean)
    train = train.join(cam_hour, on=["camera_id", "hour", "is_weekend"])
    test = test.join(cam_hour, on=["camera_id", "hour", "is_weekend"])
    test["camera_hour_mean"] = test["camera_hour_mean"].fillna(test["camera_mean"])
    train["camera_hour_mean"] = train["camera_hour_mean"].fillna(train["camera_mean"])
    train["is_weekend"], test["is_weekend"] = train["is_weekend"].astype(int), test["is_weekend"].astype(int)

    cols = _feature_columns(train)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6,
                                          random_state=config.RANDOM_STATE)
    model.fit(train[cols], train["congestion_idx"])
    test["predicted"] = model.predict(test[cols]).clip(0, 100)

    def scores(pred):
        return {"MAE": float(mean_absolute_error(test["congestion_idx"], pred)),
                "R2": float(r2_score(test["congestion_idx"], pred))}

    metrics = {
        "Persistence (last hour)": scores(test["lag1"]),
        "Camera x hour average": scores(test["camera_hour_mean"]),
        "Gradient boosting": scores(test["predicted"]),
    }
    imp = permutation_importance(model, test[cols], test["congestion_idx"], n_repeats=5,
                                 random_state=config.RANDOM_STATE, scoring="neg_mean_absolute_error")
    importance = (pd.DataFrame({"feature": cols, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    return {"metrics": metrics, "predictions": test, "importance": importance, "features": cols,
            "train_days": [str(pd.Timestamp(d).date()) for d in days if d not in test_set],
            "test_days": [str(pd.Timestamp(d).date()) for d in sorted(test_set)],
            "n_train": int(len(train)), "n_test": int(len(test))}
