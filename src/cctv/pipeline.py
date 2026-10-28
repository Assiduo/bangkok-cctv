"""Orchestration helpers shared by the CLI scripts and the tests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import analysis, config, external, modeling


def run_analysis(df: pd.DataFrame, cams: pd.DataFrame, ext: dict[str, pd.DataFrame] | None = None,
                 out_dir: Path | None = None) -> dict:
    """Aggregate -> cluster -> anomalies -> forecast -> insights; write every artefact the dashboard reads."""
    out_dir = Path(out_dir or config.PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ext or {}

    hourly = external.merge_hourly(analysis.hourly_table(df), ext)
    summary = external.merge_cameras(analysis.camera_summary(df, hourly, cams), ext)
    anomalies = analysis.detect_anomalies(hourly)
    result = modeling.train_and_evaluate(modeling.build_model_frame(hourly, summary))
    insights = analysis.build_insights(df, hourly, summary, anomalies)

    hourly.to_parquet(out_dir / config.HOURLY_PARQUET.name, index=False)
    summary.to_parquet(out_dir / config.CAMERA_PARQUET.name, index=False)
    anomalies.to_parquet(out_dir / "anomalies.parquet", index=False)
    result["predictions"].to_parquet(out_dir / "model_predictions.parquet", index=False)
    result["importance"].to_csv(out_dir / "model_importance.csv", index=False)
    meta = {k: result[k] for k in ("metrics", "features", "train_days", "test_days", "n_train", "n_test")}
    (out_dir / "model_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "insights.json").write_text(json.dumps(insights, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"hourly": hourly, "summary": summary, "anomalies": anomalies, "model": result, "insights": insights}
