from cctv import analysis
from helpers import make_cameras, make_clean_frames


def test_hourly_and_clusters_find_two_groups():
    df = make_clean_frames()
    hourly = analysis.hourly_table(df)
    assert hourly["hour"].between(6, 20).all()
    labels, prof, k = analysis.cluster_cameras(hourly)
    assert k >= 2
    # odd cameras peak at 8h, even ones at 17h -> must not share all clusters
    assert labels[labels.index % 2 == 1].nunique() >= 1
    assert set(labels[labels.index % 2 == 1]).isdisjoint(set(labels[labels.index % 2 == 0]))


def test_anomaly_detection_catches_injected_spike():
    df = make_clean_frames()
    mask = (df.camera_id == 3) & (df.date == df.date.min() + __import__("pandas").Timedelta(days=2)) & (df.hour == 12)
    df.loc[mask, ["congestion_idx"]] = 100.0
    hourly = analysis.hourly_table(df)
    found = analysis.detect_anomalies(hourly)
    assert ((found.camera_id == 3) & (found.hour == 12) & (found.kind == "unusually busy")).any()


def test_camera_summary_and_insights_run():
    df = make_clean_frames()
    df["is_frozen"] = False
    hourly = analysis.hourly_table(df)
    summary = analysis.camera_summary(df, hourly, make_cameras())
    assert {"cluster", "peak_hour", "weekend_vs_weekday_pct"} <= set(summary.columns)
    lines = analysis.build_insights(df, hourly, summary, analysis.detect_anomalies(hourly))
    assert len(lines) >= 7 and all(isinstance(x, str) for x in lines)
