from cctv import analysis, modeling
from helpers import make_cameras, make_clean_frames


def test_model_beats_naive_baseline_on_structured_data():
    hourly = analysis.hourly_table(make_clean_frames())
    frame = modeling.build_model_frame(hourly, make_cameras())
    res = modeling.train_and_evaluate(frame, test_days=2)
    m = res["metrics"]
    assert res["n_test"] > 0 and len(res["test_days"]) == 2
    assert m["Gradient boosting"]["MAE"] <= m["Persistence (last hour)"]["MAE"]
    assert "lag1" in res["features"] and "precipitation" not in res["features"]


def test_weather_columns_used_when_present():
    hourly = analysis.hourly_table(make_clean_frames())
    hourly["precipitation"] = 0.0
    res = modeling.train_and_evaluate(modeling.build_model_frame(hourly, make_cameras()))
    assert "precipitation" in res["features"]
