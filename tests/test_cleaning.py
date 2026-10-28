import pandas as pd

from cctv import cleaning


def _frames(n=6):
    return pd.DataFrame({
        "camera_id": 1, "timestamp": pd.date_range("2026-08-25 08:00", periods=n, freq="10min"),
        "img_hash": ["a", "b", "c", "c", "c", "d"][:n], "brightness": [100, 101, 102, 102, 102, 103][:n],
        "sharpness": [5000.0] * n, "contrast": [40.0, 40, 40, 40, 40, 1.0][:n]})


def test_frozen_run_detected_only_for_runs_of_three():
    frozen = cleaning.flag_frozen(_frames())
    assert frozen.tolist() == [False, False, True, True, True, False]


def test_quality_flags_blank_and_usable():
    df = cleaning.add_quality_flags(_frames())
    assert df["is_blank"].iloc[-1] and not df["usable"].iloc[-1]
    assert df["usable"].iloc[0]


def test_day_part_boundaries():
    assert cleaning.day_part(6) == "early morning"
    assert cleaning.day_part(8) == "morning peak"
    assert cleaning.day_part(17) == "evening peak"


def test_congestion_index_bounded():
    df = _frames()
    df["pce_load"] = [1, 2, 3, 4, 5, 50.0]
    df["usable"] = True
    out = cleaning.add_normalised_load(df)
    assert out["congestion_idx"].between(0, 100).all()
