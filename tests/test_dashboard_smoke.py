"""Run every dashboard page headlessly against synthetic processed data."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cctv import config
from helpers import build_processed_dir

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
PAGES = ["Overview", "Map", "Temporal patterns", "Camera groups", "Anomalies", "Forecast",
         "External data", "Image explorer", "Data quality & method"]


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    build_processed_dir(out)
    return out


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page, processed, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "CLEAN_PARQUET", processed / "images_clean.parquet")
    monkeypatch.setattr(config, "HOURLY_PARQUET", processed / "camera_hourly.parquet")
    monkeypatch.setattr(config, "CAMERA_PARQUET", processed / "camera_summary.parquet")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "no_raw")
    monkeypatch.setattr(config, "SAMPLES_DIR", tmp_path / "no_samples")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"


def test_external_page_with_scraped_data(tmp_path_factory, monkeypatch, tmp_path):
    out = tmp_path_factory.mktemp("processed_ext")
    build_processed_dir(out, with_external=True)
    for name, fname in [("PROCESSED_DIR", ""), ("CLEAN_PARQUET", "images_clean.parquet"),
                        ("HOURLY_PARQUET", "camera_hourly.parquet"), ("CAMERA_PARQUET", "camera_summary.parquet")]:
        monkeypatch.setattr(config, name, out / fname if fname else out)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "no_raw")
    monkeypatch.setattr(config, "SAMPLES_DIR", tmp_path / "no_samples")
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("External data").run()
    assert not at.exception, at.exception
    assert not at.info, "external page should show charts, not the 'no data' hint"
    at.sidebar.radio[0].set_value("Forecast").run()
    assert not at.exception, at.exception
