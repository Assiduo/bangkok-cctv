"""Run every dashboard page headlessly against synthetic processed data."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cctv import config
from helpers import build_processed_dir

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
VIEWS = ["overview", "map_view", "temporal", "profile", "compare", "clusters", "trip", "anomalies", "forecast",
         "external_data", "image_explorer", "model_check", "method", "story", "team"]

# The app uses st.navigation, so each page module is rendered directly with the same data and global filters.
SCRIPT = """
import sys
sys.path.insert(0, {app!r})
import streamlit as st
st.cache_data.clear()
from data_access import data_version, load_data, sidebar_filters
import importlib
data = load_data(data_version())
filters = sidebar_filters(data)
importlib.import_module("views." + {view!r}).render(data, filters)
"""


def _patch(monkeypatch, processed, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "CLEAN_PARQUET", processed / "images_clean.parquet")
    monkeypatch.setattr(config, "HOURLY_PARQUET", processed / "camera_hourly.parquet")
    monkeypatch.setattr(config, "CAMERA_PARQUET", processed / "camera_summary.parquet")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "no_raw")
    monkeypatch.setattr(config, "SAMPLES_DIR", tmp_path / "no_samples")


def _run(view):
    at = AppTest.from_string(SCRIPT.format(app=str(Path(APP).parent), view=view), default_timeout=180)
    return at.run()


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    build_processed_dir(out)
    return out


@pytest.mark.parametrize("view", VIEWS)
def test_page_renders_without_exception(view, processed, monkeypatch, tmp_path):
    _patch(monkeypatch, processed, tmp_path)
    at = _run(view)
    assert not at.exception, f"{view}: {at.exception}"


def test_external_page_with_scraped_data(tmp_path_factory, monkeypatch, tmp_path):
    out = tmp_path_factory.mktemp("processed_ext")
    build_processed_dir(out, with_external=True)
    for name, fname in [("PROCESSED_DIR", ""), ("CLEAN_PARQUET", "images_clean.parquet"),
                        ("HOURLY_PARQUET", "camera_hourly.parquet"), ("CAMERA_PARQUET", "camera_summary.parquet")]:
        monkeypatch.setattr(config, name, out / fname if fname else out)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "no_raw")
    monkeypatch.setattr(config, "SAMPLES_DIR", tmp_path / "no_samples")
    at = _run("external_data")
    assert not at.exception, at.exception
    assert not at.info, "external page should show charts, not the 'no data' hint"
    at = _run("forecast")
    assert not at.exception, at.exception
