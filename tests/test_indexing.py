import pandas as pd

from cctv.indexing import parse_filename


def test_parse_valid_filename():
    meta = parse_filename("11_ใกล้ซอยราชวิถี-6_รถมาจาก_20260825_061508.jpg")
    assert meta == {"camera_id": 11, "timestamp": pd.Timestamp("2026-08-25 06:15:08")}


def test_parse_rejects_garbage():
    assert parse_filename(".DS_Store") is None
    assert parse_filename("11_x_20261399_999999.jpg") is None


def test_read_bgr_handles_thai_filename(tmp_path):
    import cv2
    import numpy as np

    from cctv.imageio import read_bgr
    path = tmp_path / "11_ใกล้ซอย_20260825_061508.jpg"
    ok, buf = cv2.imencode(".jpg", np.full((20, 30, 3), 128, np.uint8))
    buf.tofile(str(path))
    assert read_bgr(path).shape == (20, 30, 3)
    assert read_bgr(tmp_path / "missing.jpg") is None


def test_fix_name_repairs_cp437_mojibake_only_when_flag_missing():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("extract_raw_zips", Path(__file__).parents[1] / "scripts" / "extract_raw_zips.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    thai = "11_ใกล้ซอยราชวิถี_20260825_061508.jpg"
    garbled = thai.encode("utf-8").decode("cp437")
    assert mod.fix_name(garbled, 0) == thai
    assert mod.fix_name(thai, 0x800) == thai
    assert mod.fix_name("plain_ascii.jpg", 0) == "plain_ascii.jpg"
