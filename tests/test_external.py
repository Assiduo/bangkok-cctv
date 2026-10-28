import pandas as pd
import pytest

from cctv import external
from helpers import make_cameras


def test_haversine_known_distance():
    # ~1.11 km per 0.01 degree latitude
    assert external.haversine_m(13.70, 100.5, 13.71, 100.5) == pytest.approx(1111, rel=0.01)


@pytest.mark.parametrize("tags,expected", [
    ({"amenity": "school"}, "school"), ({"amenity": "marketplace"}, "market"),
    ({"shop": "mall"}, "mall"), ({"railway": "station"}, "rail_station"),
    ({"highway": "bus_stop"}, "bus_stop"), ({"amenity": "cafe"}, None)])
def test_categorize_poi(tags, expected):
    assert external.categorize_poi(tags) == expected


def test_weather_parses_batched_list_response(monkeypatch):
    cams = make_cameras(3)

    def fake(url, params=None, **_):
        n = len(params["latitude"].split(","))
        item = {"hourly": {"time": ["2026-08-25T06:00", "2026-08-25T07:00"],
                           **{v: [1.0, 2.0] for v in external.WEATHER_VARS}}}
        return [item] * n if n > 1 else item

    monkeypatch.setattr(external, "http_json", fake)
    monkeypatch.setattr(external.time, "sleep", lambda *_: None)
    df = external.fetch_weather(cams, "2026-08-25", "2026-08-25")
    assert len(df) == 6 and set(df.camera_id) == {1, 2, 3}
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_pois_counts_within_radius(monkeypatch):
    cams = make_cameras(2)
    near = {"type": "node", "lat": cams.latitude[0] + 0.001, "lon": cams.longitude[0], "tags": {"amenity": "school"}}
    far = {"type": "node", "lat": cams.latitude[0] + 0.05, "lon": cams.longitude[0], "tags": {"amenity": "school"}}
    way = {"type": "way", "center": {"lat": cams.latitude[1], "lon": cams.longitude[1]}, "tags": {"shop": "mall"}}
    monkeypatch.setattr(external, "http_json", lambda *a, **k: {"elements": [near, far, way]})
    df = external.fetch_pois(cams)
    assert df.loc[0, "poi_school"] == 1 and df.loc[1, "poi_mall"] == 1 and df.loc[0, "poi_total"] == 1


def test_districts_parse(monkeypatch):
    cams = make_cameras(2)
    monkeypatch.setattr(external, "http_json", lambda *a, **k: {"address": {"city_district": "Khet Dusit", "suburb": "Dusit"}})
    df = external.fetch_districts(cams, pause=0)
    assert df["district"].tolist() == ["Khet Dusit"] * 2


def test_merge_helpers_left_join():
    hourly = pd.DataFrame({"camera_id": [1, 1], "timestamp": pd.to_datetime(["2026-08-25 06:00", "2026-08-25 07:00"])})
    weather = pd.DataFrame({"camera_id": [1], "timestamp": pd.to_datetime(["2026-08-25 06:00"]), "precipitation": [2.0]})
    merged = external.merge_hourly(hourly, {"weather": weather})
    assert len(merged) == 2 and merged["precipitation"].notna().sum() == 1
