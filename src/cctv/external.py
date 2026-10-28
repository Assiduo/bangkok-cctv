"""Step 3 – additional data collection (web scraping / public APIs).

Four independent sources are collected for the camera locations and the capture window:

1. Open-Meteo Historical Weather API   -> hourly temperature, humidity, rain, cloud, wind
2. Open-Meteo Air-Quality API          -> hourly PM2.5 / PM10
3. OpenStreetMap Overpass API          -> points of interest around each camera (schools, malls, stations ...)
4. OpenStreetMap Nominatim (reverse)   -> district / sub-district name of each camera

All network access goes through :func:`http_json`, which tests replace with a stub.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from . import config

USER_AGENT = "bangkok-cctv-course-project/1.0 (educational use; 2190513 Data Science)"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m"]
AIR_VARS = ["pm2_5", "pm10"]
POI_RADIUS_M = 500
POI_CATEGORIES = ["school", "university", "hospital", "market", "mall", "rail_station", "bus_stop"]


# --------------------------------------------------------------------------- HTTP
def http_json(url: str, params: dict | None = None, data: str | None = None,
              retries: int = 4, timeout: int = 120):
    """GET (or POST when ``data`` is given) and return parsed JSON, retrying with exponential back-off."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(retries):
        try:
            if data is None:
                resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            else:
                resp = requests.post(url, data={"data": data}, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"request failed after {retries} attempts: {url}") from exc
            time.sleep(2 ** attempt)


# --------------------------------------------------------------------------- Open-Meteo
def _open_meteo(url: str, cameras: pd.DataFrame, variables: list[str], start: str, end: str,
                batch: int = 25) -> pd.DataFrame:
    frames = []
    for i in range(0, len(cameras), batch):
        part = cameras.iloc[i:i + batch]
        params = {"latitude": ",".join(part["latitude"].astype(str)),
                  "longitude": ",".join(part["longitude"].astype(str)),
                  "hourly": ",".join(variables), "start_date": start, "end_date": end,
                  "timezone": config.TIMEZONE}
        payload = http_json(url, params=params)
        payload = payload if isinstance(payload, list) else [payload]   # one location -> dict
        for cam_id, item in zip(part["camera_id"], payload):
            hourly = pd.DataFrame(item["hourly"])
            hourly.insert(0, "camera_id", int(cam_id))
            frames.append(hourly)
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"])
    return out.rename(columns={"time": "timestamp"})


def fetch_weather(cameras: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return _open_meteo("https://archive-api.open-meteo.com/v1/archive", cameras, WEATHER_VARS, start, end)


def fetch_air_quality(cameras: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return _open_meteo("https://air-quality-api.open-meteo.com/v1/air-quality", cameras, AIR_VARS, start, end)


# --------------------------------------------------------------------------- Overpass POIs
def _overpass_query(south: float, west: float, north: float, east: float) -> str:
    bbox = f"({south},{west},{north},{east})"
    return (f'[out:json][timeout:180];('
            f'nwr["amenity"~"^(school|university|hospital|marketplace)$"]{bbox};'
            f'nwr["shop"~"^(mall|supermarket|department_store)$"]{bbox};'
            f'nwr["railway"="station"]{bbox};'
            f'nwr["highway"="bus_stop"]{bbox};);out center;')


def categorize_poi(tags: dict) -> str | None:
    if tags.get("amenity") in ("school", "university", "hospital"):
        return tags["amenity"]
    if tags.get("amenity") == "marketplace":
        return "market"
    if tags.get("shop") in ("mall", "supermarket", "department_store"):
        return "mall"
    if tags.get("railway") == "station":
        return "rail_station"
    if tags.get("highway") == "bus_stop":
        return "bus_stop"
    return None


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised, broadcasts)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def fetch_pois(cameras: pd.DataFrame, radius_m: int = POI_RADIUS_M) -> pd.DataFrame:
    """Count points of interest within ``radius_m`` of each camera."""
    margin = 0.01
    query = _overpass_query(cameras.latitude.min() - margin, cameras.longitude.min() - margin,
                            cameras.latitude.max() + margin, cameras.longitude.max() + margin)
    payload = http_json("https://overpass-api.de/api/interpreter", data=query, timeout=300)
    pois = []
    for el in payload.get("elements", []):
        cat = categorize_poi(el.get("tags", {}))
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if cat and lat is not None and lon is not None:
            pois.append((cat, lat, lon))
    poi_df = pd.DataFrame(pois, columns=["category", "lat", "lon"])
    rows = []
    for cam in cameras.itertuples():
        row = {"camera_id": cam.camera_id}
        if len(poi_df):
            near = poi_df[haversine_m(cam.latitude, cam.longitude, poi_df.lat.values, poi_df.lon.values) <= radius_m]
            counts = near["category"].value_counts()
        else:
            counts = pd.Series(dtype=int)
        for cat in POI_CATEGORIES:
            row[f"poi_{cat}"] = int(counts.get(cat, 0))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["poi_total"] = out[[f"poi_{c}" for c in POI_CATEGORIES]].sum(axis=1)
    return out


# --------------------------------------------------------------------------- Nominatim districts
def fetch_districts(cameras: pd.DataFrame, pause: float = 1.1) -> pd.DataFrame:
    """Reverse-geocode each camera to its district (Nominatim policy: max 1 request/second)."""
    rows = []
    for cam in cameras.itertuples():
        data = http_json("https://nominatim.openstreetmap.org/reverse",
                         params={"lat": cam.latitude, "lon": cam.longitude, "format": "jsonv2",
                                 "zoom": 14, "accept-language": "en"})
        addr = data.get("address", {})
        rows.append({"camera_id": cam.camera_id,
                     "district": addr.get("city_district") or addr.get("suburb") or addr.get("county"),
                     "subdistrict": addr.get("suburb") or addr.get("quarter") or addr.get("neighbourhood")})
        time.sleep(pause)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- merge helpers
def load_external() -> dict[str, pd.DataFrame]:
    """Load whichever scraped files exist in data/external (missing ones are simply absent)."""
    files = {"weather": "weather_hourly.csv", "air": "air_quality_hourly.csv",
             "pois": "osm_pois.csv", "districts": "camera_districts.csv"}
    out = {}
    for key, name in files.items():
        path = config.EXTERNAL_DIR / name
        if path.exists():
            out[key] = pd.read_csv(path, parse_dates=["timestamp"] if key in ("weather", "air") else None)
    return out


def merge_hourly(hourly: pd.DataFrame, external: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for key in ("weather", "air"):
        if key in external:
            hourly = hourly.merge(external[key], on=["camera_id", "timestamp"], how="left")
    return hourly


def merge_cameras(cams: pd.DataFrame, external: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for key in ("pois", "districts"):
        if key in external:
            cams = cams.merge(external[key], on="camera_id", how="left")
    return cams
