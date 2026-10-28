"""Collect additional data sources (weather, air quality, OSM POIs, districts).

Needs internet access. Usage:
    python scripts/03_scrape_external.py            # all four sources
    python scripts/03_scrape_external.py --only weather pois
"""
import argparse

import pandas as pd

from cctv import config, external
from cctv.indexing import load_camera_info

SOURCES = ("weather", "air", "pois", "districts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=SOURCES, default=list(SOURCES))
    ap.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = ap.parse_args()

    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    cams = load_camera_info()
    index = pd.read_parquet(config.INDEX_PARQUET)
    start, end = index.timestamp.min().strftime("%Y-%m-%d"), index.timestamp.max().strftime("%Y-%m-%d")

    jobs = {
        "weather": ("weather_hourly.csv", lambda: external.fetch_weather(cams, start, end)),
        "air": ("air_quality_hourly.csv", lambda: external.fetch_air_quality(cams, start, end)),
        "pois": ("osm_pois.csv", lambda: external.fetch_pois(cams)),
        "districts": ("camera_districts.csv", lambda: external.fetch_districts(cams)),
    }
    for name in args.only:
        filename, fetch = jobs[name]
        target = config.EXTERNAL_DIR / filename
        if target.exists() and not args.force:
            print(f"[scrape] {name}: {filename} exists, skipping (use --force to refresh)")
            continue
        print(f"[scrape] {name} ...", flush=True)
        df = fetch()
        df.to_csv(target, index=False)
        print(f"[scrape] {name}: {len(df):,} rows -> {target}")


if __name__ == "__main__":
    main()
