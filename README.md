# Bangkok CCTV Image Analysis — 2190513 Data Science (ICE)

Raw CCTV frames → vehicle detection → structured camera-hour data → insights → Streamlit dashboard.

**Data:** 61,319 frames · 100 BMA cameras · 25–31 Aug 2026 · 06:00–21:00 · ~10-minute spacing.

## Quick start (processed data already included)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
streamlit run app/dashboard.py
```

The dashboard reads `data/processed/`. The *Image explorer* page additionally needs the raw images in `data/raw/`
(`python scripts/extract_raw_zips.py --out data/raw cctv1.zip cctv2.zip cctv3.zip cctv4.zip`; on Windows use a short folder such as `C:\cctv_images`, see `docs/WINDOWS_GPU.md`).

## Re-running the pipeline  (required once: the bundled results came from the smaller YOLOX-tiny model)

```bash
bash run_pipeline.sh                   # steps 0-2, 4, 5. Step 2 uses all CPU cores (~3.4 img/s per core), resumable
python scripts/03_scrape_external.py   # step 3 – needs internet; then re-run scripts/05_analyze.py
python scripts/06_make_samples.py      # refresh the demo frames for the online dashboard
```
`python scripts/02_extract_features.py --workers 4` limits the number of CPU cores used.
`python scripts/run_pipeline.py` does the same as `run_pipeline.sh` and also works on Windows. Windows + NVIDIA GPU: see `docs/WINDOWS_GPU.md`. If your images are in a folder with a different name/location, set `CCTV_RAW_DIR` to it instead of using `data/raw`.

Detector size is chosen with the `CCTV_MODEL` environment variable (default `yolox_m`). Measured on 12 rush-hour
frames (1 CPU core): s 5.4 vehicles/img @0.29 s · m 7.3 @0.71 s · l 7.5 @1.4 s · x 9.7 @2.5 s.
```bash
CCTV_MODEL=yolox_x bash run_pipeline.sh     # most accurate, slowest (hours)
```
Use the same `CCTV_MODEL` value in every terminal where you run steps 2, 4, 5 and the dashboard.

| Step | Script | Output |
|------|--------|--------|
| 1 Index | `01_build_index.py` | camera id + timestamp parsed from filenames |
| 2 Features | `02_extract_features.py` | YOLOX-tiny vehicle counts + brightness/sharpness/hash per frame |
| 3 Scraping | `03_scrape_external.py` | weather, air quality (Open-Meteo), POIs (Overpass), districts (Nominatim) |
| 4 Cleaning | `04_clean_data.py` | quality flags, winsorising, per-camera congestion index |
| 5 Analysis | `05_analyze.py` | hourly table, clusters, anomalies, forecast model, insights |
| 6 Demo samples | `06_make_samples.py` | ~400 frames in `data/samples/` so the *public* dashboard can show images |

## Mapping to the project requirements

- **Component 1 – Data engineering:** `indexing.py`, `detector.py`, `features.py`, `cleaning.py` (blank / degraded / frozen-feed detection, duplicate and gap checks, capping, normalisation).
- **Component 2 – Data science:** `analysis.py` (KMeans camera groups, z-score anomalies), `modeling.py` (next-hour forecast, time-based split).
- **Component 3 – Visualisation:** `app/` – 9-page Streamlit dashboard.
- **Component 4 – Insights:** auto-generated on the Overview page and in `outputs/insights.md`.
- **Web scraping (separate task):** `src/cctv/external.py` – 4 sources.

## Results on the real data (first run, YOLOX-tiny – numbers change after re-running with YOLOX-s)

- 97.7% of frames usable (504 blank, 839 degraded, 586 frozen-feed frames flagged; 71 capture gaps > 30 min).
- City-wide congestion peaks at 17:00 and is lowest at 21:00; weekends are ~21% lighter than weekdays.
- Busiest day Wed 26 Aug, quietest Sun 30 Aug; 53 anomalous camera-hours, most on Mon 31 Aug.
- Cameras split into three behaviour groups: twin-peak (41), morning-peak (38), evening-peak (21).
- Forecast (held-out 30–31 Aug): gradient boosting MAE 12.64 vs 12.64 for the camera×hour average and 13.36 for
  persistence – **the model does not beat the simple seasonal baseline** on this data. Weather features may help once scraped.

## Limitations

- The detector (YOLOX family selected by `CCTV_MODEL`, 640 px input, score ≥ 0.30, tunable in `config.py`) works on 352×288 frames and still under-counts motorcycles and distant vehicles (about 2 vehicles per frame by day at threshold 0.30): counts are a relative proxy.
- The congestion index is relative to each camera's own 95th percentile.
- One week, one weekend. Scraped data has **not** been collected yet in this package (no internet where it was built);
  unit tests cover the scrapers with mocked responses only. Run step 3 yourself and check the output files.

## Share the dashboard online

See `docs/DEPLOY.md` (Streamlit Community Cloud, free; plus a temporary-link option).

## Tests

`pip install -r requirements-dev.txt && pytest` – 32 tests (parsing, cleaning rules, clustering, anomalies, model, scrapers with mocks, every dashboard page).

## Gen-AI usage

See `docs/AI_USAGE.md` (edit to match what your group actually did). Video outline: `docs/PRESENTATION_GUIDE.md`.
