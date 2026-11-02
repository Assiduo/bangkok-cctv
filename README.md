# Bangkok CCTV Image Analysis — 2190513 Data Science (ICE)

Raw CCTV frames → vehicle detection → structured camera-hour data → analysis → Streamlit dashboard.

**Live dashboard:** https://bangkok-cctv.streamlit.app  ·  **Code:** https://github.com/Assiduo/bangkok-cctv

**Data:** 61,319 frames · 100 BMA cameras · 25–31 Aug 2026 · 06:00–21:00 · ~10-minute spacing.

**Team:** Passakorn Rakwetpakorn (6638161521) · Pattamaporn Prasertkarakul (6638113421) · Tanapong Nuengsigkapian (6638080321)

## Quick start (processed data already included)

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11 recommended
pip install -r requirements.txt && pip install -e .
streamlit run app/dashboard.py
```

The dashboard reads `data/processed/`. Without the raw images it shows the ~1,900 demo frames in `data/samples/`;
with them (set `CCTV_RAW_DIR`, or put the camera folders in `data/raw/`) the *Image explorer* can also run the detector on any frame.
The raw images (~1.7 GB) are not in this repository.

## Pipeline (re-running is optional; results are included)

```bash
bash run_pipeline.sh                   # steps 1, 2, 4, 5 (step 2 is the slow one: vehicle detection, resumable)
python scripts/03_scrape_external.py   # step 3, needs internet; then re-run scripts/05_analyze.py
python scripts/06_make_samples.py      # refresh the demo frames used by the online dashboard
```
`python scripts/run_pipeline.py` does the same as `run_pipeline.sh` and also works on Windows.
The detector is chosen by `models/best_detector.json` (final: **YOLOX-X**, score threshold **0.30**);
`CCTV_MODEL` and `CCTV_DET_THRESHOLD` override it. Large `.onnx` files are downloaded by `scripts/00_download_model.py`
(only the small `yolox_s.onnx` is committed, for the online demo).

| Step | Script | Output |
|------|--------|--------|
| 0 Models | `00_download_model.py` | detector weights (ONNX) |
| 1 Index | `01_build_index.py` | camera id + timestamp parsed from file names |
| 2 Features | `02_extract_features.py` | vehicle counts by type, brightness, sharpness, hash per frame |
| 3 Scraping | `03_scrape_external.py` | weather, air quality (Open-Meteo), places (OSM Overpass), districts (OSM Nominatim) |
| 4 Cleaning | `04_clean_data.py` | quality flags, capping, per-camera congestion index |
| 5 Analysis | `05_analyze.py` | hourly table, camera groups, anomalies, forecast model, insights |
| 6 Samples | `06_make_samples.py` | demo frames in `data/samples/` for the public dashboard |
| 10–13, 20–21 | labelling, fine-tuning and detector evaluation | `outputs/detector_eval.csv`, `models/best_detector.json` |

## Mapping to the project requirements

- **Component 1 – Data engineering:** `indexing.py`, `detector.py`, `features.py`, `cleaning.py`
  (blank / degraded / frozen-feed detection, duplicate and capture-gap checks, capping, per-camera normalisation).
- **Component 2 – Data science:** `analysis.py` (KMeans camera groups, z-score anomalies), `modeling.py` (next-hour forecast, time-based split).
- **Component 3 – Visualisation:** `app/` – multi-page Streamlit dashboard.
- **Component 4 – Insights:** Overview page of the dashboard and `outputs/insights.md`.
- **Web scraping (separate task):** `src/cctv/external.py` – 4 sources, saved in `data/external/`.

## Results on the real data

- 97.7% of frames usable (504 blank, 839 degraded, 586 frozen-feed frames flagged; 71 capture gaps > 30 min).
- City-wide congestion peaks at 17:00 (index 62) and is lowest at 21:00 (29); weekends are 19% lighter than weekdays.
- Busiest day Wed 26 Aug, quietest Sun 30 Aug; 53 anomalous camera-hours (32 unusually busy), most on Mon 31 Aug.
- Cameras split into three behaviour groups (59 / 22 / 19 cameras).
- Detected mix: 70% cars, 13% motorcycles (under-counted), 13% trucks, 3% buses.
- Next-hour forecast, held-out days 30–31 Aug (MAE, lower is better): gradient boosting **11.33**,
  camera×hour average 11.58, persistence 12.15. The gain over the simple baseline is small.

## Limitations

- The detector works on 352×288 frames and under-counts motorcycles and distant vehicles; counts are a relative proxy,
  and the congestion index is relative to each camera's own 95th percentile.
- One week with one weekend, and no frames after 21:00, so no seasons, holidays or nights.
- Labels used for evaluation are AI-assisted and not verified box by box (see `docs/AI_USAGE.md`).
- The public dashboard ships only a sample of frames, so some frame previews are replaced by a note. Run it locally with the full image set to see every frame.

## Tests

`pip install -r requirements-dev.txt && pytest` – parsing, cleaning rules, clustering, anomalies, forecast model, scrapers (mocked) and a smoke test of every dashboard page.

## Gen-AI usage and licences

Where and why AI was used: `docs/AI_USAGE.md`. YOLOX is Apache-2.0; the Ultralytics YOLO11 experiment (AGPL-3.0) is not part of the final results.
