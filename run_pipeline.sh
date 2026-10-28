#!/usr/bin/env bash
# End-to-end pipeline. Steps 1-2 are slow (~75 min on 1 CPU core) but resumable.
# Detector size via CCTV_MODEL (yolox_s | yolox_m [default] | yolox_l | yolox_x), e.g.  CCTV_MODEL=yolox_x bash run_pipeline.sh
# Usage:  bash run_pipeline.sh            # everything except scraping
#         SCRAPE=1 bash run_pipeline.sh   # also run web scraping (needs internet)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/00_download_model.py
python scripts/01_build_index.py
python scripts/02_extract_features.py
if [[ "${SCRAPE:-0}" == "1" ]]; then python scripts/03_scrape_external.py; fi
python scripts/04_clean_data.py
python scripts/05_analyze.py
echo "Done. Launch the dashboard with:  streamlit run app/dashboard.py"
