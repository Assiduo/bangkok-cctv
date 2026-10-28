"""Cross-platform pipeline runner (Windows / macOS / Linux).

    python scripts/run_pipeline.py            # model download, index, features, cleaning, analysis
    python scripts/run_pipeline.py --scrape   # also run web scraping (needs internet)
Choose the detector with the CCTV_MODEL environment variable (yolox_s | yolox_m | yolox_l | yolox_x).
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["00_download_model.py", "01_build_index.py", "02_extract_features.py", "03_scrape_external.py",
         "04_clean_data.py", "05_analyze.py", "06_make_samples.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape", action="store_true", help="include 03_scrape_external.py")
    args = ap.parse_args()
    for step in STEPS:
        if step.startswith("03") and not args.scrape:
            continue
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / step)], check=True)
    print("\nDone. Launch the dashboard with:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
