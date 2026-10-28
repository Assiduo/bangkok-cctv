"""Aggregate, cluster, detect anomalies, train the forecaster and write insights."""
import json

import pandas as pd

from cctv import config, external
from cctv.indexing import load_camera_info
from cctv.pipeline import run_analysis


def main() -> None:
    df = pd.read_parquet(config.CLEAN_PARQUET)
    ext = external.load_external()
    print(f"[analyze] external sources found: {sorted(ext) or 'none (run 03_scrape_external.py)'}")
    res = run_analysis(df, load_camera_info(), ext)

    out = config.ROOT / "outputs" / "insights.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("# Key insights\n\n" + "\n".join(f"- {line}" for line in res["insights"]) + "\n")
    print("\n".join(res["insights"]))
    print(json.dumps(res["model"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
