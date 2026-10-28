"""Model check: how trustworthy are the vehicle counts? Shows the evaluation tables/pictures written by scripts 20 and 14."""
import pandas as pd
import streamlit as st

import style
from cctv import config

from data_access import Data

OUT = config.ROOT / "outputs" / "improvement"


def _table(title: str, path, note: str = "") -> None:
    if path.exists():
        st.subheader(title)
        if note:
            st.caption(note)
        st.dataframe(pd.read_csv(path), width="stretch", hide_index=True)
    # a missing optional table is simply not shown


def render(data: Data, f: dict) -> None:
    style.page_header("Model check", "How far can the vehicle counts be trusted? Measured on frames labelled by hand-review.")
    st.markdown('<div class="note">Detector in use: <b>' + str(config.MODEL_NAME) + "</b>. Metrics are precision / recall / F1 at "
                "IoU 0.5 and the error of the per-frame vehicle count. Our own labels were proposed by the detector "
                "family itself, so the independent 300-frame set below is the fairer test.</div>", unsafe_allow_html=True)
    _table("Held-out cameras (our labelled frames)", OUT / "report.csv")
    _table("By time of day", OUT / "report_by_daypart.csv", "Count error per frame (vehicles); negative bias = the detector under-counts. Low thresholds add false alarms, most visibly at night.")
    _table("Independent 300 frames (other cameras, AI-reviewed labels)", OUT / "external300_report.csv",
           "Only three coarse groups exist there; two-wheeler boxes exclude the rider, so ignore that column's F1.")
    pics = sorted(OUT.glob("pic_*.jpg"))[:6]
    if pics:
        st.subheader("Before / after examples")
        for col, p in zip(st.columns(min(3, len(pics))), pics):
            col.image(str(p), caption=p.stem, width="stretch")
