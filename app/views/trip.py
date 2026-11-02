"""Trip-time advisor: for one camera, which hours are calmest vs most jammed (real averages from the data)."""
import plotly.graph_objects as go
import streamlit as st

import style

from data_access import Data
from views.overview import frame_at


def render(data: Data, f: dict) -> None:
    style.page_header("Best time to travel", "Pick a road and day type: when is it calmest, and when should you avoid it?")
    cams = data.cameras[["camera_id", "camera_name"]].drop_duplicates().sort_values("camera_name")
    label = {int(r.camera_id): f"{r.camera_name} (#{int(r.camera_id)})" for r in cams.itertuples()}
    c1, c2, c3 = st.columns([3, 1, 2])
    cid = c1.selectbox("Camera", list(label), format_func=label.get)
    kind = c2.radio("Day", ["Weekday", "Weekend", "All"], horizontal=False)
    lo_h, hi_h = c3.slider("Time window", 6, 21, (6, 21))
    h = data.hourly
    h = h[(h.camera_id == cid) & h.hour.between(lo_h, hi_h)]
    if kind != "All":
        h = h[h["is_weekend"] == (kind == "Weekend")]
    if h.empty:
        st.info("No data for this selection.")
        return
    by = h.groupby("hour")["congestion_idx"].mean()
    best, worst = int(by.idxmin()), int(by.idxmax())
    drop = 100 * (1 - by.min() / by.max()) if by.max() else 0
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Calmest hour (index {by.min():.0f})", f"{best:02d}:00")
    k2.metric(f"Busiest hour (index {by.max():.0f})", f"{worst:02d}:00")
    k3.metric("Calm vs peak drop", f"{drop:.0f}%")
    style.download_csv(by.reset_index(), "best_time_to_travel.csv")
    colors = [style.ACCENT if x == best else style.WARN if x == worst else "#cbd5e1" for x in by.index]
    fig = go.Figure(go.Bar(x=by.index, y=by.values, marker_color=colors))
    fig.update_layout(height=320, xaxis_title="Hour of day", yaxis_title="Mean congestion index", showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Index = how busy this camera is relative to its own busiest periods (100 = as busy as this road gets). "
               "Averages cover 25-31 Aug 2026 only, so treat them as typical, not guaranteed.")
    st.caption("Each picture is the actual calmest / busiest frame of that hour. After dark, glare and dark frames make counts less "
               "reliable, so a calm night hour can still look busy in the photo.")
    a, b = st.columns(2)
    for col, hr, nm in ((a, best, "Calmest"), (b, worst, "Busiest")):
        got = frame_at(data, cid, hr, pick=nm.lower())
        if got:
            col.image(got[0], caption=f"{nm} · {hr:02d}:00 · {got[1]}", width="stretch")
        else:
            with col:
                style.no_preview(f"{nm} · {hr:02d}:00")
