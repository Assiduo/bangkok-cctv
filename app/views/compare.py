"""Compare 2-3 cameras: hourly congestion side by side + a sample frame of each at a chosen hour."""
import plotly.express as px
import streamlit as st

import style

from data_access import Data, filter_hourly
from views.overview import frame_at


def render(data: Data, f: dict) -> None:
    style.page_header("Compare cameras", "Which road is busier at which hour? Pick 2-3 cameras.")
    cams = data.cameras[["camera_id", "camera_name"]].drop_duplicates().sort_values("camera_name")
    label = {int(r.camera_id): f"{r.camera_name} (#{int(r.camera_id)})" for r in cams.itertuples()}
    chosen = st.multiselect("Cameras", list(label), default=list(label)[:2], format_func=label.get, max_selections=3)
    if len(chosen) < 2:
        st.info("Pick at least two cameras.")
        return
    h = filter_hourly(data, f)
    h = h[h.camera_id.isin(chosen)]
    if h.empty:
        st.info("No data for these cameras with the current filters.")
        return
    t = h.groupby(["camera_id", "hour"], as_index=False)["congestion_idx"].mean()
    t["camera"] = t["camera_id"].map(label)
    st.plotly_chart(px.line(t, x="hour", y="congestion_idx", color="camera", markers=True, height=360,
                            labels={"congestion_idx": "Congestion index", "hour": "Hour of day"}), width="stretch")
    style.download_csv(t[["camera", "hour", "congestion_idx"]], "compare_cameras.csv")
    mean = t.groupby("camera")["congestion_idx"].mean().sort_values(ascending=False)
    st.markdown(f'<div class="note"><b>{mean.index[0]}</b> is busiest on average (index {mean.iloc[0]:.0f}); '
                f'<b>{mean.index[-1]}</b> is the calmest ({mean.iloc[-1]:.0f}).</div>', unsafe_allow_html=True)
    hr = st.slider("Show the busiest frame at hour", f["h0"], f["h1"], min(max(17, f["h0"]), f["h1"]))
    for col, cid in zip(st.columns(len(chosen)), chosen):
        got = frame_at(data, cid, hr, pick="busiest")
        if got:
            col.image(got[0], caption=f"{label[cid]} · {hr:02d}:00 · {got[1]}", width="stretch")
        else:
            with col:
                style.no_preview(f"{label[cid]} · {hr:02d}:00")
