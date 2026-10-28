"""Camera profile: everything about one camera on a single page."""
import pandas as pd
import plotly.express as px
import streamlit as st

import style

from data_access import Data
from views.overview import frame_at

MIX = ["car", "motorcycle", "bus", "truck", "bicycle"]


def render(data: Data, f: dict) -> None:
    style.page_header("Camera profile", "One camera in detail: its rhythm, vehicle mix, unusual hours and a typical frame.")
    cams = data.cameras.sort_values("camera_name")
    label = {int(r.camera_id): f"{r.camera_name} (#{int(r.camera_id)})" for r in cams.itertuples()}
    cid = st.selectbox("Camera", list(label), format_func=label.get)
    cam = cams[cams.camera_id == cid].iloc[0]
    h = data.hourly[data.hourly.camera_id == cid]
    k = st.columns(4)
    k[0].metric("Vehicles / frame", f"{cam.vehicle_mean:.1f}")
    k[1].metric("Mean congestion index", f"{cam.congestion_mean:.0f}")
    k[2].metric("Peak hour", f"{int(cam.peak_hour):02d}:00")
    k[3].metric("Quietest hour", f"{int(cam.quietest_hour):02d}:00")
    place = " · ".join(dict.fromkeys(str(x) for x in (cam.get("district"), cam.get("subdistrict")) if isinstance(x, str) and x))
    st.markdown(f'<div class="note"><b>{cam.cluster}</b>' + (f" · {place}" if place else "")
                + f" · {int(cam.n_images):,} frames, {cam.usable_pct:.0f}% usable</div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    t = h.assign(type=h["is_weekend"].map({True: "Weekend", False: "Weekday"})).groupby(["type", "hour"], as_index=False)["congestion_idx"].mean()
    left.plotly_chart(px.line(t, x="hour", y="congestion_idx", color="type", markers=True, height=320, title="Congestion by hour",
                              labels={"congestion_idx": "Congestion index", "hour": "Hour of day"}), width="stretch")
    mix = pd.DataFrame({"type": MIX, "share": [float(cam[f"{c}_share"]) * 100 for c in MIX]})
    right.plotly_chart(px.bar(mix, x="type", y="share", height=320,
                              title="Vehicle mix (%)", labels={"share": "% of vehicles", "type": ""}), width="stretch")
    an = data.anomalies[data.anomalies.camera_id == cid]
    if len(an):
        st.subheader(f"Unusual hours ({len(an)})")
        v = an[["date", "hour", "kind", "congestion_idx", "expected", "z"]].copy()
        v["date"] = v["date"].dt.date
        st.dataframe(v.round(1), hide_index=True, width="stretch")
    hr = st.slider("Frame at hour (busiest of that hour)", 6, 21, int(cam.peak_hour))
    got = frame_at(data, cid, hr, pick="busiest")
    if got:
        st.image(got[0], caption=f"{hr:02d}:00 · {got[1]}", width=520)
    else:
        st.caption("No image for this hour on this computer.")
    style.download_csv(h[["date", "hour", "vehicle_total", "congestion_idx", "car", "motorcycle", "bus", "truck"]],
                       f"camera_{cid}_hourly.csv", "Download this camera's hourly data")
