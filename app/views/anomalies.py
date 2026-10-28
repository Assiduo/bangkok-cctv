"""Anomaly page: camera-hours that deviate from the camera's own normal."""
import plotly.express as px
import streamlit as st

import style

from data_access import Data
from views.image_explorer import show_frames
from views.overview import frame_at


def render(data: Data, f: dict) -> None:
    style.page_header("Anomalies", "Camera-hours that are much busier or quieter than that camera's normal.")
    an = data.anomalies.merge(data.cameras[["camera_id", "camera_name", "cluster"]], on="camera_id")
    an = an[an["date"].between(f["d0"], f["d1"]) & an["hour"].between(f["h0"], f["h1"]) & an["cluster"].isin(f["clusters"])]
    if an.empty:
        st.info("No anomalies for the current filters.")
        return
    st.caption("A camera-hour is anomalous when its congestion index differs from that camera's normal for the same "
               "hour and day type by more than 3 standard deviations of its residuals.")
    kind = st.radio("Type", ["both", "unusually busy", "unusually quiet"], horizontal=True)
    if kind != "both":
        an = an[an["kind"] == kind]
    c1, c2 = st.columns(2)
    per_day = an.assign(day=an["date"].dt.strftime("%a %d %b")).groupby(["day", "kind"], as_index=False).size()
    c1.plotly_chart(px.bar(per_day, x="day", y="size", color="kind", title="Anomalies per day",
                           labels={"size": "count"}), width="stretch")
    per_hour = an.groupby(["hour", "kind"], as_index=False).size()
    c2.plotly_chart(px.bar(per_hour, x="hour", y="size", color="kind", title="Anomalies per hour",
                           labels={"size": "count"}), width="stretch")

    view = an[["camera_id", "camera_name", "date", "hour", "kind", "congestion_idx", "expected", "z"]].copy()
    view["date"] = view["date"].dt.date
    st.dataframe(view.round(1), hide_index=True, width="stretch", height=320)

    style.download_csv(view, "anomalies.csv")
    st.subheader("Biggest events")
    day = view[view["hour"].between(7, 18)]          # daytime frames are clear; night ones mislead
    st.caption("Daytime events only (07:00-18:00), where the frames are clear and the counts most reliable. Each picture is the busiest (or calmest) frame of that hour on the day it happened. 'Busy' is relative to that camera's own normal, so a normally quiet junction can be unusually busy with only about ten vehicles.")
    feed = day.reindex(day["z"].abs().sort_values(ascending=False).index).head(6)
    for col, r in zip(st.columns(3) * 2, feed.itertuples()):
        got = frame_at(data, int(r.camera_id), int(r.hour), r.date, pick="busiest" if r.kind == "unusually busy" else "calmest")
        with col:
            if got:
                st.image(got[0], width="stretch")
            when = got[1] if got else f"{r.date} {r.hour:02d}:00"
            st.markdown(f"**#{r.camera_id} · {r.camera_name[:26]}**<br>{when} · {r.kind} (z={r.z:+.1f})", unsafe_allow_html=True)

    st.subheader("Inspect an event")
    options = [f"#{r.camera_id} · {r.date} {r.hour:02d}:00 · {r.kind} (z={r.z:+.1f})" for r in view.itertuples()]
    pick = st.selectbox("Event", options[:200])
    r = view.iloc[options.index(pick)]
    show_frames(data, int(r.camera_id), str(r.date), int(r.hour), around="busiest" if r.kind == "unusually busy" else "calmest")
