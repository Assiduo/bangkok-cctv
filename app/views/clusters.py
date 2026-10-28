"""Camera behaviour groups (KMeans on the 24-hour shape) and a single-camera drill-down."""
import plotly.express as px
import streamlit as st

import mapkit
import style

from data_access import Data, filter_cameras, filter_hourly


def render(data: Data, f: dict) -> None:
    style.page_header("Camera groups", "Cameras grouped by the shape of their daily rhythm, with a drill-down for any single camera.")
    cams = filter_cameras(data, f)
    h = filter_hourly(data, f)
    if h.empty:
        st.warning("No data for the current filters.")
        return
    prof = h.merge(cams[["camera_id", "cluster"]], on="camera_id").groupby(["cluster", "hour"], as_index=False)["congestion_idx"].mean()
    st.subheader("Typical daily profile of each group")
    st.plotly_chart(px.line(prof, x="hour", y="congestion_idx", color="cluster", markers=True,
                            labels={"congestion_idx": "Congestion index", "cluster": ""}), width="stretch")
    st.caption("Cameras are clustered by the *shape* of their 24h profile (KMeans, k chosen by silhouette score).")

    basemap = mapkit.basemap_picker("cluster_basemap")
    fig = mapkit.camera_map(cams, basemap, height=520, zoom=10.2, color="cluster", hover_name="camera_name",
                            hover_data=["camera_id", "peak_hour", "congestion_mean"])
    if fig is not None:
        st.plotly_chart(fig, width="stretch")

    st.subheader("Camera drill-down")
    label_map = dict(zip(cams["camera_label"], cams["camera_id"]))
    chosen = st.selectbox("Camera", sorted(label_map))
    cid = label_map[chosen]
    row = cams[cams.camera_id == cid].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Group", row["cluster"])
    c2.metric("Peak hour", f"{int(row['peak_hour']):02d}:00")
    c3.metric("Mean congestion", f"{row['congestion_mean']:.0f}")
    c4.metric("Weekend vs weekday", f"{row['weekend_vs_weekday_pct']:+.0f}%")
    ch = h[h.camera_id == cid].assign(day=lambda d: d["date"].dt.strftime("%a %d %b"))
    st.plotly_chart(px.line(ch, x="hour", y="congestion_idx", color="day", markers=True,
                            labels={"congestion_idx": "Congestion index"}), width="stretch")
    mix = ch[["car", "motorcycle", "bus", "truck", "bicycle"]].mean().rename("per frame").reset_index()
    mix = mix.assign(share=mix["per frame"] / mix["per frame"].sum() * 100).sort_values("share")
    st.plotly_chart(px.bar(mix, x="share", y="index", orientation="h", title="Detected vehicle mix (% of detections)",
                           labels={"share": "%", "index": ""}, height=300), width="stretch")
