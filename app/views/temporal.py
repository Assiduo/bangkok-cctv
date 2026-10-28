"""Temporal patterns page: heatmaps and day-part comparisons."""
import plotly.express as px
import streamlit as st

import style

from data_access import Data, filter_hourly


def render(data: Data, f: dict) -> None:
    style.page_header("Temporal patterns", "How does congestion change by hour and day, and which cameras follow the same rhythm?")
    h = filter_hourly(data, f)
    if h.empty:
        st.warning("No data for the current filters.")
        return
    h = h.assign(day=h["date"].dt.strftime("%a %d %b"))
    heat = h.groupby(["day", "hour"], as_index=False)["congestion_idx"].mean()
    order = h.sort_values("date")["day"].unique().tolist()
    fig = px.density_heatmap(heat, x="hour", y="day", z="congestion_idx", histfunc="avg", nbinsx=16,
                             category_orders={"day": order}, color_continuous_scale=style.CONGESTION,
                             labels={"congestion_idx": "Index"})
    fig.update_layout(height=380)
    st.subheader("Citywide congestion: day × hour")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Camera × hour heatmap")
    top_n = st.slider("Cameras to show (ranked by mean congestion)", 10, 100, 30, step=5)
    ranked = h.groupby("camera_id")["congestion_idx"].mean().nlargest(top_n).index
    mat = (h[h.camera_id.isin(ranked)].pivot_table(index="camera_id", columns="hour", values="congestion_idx"))
    mat = mat.loc[ranked]
    mat.index = mat.index.astype(str)
    st.plotly_chart(px.imshow(mat, aspect="auto", color_continuous_scale=style.CONGESTION,
                              labels=dict(x="Hour", y="Camera", color="Index"), height=max(380, top_n * 16)),
                    width="stretch")

    st.subheader("Rush-hour contrast")
    dp = data.hourly.merge(data.cameras[["camera_id", "cluster"]], on="camera_id")
    dp = dp[dp["camera_id"].isin(h["camera_id"].unique())]
    dp = dp[dp.hour.between(f["h0"], f["h1"])]
    bars = dp.groupby(["cluster", "is_weekend"], as_index=False)["congestion_idx"].mean()
    bars["type"] = bars["is_weekend"].map({True: "Weekend", False: "Weekday"})
    st.plotly_chart(px.bar(bars, x="cluster", y="congestion_idx", color="type", barmode="group",
                           labels={"congestion_idx": "Congestion index", "cluster": ""}), width="stretch")
