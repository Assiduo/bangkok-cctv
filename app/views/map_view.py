"""Map page: where is it busy, and how does that change through the day."""
import streamlit as st

import mapkit
import style

from data_access import Data, filter_cameras, filter_hourly

METRICS = {"Congestion index (0-100, relative)": "congestion_idx", "Vehicles per frame": "vehicle_total",
           "Passenger-car-equivalent load": "pce_load", "Motorcycles per frame": "motorcycle", "Buses per frame": "bus"}


def render(data: Data, f: dict) -> None:
    style.page_header("Camera map", "Where are the cameras, and which areas are busiest? Pick a metric or animate through the day.")
    metric_label = st.selectbox("Colour cameras by", list(METRICS))
    metric = METRICS[metric_label]
    animate = st.toggle("Animate by hour of day", value=False)
    h = filter_hourly(data, f)
    cams = filter_cameras(data, f)
    if h.empty:
        st.warning("No data for the current filters.")
        return
    keys = ["camera_id", "hour"] if animate else ["camera_id"]
    agg = h.groupby(keys, as_index=False)[[metric, "vehicle_total"]].mean() if metric != "vehicle_total" \
        else h.groupby(keys, as_index=False)[["vehicle_total"]].mean()
    agg = agg.merge(cams[["camera_id", "camera_name", "latitude", "longitude", "cluster"]], on="camera_id")
    agg["size"] = agg["vehicle_total"].clip(lower=0.5)
    basemap = mapkit.basemap_picker("map_basemap")
    fig = mapkit.camera_map(agg, basemap, color=metric, size="size", size_max=18, hover_name="camera_name",
                            hover_data={"camera_id": True, "cluster": True, "size": False,
                                        "latitude": False, "longitude": False},
                            color_continuous_scale=style.CONGESTION,
                            animation_frame="hour" if animate else None)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    st.caption("Bubble size = vehicles per frame. The congestion index compares each camera with its own busiest "
               "periods, so it is comparable across cameras with different angles/zoom.")

    st.subheader("Busiest and quietest cameras")
    rank = (h.groupby("camera_id", as_index=False)["congestion_idx"].mean()
            .merge(cams[["camera_id", "camera_name", "cluster"]], on="camera_id").sort_values("congestion_idx"))
    style.download_csv(rank, "camera_ranking.csv")
    left, right = st.columns(2)
    left.markdown("**Top 10 busiest**")
    left.dataframe(rank.tail(10).iloc[::-1], hide_index=True, width="stretch")
    right.markdown("**Top 10 quietest**")
    right.dataframe(rank.head(10), hide_index=True, width="stretch")
