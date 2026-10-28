"""Data quality and methodology page."""
import plotly.express as px
import streamlit as st

import style

from data_access import Data

PIPELINE = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f1f3f5", fontname="Helvetica"];
  raw [label="Raw CCTV JPEGs\\n(61k frames, 100 cameras)"]; idx [label="1 Index\\nparse filename"];
  det [label="2 Features\\nYOLOX detector +\\nquality metrics"]; ext [label="3 Scraping\\nweather · air · OSM · districts", fillcolor="#ffe3e3"];
  cln [label="4 Cleaning\\nflags · capping ·\\ncongestion index"]; ana [label="5 Analysis\\nclusters · anomalies ·\\nforecast model"];
  dash [label="Streamlit dashboard", fillcolor="#d3f9d8"];
  raw -> idx -> det -> cln -> ana -> dash; ext -> ana;
}
"""

LIMITS = [
    "Detections come from a COCO-pretrained YOLOX detector on low-resolution (352×288) frames: small / distant vehicles "
    "and motorcycles are under-counted (on held-out frames about 2 per frame by day and 0.6 at night at threshold 0.30). Counts are a *relative* traffic proxy, not ground truth.",
    "Cameras differ in angle and zoom, so the congestion index is normalised per camera (relative to its own 95th percentile).",
    "Capture covers 06:00–21:00 on 25–31 Aug 2026 at roughly 10-minute spacing; nothing is known about other seasons or nights.",
    "Weekend conclusions rest on a single weekend (29–30 Aug).",
]


def render(data: Data, f: dict) -> None:
    style.page_header("Data & method", "How the data was built, how clean it is, and what the numbers can and cannot tell you.")
    st.subheader("How the project fits together")
    a, b, c, d, e = st.columns(5)
    for col, title, body in (
        (a, "Data engineering", "Index 61k frames, detect vehicles, flag blank / degraded / frozen frames, build camera-hour table."),
        (b, "Web scraping", "Open-Meteo weather and air quality, OpenStreetMap places, Nominatim districts."),
        (c, "Data science", "Congestion index, KMeans camera groups, z-score anomalies, next-hour forecast."),
        (d, "Visualisation", "This Streamlit dashboard: story, map, patterns, groups, anomalies, forecast."),
        (e, "Insights", "Overview page: seven questions answered from the data."),
    ):
        col.markdown(f"**{title}**")
        col.caption(body)
    st.graphviz_chart(PIPELINE)
    q = data.quality
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames indexed", f"{q['n_indexed']:,}")
    c2.metric("Usable after cleaning", f"{q['n_usable']:,}", f"{100 * q['n_usable'] / q['n_clean']:.1f}%")
    c3.metric("Frozen-feed frames", f"{q['n_is_frozen']:,}")
    c4.metric("Blank / degraded", f"{q['n_is_blank']:,} / {q['n_is_degraded']:,}")

    st.subheader("Flagged frames by camera")
    img = data.images
    bad = img.assign(flagged=~img["usable"]).groupby("camera_id", as_index=False)["flagged"].mean()
    bad["flagged"] *= 100
    bad = bad.nlargest(15, "flagged").assign(camera_id=lambda d: d["camera_id"].astype(str))
    st.plotly_chart(px.bar(bad, x="camera_id", y="flagged", labels={"flagged": "% frames flagged", "camera_id": "camera"}),
                    width="stretch")

    st.subheader("Lighting and capture coverage")
    light = img.groupby("hour", as_index=False).agg(brightness=("brightness", "mean"), frames=("image_path", "size"))
    a, b = st.columns(2)
    a.plotly_chart(px.line(light, x="hour", y="brightness", markers=True, title="Mean image brightness by hour"), width="stretch")
    perday = img.assign(day=img["date"].dt.strftime("%a %d")).groupby("day", as_index=False).size()
    b.plotly_chart(px.bar(perday, x="day", y="size", title="Frames captured per day", labels={"size": "frames"}), width="stretch")
    if len(data.gaps):
        st.caption(f"{len(data.gaps):,} capture gaps longer than 30 minutes were found "
                   f"(e.g. {data.gaps.groupby(data.gaps.timestamp.dt.date).size().idxmax()} has the most).")

    st.subheader("Known limitations")
    for line in LIMITS:
        st.markdown(f"- {line}")
