"""External data page: weather, air quality, points of interest and districts vs traffic."""
import plotly.express as px
import streamlit as st

import style

from data_access import Data

HELP = ("No scraped data found yet. Run `python scripts/03_scrape_external.py` on a machine with internet access, "
        "then re-run `python scripts/05_analyze.py` and refresh this page.")


def render(data: Data, f: dict) -> None:
    style.page_header("External data", "Weather, air quality, districts and nearby places collected from public sources.")
    if not (data.has_weather or data.has_air or data.has_pois or data.has_districts):
        st.info(HELP)
        return
    h = data.hourly
    if data.has_weather:
        st.subheader("Weather (Open-Meteo)")
        rain = h.assign(rain=lambda d: (d["precipitation"] > 0.1).map({True: "Rain", False: "Dry"}))
        c1, c2 = st.columns(2)
        by_hour = rain.groupby(["hour", "rain"], as_index=False)["congestion_idx"].mean()
        c1.plotly_chart(px.line(by_hour, x="hour", y="congestion_idx", color="rain", markers=True,
                                title="Congestion by hour: rain vs dry"), width="stretch")
        samp = h.dropna(subset=["precipitation"]).sample(min(6000, len(h)), random_state=0)
        c2.plotly_chart(px.scatter(samp, x="precipitation", y="congestion_idx", opacity=0.35, trendline="lowess",
                                   title="Rainfall vs congestion (camera-hours)"), width="stretch")
        cols = [c for c in ["congestion_idx", "temperature_2m", "relative_humidity_2m", "precipitation",
                            "cloud_cover", "wind_speed_10m", "pm2_5", "pm10", "brightness"] if c in h.columns]
        st.plotly_chart(px.imshow(h[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, title="Correlation matrix"), width="stretch")
    if data.has_air:
        st.subheader("Air quality (Open-Meteo)")
        aq = h.groupby("hour", as_index=False)[["pm2_5", "congestion_idx"]].mean()
        st.plotly_chart(px.scatter(aq, x="congestion_idx", y="pm2_5", text="hour", trendline="ols",
                                   title="Hourly PM2.5 vs congestion (citywide mean)"), width="stretch")
    c = data.cameras
    if data.has_pois:
        st.subheader("Surroundings (OpenStreetMap, 500 m radius)")
        st.plotly_chart(px.scatter(c, x="poi_total", y="congestion_mean", color="cluster", hover_name="camera_name",
                                   trendline="ols", title="Nearby points of interest vs mean congestion"),
                        width="stretch")
    if data.has_districts:
        st.subheader("Districts (Nominatim)")
        d = c.groupby("district", as_index=False).agg(cameras=("camera_id", "size"), congestion=("congestion_mean", "mean"))
        st.plotly_chart(px.bar(d.sort_values("congestion"), x="congestion", y="district", orientation="h",
                               hover_data=["cameras"], height=max(400, 22 * len(d))), width="stretch")
