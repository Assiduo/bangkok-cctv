"""Forecast page: how we predict the next hour's congestion, how good it is, and what it relies on."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import style

from data_access import Data

NICE = {"camera_hour_mean": "Typical level of this camera at this hour", "lag1": "Congestion 1 hour ago",
        "lag2": "Congestion 2 hours ago", "camera_mean": "Camera's overall average", "hour": "Hour of day",
        "weekday": "Day of week", "is_weekend": "Weekend or weekday", "latitude": "Latitude", "longitude": "Longitude",
        "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity", "precipitation": "Rain",
        "cloud_cover": "Cloud cover", "wind_speed_10m": "Wind speed", "pm2_5": "PM2.5", "pm10": "PM10"}


def render(data: Data, f: dict) -> None:
    style.page_header("Forecast", "Can we predict next hour's congestion at a camera? How the model works, how well it does, "
                      "and where it is weak.")
    meta, m = data.model_meta, pd.DataFrame(data.model_meta["metrics"]).T
    pred = data.predictions.copy()
    pred["abs_err"] = (pred["predicted"] - pred["congestion_idx"]).abs()
    best = m["MAE"].idxmin()

    # ---- headline numbers
    style.facts([(f"{m.loc['Gradient boosting', 'MAE']:.1f}", "typical error, index points (MAE)"),
                 (f"{m.loc['Gradient boosting', 'R2']:.2f}", "R-squared on unseen days"),
                 (f"{meta['n_train']:,} / {meta['n_test']:,}", "train / test camera-hours"),
                 (f"{len(meta['features'])}", "inputs the model sees")])

    # ---- how it works
    st.subheader("How the forecast works")
    style.timeline([
        ("The question", "For one camera, given what we know up to now, what will its congestion index (0-100) be in the next hour?"),
        ("What the model sees", "Four kinds of input, all known at prediction time: "
                                "**recent history** (congestion 1 and 2 hours ago, only when those hours are consecutive); "
                                "**typical level** (the camera's average, and its average for this hour on weekdays or weekends); "
                                "**time and place** (hour, day of week, weekend flag, latitude and longitude); "
                                "**weather and air quality** (temperature, humidity, rain, cloud, wind, PM2.5, PM10 from the scraped sources)."),
        ("The model", "Gradient boosting (scikit-learn HistGradientBoostingRegressor: 300 rounds, learning rate 0.06, depth 6). "
                      "It learns many small decision trees, each correcting the errors of the previous ones."),
        ("A fair test", f"A time-based split, never a random one: trained on **{meta['train_days'][0]} to {meta['train_days'][-1]}**, "
                        f"tested on the days it never saw, **{meta['test_days'][0]} and {meta['test_days'][-1]}**. "
                        "The typical-level inputs are computed from training days only, so nothing leaks from the test days."),
        ("Compared with", "Two simple baselines: **persistence** (next hour = this hour) and the **camera x hour average** "
                          "(just the typical value for that camera, hour and day type). The model has to beat both to be worth using."),
    ])

    # ---- scores: table first, chart on its own row below
    st.subheader("How good is it?")
    show = m.reset_index().rename(columns={"index": "Model", "MAE": "MAE (lower is better)", "R2": "R-squared (higher is better)"})
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    mm = m.reset_index().rename(columns={"index": "model"}).sort_values("MAE", ascending=False)
    fig = px.bar(mm, x="MAE", y="model", orientation="h", height=250, title="Mean absolute error on unseen days (lower is better)",
                 labels={"MAE": "Mean absolute error (index points)", "model": ""}, text=mm["MAE"].round(2))
    fig.update_traces(marker_color=[style.ACCENT if x == best else "#cbd5e1" for x in mm["model"]], textposition="outside")
    fig.update_xaxes(range=[0, mm["MAE"].max() * 1.15])
    st.plotly_chart(fig, width="stretch")
    gb, base = m.loc["Gradient boosting", "MAE"], m.loc["Camera x hour average", "MAE"]
    style.callout("result", "A small but consistent gain",
                  f"Gradient boosting is wrong by **{gb:.1f}** index points on average, against {base:.1f} for the typical-value baseline "
                  f"and {m.loc['Persistence (last hour)', 'MAE']:.1f} for persistence: **{100 * (1 - gb / base):.0f}% better** than the best baseline "
                  f"and {100 * (1 - gb / m.loc['Persistence (last hour)', 'MAE']):.0f}% better than persistence. "
                  f"R-squared {m.loc['Gradient boosting', 'R2']:.2f} means it explains under half of the variation. "
                  "With only one week of data, a camera's usual daily rhythm already explains most of what is predictable.")

    # ---- what drives it
    st.subheader("What does the forecast rely on?")
    imp = data.importance.head(10).copy()
    imp["label"] = imp["feature"].map(NICE).fillna(imp["feature"])
    fig = px.bar(imp.iloc[::-1], x="importance", y="label", orientation="h", height=380,
                 labels={"importance": "Error added when this input is scrambled (index points)", "label": ""})
    fig.update_traces(marker_color=style.ACCENT)
    st.plotly_chart(fig, width="stretch")
    top = imp.iloc[0]
    style.note(f"Permutation importance: we shuffle one input on the test days and see how much the error grows. "
               f"{top['label']} dominates (+{top['importance']:.1f} points); every other input adds under 0.3, so those small differences "
               "(including weather) should not be over-read.")

    # ---- where it fails
    st.subheader("When and where is it wrong?")
    c1, c2 = st.columns(2)
    by_hour = pred.groupby("hour", as_index=False)["abs_err"].mean()
    fig = px.bar(by_hour, x="hour", y="abs_err", height=300, title="Error by hour of day",
                 labels={"abs_err": "Mean absolute error", "hour": "Hour of day"})
    fig.update_traces(marker_color="#cbd5e1")
    c1.plotly_chart(fig, width="stretch")
    worst_h = by_hour.loc[by_hour["abs_err"].idxmax()]
    by_cam = pred.groupby("camera_id")["abs_err"].mean()
    c2.plotly_chart(px.histogram(by_cam, nbins=20, height=300, title="Error per camera",
                                 labels={"value": "Mean absolute error", "count": "cameras"}).update_layout(showlegend=False),
                    width="stretch")

    style.note(f"The error is largest at {int(worst_h['hour']):02d}:00 ({worst_h['abs_err']:.1f} points), when traffic is dropping fast and the "
               f"previous hour is a poor guide. Cameras range from {by_cam.min():.1f} to {by_cam.max():.1f} points of error.")

    # ---- one camera
    st.subheader("Actual vs predicted, one camera")
    p = pred.merge(data.cameras[["camera_id", "camera_label"]], on="camera_id")
    cam = st.selectbox("Camera", sorted(p["camera_label"].unique()))
    q = p[p.camera_label == cam].sort_values("timestamp")
    k1, k2 = st.columns(2)
    k1.metric("This camera's error (MAE)", f"{q['abs_err'].mean():.1f}")
    k2.metric("Average over all cameras", f"{pred['abs_err'].mean():.1f}")
    q = q.set_index("timestamp")
    q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="h"))        # empty hours become gaps, not long diagonal lines
    fig = go.Figure()
    fig.add_scatter(x=q.index, y=q["congestion_idx"], name="actual", mode="lines+markers", line_color="#0f172a")
    fig.add_scatter(x=q.index, y=q["predicted"], name="predicted", mode="lines+markers", line_color=style.ACCENT)
    fig.update_layout(yaxis_title="Congestion index", height=380, yaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")

    style.callout("note", "Limits to keep in mind",
                  "(1) The test is two days (Sun 30 and Mon 31 Aug), and the training data holds only one weekend day, so Sunday is predicted "
                  "from a single Saturday. (2) The weather inputs are the observed values for the hour being predicted; a live system would need "
                  "a weather forecast instead. (3) The index is built from detector counts, so the forecast inherits the detector's blind "
                  "spots (small vehicles, night).")
