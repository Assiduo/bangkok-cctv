"""Overview: the project told as seven questions, each with a plain-language answer, a chart and (where useful) real frames."""
from __future__ import annotations

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

import style

from cctv import config
from cctv.imageio import read_bgr
from data_access import Data, filter_cameras, filter_hourly
from views.image_explorer import available_frames, image_root

MIX = ["car", "motorcycle", "bus", "truck", "bicycle"]
def pct(x: float) -> str:
    """Signed percent without a '-0%'."""
    r = round(x)
    return f"{r:+d}%" if r else "0%"


H = 330   # chart height used throughout the story


@st.cache_data(show_spinner=False, max_entries=64)
def load_rgb(path: str):
    img = read_bgr(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def frame_at(data: Data, camera_id: int, hour: int, date=None, pick: str = "median"):
    """A representative frame (middle of the hour, any day) for a camera, or None if images are unavailable."""
    root, is_sample = image_root()
    if root is None:
        return None
    fr = available_frames(data, root, is_sample)
    fr = fr[(fr.camera_id == camera_id) & (fr.hour == hour) & fr["usable"]].sort_values("timestamp")
    if date is not None:
        d = fr[fr["timestamp"].dt.date == pd.Timestamp(date).date()]
        fr = d if not d.empty else fr
    if fr.empty:
        return None
    if pick == "busiest":          # the frame that really shows the peak (index is capped at 100, so break ties by vehicle count)
        row = fr.sort_values(["congestion_idx", "vehicle_total"], ascending=False).iloc[0]
    elif pick == "calmest":
        row = fr.sort_values(["congestion_idx", "vehicle_total"], ascending=True).iloc[0]
    else:
        row = fr.iloc[len(fr) // 2]
    rgb = load_rgb(str(root / row["image_path"]))
    return None if rgb is None else (rgb, f"{row['timestamp']:%a %d %b %H:%M} · {int(row['vehicle_total'])} vehicles detected")


def when(h: pd.DataFrame) -> None:
    by_hour = h.groupby("hour")["congestion_idx"].mean()
    pk, lo = int(by_hour.idxmax()), int(by_hour.idxmin())
    wk = h.groupby("is_weekend")["congestion_idx"].mean()
    delta = f" Weekends run **{pct(100 * (wk[True] / wk[False] - 1))}** vs weekdays." if len(wk) == 2 else ""
    style.story_head("When", "When is Bangkok busiest?",
                     f"Congestion peaks at **{pk:02d}:00** (index {by_hour.max():.0f}) and is lowest at "
                     f"**{lo:02d}:00** ({by_hour.min():.0f}).{delta} The index compares each camera with its own "
                     "busiest periods, so 100 means 'as busy as this road gets'.")
    t = h.groupby(["is_weekend", "hour"], as_index=False)["congestion_idx"].mean()
    t["type"] = t["is_weekend"].map({True: "Weekend", False: "Weekday"})
    fig = px.line(t, x="hour", y="congestion_idx", color="type", labels={"congestion_idx": "Congestion index", "hour": "Hour of day"},
                  height=H)
    fig.add_vrect(x0=pk - 0.4, x1=pk + 0.4, fillcolor=style.ACCENT, opacity=0.12, line_width=0)
    st.plotly_chart(fig, width="stretch")


def where(data: Data, h: pd.DataFrame, cams: pd.DataFrame) -> None:
    rank = (h.groupby("camera_id", as_index=False)["congestion_idx"].mean()
            .merge(cams[["camera_id", "camera_name", "cluster"]], on="camera_id").sort_values("congestion_idx", ascending=False))
    top = rank.iloc[0]
    style.story_head("Where", "Which roads are the most congested?",
                     f"**{top.camera_name}** (camera {int(top.camera_id)}) is the busiest relative to its own normal "
                     f"(index {top.congestion_idx:.0f}). Below, the top 3 busiest roads at their busiest daytime hour.")
    left, right = st.columns([1, 1])
    show = rank.head(10).iloc[::-1].assign(camera_name=lambda d: d["camera_name"].str.slice(0, 28))
    left.plotly_chart(px.bar(show, x="congestion_idx", y="camera_name", orientation="h", height=H,
                             labels={"congestion_idx": "Mean congestion index", "camera_name": ""},
                             title="Top 10 busiest cameras"), width="stretch")
    right.plotly_chart(px.histogram(rank, x="congestion_idx", nbins=20, height=H, title="All cameras",
                                    labels={"congestion_idx": "Mean congestion index", "count": "cameras"}), width="stretch")
    # daytime only: night frames are dark and the detector misses vehicles, so they mislead
    day = h[h["hour"].between(7, 18)]
    for col, (_, r) in zip(st.columns(3), rank.head(3).iterrows()):
        by_hour = day[day.camera_id == r.camera_id].groupby("hour")["congestion_idx"].mean()
        if by_hour.empty:
            continue
        hr = int(by_hour.idxmax())
        got = frame_at(data, int(r.camera_id), hr, pick="busiest")
        if got:
            col.image(got[0], width="stretch")
        col.markdown(f"**#{rank.index.get_loc(_) + 1} · {r.camera_name[:30]}**<br>busiest daytime hour {hr:02d}:00 · {got[1] if got else ''}",
                     unsafe_allow_html=True)


def what(cams: pd.DataFrame) -> None:
    mix = (cams[[f"{c}_share" for c in MIX]].mean() * 100).rename(lambda c: c.replace("_share", ""))
    mix = mix.rename("share").reset_index().rename(columns={"index": "type"}).sort_values("share")
    top = mix.iloc[-1]
    style.story_head("Traffic mix", "What is on the road?",
                     f"**{top['type'].capitalize()}s** are {top.share:.0f}% of detected vehicles; motorcycles "
                     f"{mix.loc[mix['type'] == 'motorcycle', 'share'].iloc[0]:.0f}%.")
    st.plotly_chart(px.bar(mix, x="share", y="type", orientation="h", height=260, labels={"share": "% of detections", "type": ""}),
                    width="stretch")
    style.note("Motorcycles and distant vehicles are small in 352×288 CCTV frames, so they are the detector's weak spot. "
               "Treat the mix as a relative proxy, not a census.")


def unusual(data: Data, f: dict) -> None:
    an = data.anomalies.merge(data.cameras[["camera_id", "camera_name", "cluster"]], on="camera_id")
    an = an[an["date"].between(f["d0"], f["d1"]) & an["hour"].between(f["h0"], f["h1"]) & an["cluster"].isin(f["clusters"])]
    if an.empty:
        style.story_head("Unusual", "What looks abnormal?", "No camera-hour deviates strongly from its normal in the current selection.")
        return
    busy = int((an["kind"] == "unusually busy").sum())
    worst = an["date"].value_counts().idxmax()
    style.story_head("Unusual", "What looks abnormal?",
                     f"**{len(an):,}** camera-hours differ from that camera's normal by more than 3 standard deviations "
                     f"({busy:,} unusually busy, {len(an) - busy:,} unusually quiet). Most fall on **{worst:%a %d %b}**. "
                     "Open the Anomalies page to see the frames.")
    per_day = an.assign(day=an["date"].dt.strftime("%a %d %b")).groupby(["date", "day", "kind"], as_index=False).size().sort_values("date")
    st.plotly_chart(px.bar(per_day, x="day", y="size", color="kind", height=H, labels={"size": "camera-hours", "day": ""}),
                    width="stretch")


def context(data: Data, h: pd.DataFrame) -> None:
    if not data.has_weather:
        style.story_head("Context", "Do weather and surroundings matter?",
                         "No external data yet. Run the scraping step (see README) to add weather, air quality, districts and nearby places.")
        return
    d = h.dropna(subset=["precipitation"]).assign(wet=lambda x: x["precipitation"] > 0.1)
    if d["wet"].nunique() < 2:
        style.story_head("Context", "Do weather and surroundings matter?",
                         "There was no rain in the selected period, so a rain-versus-dry comparison is not possible. "
                         "See the External data page for temperature, humidity and air quality.")
        return
    m = d.groupby("wet")["congestion_idx"].mean()
    style.story_head("Context", "Does rain change traffic?",
                     f"Rainy camera-hours average **{m[True]:.0f}** on the congestion index versus **{m[False]:.0f}** when dry "
                     f"({pct(100 * (m[True] / m[False] - 1))}). This is a correlation over one week of weather, not proof of cause.")
    c = d.assign(weather=d["wet"].map({True: "Rain", False: "Dry"})).groupby("weather", as_index=False)["congestion_idx"].mean()
    st.plotly_chart(px.bar(c, x="weather", y="congestion_idx", height=260, labels={"congestion_idx": "Mean congestion index", "weather": ""}),
                    width="stretch")


def forecast(data: Data) -> None:
    met = pd.DataFrame(data.model_meta["metrics"]).T.reset_index().rename(columns={"index": "model"}).sort_values("MAE")
    best, naive = met.iloc[0], met[met["model"].str.contains("Persist", case=False)]
    extra = ""
    if len(naive) and best["model"] != naive.iloc[0]["model"]:
        gain = 100 * (1 - best["MAE"] / naive.iloc[0]["MAE"])
        extra = f" That is {gain:.0f}% lower error than simply repeating the last hour."
    style.story_head("Forecast", "Can we predict the next hour?",
                     f"On days the model never saw, **{best['model']}** is off by about **{best['MAE']:.1f} index points** "
                     f"on average (R² {best['R2']:.2f}).{extra}")
    st.plotly_chart(px.bar(met.sort_values("MAE", ascending=False), x="MAE", y="model", orientation="h", height=240,
                           labels={"MAE": "Mean absolute error (lower is better)", "model": ""}), width="stretch")


def trust(data: Data) -> None:
    q = data.quality
    pct = 100 * q["n_usable"] / max(q["n_clean"], 1)
    style.story_head("Data quality", "Can we trust the numbers?",
                     f"**{pct:.1f}%** of {q['n_indexed']:,} frames passed the quality checks. Blank, degraded and frozen-feed "
                     "frames are flagged and excluded; congestion is normalised per camera. Details on the last page.")
    a, b, c = st.columns(3)
    a.metric("Frames indexed", f"{q['n_indexed']:,}")
    b.metric("Usable", f"{q['n_usable']:,}")
    c.metric("Frozen / blank / degraded", f"{q['n_is_frozen']:,} / {q['n_is_blank']:,} / {q['n_is_degraded']:,}")


def render(data: Data, f: dict) -> None:
    style.page_header("Bangkok CCTV – Traffic Intelligence",
                      "From raw camera frames to vehicle counts to what the city's traffic does. "
                      "The Filters button above applies to every chart.")
    h = filter_hourly(data, f)
    cams = filter_cameras(data, f)
    if h.empty:
        st.warning("No data for the current filters.")
        return
    daily = h.groupby("date")["n_images"].sum()
    hourly_v = h.groupby("hour")["vehicle_total"].mean()
    by_hour = h.groupby("hour")["congestion_idx"].mean()
    cong = float(h.congestion_idx.mean())
    sizes = cams.groupby("cluster").size().sort_values(ascending=False).values
    style.stat_cards([
        dict(label="CAMERAS", value=f"{cams.camera_id.nunique()}", sub=f"{cams.cluster.nunique()} behaviour groups, by size",
             bottom=style.segbar(list(sizes))),
        dict(label="FRAMES ANALYSED", value=f"{int(h.n_images.sum()):,}", sub="frames per day",
             bottom=style.sparkline(daily.values, style.CATEGORICAL[3])),
        dict(label="VEHICLES / FRAME", value=f"{(h.vehicle_total * h.n_images).sum() / h.n_images.sum():.1f}", sub="average over analysed frames",
             bottom=style.sparkline(hourly_v.values, style.ACCENT)),
        dict(label="CONGESTION INDEX", value=f"{cong:.0f}<span style='font-size:1rem;color:#94a3b8;font-weight:600'> / 100</span>",
             sub="100 = as busy as each road gets", ring=style.ring(cong, style.CATEGORICAL[1]),
             bottom=style.sparkline(by_hour.values, style.CATEGORICAL[1])),
    ])
    style.rhythm_strip(by_hour)
    when(h)
    where(data, h, cams)
    what(cams)
    unusual(data, f)
    context(data, h)
    forecast(data)
    trust(data)
