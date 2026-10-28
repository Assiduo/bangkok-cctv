"""Image explorer: look at the raw frames and run the detector live."""
from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

import style

from cctv import config
from cctv.imageio import read_bgr
from cctv.indexing import sample_name
from data_access import Data


@st.cache_resource(show_spinner="Loading detector ...")
def get_detector():
    from cctv.detectors import make_detector
    try:
        return make_detector()
    except Exception:              # online demo: only the small yolox_s.onnx is uploaded, so fall back to it
        return make_detector("yolox_s")


def image_root():
    """(folder, is_sample_set). Full raw images win; otherwise the small demo sample set."""
    if config.RAW_DIR.exists() and any(config.RAW_DIR.iterdir()):
        return config.RAW_DIR, False
    if config.SAMPLES_DIR.exists() and any(config.SAMPLES_DIR.rglob("*.jpg")):
        return config.SAMPLES_DIR, True
    return None, False


def available_frames(data: Data, root, is_sample: bool) -> pd.DataFrame:
    """Frames that can actually be displayed (all of them for raw, only the shipped ones for samples)."""
    imgs = data.images
    if not is_sample:
        return imgs
    have = {p.name for p in root.rglob("*.jpg")}
    short = imgs["image_path"].map(sample_name)
    out = imgs[short.isin(have)].copy()
    out["image_path"] = short[out.index]          # shipped copies use short ASCII names (see indexing.sample_name)
    return out


@st.cache_data(show_spinner=False, max_entries=300)
def annotate(path: str, draw: bool = True, thr: float = config.DET_SCORE_THRESHOLD) -> tuple:
    """Frame as RGB (optionally with detection boxes) plus the detector's counts. Cached per file."""
    img = read_bgr(path)
    det = get_detector()
    det.score_thr = thr
    dets = det.detect(img)
    if draw:
        for cls, score, x1, y1, x2, y2 in dets:
            colour = (60, 200, 60) if cls in config.VEHICLE_CLASSES else (230, 120, 40)  # BGR: green / blue
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), det.count_from_detections(dets)


def show_frames(data: Data, camera_id: int, date: str, hour: int, limit: int = 3, around: str | None = None) -> None:
    """Show up to ``limit`` annotated frames for a camera-hour (or explain why not).

    ``around="busiest"|"calmest"`` centres the strip on the frame with the highest / lowest congestion in that hour."""
    root, is_sample = image_root()
    if root is None:
        st.info("No images found in data/raw or data/samples, so previews are unavailable.")
        return
    frames = available_frames(data, root, is_sample)
    imgs = frames[(frames.camera_id == camera_id) & (frames.date == pd.Timestamp(date))
                  & (frames.hour == hour)].sort_values("timestamp")
    if imgs.empty:
        st.info("No frame for this hour is included in the online demo's sample set.")
        return
    imgs = imgs.reset_index(drop=True)
    if around in ("busiest", "calmest") and len(imgs) > limit:
        c = int(imgs["congestion_idx"].idxmax() if around == "busiest" else imgs["congestion_idx"].idxmin())
        lo = min(max(c - limit // 2, 0), len(imgs) - limit)
        imgs = imgs.iloc[lo:lo + limit]
        centre = imgs.index[imgs.index == c][0]
    else:
        centre = None
    cols = st.columns(limit)
    for col, (i, row) in zip(cols, imgs.head(limit).iterrows()):
        rgb, counts = annotate(str(root / row["image_path"]))
        mark = f" · {around} frame" if centre is not None and i == centre else ""
        col.image(rgb, caption=f"{row['timestamp']:%H:%M:%S} · {counts['vehicle_total']} vehicles · index {row['congestion_idx']:.0f}{mark}")


def render(data: Data, f: dict) -> None:
    style.page_header("Image explorer", "Look at the raw frames of one camera and compare different times of day.")
    root, is_sample = image_root()
    if root is None:
        st.warning("Place the extracted cctv1–cctv4 folders in `data/raw/` (or run scripts/06_make_samples.py) to enable this page.")
        return
    frames = available_frames(data, root, is_sample)
    if is_sample:
        st.info("Online demo: showing a small sample of frames per camera. Run the dashboard locally with the full image set for every frame.")
    if frames.empty:
        st.warning("No displayable frames match the processed data.")
        return
    cams = data.cameras[data.cameras.camera_id.isin(frames.camera_id.unique())].sort_values("camera_id")
    label_map = dict(zip(cams["camera_label"], cams["camera_id"]))
    c1, c2 = st.columns([2, 1])
    cid = label_map[c1.selectbox("Camera", list(label_map))]
    imgs = frames[frames.camera_id == cid]
    days = sorted(imgs["date"].dt.date.unique())
    day = c2.selectbox("Date", days, format_func=lambda d: d.strftime("%a %d %b"))
    imgs = imgs[imgs["date"].dt.date == day].sort_values("timestamp").reset_index(drop=True)
    if imgs.empty:
        st.info("No frames for this camera on this date.")
        return
    d1, d2 = st.columns([1, 2])
    draw = d1.toggle("Draw detections", value=True, help="Green = vehicles, blue = other objects (people, traffic lights). Off = raw photo.")
    thr = d2.slider("Detection sensitivity (lower finds more, but may add false boxes)", 0.10, 0.60,
                    float(config.DET_SCORE_THRESHOLD), 0.05)

    mode = st.radio("View", ["Time slider", "Compare times of day"], horizontal=True)
    if mode == "Compare times of day":
        hours = sorted(imgs["hour"].unique())
        default = [h for h in (7, 12, 17, 20) if h in hours] or hours[:3]
        chosen = st.multiselect("Hours", hours, default=default, format_func=lambda h: f"{h:02d}:00")
        cols = st.columns(2)
        for i, h in enumerate(chosen):
            hr = imgs[imgs["hour"] == h]
            row = hr.iloc[len(hr) // 2]                       # the frame in the middle of that hour
            rgb, counts = annotate(str(root / row["image_path"]), draw, thr)
            cols[i % 2].image(rgb, caption=f"{row['timestamp']:%H:%M} · {counts['vehicle_total']} vehicles · "
                                           f"congestion {row['congestion_idx']:.0f}", width="stretch")
        return

    times = imgs["timestamp"].dt.strftime("%H:%M").tolist()
    pick = st.select_slider("Time of day (drag, or click the slider and use the arrow keys)", options=list(range(len(times))),
                            value=len(times) // 2, format_func=lambda i: times[i])
    row = imgs.iloc[pick]
    rgb, counts = annotate(str(root / row["image_path"]), draw, thr)
    left, right = st.columns([2, 1])
    left.image(rgb, caption=f"{row['timestamp']:%a %d %b %H:%M:%S}", width="stretch")
    with right:
        st.caption(f"Detector: {get_detector().model_path.name}")
        st.metric("Vehicles detected", counts["vehicle_total"])
        st.metric("Congestion index", f"{row['congestion_idx']:.0f} / 100")
        st.write({k2: v for k2, v in counts.items() if k2 not in ("vehicle_total", "pce_load")})
        st.write(f"Brightness {row['brightness']:.0f} · usable: **{bool(row['usable'])}**")
        st.caption("Congestion index is computed with the standard sensitivity; the slider only changes what is drawn and counted here.")
