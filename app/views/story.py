"""Project story: the whole journey in the order we lived it (maps onto the course rubric)."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import style

from data_access import Data

# Held-out-camera evaluation (200 frames: 128 day, 72 night): best F1 per setup, from outputs/improvement/report.csv
SETUPS = [
    ("YOLOX-X (pretrained)", 0.840, "0.20"), ("YOLOX-X + fine-tuned YOLO11-L (box fusion)", 0.817, "0.20"),
    ("YOLOX-X + tiling + fine-tuned L", 0.798, "0.25"), ("YOLOX-X + fine-tuned (first attempt)", 0.789, "0.25"),
    ("YOLOX-X + tiling", 0.786, "0.30"), ("YOLOX-X tiling + fine-tuned (first)", 0.780, "0.30"),
    ("Fine-tuned YOLO11-L, 960 px", 0.761, "0.25"), ("Fine-tuned YOLO11 (first attempt)", 0.727, "0.40"),
]
# YOLOX-X pretrained on the same frames, by threshold: precision, recall, F1
PRF = {0.15: (0.785, 0.881, 0.830), 0.20: (0.831, 0.848, 0.840), 0.25: (0.856, 0.811, 0.833),
       0.30: (0.883, 0.772, 0.824), 0.40: (0.907, 0.693, 0.786), 0.50: (0.920, 0.603, 0.728)}


def _facts(items):
    style.facts([(v, l) for l, v in items])


def render(data: Data, f: dict) -> None:
    q = data.quality
    h = data.hourly
    cams = data.cameras
    style.page_header("Project story", "From 61,319 raw CCTV frames to a traffic dashboard: what we did, what we tried, "
                      "what went wrong and what we learned. Written in the order it happened.")

    st.markdown('<div class="rubric">'
                '<div><b>Data engineering</b><span>Chapters 2-3: indexing, quality flags, per-camera normalisation, four scraped sources</span></div>'
                '<div><b>Data science</b><span>Chapters 4-5: detector comparison, congestion index, camera groups, anomalies, forecast</span></div>'
                '<div><b>Visualisation</b><span>Chapter 6: this Streamlit dashboard</span></div>'
                '<div><b>Insights</b><span>Chapter 7: what the data says, with honest caveats</span></div>'
                '</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------ 1
    style.chapter(1, "The brief and the data",
                  "The course project: take raw CCTV images and turn them into structured data and insight, "
                  "using data engineering, data science, visualisation, and at least three scraped data sources.")
    _facts([("frames", f"{q['n_indexed']:,}"), ("BMA cameras", f"{cams.camera_id.nunique()}"), ("days", "7 (25-31 Aug 2026)"),
                 ("capture window", "06:00-21:00"), ("spacing", "~10 min"), ("frame size", "352 x 288 px")])
    st.markdown("The images arrive as JPEG files whose names carry the camera id, the Thai location and the timestamp. "
                "There are **no labels and no vehicle counts**, so everything about traffic has to be *measured* from pixels. "
                "That shaped the plan: a five-step, scripted and resumable pipeline (index, detect, scrape, clean, analyse) "
                "so every number on this dashboard can be reproduced from the raw frames.")

    # ------------------------------------------------------------------ 2
    style.chapter(2, "Data engineering: from JPEGs to a trustworthy table",
                  "Before any chart, we had to know which frames can be trusted.")
    style.timeline([
        ("Index every frame", "Parse camera id and timestamp from each file name. Checks: **0** undecodable files, **0** duplicate "
                              "timestamps, **0** unknown cameras."),
        ("Measure each frame", "Detect vehicles (cars, motorcycles, buses, trucks, bicycles) and compute image-quality metrics: "
                               "brightness, contrast, sharpness, edge density, glare, and an image hash."),
        ("Flag bad frames", f"**{q['n_is_blank']:,}** blank (almost no contrast), **{q['n_is_degraded']:,}** degraded (much blurrier than that "
                            f"camera's normal), **{q['n_is_frozen']:,}** frozen-feed (runs of pixel-identical frames, i.e. a stuck camera). "
                            f"**{q['n_is_low_light']:,}** low-light frames are kept but flagged. **{q['n_gaps_over_threshold']}** capture gaps longer than 30 min."),
        ("Keep what is usable", f"**{q['n_usable']:,}** of {q['n_clean']:,} frames (**{100 * q['n_usable'] / q['n_clean']:.1f}%**) pass every check. A frame can carry several flags, so the flag counts above add up to more than the {q['n_clean'] - q['n_usable']:,} frames removed."),
        ("Make cameras comparable", "Cameras differ in angle and zoom, so raw counts are not comparable. Each camera's load is capped at an "
                                    "upper percentile (winsorising) and scaled to a **0-100 congestion index** relative to its own 95th percentile. "
                                    "Vehicles are weighted by size (passenger-car equivalents), so a bus counts more than a motorcycle."),
        ("Build the camera-hour table", f"Aggregate to **{len(h):,}** camera-hour rows: the table every analysis and chart uses."),
    ])
    flags = pd.DataFrame({"flag": ["Low light (kept)", "Degraded", "Frozen feed", "Blank", "Gaps > 30 min"],
                          "frames": [q["n_is_low_light"], q["n_is_degraded"], q["n_is_frozen"], q["n_is_blank"], q["n_gaps_over_threshold"]]})
    st.plotly_chart(px.bar(flags, x="frames", y="flag", orientation="h", height=300, title="What the quality checks found",
                           labels={"frames": "frames (gaps: count)", "flag": ""}), width="stretch")
    style.callout("obstacle", "Frozen cameras look like steady traffic",
                  "A stuck feed repeats one image forever, which a detector happily counts as constant traffic. We detect runs of identical "
                  "frames per camera and exclude them, instead of trusting the counts blindly.")
    style.callout("obstacle", "Thai file names and long Windows paths",
                  "OpenCV cannot open paths with Thai characters on Windows, so images are decoded from raw bytes in our own reader, "
                  "and the pipeline can read the images from a short folder outside the project.")

    # ------------------------------------------------------------------ 3
    style.chapter(3, "Web scraping: four public sources",
                  "Cameras tell us how busy a road is, not why. We added context from four independent public sources and joined them "
                  "to the camera-hour table by location and hour.")
    style.table(["Source", "What we collected", "Joined by", "Why it matters"], [
        ["**Open-Meteo** historical weather", "Hourly temperature, humidity, rain, cloud cover, wind", "hour (+ camera location)", "Does rain change traffic?"],
        ["**Open-Meteo** air quality", "Hourly PM2.5 and PM10", "hour (+ camera location)", "Pollution is a traffic outcome and a possible predictor"],
        ["**OpenStreetMap Overpass**", "Places within 500 m: schools, universities, hospitals, markets, malls, rail and bus stops", "camera location",
         "Do cameras near schools or stations behave differently?"],
        ["**OpenStreetMap Nominatim**", "District and sub-district of each camera (reverse geocoding)", "camera location", "Compare areas of the city"],
    ])
    rain = h.dropna(subset=["precipitation"]) if "precipitation" in h.columns else h.iloc[0:0]
    if len(rain):
        wet = rain["precipitation"] > 0.1
        facts = [("districts covered", f"{cams['district'].nunique()}" if "district" in cams.columns else "n/a"),
                 ("places per camera (avg)", f"{cams['poi_total'].mean():.0f}" if "poi_total" in cams.columns else "n/a"),
                 ("camera-hours with rain", f"{100 * wet.mean():.0f}%"),
                 ("temperature range", f"{rain['temperature_2m'].min():.0f}-{rain['temperature_2m'].max():.0f} C")]
        _facts(facts)
        if wet.nunique() == 2:
            m = rain.groupby(wet)["congestion_idx"].mean()
            style.callout("result", "Rain and congestion",
                          f"Rainy camera-hours average **{m[True]:.0f}** on the congestion index versus **{m[False]:.0f}** when dry. "
                          "This is one week of weather and a correlation, not proof that rain causes the difference.")
    style.callout("note", "How the scrapers behave",
                  "All network calls go through one function with retries and exponential back-off and a descriptive User-Agent. "
                  "Reverse geocoding pauses about one second between requests, as the Nominatim usage policy asks. The scrapers are "
                  "covered by unit tests that use mocked responses, so the tests run without internet.")

    # ------------------------------------------------------------------ 4
    style.chapter(4, "Choosing the vehicle detector",
                  "This was the hardest and most important decision, because every number downstream depends on the counts.")
    st.markdown("**Step 1: which YOLOX size?** Without labels we start from a detector pretrained on COCO (YOLOX, open licence). "
                "We ran four sizes on the same 12 rush-hour frames on one CPU core:")
    sizes = [("YOLOX-S", 5.4, 0.29), ("YOLOX-M", 7.3, 0.71), ("YOLOX-L", 7.5, 1.4), ("YOLOX-X", 9.7, 2.5)]
    style.table(["Detector", "Vehicles found per frame", "Time per frame (1 CPU core)", "All 61,319 frames on 1 core"],
                [[n, f"{v}", f"{t} s", f"~{61319 * t / 3600:.0f} hours"] for n, v, t in sizes])
    style.callout("decision", "Why YOLOX-X, and why it was painful",
                  "YOLOX-X finds far more vehicles (9.7 per frame against 5.4 for the smallest), mostly distant and small ones that the "
                  "smaller models miss. The price is speed: about **43 hours on a single CPU core**. We solved it by running ONNX Runtime on "
                  "an NVIDIA RTX 3080 Ti laptop GPU, which also meant fixing GPU setup and an input-size bug when we tried a 960 px model.")

    st.markdown("**Step 2: can we beat the pretrained detector?** We built an evaluation set of **1,004 frames** emphasising rain, night "
                "and dense, stopped traffic. A detector proposed boxes and an AI assistant reviewed each image to keep, drop or re-class "
                "boxes (disclosed in the Gen-AI statement). We tested on **held-out cameras only** (200 frames: 128 day, 72 night) and "
                "compared eight setups: plain YOLOX-X, tiled inference, two fine-tuned YOLO11 detectors, and box-fusion ensembles.")
    comp = pd.DataFrame(SETUPS, columns=["setup", "F1", "best threshold"]).sort_values("F1")
    fig = px.bar(comp, x="F1", y="setup", orientation="h", height=380, title="Best F1 per setup on held-out cameras",
                 labels={"F1": "F1 (IoU 0.5)", "setup": ""}, hover_data=["best threshold"])
    fig.update_traces(marker_color=[style.ACCENT if s.startswith("YOLOX-X (pre") else "#cbd5e1" for s in comp["setup"]])
    fig.update_xaxes(range=[0.6, 0.9])
    st.plotly_chart(fig, width="stretch")
    style.callout("result", "The plain pretrained model won",
                  "Pretrained YOLOX-X reached **F1 0.84** and the lowest count error. Our fine-tuned detectors scored lower, and tiling and "
                  "ensembles added cost without a clear gain.")
    style.callout("obstacle", "Why fine-tuning did not help",
                  "Four likely reasons, which we could not fully separate: (1) our labels were **proposed by the very detector family we "
                  "evaluate**, so they favour it; (2) small and distant vehicles are only partly labelled; (3) about 1,000 images is little "
                  "for fine-tuning; (4) held-out cameras show different views from the training cameras. We therefore kept the pretrained model "
                  "and report this as a negative result rather than hiding it.")

    st.markdown("**Step 3: one more independent check.** Because our labels favour YOLOX, we also tested on a separate set of **300 frames** "
                "from other cameras, labelled from a different detector (YOLOv8) with AI review. It has three coarse groups only, "
                "and is not human-verified, so we treat it as a sanity check, not as ground truth.")

    st.markdown("**Step 4: the score threshold.** A threshold decides how confident the detector must be before we count a vehicle. "
                "Lower means more vehicles found but more false alarms.")
    pr = pd.DataFrame([(t, "Precision", v[0]) for t, v in PRF.items()] + [(t, "Recall", v[1]) for t, v in PRF.items()]
                      + [(t, "F1", v[2]) for t, v in PRF.items()], columns=["threshold", "metric", "value"])
    st.plotly_chart(px.line(pr, x="threshold", y="value", color="metric", markers=True, height=330,
                            title="YOLOX-X on held-out cameras: precision / recall / F1 by threshold",
                            labels={"value": "", "threshold": "score threshold"}), width="stretch")
    style.table(["Threshold", "Our 1,004 frames (precision / recall / F1)", "Independent 300 frames (vehicle F1)"], [
        ["0.20", "0.887 / 0.892 / **0.890**", "0.768"], ["0.25", "0.917 / 0.850 / 0.882", "0.779"],
        ["**0.30 (chosen)**", "0.943 / 0.808 / 0.870", "0.790"]])
    style.callout("decision", "We use 0.30",
                  "The two test sets disagree on the best value (our labels like 0.20, the independent set likes 0.40), so we chose the value that "
                  "is never far from the best on either: 0.30 loses only 0.02 F1 on our labels, is within 0.005 of the best on the independent set, "
                  "and makes the fewest false detections. We also looked at 30 frames side by side at 0.20, 0.25 and 0.30: roughly 4 in 10 of the "
                  "extra detections at 0.20 were not vehicles (roadside boxes, buildings, duplicate boxes).")
    style.callout("obstacle", "We nearly mislabelled our own results",
                  "The stored counts do not record which threshold made them. We assumed 0.20, then re-ran the detector on a random sample at "
                  "several thresholds and compared with the stored counts: **30 of 30 frames matched 0.30 exactly** (0.25 matched 18, 0.20 only 7). "
                  "Lesson: save settings next to results. At 0.30 the detector is conservative: it misses about 2 vehicles per frame by day (bias -1.9) and about 0.6 at night (bias -0.6) on our held-out frames.")

    # ------------------------------------------------------------------ 5
    style.chapter(5, "Data science: turning counts into patterns",
                  "With a clean camera-hour table we asked four questions.")
    sizes_c = cams.groupby("cluster").size().sort_values(ascending=False)
    n_anom, n_busy = len(data.anomalies), int((data.anomalies["kind"] == "unusually busy").sum())
    style.timeline([
        ("A relative congestion index", "0-100 for each camera, so busy and quiet roads can be compared fairly (chapter 2)."),
        ("Camera behaviour groups", "KMeans on each camera's 24-hour shape (z-scored), with the number of groups chosen by silhouette score. Result: "
                                    + ", ".join(f"**{n}** ({c} cameras)" for n, c in sizes_c.items()) + "."),
        ("Anomalies", f"A camera-hour is unusual when it differs from that camera's normal for the same hour and day type by more than 3 standard "
                      f"deviations: **{n_anom}** found, **{n_busy}** unusually busy."),
        ("Next-hour forecast", "Gradient boosting with lags, camera and hour averages, location, weather and air quality, trained on 25-29 Aug "
                               "and tested on the unseen 30-31 Aug (a time-based split, so no peeking at the future)."),
    ])
    met = pd.DataFrame(data.model_meta["metrics"]).T.reset_index().rename(columns={"index": "model"}).sort_values("MAE", ascending=False)
    fig = px.bar(met, x="MAE", y="model", orientation="h", height=240, title="Forecast error on unseen days (lower is better)",
                 labels={"MAE": "Mean absolute error (index points)", "model": ""})
    fig.update_traces(marker_color=["#cbd5e1", "#cbd5e1", style.ACCENT])
    st.plotly_chart(fig, width="stretch")
    best = met.sort_values("MAE").iloc[0]
    style.callout("result", "A modest, honest gain",
                  f"Gradient boosting reaches an error of **{best['MAE']:.1f}** index points (R-squared {best['R2']:.2f}), slightly better than the "
                  "camera-by-hour average and simple persistence. With one week of data the typical daily rhythm already explains most of the "
                  "pattern, so we do not claim more than that.")

    # ------------------------------------------------------------------ 6
    style.chapter(6, "Visualisation: the dashboard",
                  "We designed it as a story first and a toolbox second.")
    style.timeline([
        ("Overview first", "Seven questions answered in plain language, each with a chart and, where possible, real frames from the camera."),
        ("Then tools", "Map with hour-by-hour animation, temporal heatmaps, camera groups, camera profile, side-by-side comparison, "
                       "best-time-to-travel, anomalies with frames, forecast, external data, image explorer."),
        ("Honest by design", "A Model check page shows the evaluation numbers, and every page states what the index means and where it is weak."),
        ("Readable colour", "One blue ramp from light (free-flowing) to dark (jammed); categorical colours were checked for colour-blind safety."),
        ("Practical", "Global filters (dates, hours, day type, camera group), CSV download on key tables, cached data loading so it stays fast on 59,894 frames."),
    ])

    # ------------------------------------------------------------------ 7
    style.chapter(7, "Insights", "What the data says.")
    style.insight_cards(data.insights)
    style.note("These come from one week (one weekend) in August 2026 and a detector that under-counts small vehicles, so read them as relative patterns.")

    # ------------------------------------------------------------------ 8
    style.chapter(8, "What went wrong, and what we did about it")
    style.table(["Problem", "What happened", "What we did"], [
        ["Counts are only a proxy", "352 x 288 frames: motorcycles and distant vehicles are small and get missed (about 2 per frame by day at 0.30)",
         "Report a relative index per camera; state the limit on every page"],
        ["Best detector is slow", "YOLOX-X needs about 43 CPU-hours", "Ran on a GPU with ONNX Runtime; pipeline is sharded and resumable"],
        ["Fine-tuning lost to pretrained", "Biased and small label set", "Kept the pretrained model; documented as a negative result"],
        ["Our test favoured the model", "Labels were proposed by the detector family under test", "Added an independent 300-frame test"],
        ["Unknown threshold in stored results", "Results did not record the setting", "Verified by re-detection (30 of 30 = 0.30); keep settings with results"],
        ["Frozen and blank cameras", "Looked like steady or empty traffic", "Detected by identical-frame runs and contrast; excluded"],
        ["Windows quirks", "Thai file names, long paths, nested folders, file encodings", "Byte-level image reader, short image folder, UTF-8 outputs"],
    ])

    # ------------------------------------------------------------------ 9
    style.chapter(9, "Tools and AI assistance")
    style.table(["Area", "What we used"], [
        ["Data engineering", "Python, pandas, PyArrow (Parquet), NumPy, OpenCV"],
        ["Detection", "YOLOX (Apache-2.0) via ONNX Runtime with CUDA; Ultralytics YOLO11 for the fine-tuning experiment only (AGPL-3.0)"],
        ["Data science", "scikit-learn (KMeans, gradient boosting), statsmodels (LOWESS trend lines)"],
        ["Scraping", "requests; Open-Meteo, OpenStreetMap Overpass and Nominatim"],
        ["Visualisation", "Streamlit, Plotly, Leaflet map component"],
        ["Quality", "pytest unit tests, including scrapers with mocked responses"],
        ["AI assistant", "Claude (Anthropic) helped with code, label review, analysis and dashboard design. Every output was run and checked "
                         "by us. Details are in the Gen-AI usage statement (docs/AI_USAGE.md)."],
    ])

    # ------------------------------------------------------------------ 10
    style.chapter(10, "Limits and next steps")
    for line in [
        "One week and one weekend: seasonality and holidays are unknown.",
        "Nothing is known about nights (capture stops at 21:00).",
        "Labels are AI-assisted and not verified box by box by a human. Next: human review of a few hundred frames, then retry fine-tuning.",
        "Counts under-estimate motorcycles and distant vehicles. Next: higher-resolution sources or a detector trained for top-down CCTV.",
        "Extend the scraped context (events, road works, public transport data) and test whether it improves the forecast.",
    ]:
        st.markdown(f"- {line}")
