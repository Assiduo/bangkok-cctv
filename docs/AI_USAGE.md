# AI usage disclosure / การใช้ AI ช่วยในโปรเจกต์

This project (2190513 Data Science, Bangkok CCTV Image Analysis) used AI assistance. This page states exactly where.
โปรเจกต์นี้ใช้ AI ช่วย ระบุไว้ตรงนี้ว่าใช้ตรงไหนบ้าง

## 1. Vehicle detector (final pipeline)
- Model: **YOLOX-X** (pretrained on COCO, ONNX, 640 px input). Used as-is, **not fine-tuned**.
- Counting threshold: **0.30**. Chosen by comparing 0.20 / 0.25 / 0.30 on (a) our 1,004 labelled frames and
  (b) an independent 300-frame set, plus visual inspection of 30 frames. On our frames 0.30 gave precision 0.943,
  recall 0.808, F1 0.870 (0.20: F1 0.890 but ~43% of the extra detections are false); on the independent set 0.30 was
  within 0.005 F1 of the best value. 0.30 also gave the lowest worst-case count error across both sets.
- Known limits: the detector under-counts (about 2 vehicles per frame by day, 0.6 at night at threshold 0.30 on held-out frames); very small or far vehicles are missed; tightly packed
  vehicles can be merged or duplicated. Counts are therefore reported as a **relative congestion index per camera**.

## 2. AI-assisted labelling (evaluation and fine-tuning experiments only)
- 1,004 frames were labelled with AI assistance: a detector proposed boxes, and **Claude (an AI assistant)** looked
  at the images and decided which boxes to keep, drop or re-class (car, motorcycle, bus, truck, bicycle).
  These labels were **not verified box-by-box by a human**.
- Because the detector family proposed the boxes, scores on this set favour YOLOX. We therefore also tested on an
  **independent 300-frame set** (other cameras, labelled from YOLOv8s proposals with AI review, also **not human-verified**).
- A fine-tuned YOLO11 detector (Ultralytics, AGPL-3.0) was trained as an experiment. It did **not** beat YOLOX-X
  on held-out cameras and is **not used** in the final results.

## 3. AI help with code, analysis and design
- Claude (Anthropic) assisted with: pipeline scripts, evaluation scripts, dashboard code and visual design,
  and drafting explanations. All outputs were run and checked by the team; numbers in the report come from our own data.

## 4. Data sources
Camera frames: Bangkok CCTV (25-31 Aug 2026). External data: weather, air quality, OpenStreetMap places, district
boundaries (scraped; see Data & method page).

## 5. Licences
YOLOX: Apache-2.0. Ultralytics YOLO: AGPL-3.0 (experiment only). Dashboard: Streamlit, Plotly (open source).

## 6. Team
Passakorn Rakwetpakorn (6638161521), Pattamaporn Prasertkarakul (6638113421), Tanapong Nuengsigkapian (6638080321).
Who reviewed what: ________________________
