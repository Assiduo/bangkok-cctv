"""Team page. Edit the TEAM list below to change names, ids, photos (app/assets/) or roles."""
import base64
from pathlib import Path

import streamlit as st

import style

ASSETS = Path(__file__).resolve().parent.parent / "assets"

TEAM = [
    {"name": "Passakorn Rakwetpakorn", "id": "6638161521", "photo": "team_passakorn.jpg",
     "role": "Data pipeline, vehicle detection and evaluation, dashboard design and build."},
    {"name": "Pattamaporn Prasertkarakul", "id": "6638113421", "photo": "team_pattamaporn.jpg",
     "role": "Slides and presentation: storyline, slide design, speaker notes and the 15-minute talk."},
    {"name": "Tanapong Nuengsigkapian", "id": "6638080321", "photo": "team_tanapong.jpg",
     "role": "Model review: checked the forecast model, its metrics and the detector results, and the limitations we report."},
]


def _photo(name: str | None) -> str:
    if name and (ASSETS / name).exists():
        b64 = base64.b64encode((ASSETS / name).read_bytes()).decode()
        return f'<img src="data:image/jpeg;base64,{b64}" alt="">'
    return ('<div style="aspect-ratio:1/1;background:#f1f4f9;display:flex;align-items:center;justify-content:center;'
            'color:#94a3b8;font-size:3rem;font-weight:300">?</div>')


def render(data, f: dict) -> None:
    st.markdown('<div class="hero"><small>Final project · 2190513 Data Science (ICE)</small>'
                '<h1>Bangkok CCTV Image Analysis</h1>'
                '<p>The final project of the Data Science course (2190513, ICE, Chulalongkorn University). '
                'We turned 61,319 CCTV frames from 100 Bangkok cameras (25-31 August 2026) into a traffic dataset, '
                'compared vehicle detectors, joined four scraped data sources, and built this dashboard.</p></div>',
                unsafe_allow_html=True)

    st.subheader("Team")
    cards = "".join(
        f'<div class="member">{_photo(m["photo"])}<div class="mi"><h3>{m["name"]}</h3>'
        f'<div class="sid">{m["id"]}</div><p>{m["role"]}</p></div></div>' for m in TEAM)
    st.markdown(f'<div class="members">{cards}</div>', unsafe_allow_html=True)

    st.subheader("About the project")
    style.table(["", ""], [
        ["Course", "2190513 Data Science (ICE)"],
        ["Project type", "Final project"],
        ["Data", "61,319 CCTV frames · 100 BMA cameras · 25-31 Aug 2026, 06:00-21:00"],
        ["Scraped sources", "Open-Meteo weather · Open-Meteo air quality · OpenStreetMap Overpass · OpenStreetMap Nominatim"],
        ["Detector", "YOLOX-X (ONNX Runtime, GPU), score threshold 0.30"],
        ["Built with", "Python · pandas · scikit-learn · Streamlit · Plotly"],
        ["AI assistance", "Claude (Anthropic): see the Gen-AI usage statement, docs/AI_USAGE.md"],
    ])
    style.note("Read the full journey on the Project story page.")
