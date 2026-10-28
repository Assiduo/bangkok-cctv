"""Bangkok CCTV dashboard entry point:  streamlit run app/dashboard.py"""
import streamlit as st

st.set_page_config(page_title="Bangkok CCTV Traffic Intelligence", page_icon="📹", layout="wide", initial_sidebar_state="expanded")

import style  # noqa: E402
from data_access import data_version, load_data, pipeline_ready, sidebar_filters  # noqa: E402
from cctv import config  # noqa: E402  (data_access puts src/ on sys.path)
from views import (anomalies, clusters, compare, external_data, forecast, image_explorer, map_view, method,  # noqa: E402
                   model_check, overview, profile, story, team, temporal, trip)

# section -> (title, material icon, module)
SECTIONS = {
    "Explore": [("Overview", "dashboard", overview), ("Map", "map", map_view), ("Temporal patterns", "schedule", temporal)],
    "Cameras": [("Camera profile", "videocam", profile), ("Compare cameras", "compare_arrows", compare),
                ("Camera groups", "hub", clusters)],
    "Insights": [("Best time to travel", "route", trip), ("Anomalies", "warning", anomalies), ("Forecast", "trending_up", forecast),
                 ("External data", "cloud", external_data)],
    "Data": [("Image explorer", "image_search", image_explorer), ("Model check", "fact_check", model_check),
             ("Data & method", "menu_book", method)],
    "About": [("Project story", "auto_stories", story), ("Team", "groups", team)],
}


def topbar(data) -> dict:
    """Filter summary chips on the left, the filter popover on the right."""
    left, right = st.columns([6, 1.2], vertical_alignment="center")
    with right:
        with st.popover("Filters", icon=":material/tune:", use_container_width=True):
            f = sidebar_filters(data, st)
    n_all = f.get("n_clusters", len(f["clusters"]))
    chips = [("Dates", f"{f['d0']:%d %b} – {f['d1']:%d %b}"), ("Hours", f"{f['h0']:02d}:00–{f['h1']:02d}:00"),
             ("Days", f["day_type"]), ("Groups", f"{len(f['clusters'])}/{n_all}")]
    left.markdown('<div class="chips">' + "".join(f'<span class="chip"><i>{k}</i>{v}</span>' for k, v in chips) + "</div>",
                  unsafe_allow_html=True)
    return f


def main() -> None:
    style.setup()
    if not pipeline_ready():
        st.error("Processed data not found. Run the pipeline first:  `bash run_pipeline.sh` (see README).")
        st.stop()
    data = load_data(data_version())
    filters = topbar(data)

    def make(title, icon, mod):
        def run():
            mod.render(data, filters)
        run.__name__ = mod.__name__.split(".")[-1]
        return st.Page(run, title=title, icon=f":material/{icon}:", url_path=run.__name__, default=(mod is overview))

    pages = {sec: [make(*item) for item in items] for sec, items in SECTIONS.items()}
    try:
        nav = st.navigation(pages, expanded=True)     # show every page, no 'View N more'
    except TypeError:                                  # older Streamlit without the argument
        nav = st.navigation(pages)
    q = data.quality
    st.sidebar.markdown(
        f'<div class="side-foot"><b>{q["n_indexed"]:,}</b> frames · <b>100</b> cameras<br>25–31 Aug 2026<br>'
        f'Detector <b>{config.MODEL_VARIANT.upper().replace("YOLOX_", "YOLOX-")}</b> · threshold <b>{config.DET_SCORE_THRESHOLD:.2f}</b></div>',
        unsafe_allow_html=True)
    nav.run()


main()
