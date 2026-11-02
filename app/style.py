"""Visual identity: clean light tech UI (slate neutrals, one blue accent). One place for colours, CSS, logo and the shared Plotly template."""
from __future__ import annotations

import math
import re
import urllib.parse

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK, MUTED, GRID = "#0f172a", "#64748b", "#e8ecf2"
LINE, PAGE, SURFACE = "#e5e9f0", "#f6f7fb", "#ffffff"
ACCENT, ACCENT_SOFT = "#3b6ef6", "#eef3ff"
SIDEBAR = "#0f172a"            # dark ink: map marker rings, tooltips
WARN = "#e8590c"
# categorical order is fixed (validated for colour-blind separation); never cycle past 5
CATEGORICAL = ["#3b6ef6", "#e8590c", "#7c4dff", "#0a89a6", "#d6336c"]
# sequential: light = free-flowing, dark = jammed
CONGESTION = ["#eef3ff", "#c7d7fe", "#8fb0fb", "#4f80f7", "#1e3fa8"]


def register_plotly_template() -> None:
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED, size=12),
                title=dict(font=dict(color=MUTED, size=12)))
    pio.templates["cctv"] = go.layout.Template(layout=go.Layout(
        font=dict(family="Inter, 'Noto Sans Thai', 'Segoe UI', system-ui, sans-serif", color=INK, size=13),
        colorway=CATEGORICAL, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=axis, yaxis=axis, margin=dict(l=10, r=10, t=44, b=120),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.34, xanchor="left", x=0, title_text="", font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
        coloraxis=dict(colorscale=[[i / (len(CONGESTION) - 1), c] for i, c in enumerate(CONGESTION)],
                       colorbar=dict(outlinewidth=0, thickness=12)),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#fff", size=12), bordercolor=INK),
    ))
    pio.templates.default = "cctv"


# ---------------------------------------------------------------- logo
def _logo_svg() -> str:
    """Minimal black-and-white mark: a wall-mounted CCTV (bullet) camera with a square lens, plain wordmark."""
    f = "Inter, Segoe UI, system-ui, sans-serif"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="236" height="52" viewBox="0 0 236 52">'
        '<g transform="translate(2 3)" fill="#0f172a">'
        '<g transform="rotate(8 26 22)"><rect x="13" y="13" width="25" height="14" rx="4"/><rect x="15" y="9.5" width="19" height="4" rx="1.5"/>'
        '<rect x="32" y="15.2" width="8.4" height="8.4" rx="1.6" fill="#fff"/><rect x="34.5" y="17.7" width="3.4" height="3.4" rx=".6"/></g>'
        '<rect x="3" y="9" width="4.5" height="28" rx="1.8"/>'
        '<path d="M7.5 23H16" stroke="#0f172a" stroke-width="4" stroke-linecap="round"/></g>'
        f'<text x="60" y="26" font-family="{f}" font-size="20" font-weight="800" fill="#0f172a" letter-spacing="-0.4">'
        f'Bangkok <tspan font-weight="400">CCTV</tspan></text>'
        f'<text x="60" y="42" font-family="{f}" font-size="9.5" font-weight="600" fill="#64748b" letter-spacing="1.8">TRAFFIC INTELLIGENCE</text></svg>')


PANEL_URI = "data:image/svg+xml;utf8," + urllib.parse.quote(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" stroke="black" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round"><rect x="1.8" y="2.5" width="12.4" height="11" rx="2.2"/><path d="M6.2 2.8v10.4"/></svg>')


LOGO_URI = "data:image/svg+xml;utf8," + urllib.parse.quote(_logo_svg())

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, label, h1, h2, h3, h4, button, input, textarea, [data-testid="stMetric"] {{
  font-family: Inter, 'Noto Sans Thai', 'Segoe UI', system-ui, sans-serif; }}
.stApp {{ background:{PAGE}; }}
/* header: keep it (it holds the sidebar re-open button) but make it invisible; only drop the deploy button */
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stAppDeployButton"], .stAppDeployButton, #MainMenu {{ display:none; }}
.block-container {{ padding-top:3.4rem; padding-bottom:4rem; max-width:1180px; }}
h1, h2, h3 {{ letter-spacing:-0.02em; color:{INK}; }}

/* sidebar */
[data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {LINE}; }}
[data-testid="stSidebarHeader"] {{ position:absolute; top:29px; right:12px; z-index:5; padding:0; height:auto; width:auto; background:transparent; }}
[data-testid="stSidebarNav"]::before {{ content:""; display:block; height:56px; margin:4px 12px 12px;
  background:url("{LOGO_URI}") no-repeat left center / contain; }}
[data-testid="stSidebarNav"] {{ padding-top:.9rem; }}
[data-testid="stSidebarUserContent"] {{ padding-top:0; }}
[data-testid="stSidebarNavLink"] {{ border-radius:9px; padding:.42rem .7rem; color:{MUTED}; font-weight:500; margin:1px 0; }}
[data-testid="stSidebarNavLink"]:hover {{ background:#f1f4f9; color:{INK}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background:{ACCENT_SOFT}; color:{ACCENT}; font-weight:650; }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color:{ACCENT}; }}
[data-testid="stNavSectionHeader"], [data-testid="stSidebarNav"] header {{ color:#94a3b8; font-size:.68rem; letter-spacing:.12em;
  text-transform:uppercase; font-weight:700; padding:.9rem .7rem .2rem; }}
.side-foot {{ color:{MUTED}; font-size:.78rem; line-height:1.55; border-top:1px solid {LINE}; padding-top:12px; margin-top:8px; }}
.side-foot b {{ color:{INK}; }}

/* top bar with filter summary */
.chips {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; min-height:40px; }}
.chip {{ background:{SURFACE}; border:1px solid {LINE}; color:{INK}; border-radius:999px; padding:5px 12px; font-size:.8rem; font-weight:500; }}
.chip i {{ font-style:normal; color:{MUTED}; margin-right:6px; }}

/* page header */
.pagehead {{ margin:6px 0 20px; }}
.pagehead h2 {{ margin:0; padding:0; font-size:1.85rem; font-weight:750; letter-spacing:-0.025em; }}
.pagehead p {{ color:{MUTED}; margin:.35rem 0 0; font-size:1rem; max-width:80ch; }}

/* cards */
[data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; padding:16px 18px;
  box-shadow:0 1px 2px rgba(15,23,42,.04); }}
[data-testid="stMetricLabel"] {{ color:{MUTED}; }}
[data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; font-weight:700; letter-spacing:-0.02em; }}
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {{ background:{SURFACE}; border:1px solid {LINE};
  border-radius:14px; padding:14px 16px 8px; box-shadow:0 1px 2px rgba(15,23,42,.04); margin-bottom:6px; overflow:visible; }}
[data-testid="stPlotlyChart"] {{ padding:0; }}
[data-testid="stPlotlyChart"] svg.main-svg {{ background:transparent !important; }}
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:12px; overflow:hidden; }}
.insights {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; padding:4px 18px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.insights div {{ padding:10px 0; border-bottom:1px solid {LINE}; }}
.insights div:last-child {{ border-bottom:0; }}
[data-testid="stImage"] img {{ border-radius:12px; }}
.stDownloadButton button, [data-testid="stPopover"] button {{ border-radius:10px; border:1px solid {LINE}; background:{SURFACE}; font-weight:600; }}

/* story sections */
.kicker {{ display:inline-block; background:{ACCENT_SOFT}; color:{ACCENT}; font-size:.7rem; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; padding:4px 11px; border-radius:999px; margin:38px 0 8px; }}
.story h3 {{ margin:0 0 6px; font-size:1.4rem; font-weight:750; }}
.story p {{ color:{MUTED}; margin:0 0 16px; max-width:78ch; line-height:1.6; }}
.story p b {{ color:{INK}; }}
.note {{ background:#f1f4f9; border:1px solid {LINE}; border-radius:12px; padding:11px 15px; color:{INK}; font-size:.9rem; margin:6px 0 12px; }}

/* stat cards + infographics */
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin:4px 0 10px; }}
.stat {{ display:flex; flex-direction:column; gap:12px; background:{SURFACE}; border:1px solid {LINE};
  border-radius:16px; padding:18px 20px; box-shadow:0 1px 2px rgba(15,23,42,.04); min-width:0; }}
.stat-top {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
.stat-bottom svg {{ width:100%; height:44px; display:block; }}
.segbar {{ display:flex; gap:3px; height:12px; }} .segbar i {{ border-radius:6px; display:block; }}
.stat-label {{ display:block; color:{MUTED}; font-size:.78rem; font-weight:600; letter-spacing:.02em; }}
.stat-value {{ display:block; font-size:2rem; font-weight:750; letter-spacing:-0.03em; line-height:1.15; margin-top:4px; font-variant-numeric:tabular-nums; }}
.stat-sub {{ display:block; color:{MUTED}; font-size:.78rem; margin-top:2px; }}
.rhythm {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:16px; padding:18px 20px 14px; box-shadow:0 1px 2px rgba(15,23,42,.04); margin:10px 0 6px; }}
.rh-head {{ display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:6px; margin-bottom:12px; }}
.rh-head b {{ font-size:.95rem; }} .rh-head span {{ color:{MUTED}; font-size:.8rem; }}
.rh-cells {{ display:flex; gap:4px; }}
.rh-cell {{ flex:1; min-width:0; text-align:center; }}
.rh-bar {{ height:46px; border-radius:8px; }}
.rh-cell small {{ display:block; color:{MUTED}; font-size:.68rem; margin-top:5px; font-variant-numeric:tabular-nums; }}
.rh-cell.peak .rh-bar {{ outline:2px solid {INK}; outline-offset:2px; }}
.rh-legend {{ display:flex; align-items:center; gap:8px; color:{MUTED}; font-size:.72rem; margin-top:10px; }}
.rh-legend div {{ width:120px; height:8px; border-radius:99px; background:linear-gradient(90deg,{",".join(CONGESTION)}); }}

/* story components */
.chap {{ display:flex; gap:16px; align-items:flex-start; margin:54px 0 18px; }}
.chap-n {{ flex:0 0 auto; width:46px; height:46px; border-radius:13px; background:{INK}; color:#fff; font-weight:750; display:flex; align-items:center;
  justify-content:center; font-variant-numeric:tabular-nums; letter-spacing:-0.02em; }}
.chap h2 {{ margin:0; padding:0; font-size:1.55rem; font-weight:750; letter-spacing:-0.025em; }}
.chap p {{ margin:.3rem 0 0; color:{MUTED}; max-width:78ch; line-height:1.6; }}
.callout {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; padding:14px 18px; margin:12px 0; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.callout .tag {{ display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; padding:3px 10px; border-radius:999px;
  background:#f1f4f9; color:{MUTED}; margin-bottom:8px; }}
.callout.obstacle .tag {{ background:#fff1ea; color:{WARN}; }}
.callout.decision .tag {{ background:{ACCENT_SOFT}; color:{ACCENT}; }}
.callout.result .tag {{ background:#eaf7f9; color:#0a89a6; }}
.callout b.t {{ display:block; font-size:1rem; margin-bottom:4px; }}
.callout p {{ margin:0; color:#334155; line-height:1.6; }}
.facts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:14px 0; }}
.fact {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; padding:14px 16px; }}
.fact b {{ display:block; font-size:1.6rem; font-weight:750; letter-spacing:-0.03em; font-variant-numeric:tabular-nums; }}
.fact span {{ color:{MUTED}; font-size:.8rem; }}
.tl {{ position:relative; margin:14px 0 6px; padding-left:34px; }}
.tl::before {{ content:""; position:absolute; left:11px; top:6px; bottom:6px; width:2px; background:{LINE}; }}
.tl-i {{ position:relative; padding:0 0 18px; }}
.tl-i::before {{ content:""; position:absolute; left:-30px; top:5px; width:12px; height:12px; border-radius:50%; background:{SURFACE}; border:3px solid {INK}; }}
.tl-i > b {{ display:block; font-size:.98rem; }} .tl-i span b {{ display:inline; color:#0f172a; }} .tl-i span {{ color:#475569; line-height:1.6; font-size:.93rem; }}
.pills {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 4px; }}
.pill {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:999px; padding:5px 12px; font-size:.82rem; font-weight:500; }}
.rubric {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:10px 0 6px; }}
.rubric div {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; padding:14px 16px; }}
.rubric b {{ display:block; margin-bottom:4px; }} .rubric span {{ color:{MUTED}; font-size:.85rem; line-height:1.5; }}
.tbl {{ width:100%; border-collapse:separate; border-spacing:0; background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; overflow:hidden; font-size:.9rem; margin:10px 0; }}
.tbl th {{ text-align:left; background:#f8fafc; color:{MUTED}; font-size:.74rem; letter-spacing:.06em; text-transform:uppercase; padding:10px 14px; border-bottom:1px solid {LINE}; }}
.tbl td {{ padding:11px 14px; border-bottom:1px solid {LINE}; vertical-align:top; line-height:1.5; }}
.tbl tr:last-child td {{ border-bottom:0; }}
.tblwrap {{ overflow-x:auto; }}
/* team */
.hero {{ background:{INK}; color:#fff; border-radius:20px; padding:34px 36px; margin:6px 0 22px; }}
.hero small {{ color:#94a3b8; letter-spacing:.14em; text-transform:uppercase; font-weight:700; font-size:.72rem; }}
.hero h1 {{ font-family:inherit; color:#fff; margin:10px 0 8px; font-size:2.1rem; font-weight:800; letter-spacing:-0.03em; line-height:1.15; }}
.hero p {{ color:#cbd5e1; max-width:70ch; line-height:1.6; margin:0; }}
.members {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }}
.member {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:18px; overflow:hidden; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.member img {{ width:100%; aspect-ratio:1/1; object-fit:cover; display:block; }}
.member .mi {{ padding:16px 18px 18px; }}
.member h3 {{ margin:0; font-size:1.15rem; font-weight:750; letter-spacing:-0.02em; }}
.member .sid {{ color:{MUTED}; font-size:.85rem; font-variant-numeric:tabular-nums; margin:2px 0 10px; }}
.member p {{ margin:0; color:#334155; font-size:.9rem; line-height:1.55; }}
/* sidebar toggle: a panel icon instead of the double arrows (top-right of the logo row when open, top-left when hidden) */
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] {{ width:32px; height:32px; border-radius:9px; background:#f1f4f9; display:grid; place-items:center; }}
[data-testid="stSidebarCollapseButton"] button:hover, [data-testid="stExpandSidebarButton"]:hover {{ background:#e3e9f3; }}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"], [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{
  font-size:0 !important; color:transparent !important; display:block; width:18px; height:18px; background:{MUTED};
  -webkit-mask:url("{PANEL_URI}") center/contain no-repeat; mask:url("{PANEL_URI}") center/contain no-repeat; }}
</style>
"""


def setup() -> None:
    register_plotly_template()
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    page_header(title, subtitle)


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="pagehead"><h2>{title}</h2>{sub}</div>', unsafe_allow_html=True)


_BOLD = re.compile(r"\*\*(.+?)\*\*")


def insight_cards(lines: list[str]) -> None:
    rows = "".join("<div>" + _BOLD.sub(r"<b>\1</b>", line) + "</div>" for line in lines)
    st.markdown(f'<div class="insights">{rows}</div>', unsafe_allow_html=True)


def story_head(kicker: str, question: str, answer: str) -> None:
    """One chapter of the overview: topic label, the question, and the answer in plain words."""
    body = _BOLD.sub(r"<b>\1</b>", answer)
    st.markdown(f'<div class="kicker">{kicker}</div><div class="story"><h3>{question}</h3><p>{body}</p></div>',
                unsafe_allow_html=True)


def no_preview(caption: str = "") -> None:
    """Shown instead of a frame when the online demo does not ship that image (the full set is ~1.7 GB)."""
    cap = f"<br><span>{caption}</span>" if caption else ""
    st.markdown(
        '<div style="border:1px dashed #cbd5e1;border-radius:12px;background:#f8fafc;padding:18px 16px;color:#475569;'
        'font-size:.85rem;line-height:1.45;text-align:center"><b style="color:#334155">Preview not included in the online demo</b><br>'
        'The full image set (61,319 frames, about 1.7 GB) and the large YOLOX-X detector are too big for free hosting, '
        'so only a sample of frames is shipped. Run the dashboard locally to see every frame.' + cap + '</div>',
        unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def download_csv(df, filename: str, label: str = "Download CSV") -> None:
    """Small CSV export button (UTF-8 with BOM so Excel shows Thai correctly)."""
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv",
                       key=f"dl_{filename}")


# ---------------------------------------------------------------- infographics
def sparkline(values, color: str = ACCENT, w: int = 240, h: int = 44) -> str:
    v = [float(x) for x in values]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = [(2 + i * (w - 4) / (len(v) - 1), h - 4 - (x - lo) / span * (h - 10)) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"2,{h} " + line + f" {w - 2},{h}"
    gid = f"sp{abs(hash(line)) % 10**6}"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".22"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="{area}" fill="url(#{gid})"/><polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/><circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="3" fill="{color}"/></svg>')


def ring(pct: float, color: str = ACCENT, size: int = 64) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    r = 26
    c = 2 * math.pi * r
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 64 64"><circle cx="32" cy="32" r="{r}" fill="none" stroke="#e8ecf4" stroke-width="7"/>'
            f'<circle cx="32" cy="32" r="{r}" fill="none" stroke="{color}" stroke-width="7" stroke-linecap="round" '
            f'stroke-dasharray="{c * pct / 100:.1f} {c:.1f}" transform="rotate(-90 32 32)"/></svg>')


def stat_cards(cards: list[dict]) -> None:
    """cards: [{label, value, sub, ring(html, optional), bottom(html, optional)}]"""
    html = "".join(
        f'<div class="stat"><div class="stat-top"><div><span class="stat-label">{c["label"]}</span><span class="stat-value">{c["value"]}</span>'
        f'<span class="stat-sub">{c.get("sub", "")}</span></div><div>{c.get("ring", "")}</div></div>'
        f'<div class="stat-bottom">{c.get("bottom", "")}</div></div>' for c in cards)
    st.markdown(f'<div class="stats">{html}</div>', unsafe_allow_html=True)


def segbar(values, colors=None) -> str:
    """Proportional segmented bar (e.g. how many cameras are in each behaviour group)."""
    colors = colors or CATEGORICAL
    tot = float(sum(values)) or 1.0
    return '<div class="segbar">' + "".join(f'<i style="flex:{v / tot:.4f};background:{colors[i % len(colors)]}"></i>' for i, v in enumerate(values)) + "</div>"


def _ramp(t: float) -> str:
    t = max(0.0, min(1.0, t)) * (len(CONGESTION) - 1)
    i = min(int(t), len(CONGESTION) - 2)
    a, b = CONGESTION[i], CONGESTION[i + 1]
    f = t - i
    mix = lambda k: int(int(a[k:k + 2], 16) * (1 - f) + int(b[k:k + 2], 16) * f)  # noqa: E731
    return f"#{mix(1):02x}{mix(3):02x}{mix(5):02x}"


def rhythm_strip(by_hour, title: str = "A day in Bangkok traffic") -> None:
    """One cell per hour, coloured by mean congestion; the peak hour is outlined."""
    if len(by_hour) == 0:
        return
    lo, hi = float(by_hour.min()), float(by_hour.max())
    pk = by_hour.idxmax()
    cells = "".join(
        f'<div class="rh-cell{" peak" if h == pk else ""}" title="{int(h):02d}:00 · index {v:.0f}"><div class="rh-bar" '
        f'style="background:{_ramp((v - lo) / ((hi - lo) or 1))}"></div><small>{int(h):02d}</small></div>' for h, v in by_hour.items())
    st.markdown(f'<div class="rhythm"><div class="rh-head"><b>{title}</b><span>Busiest at {int(pk):02d}:00 · quietest at '
                f'{int(by_hour.idxmin()):02d}:00</span></div><div class="rh-cells">{cells}</div>'
                f'<div class="rh-legend">free<div></div>jammed</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- story components
def _b(text: str) -> str:
    """**bold** markers to <b> (kept outside f-strings: Python 3.11 forbids backslashes inside them)."""
    return _BOLD.sub(r"<b>\1</b>", text)


def chapter(num: int, title: str, lede: str = "") -> None:
    p = "<p>" + _b(lede) + "</p>" if lede else ""
    st.markdown(f'<div class="chap"><div class="chap-n">{num:02d}</div><div><h2>{title}</h2>{p}</div></div>', unsafe_allow_html=True)


def callout(kind: str, title: str, body: str) -> None:
    """kind: obstacle | decision | result | note"""
    label = {"obstacle": "Obstacle", "decision": "Decision", "result": "Result", "note": "Note"}.get(kind, kind)
    st.markdown(f'<div class="callout {kind}"><span class="tag">{label}</span><b class="t">{title}</b><p>{_b(body)}</p></div>',
                unsafe_allow_html=True)


def facts(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="facts">' + "".join(f'<div class="fact"><b>{v}</b><span>{k}</span></div>' for v, k in items) + "</div>",
                unsafe_allow_html=True)


def timeline(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="tl">' + "".join(f'<div class="tl-i"><b>{t}</b><span>{_b(x)}</span></div>' for t, x in items) + "</div>",
                unsafe_allow_html=True)


def pills(items: list[str]) -> None:
    st.markdown('<div class="pills">' + "".join(f'<span class="pill">{i}</span>' for i in items) + "</div>", unsafe_allow_html=True)


def table(headers: list[str], rows: list[list[str]]) -> None:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join("<td>" + _b(str(c)) + "</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(f'<div class="tblwrap"><table class="tbl"><tr>{head}</tr>{body}</table></div>', unsafe_allow_html=True)
