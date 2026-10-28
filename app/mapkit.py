"""Camera maps with a basemap choice. 'No basemap' needs no tiles and no WebGL, so it always renders."""
from __future__ import annotations

import json
import math

import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

import style

NONE, LEAFLET = "No basemap (always works)", "OpenStreetMap (Leaflet, no WebGL)"
BASEMAPS = {NONE: None, LEAFLET: "leaflet", "Light (CARTO, WebGL)": "carto-positron",
            "OpenStreetMap (WebGL)": "open-street-map"}
RING = dict(width=0.8, color=style.SIDEBAR)


def view_box(df) -> tuple[float, float, float, float]:
    """(south, west, north, east) around the cameras, ignoring stray outliers."""
    lat, lon = df["latitude"], df["longitude"]
    return lat.quantile(0.02), lon.quantile(0.02), lat.quantile(0.98), lon.quantile(0.98)


def fit_zoom(df, width_px: int = 900, height_px: int = 600) -> tuple[dict, float]:
    """Centre and zoom that frame all cameras (Mapbox/MapLibre zoom: 256px world at zoom 0)."""
    s, w, n, e = view_box(df)
    lon_span, lat_span = max(e - w, 0.01), max((n - s) / max(math.cos(math.radians((n + s) / 2)), 0.2), 0.01)
    zoom = min(math.log2(width_px / 256 * 360 / lon_span), math.log2(height_px / 256 * 180 / lat_span)) - 0.35
    return dict(lat=(n + s) / 2, lon=(e + w) / 2), round(min(max(zoom, 8), 16), 2)


def basemap_picker(key: str) -> str | None:
    label = st.radio("Basemap", list(BASEMAPS), horizontal=True, key=key,
                     help="If a map area is blank, your browser or network is blocking WebGL or map tiles. "
                          "'No basemap' always works.")
    return BASEMAPS[label]


def _ramp(frac: float) -> str:
    cols = style.CONGESTION
    x = min(max(frac, 0), 1) * (len(cols) - 1)
    i = min(int(x), len(cols) - 2)
    a, b = cols[i], cols[i + 1]
    t = x - i
    mix = lambda k: round(int(a[k:k + 2], 16) * (1 - t) + int(b[k:k + 2], 16) * t)  # noqa: E731
    return "#%02x%02x%02x" % (mix(1), mix(3), mix(5))


def leaflet_map(df, color: str | None, height: int, zoom: float) -> None:
    df = df.drop_duplicates("camera_id")
    lo, hi = (df[color].min(), df[color].max()) if color and df[color].dtype.kind in "fi" else (0, 1)
    pts = []
    for r in df.itertuples():
        v = getattr(r, color) if color else 0
        fill = _ramp((v - lo) / (hi - lo + 1e-9)) if color and df[color].dtype.kind in "fi" else style.ACCENT
        pts.append([r.latitude, r.longitude, fill, f"{r.camera_name}<br>{color}: {v:.1f}" if color and
                    df[color].dtype.kind in "fi" else str(getattr(r, "camera_name", ""))])
    html = f"""<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="m" style="height:{height}px;border-radius:12px"></div>
<script>
var m=L.map('m');
m.fitBounds([[{view_box(df)[0]},{view_box(df)[1]}],[{view_box(df)[2]},{view_box(df)[3]}]],{{padding:[30,30]}});
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'&copy; OpenStreetMap'}}).addTo(m);
{json.dumps(pts)}.forEach(function(p){{L.circleMarker([p[0],p[1]],{{radius:8,color:'{style.SIDEBAR}',weight:1,fillColor:p[2],fillOpacity:.95}}).bindTooltip(p[3]).addTo(m);}});
</script>"""
    components.html(html, height=height + 10)


def camera_map(df, basemap: str | None, *, height: int = 620, zoom: float = 10.3, **kw):
    """Returns a Plotly figure, or None when it already drew a Leaflet map."""
    if basemap == "leaflet":
        leaflet_map(df, kw.get("color"), height, zoom)
        return None
    if basemap is None:
        fig = px.scatter(df, x="longitude", y="latitude", height=height, **kw)
        fig.update_traces(marker=dict(line=RING, opacity=0.95))
        fig.update_xaxes(title="longitude")
        fig.update_yaxes(title="latitude", scaleanchor="x", scaleratio=1.03)
        return fig
    center, zoom = fit_zoom(df, height_px=height)
    fig = px.scatter_map(df, lat="latitude", lon="longitude", zoom=zoom, center=center, height=height, **kw)
    fig.update_layout(map_style=basemap, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,.9)"))
    return fig
