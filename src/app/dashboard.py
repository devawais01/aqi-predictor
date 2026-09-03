"""Streamlit dashboard for the Pearls AQI Predictor.

Loads models from the Supabase registry and features from the Supabase
feature store, computes a live three-day forecast, and presents it with
alerts, EDA, model telemetry and SHAP explanations.

Every forecast value is the direct output of a model trained for that
specific horizon. Nothing is interpolated, tiled or noise-injected.

    streamlit run src/app/dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from plotly.subplots import make_subplots

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import config
from src.models import predict as predictor
from src.utils import aqi_calculator as calc
from src.utils import db_client
from src.utils import openweather

st.set_page_config(
    page_title="Lahore AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_COLOURS = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
    "Unknown": "#8899AA",
}

ALERT_STYLES = {
    0: ("✅", "#00E400", "Air quality acceptable"),
    1: ("⚠️", "#FF7E00", "Unhealthy for sensitive groups"),
    2: ("🔴", "#FF0000", "Unhealthy"),
    3: ("🟣", "#8F3F97", "Very unhealthy"),
    4: ("☠️", "#7E0023", "Hazardous"),
}

# Plot palette, kept in one place so charts match the CSS theme.
PLOT_BG = "rgba(0,0,0,0)"
GRID = "rgba(56,189,248,0.10)"
AXIS = "#8296B4"
CYAN = "#38BDF8"
AMBER = "#FF8A3D"

st.markdown("""
<style>
    /* ------------------------------------------------------------------
       Palette is declared here rather than relying only on
       .streamlit/config.toml, because Streamlit Cloud's own Appearance
       setting can override the config file.
       ------------------------------------------------------------------ */
    :root {
        --bg-1:     #070B14;
        --panel:    rgba(20, 30, 48, 0.55);
        --panel-2:  rgba(12, 19, 34, 0.72);
        --line:     rgba(56, 189, 248, 0.15);
        --line-lit: rgba(56, 189, 248, 0.42);
        --cyan:     #38BDF8;
        --indigo:   #6366F1;
        --violet:   #A78BFA;
        --ink:      #E8F0FA;
        --muted:    #8296B4;
    }

    .stApp {
        background:
            radial-gradient(1100px 620px at 12% -8%,
                rgba(56,189,248,0.10) 0%, rgba(56,189,248,0) 62%),
            radial-gradient(900px 520px at 88% 4%,
                rgba(99,102,241,0.12) 0%, rgba(99,102,241,0) 60%),
            linear-gradient(180deg, #070B14 0%, #0A1120 55%, #070B14 100%);
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px;
    }
    [data-testid="stHeader"] { background: transparent; height: 2rem; }
    #MainMenu, footer { visibility: hidden; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,
            rgba(9,14,26,0.96) 0%, rgba(7,11,20,0.96) 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }

    .hero-title {
        font-size: 2.5rem; font-weight: 800; letter-spacing: -0.028em;
        line-height: 1.08; margin: 0 0 0.3rem 0;
        background: linear-gradient(102deg,
            #F0F7FF 0%, var(--cyan) 46%, var(--indigo) 78%, var(--violet) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        color: var(--muted); font-size: 0.89rem; letter-spacing: 0.014em;
        margin: 0;
    }
    .hero-rule {
        height: 2px; width: 100%; margin: 0.85rem 0 0.15rem 0;
        border-radius: 2px;
        background: linear-gradient(90deg,
            var(--cyan) 0%, var(--indigo) 30%, rgba(167,139,250,0.35) 55%,
            rgba(99,102,241,0) 85%);
        box-shadow: 0 0 16px rgba(56,189,248,0.35);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.9rem 1.05rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.045);
        transition: border-color .2s ease, transform .2s ease,
                    box-shadow .2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--line-lit);
        transform: translateY(-2px);
        box-shadow: 0 8px 26px rgba(0,0,0,0.42);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
        /* No uppercase transform: it turns the micro prefix into M. */
        text-transform: none;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700 !important; letter-spacing: -0.022em;
        white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important;
        font-size: clamp(1.3rem, 2.05vw, 1.95rem) !important;
        line-height: 1.12 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 0.5rem 1.05rem;
        color: var(--muted); font-weight: 600; font-size: 0.89rem;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--ink); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg,
            rgba(56,189,248,0.14) 0%, rgba(56,189,248,0.04) 100%);
        color: var(--cyan) !important;
        box-shadow: inset 0 -2px 0 var(--cyan);
    }

    .alert-banner {
        border-radius: 14px; padding: 1rem 1.3rem;
        margin: 0.5rem 0 0.9rem 0;
        font-weight: 600; font-size: 0.95rem; color: #fff;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 10px 30px rgba(0,0,0,0.42),
                    inset 0 1px 0 rgba(255,255,255,0.14);
    }

    .side-stat {
        background: var(--panel-2);
        border: 1px solid var(--line);
        border-radius: 13px; padding: 0.8rem 0.95rem;
        margin-bottom: 0.6rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .side-stat-label {
        color: var(--muted); font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.09em; text-transform: uppercase;
        margin-bottom: 0.18rem;
    }
    .side-stat-value {
        font-size: 1.95rem; font-weight: 800; line-height: 1.02;
        letter-spacing: -0.025em;
    }
    .side-meta { color: var(--muted); font-size: 0.76rem; line-height: 1.65; }
    .side-meta code {
        background: rgba(56,189,248,0.12); color: var(--cyan);
        padding: 1px 6px; border-radius: 5px; font-size: 0.72rem;
        border: 1px solid rgba(56,189,248,0.2);
    }

    div[data-testid="stExpander"] details {
        border: 1px solid var(--line); border-radius: 12px;
        background: var(--panel-2);
    }
    div[data-testid="stExpander"] summary:hover { color: var(--cyan); }

    .stButton > button {
        background: linear-gradient(180deg,
            rgba(56,189,248,0.14), rgba(99,102,241,0.10));
        border: 1px solid var(--line-lit);
        color: var(--ink); font-weight: 600; border-radius: 11px;
        transition: all .18s ease;
    }
    .stButton > button:hover {
        border-color: var(--cyan); color: var(--cyan);
        box-shadow: 0 0 18px rgba(56,189,248,0.28);
    }

    [data-testid="stAlert"] {
        border-radius: 12px; border: 1px solid var(--line);
        background: rgba(56,189,248,0.07);
    }
    hr { border-color: var(--line) !important; }
    .footnote { color: var(--muted); font-size: 0.79rem; }
</style>
""", unsafe_allow_html=True)


# Never cache without a TTL: the underlying data updates hourly, and an
# unbounded cache would serve stale forecasts until the app restarts.
@st.cache_data(ttl=1800, show_spinner="Computing live forecast...")
def load_forecast() -> dict:
    return predictor.forecast()


@st.cache_data(ttl=1800, show_spinner="Loading recent observations...")
def load_history(hours: int = 336) -> pd.DataFrame:
    return predictor.recent_history(hours)


@st.cache_data(ttl=3600, show_spinner="Loading model metrics...")
def load_metrics() -> dict | None:
    payload = db_client.download_json("metrics.json")
    if payload is None and os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as handle:
            payload = json.load(handle)
    return payload


@st.cache_data(ttl=3600)
def load_shap() -> dict | None:
    payload = db_client.download_json("shap_summary.json")
    if payload is None and os.path.exists("models/shap_summary.json"):
        with open("models/shap_summary.json") as handle:
            payload = json.load(handle)
    return payload


@st.cache_data(ttl=1800)
def load_crosscheck(primary_current: dict) -> dict | None:
    """Second-source comparison. Returns None when unavailable."""
    if not openweather.is_configured():
        return None
    try:
        return openweather.compare(primary_current)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading feature history...")
def load_features_sample(rows: int = 17376) -> pd.DataFrame:
    """Feature store rows for the EDA tab. Bounded to protect egress quota."""
    frame = db_client.fetch_features()
    return frame.iloc[-rows:] if not frame.empty else frame


def style_axes(figure: go.Figure) -> go.Figure:
    """Apply the dark theme consistently to any Plotly figure."""
    figure.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=AXIS, size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID,
                        linecolor=GRID, tickfont=dict(color=AXIS))
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID,
                        linecolor=GRID, tickfont=dict(color=AXIS))
    return figure


def aqi_colour(value: float) -> str:
    """Category colour for a given AQI, used in the sidebar readout."""
    return CATEGORY_COLOURS.get(calc.category(value), CYAN)


def aqi_gauge(value: float, title: str = "Current AQI") -> go.Figure:
    """Radial gauge coloured by EPA category band."""
    category = calc.category(value)
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"{title}<br><span style='font-size:0.72em;"
                       f"color:{AXIS}'>{category}</span>"},
        number={"font": {"size": 46, "color": "#F0F7FF"}},
        gauge={
            "axis": {"range": [0, 400], "tickwidth": 1, "tickcolor": AXIS},
            "bar": {"color": CATEGORY_COLOURS.get(category, CYAN),
                    "thickness": 0.75},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(0,228,0,0.16)"},
                {"range": [50, 100], "color": "rgba(255,255,0,0.16)"},
                {"range": [100, 150], "color": "rgba(255,126,0,0.16)"},
                {"range": [150, 200], "color": "rgba(255,0,0,0.16)"},
                {"range": [200, 300], "color": "rgba(143,63,151,0.16)"},
                {"range": [300, 400], "color": "rgba(126,0,35,0.16)"},
            ],
            "threshold": {"line": {"color": "#F0F7FF", "width": 3},
                          "thickness": 0.8, "value": value},
        },
    ))
    figure.update_layout(height=280, margin=dict(t=70, b=10, l=20, r=20),
                         paper_bgcolor=PLOT_BG, font=dict(color=AXIS))
    return figure


def forecast_chart(history: pd.DataFrame, forecast: dict) -> go.Figure:
    """Observed history plus the three model predictions.

    The forecast is drawn as three discrete markers joined by a dashed line.
    The dashes are a visual connector only - no value between the markers is
    a model output, and the chart must not imply otherwise.
    """
    figure = go.Figure()

    if not history.empty:
        local_history = history.copy()
        local_history.index = local_history.index.tz_convert(config.TIMEZONE)
        figure.add_trace(go.Scatter(
            x=local_history.index,
            y=local_history["us_aqi"],
            name="Observed",
            line=dict(color=CYAN, width=2),
            hovertemplate="%{x|%d %b %H:%M}<br>AQI %{y:.0f}<extra></extra>",
        ))

    observation_time = pd.Timestamp(forecast["observation_time_local"])
    current_aqi = forecast["current"]["aqi"]

    times = [observation_time]
    values = [current_aqi]
    labels = ["now"]
    for entry in forecast["forecast"]:
        times.append(pd.Timestamp(entry["valid_at_local"]))
        values.append(entry["aqi"])
        labels.append(f"+{entry['horizon_hours']}h ({entry['model']})")

    figure.add_trace(go.Scatter(
        x=times, y=values, name="Forecast",
        mode="lines+markers+text",
        text=[""] + [f"{v:.0f}" for v in values[1:]],
        textposition="top center",
        textfont=dict(size=13, color=AMBER),
        line=dict(color=AMBER, width=2.5, dash="dash"),
        marker=dict(size=13, symbol="diamond", color=AMBER,
                    line=dict(width=2, color="#0A1120")),
        customdata=labels,
        hovertemplate="%{customdata}<br>%{x|%d %b %H:%M}"
                      "<br>AQI %{y:.1f}<extra></extra>",
    ))

    for threshold, label, colour in [
        (100, "USG", "#FF7E00"),
        (150, "Unhealthy", "#FF4D4D"),
        (200, "Very Unhealthy", "#B06BC7"),
        (300, "Hazardous", "#C2436A"),
    ]:
        figure.add_hline(y=threshold, line_dash="dot", line_color=colour,
                         line_width=1, opacity=0.55,
                         annotation_text=label,
                         annotation_position="right",
                         annotation_font_size=10,
                         annotation_font_color=colour)

    figure.add_vline(x=observation_time.timestamp() * 1000,
                     line_dash="solid", line_color="rgba(130,150,180,0.5)",
                     line_width=1.5)

    figure.update_layout(
        height=460,
        margin=dict(t=30, b=40, l=50, r=90),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="",
        yaxis_title="AQI (US EPA scale)",
    )
    return style_axes(figure)


def render_alerts(forecast: dict) -> None:
    """Alert banners for current and forecast conditions."""
    alerts = []
    current = forecast["current"]
    if current["alert"]["severity"] > 0:
        alerts.append(("Now", current["aqi"], current["alert"]))
    for entry in forecast["forecast"]:
        if entry["alert"]["severity"] > 0:
            alerts.append((f"+{entry['horizon_hours']}h",
                           entry["aqi"], entry["alert"]))

    if not alerts:
        st.success("No health alerts over the next 72 hours.")
        return

    peak = max(alerts, key=lambda a: a[2]["severity"])
    icon, colour, label = ALERT_STYLES[peak[2]["severity"]]

    peak_value = forecast["peak_forecast_aqi"]
    current_value = current["aqi"]
    if peak_value > current_value + 0.5:
        summary = f"worst AQI {peak_value:.0f} expected within 72 hours"
    else:
        summary = (f"AQI {current_value:.0f} now; forecast to improve over "
                   f"the next 72 hours")

    st.markdown(
        f"<div class='alert-banner' style='background:{colour}'>"
        f"{icon} <strong>{label}</strong> &mdash; {summary}. "
        f"{peak[2]['message']}</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"All active alerts ({len(alerts)})"):
        for when, value, alert in alerts:
            icon, _, _ = ALERT_STYLES[alert["severity"]]
            st.write(f"{icon} **{when}** — AQI {value:.0f} "
                     f"({alert['level']}). {alert['message']}")


def tab_forecast(forecast: dict, history: pd.DataFrame) -> None:
    """Live conditions, alerts, and the three-day forecast."""
    current = forecast["current"]

    render_alerts(forecast)

    left, right = st.columns([1, 2])

    with left:
        st.plotly_chart(aqi_gauge(current["aqi"]), width="stretch")
        observed_at = pd.Timestamp(forecast["observation_time_local"])
        st.caption(f"Observed {observed_at.strftime('%d %b %Y, %H:%M')} PKT")

    with right:
        row1 = st.columns(3)
        row1[0].metric("Dominant pollutant", current["dominant_pollutant"])
        row1[1].metric("PM2.5 · µg/m³", f"{current['pm2_5']:.1f}")
        row1[2].metric("PM10 · µg/m³", f"{current['pm10']:.1f}")

        row2 = st.columns(3)
        row2[0].metric("Temperature · °C", f"{current['temperature_2m']:.1f}")
        row2[1].metric("Humidity · %", f"{current['relative_humidity_2m']:.0f}")
        row2[2].metric("Wind · km/h", f"{current['wind_speed_10m']:.1f}")

        st.markdown("**Three-day forecast**")
        cards = st.columns(3)
        for column, entry in zip(cards, forecast["forecast"]):
            delta = entry["aqi"] - current["aqi"]
            column.metric(f"+{entry['horizon_hours']}h",
                          f"{entry['aqi']:.0f}", f"{delta:+.0f} vs now",
                          delta_color="inverse")
            column.caption(f"{entry['category']}  ·  {entry['model']}")

    st.markdown("---")
    st.plotly_chart(forecast_chart(history, forecast), width="stretch")
    st.caption(
        "Diamonds are model outputs at +24h, +48h and +72h. The dashed line "
        "is a visual connector only — no intermediate value is predicted. "
        "Each horizon uses a separately trained model."
    )

    cross = load_crosscheck(current)
    if cross is not None:
        second = cross["secondary"]
        pm_pct = (abs(cross["delta_pm2_5"]) / max(current["pm2_5"], 1e-6)) * 100
        headline = ("close agreement" if pm_pct <= 15
                    else "moderate divergence" if pm_pct <= 35
                    else "substantial divergence")

        with st.expander(
            f"Independent cross-check — PM2.5 differs by "
            f"{cross['delta_pm2_5']:+.1f} µg/m³ ({pm_pct:.0f}%), {headline}"
        ):
            st.markdown(
                "Open-Meteo supplies every value the models use. OpenWeather "
                "is queried separately for the same hour and location, and "
                "its concentrations are scored with the **same** EPA "
                "implementation."
            )
            left, right = st.columns(2)
            with left:
                st.markdown("**Open-Meteo** (primary)")
                st.metric("PM2.5 · µg/m³", f"{current['pm2_5']:.1f}")
                st.metric("PM10 · µg/m³", f"{current['pm10']:.1f}")
                st.caption(f"Dominant: {current['dominant_pollutant']}")
                st.caption(f"AQI {current['aqi']:.0f} · 24-hour mean basis")
            with right:
                st.markdown("**OpenWeather** (independent)")
                st.metric("PM2.5 · µg/m³", f"{second['pm2_5']:.1f}",
                          f"{cross['delta_pm2_5']:+.1f}")
                st.metric("PM10 · µg/m³", f"{second['pm10']:.1f}")
                st.caption(f"Dominant: {second['dominant_pollutant']}")
                st.caption(f"AQI {second['computed_aqi']:.0f} · "
                           f"instantaneous basis")

            st.info(
                "**The concentrations are the fair comparison.** The two AQI "
                "figures are not directly comparable: EPA defines the PM2.5 "
                "sub-index against a 24-hour rolling mean, which Open-Meteo "
                "applies, whereas OpenWeather returns a single instantaneous "
                "reading. A 3 µg/m³ concentration difference also produces "
                "roughly 8 AQI points on its own, because the index climbs "
                "about 2.5 points per µg/m³ in this band."
            )

    with st.expander("How this forecast is produced"):
        st.write(forecast["method"])
        st.write(f"**Feature source:** `{forecast.get('feature_source', 'n/a')}`")
        st.caption(
            "The feature store is the primary source. If the hourly pipeline "
            "has not run recently, the app falls back to computing features "
            "live from Open-Meteo and reports that here."
        )


def tab_performance(metrics: dict | None) -> None:
    """Model comparison against the persistence baseline."""
    if metrics is None:
        st.warning("No metrics available. Run the training pipeline.")
        return

    st.caption(f"Trained {metrics['trained_at'][:19]} UTC  ·  "
               f"{metrics['n_rows']:,} rows  ·  "
               f"{metrics['n_features']} features")

    frame = pd.DataFrame(metrics["results"])

    st.markdown("""
    **Why the persistence baseline matters.** Current AQI is one of the
    model inputs, so any model can score respectably by echoing it. The only
    meaningful question is whether a model beats *"three days from now will
    look like right now."*
    """)

    for horizon in config.HORIZONS:
        subset = frame[frame["horizon"] == horizon].copy()
        if subset.empty:
            continue

        baseline_rows = subset[subset["model"] == "Persistence"]
        baseline_rmse = (float(baseline_rows["rmse"].iloc[0])
                         if not baseline_rows.empty else None)

        st.markdown(f"#### +{horizon} hours")
        display = subset[["model", "rmse", "mae", "r2"]].copy()
        if "cv_r2_mean" in subset.columns:
            display["cv_r2"] = subset["cv_r2_mean"]
        if baseline_rmse:
            display["vs_baseline_%"] = (
                (baseline_rmse - display["rmse"]) / baseline_rmse * 100
            ).round(1)
        display = display.sort_values("rmse").round(3)
        st.dataframe(display, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("**Selected model per horizon**")
    selected = metrics.get("best_per_horizon", {})
    if selected:
        columns = st.columns(len(selected))
        for column, (horizon, best) in zip(columns, selected.items()):
            column.metric(f"+{horizon}h", best["model"],
                          f"R² {best['r2']:.3f}")

    st.info(
        "Selection uses the **highest mean walk-forward CV R²**, not the "
        "lowest single-split RMSE. A single 80/20 split on two years of "
        "seasonal data places the whole test set in one season. At +72h "
        "Random Forest wins the split while scoring a negative mean CV R²; "
        "Ridge loses the split by a fraction of an RMSE point and stays "
        "positive."
    )


def tab_explain(shap_summary: dict | None) -> None:
    """SHAP feature attribution."""
    if shap_summary is None:
        st.warning("No SHAP summary available. Run `python -m src.models.explain`.")
        return

    st.markdown("""
    SHAP values decompose each prediction into per-feature contributions.
    Computed over **1,000 test rows**, so these reflect general behaviour
    rather than one prediction.
    """)

    horizon = st.selectbox("Horizon", list(shap_summary.keys()),
                           format_func=lambda h: f"+{h} hours")
    entry = shap_summary[str(horizon)]

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Top features by mean |SHAP|**")
        top = pd.Series(entry["top_features"]).head(15).sort_values()
        figure = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker=dict(color=top.values, colorscale="Blues",
                        line=dict(width=0)),
        ))
        figure.update_layout(height=440, margin=dict(t=10, b=30, l=10, r=10),
                             xaxis_title="mean |SHAP|", showlegend=False)
        st.plotly_chart(style_axes(figure), width="stretch")

    with right:
        st.markdown("**Contribution by feature group**")
        groups = pd.Series(entry["group_percentages"])
        groups = groups[groups > 0.05].sort_values(ascending=False)
        figure = go.Figure(go.Pie(
            labels=groups.index, values=groups.values, hole=0.45,
            marker=dict(colors=["#38BDF8", "#6366F1", "#A78BFA", "#22D3EE",
                                "#818CF8", "#2DD4BF", "#F472B6", "#FBBF24"],
                        line=dict(color="#0A1120", width=2)),
        ))
        figure.update_layout(height=440, margin=dict(t=10, b=10, l=10, r=10),
                             showlegend=True,
                             legend=dict(orientation="v", font=dict(size=10)),
                             paper_bgcolor=PLOT_BG, font=dict(color=AXIS))
        st.plotly_chart(figure, width="stretch")

    st.info(
        "Future weather (perfect prognosis) rises from **13.4%** of total "
        "importance at +24h to **25.2%** at +72h, while current pollutant "
        "state falls from 52.3% to 32.7%. Short horizons are momentum; long "
        "horizons are driven by forecast conditions."
    )

    for figure_path in [
        f"reports/figures/08_shap_beeswarm_t{horizon}.png",
        f"reports/figures/09_shap_bar_t{horizon}.png",
    ]:
        if os.path.exists(figure_path):
            st.image(figure_path, width="stretch")


def tab_eda(features: pd.DataFrame) -> None:
    """Exploratory analysis of the two-year record."""
    if features.empty:
        st.warning("Feature store is empty.")
        return

    local = features.index.tz_convert(config.TIMEZONE)
    frame = features.copy()
    frame["hour"] = local.hour
    frame["month"] = local.month

    columns = st.columns(4)
    columns[0].metric("Observations", f"{len(frame):,}")
    columns[1].metric("Mean AQI (US EPA)", f"{frame['us_aqi'].mean():.0f}")
    columns[2].metric("Minimum AQI", f"{frame['us_aqi'].min():.0f}")
    columns[3].metric("Hours ≥ Unhealthy",
                      f"{(frame['us_aqi'] >= 151).mean():.0%}")

    st.info(
        f"Across {len(frame):,} hours the AQI never once fell into the EPA "
        f"'Good' band (0–50). The minimum observed value was "
        f"{frame['us_aqi'].min():.0f}."
    )

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    figure = make_subplots(rows=1, cols=2,
                           subplot_titles=("Seasonal cycle", "Daily cycle"))
    monthly = frame.groupby("month")["us_aqi"].mean()
    figure.add_trace(go.Scatter(
        x=[months[m - 1] for m in monthly.index], y=monthly.values,
        mode="lines+markers", line=dict(color="#F472B6", width=3),
        marker=dict(size=7), showlegend=False,
    ), row=1, col=1)

    hourly = frame.groupby("hour")["us_aqi"].mean()
    figure.add_trace(go.Scatter(
        x=hourly.index, y=hourly.values,
        mode="lines+markers", line=dict(color="#A78BFA", width=3),
        marker=dict(size=6), showlegend=False,
    ), row=1, col=2)

    figure.update_yaxes(title_text="AQI (US EPA scale)", row=1, col=1)
    figure.update_xaxes(title_text="hour (PKT)", row=1, col=2)
    figure.update_layout(height=380, margin=dict(t=50, b=40))
    st.plotly_chart(style_axes(figure), width="stretch")

    st.caption(
        f"Seasonal swing of {monthly.max() - monthly.min():.0f} AQI points "
        f"(worst {months[monthly.idxmax() - 1]} {monthly.max():.0f}, best "
        f"{months[monthly.idxmin() - 1]} {monthly.min():.0f}) against a "
        f"daily range of only {hourly.max() - hourly.min():.0f} points. "
        f"Lahore's air is governed by season, not time of day."
    )

    if "dominant_pollutant" in frame.columns:
        st.markdown("**Dominant pollutant**")
        counts = frame["dominant_pollutant"].value_counts()
        figure = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.45,
            marker=dict(colors=["#38BDF8", "#A78BFA", "#22D3EE", "#6366F1"],
                        line=dict(color="#0A1120", width=2)),
        ))
        figure.update_layout(height=320, margin=dict(t=10, b=10),
                             paper_bgcolor=PLOT_BG, font=dict(color=AXIS))
        st.plotly_chart(figure, width="stretch")

    for path, caption in [
        ("reports/figures/03_full_timeseries.png",
         "Two years of daily mean AQI"),
        ("reports/figures/04_autocorrelation.png",
         "Autocorrelation decay — why persistence fails at +72h"),
        ("reports/figures/06_drivers.png", "Correlation with AQI"),
    ]:
        if os.path.exists(path):
            st.image(path, caption=caption, width="stretch")


def tab_about() -> None:
    """Methodology, honest limitations, and attribution."""
    st.markdown(f"""
### Pearls AQI Predictor — {config.CITY}, Pakistan

Three-day AQI forecasting (US EPA scale) on a fully serverless stack.

#### Architecture

| Layer | Implementation |
|---|---|
| Data source | Open-Meteo archive + air-quality APIs (free, no key) |
| Cross-validation | OpenWeather Air Pollution API (independent second source) |
| Feature store | Supabase Postgres (`feature_store`, jsonb) |
| Model registry | Supabase Storage (`model-registry`) |
| Training | scikit-learn, XGBoost, statsmodels, TensorFlow |
| Serving | FastAPI + Streamlit |
| Automation | GitHub Actions (hourly features, daily training) |

#### Method

Three **independent** models, one per horizon, with targets
`us_aqi.shift(-24)`, `shift(-48)` and `shift(-72)`. There is no recursive
forecasting and no fabricated future pollutant data. Every number shown is
a direct model output.

**Perfect prognosis.** Weather at time *t* cannot explain air quality three
days later — weather at *t+72* can. Training uses archived actual weather at
the target hour; inference substitutes the live Open-Meteo forecast.

**Independent cross-validation.** OpenWeather is queried separately for the
same location and hour. Its concentrations are scored with the *same* EPA
implementation, so any divergence reflects the measurements rather than the
index definition.

#### Honest limitations

1. **Train/serve mismatch.** Real forecasts carry error that archived
   actuals do not, so live accuracy will be modestly below reported test
   accuracy.
2. **Seasonal variance.** Walk-forward CV shows R² ranging from −0.14 to
   0.80 across folds. Summer is hardest because AQI is flat and there is
   little variance to explain.
3. **LSTM instability.** The same LSTM scored R² between 0.007 and 0.563 at
   +48h across four identical runs. It is not deployed at any horizon.
4. **Computed AQI clips at 500**; Open-Meteo extrapolates beyond the formal
   EPA scale maximum.
5. **Single coordinate pair.** Lahore varies spatially; the two providers
   differ by 37.8 µg/m³ on PM10 for the same point, which indicates how much
   coarse particulate depends on sensor placement.

#### Data attribution

Weather and air quality data by [Open-Meteo](https://open-meteo.com),
licensed CC-BY 4.0. Cross-validation data from
[OpenWeather](https://openweathermap.org).
""")


def main() -> None:
    st.markdown(
        "<div class='hero-title'>Lahore Air Quality Forecast</div>"
        "<div class='hero-sub'>Three-day AQI prediction on the US EPA scale "
        "&middot; direct multi-horizon models &middot; data by Open-Meteo "
        "(CC-BY 4.0)</div>"
        "<div class='hero-rule'></div>",
        unsafe_allow_html=True,
    )

    try:
        forecast = load_forecast()
    except Exception as exc:
        st.error(f"Could not compute a forecast: {exc}")
        st.stop()

    current = forecast["current"]
    peak = forecast["peak_forecast_aqi"]
    now_local = datetime.now(pytz.timezone(config.TIMEZONE))

    with st.sidebar:
        st.markdown(
            f"<div class='side-stat'>"
            f"<div class='side-stat-label'>Current AQI</div>"
            f"<div class='side-stat-value' "
            f"style='color:{aqi_colour(current['aqi'])}'>"
            f"{current['aqi']:.0f}</div>"
            f"<div class='side-meta'>{current['category']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='side-stat'>"
            f"<div class='side-stat-label'>Worst over 72 h</div>"
            f"<div class='side-stat-value' style='color:{aqi_colour(peak)}'>"
            f"{peak:.0f}</div>"
            f"<div class='side-meta'>{calc.category(peak)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("Refresh data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            f"<div class='side-meta' style='margin-top:0.7rem'>"
            f"Source <code>{forecast.get('feature_source', 'n/a')}</code><br/>"
            f"Updated {now_local.strftime('%H:%M')} PKT "
            f"&middot; {now_local.strftime('%d %b')}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<div class='side-stat-label'>Alert thresholds (US EPA)</div>",
            unsafe_allow_html=True,
        )
        tiers = [
            ("Sensitive groups", 101, "#FF7E00"),
            ("Unhealthy", 151, "#FF4D4D"),
            ("Very unhealthy", 201, "#B06BC7"),
            ("Hazardous", 301, "#C2436A"),
        ]
        rows = "".join(
            f"<div style='display:flex;align-items:center;gap:8px;"
            f"margin:5px 0;font-size:0.79rem;color:#8296B4'>"
            f"<span style='width:9px;height:9px;border-radius:50%;"
            f"background:{colour};display:inline-block;"
            f"box-shadow:0 0 8px {colour}'></span>"
            f"{label}<span style='margin-left:auto;color:#6B7A94'>"
            f"{threshold}+</span></div>"
            for label, threshold, colour in tiers
        )
        st.markdown(f"<div style='margin-top:0.4rem'>{rows}</div>",
                    unsafe_allow_html=True)

    tabs = st.tabs(["Forecast", "Model performance", "Explainability",
                    "Data analysis", "About"])

    with tabs[0]:
        try:
            history = load_history(336)
        except Exception as exc:
            st.warning(f"History unavailable: {exc}")
            history = pd.DataFrame()
        tab_forecast(forecast, history)

    with tabs[1]:
        tab_performance(load_metrics())

    with tabs[2]:
        tab_explain(load_shap())

    with tabs[3]:
        try:
            tab_eda(load_features_sample())
        except Exception as exc:
            st.warning(f"Feature store unavailable: {exc}")

    with tabs[4]:
        tab_about()


main()