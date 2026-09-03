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

import pytz
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

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
    "Unknown": "#888888",
}


ALERT_STYLES = {
    0: ("✅", "#00E400", "Air quality acceptable"),
    1: ("⚠️", "#FF7E00", "Unhealthy for sensitive groups"),
    2: ("🔴", "#FF0000", "Unhealthy"),
    3: ("🟣", "#8F3F97", "Very unhealthy"),
    4: ("☠️", "#7E0023", "Hazardous"),
}


st.markdown(
    """
<style>

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 1350px;
}

[data-testid="stHeader"] {
    background: transparent;
    height: 2.2rem;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}


.hero-title {
    font-size: 2.45rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}


.hero-sub {
    color: #7D8DA6;
    font-size: 0.9rem;
}


.hero-rule {
    height: 2px;
    width: 100%;
    margin: 0.9rem 0 0.2rem 0;
}


[data-testid="stMetric"] {
    border-radius: 13px;
    padding: 0.85rem 1rem;
}


[data-testid="stMetricLabel"] p {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    text-transform: none;
}


[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    white-space: normal !important;
}


.alert-banner {
    border-radius: 13px;
    padding: 0.95rem 1.25rem;
    margin: 0.5rem 0 0.9rem 0;
    font-weight: 600;
}


.side-stat {
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
}


.side-stat-label {
    font-size: 0.7rem;
    font-weight: 700;
}


.side-stat-value {
    font-size: 1.85rem;
    font-weight: 700;
}


.side-meta {
    font-size: 0.76rem;
}


.footnote {
    font-size: 0.8rem;
}


#MainMenu, footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(
    ttl=1800,
    show_spinner="Computing live forecast..."
)
def load_forecast() -> dict:
    return predictor.forecast()


@st.cache_data(
    ttl=1800,
    show_spinner="Loading recent observations..."
)
def load_history(hours: int = 336) -> pd.DataFrame:
    return predictor.recent_history(hours)


@st.cache_data(
    ttl=3600,
    show_spinner="Loading model metrics..."
)
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


@st.cache_data(
    ttl=3600,
    show_spinner="Loading feature history..."
)
def load_features_sample(rows: int = 17376) -> pd.DataFrame:
    """Feature store rows for EDA tab."""

    frame = db_client.fetch_features()

    return frame.iloc[-rows:] if not frame.empty else frame

def aqi_gauge(value: float, title: str = "Current AQI") -> go.Figure:
    """Radial gauge coloured by EPA category band."""

    category = calc.category(value)

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={
                "text": (
                    f"{title}<br>"
                    f"<span style='font-size:0.75em'>{category}</span>"
                )
            },
            number={"font": {"size": 46}},
            gauge={
                "axis": {"range": [0, 400]},
                "bar": {
                    "color": CATEGORY_COLOURS.get(
                        category,
                        "#888"
                    ),
                    "thickness": 0.75,
                },
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "rgba(0,228,0,0.25)",
                    },
                    {
                        "range": [50, 100],
                        "color": "rgba(255,255,0,0.25)",
                    },
                    {
                        "range": [100, 150],
                        "color": "rgba(255,126,0,0.25)",
                    },
                    {
                        "range": [150, 200],
                        "color": "rgba(255,0,0,0.25)",
                    },
                    {
                        "range": [200, 300],
                        "color": "rgba(143,63,151,0.25)",
                    },
                    {
                        "range": [300, 400],
                        "color": "rgba(126,0,35,0.25)",
                    },
                ],
            },
        )
    )

    figure.update_layout(
        height=280,
        margin=dict(
            t=70,
            b=10,
            l=20,
            r=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return figure



def forecast_chart(
    history: pd.DataFrame,
    forecast: dict
) -> go.Figure:
    """Observed AQI history plus model predictions."""

    figure = go.Figure()

    local_history = history.copy()

    if not local_history.empty:
        local_history.index = (
            local_history.index
            .tz_convert(config.TIMEZONE)
        )

        figure.add_trace(
            go.Scatter(
                x=local_history.index,
                y=local_history["us_aqi"],
                name="Observed",
                line=dict(
                    color="#4A9EFF",
                    width=2,
                ),
            )
        )


    observation_time = pd.Timestamp(
        forecast["observation_time_local"]
    )

    current_aqi = forecast["current"]["aqi"]

    times = [observation_time]
    values = [current_aqi]


    for entry in forecast["forecast"]:

        times.append(
            pd.Timestamp(
                entry["valid_at_local"]
            )
        )

        values.append(
            entry["aqi"]
        )


    figure.add_trace(
        go.Scatter(
            x=times,
            y=values,
            name="Forecast",
            mode="lines+markers+text",
            text=[
                ""
            ] + [
                f"{v:.0f}"
                for v in values[1:]
            ],
            textposition="top center",
            line=dict(
                color="#FF6B35",
                width=2.5,
                dash="dash",
            ),
            marker=dict(
                size=13,
                symbol="diamond",
            ),
        )
    )


    for threshold, label in [
        (100, "USG"),
        (150, "Unhealthy"),
        (200, "Very Unhealthy"),
        (300, "Hazardous"),
    ]:

        figure.add_hline(
            y=threshold,
            line_dash="dot",
            annotation_text=label,
        )


    figure.update_layout(
        height=460,
        hovermode="x unified",
        margin=dict(
            t=30,
            b=40,
            l=50,
            r=90,
        ),
        yaxis_title="AQI (US EPA scale)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return figure



def render_alerts(forecast: dict) -> None:
    """Alert banners."""

    alerts = []

    current = forecast["current"]

    if current["alert"]["severity"] > 0:
        alerts.append(
            (
                "Now",
                current["aqi"],
                current["alert"],
            )
        )


    for entry in forecast["forecast"]:

        if entry["alert"]["severity"] > 0:

            alerts.append(
                (
                    f"+{entry['horizon_hours']}h",
                    entry["aqi"],
                    entry["alert"],
                )
            )


    if not alerts:
        st.success(
            "No health alerts over the next 72 hours."
        )
        return


    peak = max(
        alerts,
        key=lambda x: x[2]["severity"]
    )


    icon, colour, label = ALERT_STYLES[
        peak[2]["severity"]
    ]


    st.markdown(
        f"""
        <div class="alert-banner"
        style="background:{colour}">
        {icon}
        <strong>{label}</strong>
        —
        AQI {peak[1]:.0f}.
        {peak[2]['message']}
        </div>
        """,
        unsafe_allow_html=True,
    )



def tab_forecast(
    forecast: dict,
    history: pd.DataFrame
) -> None:

    """Forecast tab."""

    current = forecast["current"]

    render_alerts(forecast)


    left, right = st.columns(
        [1, 2]
    )


    with left:

        st.plotly_chart(
            aqi_gauge(
                current["aqi"]
            ),
            width="stretch",
        )

        st.caption(
            forecast["observation_time_local"]
        )


    with right:

        row1 = st.columns(3)

        row1[0].metric(
            "Dominant pollutant",
            current["dominant_pollutant"],
        )

        row1[1].metric(
            "PM2.5 · µg/m³",
            f"{current['pm2_5']:.1f}",
        )

        row1[2].metric(
            "PM10 · µg/m³",
            f"{current['pm10']:.1f}",
        )


        row2 = st.columns(3)


        row2[0].metric(
            "Temperature °C",
            f"{current['temperature_2m']:.1f}",
        )

        row2[1].metric(
            "Humidity %",
            f"{current['relative_humidity_2m']:.0f}",
        )

        row2[2].metric(
            "Wind km/h",
            f"{current['wind_speed_10m']:.1f}",
        )


        st.markdown(
            "**Three-day forecast**"
        )


        cards = st.columns(3)


        for column, entry in zip(
            cards,
            forecast["forecast"]
        ):

            column.metric(
                f"+{entry['horizon_hours']}h",
                f"{entry['aqi']:.0f}",
                f"{entry['aqi'] - current['aqi']:+.0f}",
                delta_color="inverse",
            )

            column.caption(
                entry["category"]
            )


    st.markdown("---")


    st.plotly_chart(
        forecast_chart(
            history,
            forecast
        ),
        width="stretch",
    )


    # ==============================
    # OPENWEATHER CROSS CHECK
    # ==============================

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


    with st.expander(
        "How this forecast is produced"
    ):

        st.write(
            forecast["method"]
        )

        st.caption(
            "The feature store is the primary source. "
            "If unavailable, features are computed live."
        )
def tab_performance(metrics: dict | None) -> None:
    """Model performance tab."""

    if metrics is None:
        st.warning(
            "No metrics available. Run the training pipeline."
        )
        return


    st.caption(
        f"Trained {metrics['trained_at'][:19]} UTC · "
        f"{metrics['n_rows']:,} rows · "
        f"{metrics['n_features']} features"
    )


    frame = pd.DataFrame(
        metrics["results"]
    )


    st.markdown(
        """
        **Why the persistence baseline matters.**

        Current AQI is one of the model inputs, so a model can
        appear strong by simply repeating the present condition.
        The meaningful question is whether the model improves
        prediction beyond "future AQI will look like current AQI".
        """
    )


    for horizon in config.HORIZONS:

        subset = frame[
            frame["horizon"] == horizon
        ].copy()


        if subset.empty:
            continue


        st.markdown(
            f"#### +{horizon} hours"
        )


        display = subset[
            [
                "model",
                "rmse",
                "mae",
                "r2",
            ]
        ]


        st.dataframe(
            display.round(3),
            width="stretch",
            hide_index=True,
        )



def tab_explain(
    shap_summary: dict | None
) -> None:

    """SHAP explanation tab."""

    if shap_summary is None:

        st.warning(
            "No SHAP summary available."
        )

        return


    horizon = st.selectbox(
        "Forecast Horizon",
        list(shap_summary.keys()),
    )


    entry = shap_summary[
        str(horizon)
    ]


    st.markdown(
        "**Top features by mean SHAP importance**"
    )


    top = pd.Series(
        entry["top_features"]
    ).head(15)


    figure = go.Figure(
        go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
        )
    )


    figure.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
    )


    st.plotly_chart(
        figure,
        width="stretch",
    )



def tab_eda(
    features: pd.DataFrame
) -> None:

    """EDA tab."""

    if features.empty:

        st.warning(
            "Feature store is empty."
        )

        return


    frame = features.copy()


    columns = st.columns(4)


    columns[0].metric(
        "Observations",
        f"{len(frame):,}"
    )


    columns[1].metric(
        "Mean AQI",
        f"{frame['us_aqi'].mean():.0f}"
    )


    columns[2].metric(
        "Minimum AQI",
        f"{frame['us_aqi'].min():.0f}"
    )


    columns[3].metric(
        "Hours ≥ Unhealthy",
        f"{(frame['us_aqi'] >= 151).mean():.0%}"
    )


    figure = go.Figure()


    figure.add_trace(
        go.Histogram(
            x=frame["us_aqi"],
            nbinsx=50,
        )
    )


    figure.update_layout(
        height=400,
        xaxis_title="AQI",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
    )


    st.plotly_chart(
        figure,
        width="stretch",
    )



def tab_about() -> None:

    """About section."""

    st.markdown(
        f"""
## Pearls AQI Predictor — {config.CITY}

Three-day AQI forecasting system using machine learning.

### Architecture

| Layer | Implementation |
|---|---|
| Data | Open-Meteo |
| Feature Store | Supabase |
| Models | Scikit-learn / XGBoost |
| API | FastAPI |
| Dashboard | Streamlit |

### Data Validation

Open-Meteo is the primary source.

OpenWeather is used only as an independent
cross-validation source.

Both providers are converted using the same
US EPA AQI calculation.

"""
    )



def aqi_colour(
    value: float
) -> str:

    return CATEGORY_COLOURS.get(
        calc.category(value),
        "#38BDF8"
    )



def main() -> None:


    st.markdown(
        """
        <div class="hero-title">
        Lahore Air Quality Forecast
        </div>

        <div class="hero-sub">
        Three-day AQI prediction on US EPA scale
        </div>
        """,
        unsafe_allow_html=True,
    )


    try:

        forecast = load_forecast()

    except Exception as exc:

        st.error(
            f"Could not compute forecast: {exc}"
        )

        st.stop()



    current = forecast["current"]

    peak = forecast["peak_forecast_aqi"]


    now_local = datetime.now(
        pytz.timezone(
            config.TIMEZONE
        )
    )



    with st.sidebar:


        st.markdown(
            f"""
            <div class="side-stat">

            <div class="side-stat-label">
            Current AQI
            </div>

            <div class="side-stat-value"
            style="color:{aqi_colour(current['aqi'])}">
            {current['aqi']:.0f}
            </div>

            <div class="side-meta">
            {current['category']}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown(
            f"""
            <div class="side-stat">

            <div class="side-stat-label">
            Worst 72h AQI
            </div>

            <div class="side-stat-value"
            style="color:{aqi_colour(peak)}">
            {peak:.0f}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        if st.button(
            "Refresh data"
        ):

            st.cache_data.clear()

            st.rerun()



        st.caption(
            f"Updated {now_local.strftime('%H:%M')} PKT"
        )



    tabs = st.tabs(
        [
            "Forecast",
            "Model performance",
            "Explainability",
            "Data analysis",
            "About",
        ]
    )



    with tabs[0]:

        try:

            history = load_history()

        except Exception:

            history = pd.DataFrame()


        tab_forecast(
            forecast,
            history,
        )



    with tabs[1]:

        tab_performance(
            load_metrics()
        )



    with tabs[2]:

        tab_explain(
            load_shap()
        )



    with tabs[3]:

        try:

            tab_eda(
                load_features_sample()
            )

        except Exception as exc:

            st.warning(
                str(exc)
            )



    with tabs[4]:

        tab_about()



if __name__ == "__main__":
    main()