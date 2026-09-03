"""OpenWeather client used as an independent second data source.

Open-Meteo supplies all historical and forecast data for the models. This
module queries OpenWeather for the same location and hour so that current
conditions can be cross-checked against a provider with a different sensor
network and a different assimilation pipeline.

OpenWeather's own `main.aqi` field is a 1-5 band index, not the US EPA AQI,
so it is deliberately ignored. Instead the raw pollutant concentrations are
passed through our own EPA implementation (src/utils/aqi_calculator.py).

Both sources are therefore scored by the same algorithm and any divergence
is attributable to the measurements rather than to the index definition.

Every function degrades gracefully: if no API key is configured, or the
request fails, None is returned and the dashboard simply omits the panel.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from src import config
from src.utils import aqi_calculator as calc


AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
TIMEOUT = 15


# OpenWeather field name -> AQI calculator expected name
POLLUTANT_MAP = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "co": "carbon_monoxide",
    "no2": "nitrogen_dioxide",
    "so2": "sulphur_dioxide",
    "o3": "ozone",
}


def is_configured() -> bool:
    """True when an OpenWeather API key is available."""
    return bool(config.OPENWEATHER_API_KEY)


def _get(url: str, params: dict) -> Optional[dict]:
    """GET request returning None on failure."""
    try:
        response = requests.get(
            url,
            params=params,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except Exception:
        return None


def current_conditions() -> Optional[dict]:
    """Get current OpenWeather pollutants/weather and calculate EPA AQI."""

    if not is_configured():
        return None

    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "appid": config.OPENWEATHER_API_KEY,
    }

    # Get pollutant data
    air = _get(AIR_URL, params)

    if not air or not air.get("list"):
        return None

    entry = air["list"][0]

    components = entry.get("components", {})

    observed_at = pd.to_datetime(
        entry.get("dt", 0),
        unit="s",
        utc=True,
    )

    # Convert OpenWeather names to AQI calculator names
    row = {
        our_name: float(
            components.get(ow_name, 0.0)
        )
        for ow_name, our_name in POLLUTANT_MAP.items()
    }

    row["dust"] = 0.0

    frame = pd.DataFrame(
        [row],
        index=[observed_at],
    )

    try:
        computed = calc.compute(frame)

        aqi_value = float(
            computed["computed_aqi"].iloc[0]
        )

        dominant = computed["dominant_pollutant"].iloc[0]

    except Exception:
        return None


    # Get weather data
    weather = _get(
        WEATHER_URL,
        {
            **params,
            "units": "metric",
        },
    ) or {}


    main = weather.get("main", {})
    wind = weather.get("wind", {})


    return {
        "provider": "OpenWeather",

        "observed_at_utc": observed_at.isoformat(),

        "observed_at_local": observed_at.tz_convert(
            config.TIMEZONE
        ).isoformat(),

        "computed_aqi": round(
            aqi_value,
            1,
        ),

        "category": calc.category(aqi_value),

        "dominant_pollutant": dominant,

        "pm2_5": round(
            row["pm2_5"],
            1,
        ),

        "pm10": round(
            row["pm10"],
            1,
        ),

        "temperature_2m": (
            round(float(main["temp"]), 1)
            if "temp" in main
            else None
        ),

        "relative_humidity_2m": (
            round(float(main["humidity"]), 1)
            if "humidity" in main
            else None
        ),

        "wind_speed_10m": (
            round(float(wind["speed"]) * 3.6, 1)
            if "speed" in wind
            else None
        ),

        "note": (
            "Pollutant concentrations from OpenWeather are "
            "scored using the same EPA AQI breakpoint implementation "
            "used for Open-Meteo. OpenWeather's own 1-5 AQI index "
            "is ignored."
        ),
    }


def compare(primary: dict) -> Optional[dict]:
    """Compare Open-Meteo current data with OpenWeather."""

    secondary = current_conditions()

    if secondary is None:
        return None


    delta_aqi = (
        secondary["computed_aqi"]
        - float(primary["aqi"])
    )

    delta_pm25 = (
        secondary["pm2_5"]
        - float(primary["pm2_5"])
    )


    # Agreement is judged on concentration, not on AQI. The two AQI values
    # rest on different averaging windows -- Open-Meteo applies EPA's
    # 24-hour rolling mean for PM2.5, OpenWeather returns an instantaneous
    # reading -- so comparing them directly overstates any disagreement.
    reference = max(float(primary["pm2_5"]), 1e-6)
    pm_pct = abs(delta_pm25) / reference * 100

    if pm_pct <= 15:
        verdict = "close agreement"
    elif pm_pct <= 35:
        verdict = "moderate divergence"
    else:
        verdict = "substantial divergence"


    return {
        "secondary": secondary,

        "delta_aqi": round(
            delta_aqi,
            1,
        ),

        "delta_pm2_5": round(
            delta_pm25,
            1,
        ),

        "agreement": verdict,
        "pm2_5_pct_difference": round(pm_pct, 1),
        "basis": (
            "Agreement judged on PM2.5 concentration. The AQI values differ "
            "in averaging window: Open-Meteo applies EPA's 24-hour rolling "
            "mean, OpenWeather reports an instantaneous value."
        ),
    }