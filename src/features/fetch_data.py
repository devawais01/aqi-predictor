"""Fetch raw weather and air-quality observations from Open-Meteo.

Two separate APIs are queried and joined on timestamp:
  archive-api    -> historical weather (ERA5 reanalysis)
  air-quality    -> historical pollutants + Open-Meteo's own us_aqi

All timestamps are handled in UTC. Conversion to Asia/Karachi happens only
at presentation time.

Data licensed CC-BY 4.0 by Open-Meteo (https://open-meteo.com).
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from src import config

REQUEST_TIMEOUT = 60
MAX_RETRIES = 4


def _get(url: str, params: dict) -> dict:
    """GET with exponential backoff. Open-Meteo is free but not guaranteed."""
    delay = 2.0
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                raise RuntimeError("rate limited (429)")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                print(f"    attempt {attempt} failed ({exc}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(
        f"Open-Meteo request failed after {MAX_RETRIES} attempts: {last_error}"
    )


def _hourly_to_frame(payload: dict, variables: list[str]) -> pd.DataFrame:
    """Turn an Open-Meteo hourly block into a timestamp-indexed DataFrame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Response contained no hourly data.")

    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True)})
    for name in variables:
        frame[name] = hourly.get(name, [None] * len(frame))
    return frame.set_index("timestamp").sort_index()


def _date_chunks(start: date, end: date, size_days: int):
    """Yield (start, end) date pairs covering [start, end] inclusive."""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=size_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def fetch_weather(start: date, end: date) -> pd.DataFrame:
    """Historical weather from the ERA5 archive, chunked to keep requests small."""
    frames = []
    for chunk_start, chunk_end in _date_chunks(start, end, config.CHUNK_DAYS):
        print(f"  weather   {chunk_start} -> {chunk_end}")
        payload = _get(
            config.WEATHER_ARCHIVE_URL,
            {
                "latitude": config.LATITUDE,
                "longitude": config.LONGITUDE,
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(config.WEATHER_VARS),
                "timezone": "UTC",
            },
        )
        frames.append(_hourly_to_frame(payload, config.WEATHER_VARS))
        time.sleep(1)
    return pd.concat(frames).sort_index()


def fetch_air_quality(start: date, end: date) -> pd.DataFrame:
    """Historical pollutants and us_aqi from the air-quality API."""
    frames = []
    for chunk_start, chunk_end in _date_chunks(start, end, config.CHUNK_DAYS):
        print(f"  pollutant {chunk_start} -> {chunk_end}")
        payload = _get(
            config.AIR_QUALITY_URL,
            {
                "latitude": config.LATITUDE,
                "longitude": config.LONGITUDE,
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(config.AIR_QUALITY_VARS),
                "timezone": "UTC",
            },
        )
        frames.append(_hourly_to_frame(payload, config.AIR_QUALITY_VARS))
        time.sleep(1)
    return pd.concat(frames).sort_index()


def fetch_observations(start: date, end: date) -> pd.DataFrame:
    """Fetch and join weather + air quality for a date range."""
    print(f"Fetching {start} -> {end}")
    weather = fetch_weather(start, end)
    air = fetch_air_quality(start, end)

    merged = weather.join(air, how="inner")
    merged = merged[~merged.index.duplicated(keep="last")]
    print(
        f"  joined: {len(merged)} rows "
        f"({len(weather)} weather, {len(air)} air quality)"
    )
    return merged


def fetch_weather_forecast(hours: int = 96) -> pd.DataFrame:
    """Live weather forecast, used at inference for the perfect-prog features."""
    payload = _get(
        config.WEATHER_FORECAST_URL,
        {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(config.WEATHER_VARS),
            "forecast_days": max(1, (hours // 24) + 1),
            "timezone": "UTC",
        },
    )
    return _hourly_to_frame(payload, config.WEATHER_VARS)


def fetch_recent(days_back: int = 7) -> pd.DataFrame:
    """Most recent observations, for the hourly incremental pipeline.

    Uses past_days on the live endpoints rather than the archive, because the
    archive trails real time by several days.
    """
    weather_payload = _get(
        config.WEATHER_FORECAST_URL,
        {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(config.WEATHER_VARS),
            "past_days": days_back,
            "forecast_days": 1,
            "timezone": "UTC",
        },
    )
    air_payload = _get(
        config.AIR_QUALITY_URL,
        {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(config.AIR_QUALITY_VARS),
            "past_days": days_back,
            "forecast_days": 1,
            "timezone": "UTC",
        },
    )

    weather = _hourly_to_frame(weather_payload, config.WEATHER_VARS)
    air = _hourly_to_frame(air_payload, config.AIR_QUALITY_VARS)

    merged = weather.join(air, how="inner")
    merged = merged[~merged.index.duplicated(keep="last")]

    now = pd.Timestamp.now(tz="UTC").floor("h")
    return merged[merged.index <= now]