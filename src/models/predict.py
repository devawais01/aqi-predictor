"""Live inference: produce a real 3-day AQI forecast.

This is where perfect prognosis is applied. Training used *actual* archived
weather at t+24/48/72. At inference those same columns are filled from the
live Open-Meteo forecast, because the actuals do not exist yet.

Every returned number is the direct output of a model trained for that
specific horizon. Nothing is interpolated, tiled, noise-injected or
bias-anchored.

    python -m src.models.predict
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import build_features as bf
from src.features import fetch_data
from src.utils import aqi_calculator as calc
from src.utils import db_client

MODEL_DIR = "models"
_cache: dict = {}


def load_selected(horizon: int, prefer_registry: bool = False):
    """Load the model chosen for one horizon, plus its metadata.

    Looks on local disk first; falls back to the Supabase registry so the
    dashboard works on a fresh Streamlit Cloud container with no models
    baked into the image.
    """
    key = f"model_{horizon}"
    if key in _cache:
        return _cache[key]

    os.makedirs(MODEL_DIR, exist_ok=True)
    meta_path = f"{MODEL_DIR}/meta_t{horizon}.json"

    if prefer_registry or not os.path.exists(meta_path):
        db_client.download_artifact(f"meta_t{horizon}.json", meta_path)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No metadata for +{horizon}h. Run the training pipeline."
        )

    with open(meta_path) as handle:
        meta = json.load(handle)

    if meta["model"] == "LSTM":
        model_path = f"{MODEL_DIR}/best_t{horizon}.keras"
        if not os.path.exists(model_path):
            db_client.download_artifact(f"best_t{horizon}.keras", model_path)
            db_client.download_artifact(f"xscaler_t{horizon}.pkl",
                                        f"{MODEL_DIR}/xscaler_t{horizon}.pkl")
            db_client.download_artifact(f"yscaler_t{horizon}.pkl",
                                        f"{MODEL_DIR}/yscaler_t{horizon}.pkl")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        extras = (
            joblib.load(f"{MODEL_DIR}/xscaler_t{horizon}.pkl"),
            joblib.load(f"{MODEL_DIR}/yscaler_t{horizon}.pkl"),
        )
    else:
        model_path = f"{MODEL_DIR}/best_t{horizon}.pkl"
        if not os.path.exists(model_path):
            db_client.download_artifact(f"best_t{horizon}.pkl", model_path)
        model = joblib.load(model_path)
        extras = None

    _cache[key] = (model, meta, extras)
    return _cache[key]


def build_inference_row(lookback_days: int = 14,
                        max_staleness_hours: int = 3,
                        force_live: bool = False) -> tuple:
    """Get the latest engineered feature row.

    Primary path: read from the Supabase feature store, which is what the
    hourly pipeline writes. This is the correct architecture — the dashboard
    consumes the same features the models trained on, so there is exactly one
    feature definition in the system.

    Fallback: if the store is missing or stale (the hourly workflow has not
    run recently), recompute from the live API so the dashboard still works.
    The fallback is reported in the response so staleness is visible.

    Either way the future-weather columns are overwritten with the live
    Open-Meteo forecast, because at inference the archived actuals for
    t+24/48/72 do not exist yet.

    Returns (feature_row, history, weather_forecast, source).
    """
    source = "feature_store"
    frame = None

    if not force_live:
        try:
            print("Reading latest features from Supabase feature store...")
            stored = db_client.fetch_features()
            if not stored.empty:
                age = (pd.Timestamp.now(tz="UTC") - stored.index.max())
                age_hours = age.total_seconds() / 3600
                print(f"  {len(stored)} rows, latest {stored.index.max()} "
                      f"({age_hours:.1f}h old)")
                if age_hours <= max_staleness_hours:
                    frame = stored.iloc[-(24 * lookback_days):].copy()
                else:
                    print(f"  stale (> {max_staleness_hours}h); "
                          "falling back to live computation")
                    source = "live_fallback_stale"
            else:
                print("  feature store empty; falling back to live")
                source = "live_fallback_empty"
        except Exception as exc:
            print(f"  feature store unavailable ({exc}); falling back to live")
            source = "live_fallback_error"

    if frame is None:
        print("Fetching recent observations from Open-Meteo...")
        recent = fetch_data.fetch_recent(days_back=lookback_days)
        print(f"  {len(recent)} rows, latest {recent.index.max()}")
        frame = bf.add_aqi_features(recent.copy())
        frame = bf.add_time_features(frame)
        frame = bf.add_lag_features(frame)
        frame = bf.add_rolling_features(frame)
        frame = bf.add_derived_features(frame)
        if source == "feature_store":
            source = "live_fallback"

    print("Fetching live weather forecast...")
    forecast_weather = fetch_data.fetch_weather_forecast(hours=96)
    print(f"  {len(forecast_weather)} forecast hours, "
          f"to {forecast_weather.index.max()}")

    # dominant_pollutant is a string, so feature_columns() excludes it and the
    # feature store never receives it. Recompute it from the EPA sub-index
    # columns, which are numeric and therefore are stored.
    if "dominant_pollutant" not in frame.columns:
        sub_index_columns = [
            f"{pollutant}_aqi" for pollutant in calc.BREAKPOINTS
            if f"{pollutant}_aqi" in frame.columns
        ]
        if sub_index_columns:
            winner = frame[sub_index_columns].idxmax(axis=1)
            frame["dominant_pollutant"] = winner.map(
                lambda name: calc.DISPLAY_NAME.get(
                    str(name).replace("_aqi", ""), None
                ) if pd.notna(name) else None
            )
        else:
            frame["dominant_pollutant"] = None

    # Perfect prognosis: the stored rows carry archived actuals (or nulls at
    # the tail). Overwrite with the live forecast for the latest row.
    now = frame.index.max()
    for horizon in config.HORIZONS:
        target_time = now + pd.Timedelta(hours=horizon)
        for var in bf.FUTURE_WEATHER_VARS:
            column = f"{var}_t{horizon}"
            if target_time in forecast_weather.index:
                frame.loc[now, column] = forecast_weather.loc[target_time, var]
            else:
                nearest = forecast_weather.index[
                    np.abs(forecast_weather.index - target_time).argmin()
                ]
                frame.loc[now, column] = forecast_weather.loc[nearest, var]

    return frame.loc[[now]], frame, forecast_weather, source

def predict_horizon(row: pd.DataFrame, horizon: int,
                    history: pd.DataFrame | None = None) -> dict:
    """Run the selected model for one horizon.

    The feature list and its ORDER come from the training metadata, not from
    whatever order the inference frame happens to have. Scikit-learn matches
    on both name and position, so recomputing the list at inference time is
    a source of silent (or, here, loud) mismatch.
    """
    model, meta, extras = load_selected(horizon)
    features = meta["features"]

    missing = [c for c in features if c not in row.columns]
    if missing:
        raise ValueError(
            f"+{horizon}h missing {len(missing)} feature(s): {missing[:5]}"
        )

    X = row[features].astype(float)
    null_columns = X.columns[X.isna().any()].tolist()
    if null_columns:
        raise ValueError(f"+{horizon}h has null features: {null_columns[:5]}")

    if meta["model"] == "LSTM":
        if history is None:
            raise ValueError("LSTM inference requires observation history.")
        x_scaler, y_scaler = extras
        window = history[features].astype(float).iloc[-24:]
        if len(window) < 24:
            raise ValueError("LSTM needs 24 hours of history.")
        scaled = x_scaler.transform(window).astype(np.float32)
        sequence = scaled.reshape(1, 24, len(features))
        scaled_prediction = model.predict(sequence, verbose=0)
        value = float(y_scaler.inverse_transform(scaled_prediction).ravel()[0])
    else:
        value = float(model.predict(X)[0])

    value = float(np.clip(value, 0, 500))
    valid_at = row.index[0] + pd.Timedelta(hours=horizon)

    return {
        "horizon_hours": horizon,
        "valid_at_utc": valid_at.isoformat(),
        "valid_at_local": valid_at.tz_convert(config.TIMEZONE).isoformat(),
        "aqi": round(value, 1),
        "category": calc.category(value),
        "model": meta["model"],
    }

def alert_level(aqi_value: float) -> dict:
    """Map an AQI value onto an alert tier with health guidance."""
    thresholds = config.ALERT_THRESHOLDS
    if aqi_value >= thresholds["hazardous"]:
        return {
            "level": "hazardous",
            "severity": 4,
            "message": "Health emergency. Everyone should remain indoors.",
        }
    if aqi_value >= thresholds["very_unhealthy"]:
        return {
            "level": "very_unhealthy",
            "severity": 3,
            "message": "Avoid all outdoor exertion. Use air purification indoors.",
        }
    if aqi_value >= thresholds["unhealthy"]:
        return {
            "level": "unhealthy",
            "severity": 2,
            "message": "Everyone may experience effects. Limit outdoor activity.",
        }
    if aqi_value >= thresholds["sensitive"]:
        return {
            "level": "sensitive",
            "severity": 1,
            "message": ("Sensitive groups (children, elderly, respiratory or "
                        "cardiac conditions) should reduce outdoor exertion."),
        }
    return {"level": "none", "severity": 0, "message": "Air quality acceptable."}


def forecast(lookback_days: int = 14) -> dict:
    """Full live forecast: current conditions plus +24h, +48h, +72h."""
    row, history, weather, source = build_inference_row(lookback_days)

    now = row.index[0]
    current_aqi = float(row["us_aqi"].iloc[0])
    dominant = row["dominant_pollutant"].iloc[0]

    print(f"\nCurrent: AQI {current_aqi:.0f} ({calc.category(current_aqi)}), "
          f"dominant {dominant}")

    predictions = []
    for horizon in config.HORIZONS:
        try:
            result = predict_horizon(row, horizon, history)
            result["alert"] = alert_level(result["aqi"])
            predictions.append(result)
            print(f"  +{horizon:>2}h  AQI {result['aqi']:6.1f}  "
                  f"{result['category']:<32} [{result['model']}]")
        except Exception as exc:
            print(f"  +{horizon}h failed: {exc}")

    peak = max([p["aqi"] for p in predictions], default=current_aqi)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": config.CITY,
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
            "feature_source": source,
        "observation_time_utc": now.isoformat(),
        "observation_time_local": now.tz_convert(config.TIMEZONE).isoformat(),
        "current": {
            "aqi": round(current_aqi, 1),
            "category": calc.category(current_aqi),
            "dominant_pollutant": dominant,
            "computed_aqi": round(float(row["computed_aqi"].iloc[0]), 1),
            "pm2_5": round(float(row["pm2_5"].iloc[0]), 1),
            "pm10": round(float(row["pm10"].iloc[0]), 1),
            "temperature_2m": round(float(row["temperature_2m"].iloc[0]), 1),
            "wind_speed_10m": round(float(row["wind_speed_10m"].iloc[0]), 1),
            "relative_humidity_2m": round(
                float(row["relative_humidity_2m"].iloc[0]), 1
            ),
            "alert": alert_level(current_aqi),
        },
        "forecast": predictions,
        "peak_forecast_aqi": round(peak, 1),
        "peak_alert": alert_level(peak),
        "method": (
            "Direct multi-horizon forecasting. Each value is the output of a "
            "model trained specifically for that horizon. Future weather "
            "features use the live Open-Meteo forecast (perfect prognosis); "
            "training used archived actuals, so live error will be modestly "
            "higher than reported test error."
        ),
    }


def recent_history(hours: int = 168) -> pd.DataFrame:
    """Recent observed AQI, for plotting context behind the forecast."""
    days = max(2, hours // 24 + 1)
    observations = fetch_data.fetch_recent(days_back=days)
    return observations[["us_aqi", "pm2_5", "pm10"]].iloc[-hours:]


def main() -> int:
    print("=" * 62)
    print(f"LIVE FORECAST  {config.CITY}")
    print("=" * 62)

    result = forecast()

    print("\n" + "-" * 62)
    print(f"Peak over next 72h: AQI {result['peak_forecast_aqi']:.0f} "
          f"({result['peak_alert']['level']})")
    print(f"  {result['peak_alert']['message']}")
    print("-" * 62)

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/latest_forecast.json", "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    print("\nWrote data/processed/latest_forecast.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())