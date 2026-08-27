"""Transform raw observations into the model-ready feature set.

Design rules, all of which exist to prevent a specific failure:

  * TARGETS are us_aqi shifted -24, -48, -72. Three independent horizons,
    three model sets. No recursive forecasting, no fabricated future inputs.

  * CURRENT POLLUTANT STATE STAYS. us_aqi[t], pm2_5[t] and their lags are
    known, observed values at prediction time t. Using them to predict
    t+24 is forecasting, not leakage.

  * ROLLING WINDOWS ARE SHIFTED. Every rolling/diff feature gets .shift(1)
    so the window ending at t excludes t itself.

  * FUTURE WEATHER via perfect prognosis. Training uses actual archived
    weather at t+h. Inference substitutes the live forecast.

  * TIME FEATURES USE LOCAL TIME. Diurnal and seasonal cycles are local
    phenomena, so hour and month come from Asia/Karachi, not UTC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import aqi_calculator as calc

AQI_LAGS = [1, 3, 6, 12, 24, 48, 72, 168]
PM25_LAGS = [24, 48, 168]
PM10_LAGS = [24, 168]
OZONE_LAGS = [24]

AQI_ROLL_WINDOWS = [3, 6, 12, 24, 72, 168]
AQI_STD_WINDOW = 24

FUTURE_WEATHER_VARS = [
    "wind_speed_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "boundary_layer_height",
]

WARMUP_HOURS = max(AQI_LAGS + PM25_LAGS + AQI_ROLL_WINDOWS)


def add_aqi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach our own EPA computation: sub-indices, computed_aqi, dominant."""
    computed = calc.compute(df)
    return df.join(computed)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encodings of local time."""
    local = df.index.tz_convert(config.TIMEZONE)

    hour = local.hour.values
    month = local.month.values
    dow = local.dayofweek.values

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historical values at fixed offsets. All strictly backward-looking."""
    for lag in AQI_LAGS:
        df[f"aqi_lag_{lag}"] = df["us_aqi"].shift(lag)
    for lag in PM25_LAGS:
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag)
    for lag in PM10_LAGS:
        df[f"pm10_lag_{lag}"] = df["pm10"].shift(lag)
    for lag in OZONE_LAGS:
        df[f"ozone_lag_{lag}"] = df["ozone"].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling statistics, every one shifted so the window excludes t."""
    for window in AQI_ROLL_WINDOWS:
        df[f"aqi_roll_{window}"] = (
            df["us_aqi"].rolling(window=window).mean().shift(1)
        )
    df[f"aqi_std_{AQI_STD_WINDOW}"] = (
        df["us_aqi"].rolling(window=AQI_STD_WINDOW).std().shift(1)
    )
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of change and acceleration, both shifted."""
    df["aqi_change_rate"] = df["us_aqi"].diff().shift(1)
    df["aqi_accel"] = df["aqi_change_rate"].diff()
    df["wind_u"] = df["wind_speed_10m"] * np.cos(
        np.deg2rad(df["wind_direction_10m"])
    )
    df["wind_v"] = df["wind_speed_10m"] * np.sin(
        np.deg2rad(df["wind_direction_10m"])
    )
    return df


def add_future_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Perfect-prognosis features: actual weather at t+h from the archive."""
    for horizon in config.HORIZONS:
        for var in FUTURE_WEATHER_VARS:
            df[f"{var}_t{horizon}"] = df[var].shift(-horizon)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """The three forecast targets."""
    for horizon in config.HORIZONS:
        df[config.TARGET_COLUMNS[horizon]] = df["us_aqi"].shift(-horizon)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything except targets and non-numeric fields."""
    excluded = set(config.TARGET_COLUMNS.values()) | {"dominant_pollutant"}
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def audit_leakage(df: pd.DataFrame) -> None:
    """Assert that no feature can see beyond time t.

    MUST be called on the untrimmed frame. Recomputing a rolling window on a
    truncated frame gives different values at the head.
    """
    print("\n--- Leakage audit ---")
    failures = []

    for window in AQI_ROLL_WINDOWS:
        name = f"aqi_roll_{window}"
        expected = df["us_aqi"].rolling(window=window).mean().shift(1)
        if not df[name].equals(expected):
            failures.append(f"{name} is not .shift(1)ed")

    expected_std = df["us_aqi"].rolling(window=AQI_STD_WINDOW).std().shift(1)
    if not df[f"aqi_std_{AQI_STD_WINDOW}"].equals(expected_std):
        failures.append(f"aqi_std_{AQI_STD_WINDOW} is not .shift(1)ed")

    expected_change = df["us_aqi"].diff().shift(1)
    if not df["aqi_change_rate"].equals(expected_change):
        failures.append("aqi_change_rate is not .shift(1)ed")

    allowed_future = set(config.TARGET_COLUMNS.values())
    for horizon in config.HORIZONS:
        for var in FUTURE_WEATHER_VARS:
            allowed_future.add(f"{var}_t{horizon}")

    for column in df.columns:
        if column in allowed_future:
            continue
        for horizon in config.HORIZONS:
            if column.endswith(f"_t{horizon}"):
                failures.append(f"{column} references the future unexpectedly")

    for lag in AQI_LAGS:
        name = f"aqi_lag_{lag}"
        if not df[name].equals(df["us_aqi"].shift(lag)):
            failures.append(f"{name} is misaligned")

    if failures:
        for message in failures:
            print(f"  FAIL: {message}")
        raise AssertionError("Leakage audit failed.")

    print(f"  {len(AQI_ROLL_WINDOWS)} rolling means shifted: OK")
    print("  rolling std shifted: OK")
    print("  aqi_change_rate shifted: OK")
    print(f"  {len(AQI_LAGS)} AQI lags aligned: OK")
    print("  future columns limited to perfect-prog weather: OK")
    print("  Audit passed.")
    

def build(df: pd.DataFrame, trim_warmup: bool = True) -> pd.DataFrame:
    """Full pipeline: raw observations -> features + targets."""
    work = df.copy().sort_index()
    work = work[~work.index.duplicated(keep="last")]

    print(f"Building features from {len(work)} raw rows...")

    work = add_aqi_features(work)
    work = add_time_features(work)
    work = add_lag_features(work)
    work = add_rolling_features(work)
    work = add_derived_features(work)
    work = add_future_weather(work)
    work = add_targets(work)

    audit_leakage(work)

    if trim_warmup:
        before = len(work)
        work = work.iloc[WARMUP_HOURS:]
        print(f"\n  dropped {before - len(work)} warm-up rows "
              f"({WARMUP_HOURS}h of lag history)")

    features = feature_columns(work)
    print(f"  {len(features)} feature columns, "
          f"{len(config.HORIZONS)} targets, {len(work)} rows")
    return work


def summarise(df: pd.DataFrame) -> None:
    """Print a compact description of the built feature set."""
    features = feature_columns(df)

    print("\n--- Feature groups ---")
    groups = {
        "weather (current)": [c for c in features if c in config.WEATHER_VARS],
        "weather (future)": [c for c in features
                             if any(c.startswith(v + "_t")
                                    for v in FUTURE_WEATHER_VARS)],
        "pollutants": [c for c in features if c in config.POLLUTANT_VARS],
        "aqi + sub-indices": [c for c in features
                              if c.endswith("_aqi") or c == "us_aqi"],
        "lags": [c for c in features if "_lag_" in c],
        "rolling": [c for c in features
                    if c.startswith("aqi_roll") or c.startswith("aqi_std")],
        "time": [c for c in features
                 if c.endswith(("_sin", "_cos")) or c == "is_weekend"],
        "derived": [c for c in features
                    if c in ("aqi_change_rate", "aqi_accel", "wind_u", "wind_v")],
    }
    for name, columns in groups.items():
        print(f"  {name:<20} {len(columns):>3}")

    print("\n--- Target availability ---")
    for horizon, target in config.TARGET_COLUMNS.items():
        valid = int(df[target].notna().sum())
        print(f"  {target:<10} {valid:>6} non-null "
              f"({df[target].isna().sum()} null at series tail)")

    print("\n--- Null fraction, worst 10 features ---")
    nulls = df[features].isna().mean().sort_values(ascending=False)
    for name, fraction in nulls.head(10).items():
        print(f"  {name:<30} {fraction:6.2%}")