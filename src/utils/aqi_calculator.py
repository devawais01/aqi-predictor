"""US EPA Air Quality Index computation from pollutant concentrations.

Two things are easy to get wrong here and both silently corrupt the result:

  1. UNITS. Open-Meteo returns every pollutant in ug/m3. EPA breakpoints
     expect ppm for CO and O3, and ppb for SO2 and NO2. Only PM2.5 and PM10
     are already in the right unit.

  2. AVERAGING WINDOWS. EPA breakpoints are defined against specific
     averaging periods, not instantaneous readings. PM2.5 and PM10 use a
     24-hour mean, CO and O3 use an 8-hour mean, SO2 and NO2 use the 1-hour
     value. Feeding hourly PM2.5 into 24-hour breakpoints inflates the index
     as badly as a unit error does.

Final AQI is the maximum of the sub-indices; the dominant pollutant is the
argmax. Reference: EPA 454/B-24-002 (PM2.5 breakpoints revised May 2024).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Molar volume of an ideal gas at 25 C and 1 atm, in litres per mole.
MOLAR_VOLUME = 24.45

MOLECULAR_WEIGHT = {
    "carbon_monoxide": 28.01,
    "nitrogen_dioxide": 46.01,
    "sulphur_dioxide": 64.06,
    "ozone": 48.00,
}

# (concentration_low, concentration_high, index_low, index_high)
BREAKPOINTS = {
    "pm2_5": [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    "ozone": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "carbon_monoxide": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],
    "sulphur_dioxide": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ],
    "nitrogen_dioxide": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
}

# Hours of rolling mean each pollutant's breakpoints are defined against.
AVERAGING_HOURS = {
    "pm2_5": 24,
    "pm10": 24,
    "ozone": 8,
    "carbon_monoxide": 8,
    "sulphur_dioxide": 1,
    "nitrogen_dioxide": 1,
}

# EPA truncation: decimal places the concentration is truncated to before lookup.
TRUNCATION = {
    "pm2_5": 1,
    "pm10": 0,
    "ozone": 3,
    "carbon_monoxide": 1,
    "sulphur_dioxide": 0,
    "nitrogen_dioxide": 0,
}

# Human-readable names for the dashboard and report.
DISPLAY_NAME = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "ozone": "O3",
    "carbon_monoxide": "CO",
    "sulphur_dioxide": "SO2",
    "nitrogen_dioxide": "NO2",
}


def ugm3_to_ppb(values: pd.Series, pollutant: str) -> pd.Series:
    """Convert ug/m3 to parts per billion at 25 C, 1 atm."""
    weight = MOLECULAR_WEIGHT[pollutant]
    return values * MOLAR_VOLUME / weight


def ugm3_to_ppm(values: pd.Series, pollutant: str) -> pd.Series:
    """Convert ug/m3 to parts per million at 25 C, 1 atm."""
    return ugm3_to_ppb(values, pollutant) / 1000.0


def _truncate(values: pd.Series, decimals: int) -> pd.Series:
    """EPA truncation: drop digits rather than round them."""
    factor = 10 ** decimals
    return np.floor(values * factor) / factor


def to_epa_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every pollutant column from ug/m3 into its EPA unit."""
    out = pd.DataFrame(index=df.index)
    out["pm2_5"] = df["pm2_5"]                                   # already ug/m3
    out["pm10"] = df["pm10"]                                     # already ug/m3
    out["ozone"] = ugm3_to_ppm(df["ozone"], "ozone")             # ppm
    out["carbon_monoxide"] = ugm3_to_ppm(
        df["carbon_monoxide"], "carbon_monoxide"
    )                                                            # ppm
    out["sulphur_dioxide"] = ugm3_to_ppb(
        df["sulphur_dioxide"], "sulphur_dioxide"
    )                                                            # ppb
    out["nitrogen_dioxide"] = ugm3_to_ppb(
        df["nitrogen_dioxide"], "nitrogen_dioxide"
    )                                                            # ppb
    return out


def apply_averaging(df: pd.DataFrame) -> pd.DataFrame:
    """Apply each pollutant's required rolling mean.

    Uses min_periods=1 so early rows produce a value rather than NaN. The
    first 23 rows of a series are therefore based on a partial window; the
    feature builder drops the warm-up period anyway.
    """
    out = pd.DataFrame(index=df.index)
    for pollutant, hours in AVERAGING_HOURS.items():
        if pollutant not in df.columns:
            continue
        if hours == 1:
            out[pollutant] = df[pollutant]
        else:
            out[pollutant] = (
                df[pollutant].rolling(window=hours, min_periods=1).mean()
            )
    return out


def sub_index(values: pd.Series, pollutant: str) -> pd.Series:
    """Piecewise-linear EPA sub-index for one pollutant."""
    truncated = _truncate(values, TRUNCATION[pollutant])
    result = pd.Series(np.nan, index=values.index, dtype=float)

    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        mask = (truncated >= c_low) & (truncated <= c_high) & result.isna()
        if mask.any():
            span = c_high - c_low
            slope = (i_high - i_low) / span if span else 0.0
            result[mask] = slope * (truncated[mask] - c_low) + i_low

    # Anything above the top breakpoint is pinned to the scale maximum.
    top_concentration = BREAKPOINTS[pollutant][-1][1]
    top_index = BREAKPOINTS[pollutant][-1][3]
    result[(truncated > top_concentration) & result.isna()] = top_index

    return result.round(0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all sub-indices, the overall AQI, and the dominant pollutant.

    Expects raw ug/m3 columns. Returns a frame with one <pollutant>_aqi
    column each, plus computed_aqi and dominant_pollutant.
    """
    converted = to_epa_units(df)
    averaged = apply_averaging(converted)

    sub_indices = pd.DataFrame(index=df.index)
    for pollutant in BREAKPOINTS:
        if pollutant in averaged.columns:
            sub_indices[f"{pollutant}_aqi"] = sub_index(
                averaged[pollutant], pollutant
            )

    out = sub_indices.copy()
    out["computed_aqi"] = sub_indices.max(axis=1)

    winner = sub_indices.idxmax(axis=1)
    out["dominant_pollutant"] = winner.map(
        lambda name: DISPLAY_NAME.get(str(name).replace("_aqi", ""), None)
        if pd.notna(name)
        else None
    )
    return out


def category(aqi_value: float) -> str:
    """EPA category label for a single AQI value."""
    if pd.isna(aqi_value):
        return "Unknown"
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Moderate"
    if aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi_value <= 200:
        return "Unhealthy"
    if aqi_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def category_series(values: pd.Series) -> pd.Series:
    """EPA category labels for a Series of AQI values."""
    return values.apply(category)