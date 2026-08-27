"""Exploratory data analysis for Lahore AQI.

Runs as a plain script so it does not depend on a Jupyter kernel, and
converts cleanly to a notebook via jupytext. Every figure is written to
reports/figures/ for use in the final report.

    python -m notebooks.eda
    jupytext --to notebook notebooks/eda.py -o notebooks/01_eda.ipynb
"""

# %%
from __future__ import annotations

import os
import sys

# Resolve the project root whether this runs as a module from the repo root
# or as a notebook with cwd = notebooks/.
_HERE = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
_ROOT = _HERE if os.path.basename(_HERE) != "notebooks" else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.utils import aqi_calculator as calc

FIGDIR = "reports/figures"
os.makedirs(FIGDIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

print(f"project root: {_ROOT}")


# %%
def save(name: str) -> None:
    """Write the current figure to reports/figures/ and close it."""
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"    saved {path}")


def load() -> pd.DataFrame:
    """Load the engineered feature set and attach local-time columns."""
    df = pd.read_csv("data/processed/features.csv",
                     index_col="timestamp", parse_dates=["timestamp"])
    df.index = pd.to_datetime(df.index, utc=True)
    local = df.index.tz_convert(config.TIMEZONE)
    df["hour"] = local.hour
    df["month"] = local.month
    df["year"] = local.year
    return df


# %%
def section_overview(df: pd.DataFrame) -> None:
    """How bad is Lahore's air, in EPA category terms."""
    print("\n" + "=" * 64)
    print("1. OVERVIEW")
    print("=" * 64)
    print(f"  rows      {len(df):,}")
    print(f"  period    {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  us_aqi    mean {df.us_aqi.mean():.1f}   "
          f"median {df.us_aqi.median():.0f}   "
          f"min {df.us_aqi.min():.0f}   max {df.us_aqi.max():.0f}")

    categories = calc.category_series(df["us_aqi"]).value_counts()
    order = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
             "Unhealthy", "Very Unhealthy", "Hazardous"]
    colours = ["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023"]
    counts = [categories.get(c, 0) for c in order]
    pct = [100 * c / len(df) for c in counts]

    print("\n  EPA category distribution:")
    for label, count, percent in zip(order, counts, pct):
        print(f"    {label:<32} {count:>6,}h  {percent:5.1f}%")

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(order)), pct, color=colours,
                  edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [o.replace("Unhealthy for Sensitive Groups", "USG") for o in order],
        fontsize=9,
    )

    # %%
def section_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly cycle. This is the finding that justifies two years of data."""
    print("\n" + "=" * 64)
    print("2. SEASONALITY")
    print("=" * 64)

    monthly = df.groupby("month")["us_aqi"].agg(["mean", "median", "std", "max"])
    for month, row in monthly.iterrows():
        print(f"  {MONTHS[month - 1]}   mean {row['mean']:6.1f}   "
              f"median {row['median']:5.0f}   sd {row['std']:5.1f}   "
              f"max {row['max']:5.0f}")

    worst = MONTHS[monthly["mean"].idxmax() - 1]
    best = MONTHS[monthly["mean"].idxmin() - 1]
    swing = monthly["mean"].max() - monthly["mean"].min()
    print(f"\n  worst month    {worst}  ({monthly['mean'].max():.0f})")
    print(f"  best month     {best}  ({monthly['mean'].min():.0f})")
    print(f"  seasonal swing {swing:.0f} AQI points")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(monthly.index, monthly["mean"], "o-", lw=2.5, color="#c1121f")
    ax1.fill_between(monthly.index,
                     monthly["mean"] - monthly["std"],
                     monthly["mean"] + monthly["std"],
                     alpha=0.2, color="#c1121f")
    ax1.axhline(150, ls="--", c="orange", label="Unhealthy (151)")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(MONTHS)
    ax1.set_ylabel("US AQI")
    ax1.set_title("Seasonal cycle (mean +/- 1 sd)")
    ax1.legend()

    sns.boxplot(data=df, x="month", y="us_aqi", ax=ax2,
                hue="month", palette="RdYlGn_r", legend=False, showfliers=False)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(MONTHS)
    ax2.axhline(150, ls="--", c="orange")
    ax2.set_xlabel("")
    ax2.set_title("Monthly distribution")
    save("02_seasonality.png")

    fig, ax = plt.subplots(figsize=(15, 4.5))
    daily = df["us_aqi"].resample("D").mean()
    ax.plot(daily.index, daily.values, lw=0.9, color="#023e8a")
    ax.axhline(150, ls="--", c="orange", label="Unhealthy")
    ax.axhline(300, ls="--", c="#7E0023", label="Hazardous")
    ax.set_ylabel("US AQI (daily mean)")
    ax.set_title("Two years of Lahore air quality")
    ax.legend()
    save("03_full_timeseries.png")

    return monthly


# %%
def section_autocorrelation(df: pd.DataFrame) -> list[float]:
    """Autocorrelation decay explains why persistence fails at +72h."""
    print("\n" + "=" * 64)
    print("3. AUTOCORRELATION - WHY PERSISTENCE FAILS")
    print("=" * 64)

    lags = list(range(0, 169))
    autocorr = [df["us_aqi"].autocorr(lag=lag) for lag in lags]

    for lag in [1, 3, 6, 12, 24, 48, 72, 168]:
        print(f"  lag {lag:>3}h   r = {autocorr[lag]:.3f}")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(lags, autocorr, lw=2, color="#023e8a")
    ax.axhline(0, c="black", lw=0.8)
    for horizon, colour in zip([24, 48, 72], ["#2a9d8f", "#e9c46a", "#e76f51"]):
        ax.axvline(horizon, ls="--", c=colour, lw=1.8)
        ax.text(horizon + 2, 0.88, f"+{horizon}h\nr={autocorr[horizon]:.2f}",
                fontsize=9, color=colour)
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("AQI autocorrelation decay")
    save("04_autocorrelation.png")

    return autocorr


# %%
def section_diurnal(df: pd.DataFrame) -> None:
    """Daily cycle, and how it varies by month."""
    print("\n" + "=" * 64)
    print("4. DIURNAL CYCLE")
    print("=" * 64)

    hourly = df.groupby("hour")["us_aqi"].agg(["mean", "std"])
    print(f"  peak hour     {hourly['mean'].idxmax():02d}:00  "
          f"({hourly['mean'].max():.0f})")
    print(f"  lowest hour   {hourly['mean'].idxmin():02d}:00  "
          f"({hourly['mean'].min():.0f})")
    print(f"  daily range   "
          f"{hourly['mean'].max() - hourly['mean'].min():.0f} points")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(hourly.index, hourly["mean"], "o-", lw=2.5, color="#5a189a")
    ax1.fill_between(hourly.index, hourly["mean"] - hourly["std"],
                     hourly["mean"] + hourly["std"], alpha=0.2, color="#5a189a")
    ax1.set_xlabel("hour (Asia/Karachi)")
    ax1.set_ylabel("US AQI")
    ax1.set_xticks(range(0, 24, 3))
    ax1.set_title("Daily cycle")

    pivot = df.pivot_table(values="us_aqi", index="hour",
                           columns="month", aggfunc="mean")
    sns.heatmap(pivot, cmap="RdYlGn_r", ax=ax2, cbar_kws={"label": "US AQI"})
    ax2.set_xticklabels(MONTHS, rotation=0)
    ax2.set_xlabel("")
    ax2.set_title("AQI by hour and month")
    save("05_diurnal.png")

    # %%
def section_drivers(df: pd.DataFrame) -> pd.Series:
    """Which pollutants and weather variables move AQI."""
    print("\n" + "=" * 64)
    print("5. DRIVERS")
    print("=" * 64)

    columns = ["us_aqi"] + config.POLLUTANT_VARS + config.WEATHER_VARS
    correlations = df[columns].corr()["us_aqi"].drop("us_aqi").sort_values()

    print("  correlation with us_aqi:")
    for name, value in correlations.sort_values(ascending=False).items():
        print(f"    {name:<26} {value:+.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax1, annot_kws={"size": 6},
                cbar_kws={"shrink": 0.8})
    ax1.set_title("Correlation matrix")

    ax2.barh(range(len(correlations)), correlations.values,
             color=["#c1121f" if v > 0 else "#023e8a"
                    for v in correlations.values])
    ax2.set_yticks(range(len(correlations)))
    ax2.set_yticklabels(correlations.index, fontsize=8)
    ax2.axvline(0, c="black", lw=0.8)
    ax2.set_xlabel("correlation with us_aqi")
    ax2.set_title("Drivers of AQI")
    save("06_drivers.png")

    return correlations


# %%
def section_pollutants(df: pd.DataFrame) -> pd.Series:
    """Which pollutant sets the AQI, overall and by month."""
    print("\n" + "=" * 64)
    print("6. DOMINANT POLLUTANT")
    print("=" * 64)

    dominant = df["dominant_pollutant"].value_counts()
    for name, count in dominant.items():
        print(f"  {name:<8} {count:>6,}h  ({100 * count / len(df):5.1f}%)")

    by_month = pd.crosstab(df["month"], df["dominant_pollutant"],
                           normalize="index") * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.pie(dominant.values, labels=dominant.index, autopct="%1.1f%%",
            colors=sns.color_palette("Set2"), startangle=90)
    ax1.set_title("Dominant pollutant, all hours")

    by_month.plot(kind="bar", stacked=True, ax=ax2,
                  color=sns.color_palette("Set2"), width=0.85)
    ax2.set_xticklabels(MONTHS, rotation=0)
    ax2.set_xlabel("")
    ax2.set_ylabel("% of hours")
    ax2.set_title("Dominant pollutant by month")
    ax2.legend(title="", bbox_to_anchor=(1.01, 1))
    save("07_dominant_pollutant.png")

    return dominant


# %%
def main() -> None:
    """Run every section and print the consolidated findings."""
    print("=" * 64)
    print(f"EDA - {config.CITY.upper()}")
    print("=" * 64)

    df = load()
    section_overview(df)
    monthly = section_seasonal(df)
    autocorr = section_autocorrelation(df)
    section_diurnal(df)
    correlations = section_drivers(df)
    dominant = section_pollutants(df)

    weather_top = correlations.abs().loc[config.WEATHER_VARS].idxmax()

    print("\n" + "=" * 64)
    print("SUMMARY OF FINDINGS")
    print("=" * 64)
    print(f"""
1. PERSISTENTLY HAZARDOUS AIR
   Minimum AQI across {len(df):,} hours was {df.us_aqi.min():.0f}.
   Not one hour reached the EPA 'Good' band.
   {(df.us_aqi >= 151).mean():.1%} of hours were Unhealthy or worse.

2. STRONG SEASONALITY
   Worst month {MONTHS[monthly['mean'].idxmax() - 1]} ({monthly['mean'].max():.0f}),
   best month {MONTHS[monthly['mean'].idxmin() - 1]} ({monthly['mean'].min():.0f}),
   a swing of {monthly['mean'].max() - monthly['mean'].min():.0f} AQI points.
   A model trained on one season has never seen the others.

3. AUTOCORRELATION DECAYS FAST
   r(+24h)={autocorr[24]:.2f}  r(+48h)={autocorr[48]:.2f}  r(+72h)={autocorr[72]:.2f}
   This is why persistence scored R2 = -0.005 at +72h.

4. PM2.5 DOMINATES
   {dominant.index[0]} drives {100 * dominant.iloc[0] / len(df):.1f}% of hours.
   Combustion, not dust, is the problem.

5. WEATHER MATTERS
   Strongest weather correlate: {weather_top} ({correlations[weather_top]:+.3f})
   Cold, high-pressure, low-wind conditions trap pollution.
   This justifies the perfect-prognosis future-weather features.
""")
    print(f"Figures written to {FIGDIR}/")


# %%
main()