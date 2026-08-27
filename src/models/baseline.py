"""Persistence baseline: the bar every model must clear.

    prediction(t + h) = us_aqi(t)

Because current AQI is a legitimate feature, a model can score well simply
by echoing its input. Reporting R2 without this comparison is meaningless.
Any model that fails to beat persistence has learned nothing.

A seasonal-naive variant (same hour yesterday) is included for contrast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


def metrics(y_true: pd.Series, y_pred: pd.Series, label: str) -> dict:
    """RMSE, MAE and R2 for one prediction series."""
    mask = y_true.notna() & y_pred.notna()
    truth, pred = y_true[mask], y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    mae = float(mean_absolute_error(truth, pred))
    r2 = float(r2_score(truth, pred))

    return {"model": label, "rmse": rmse, "mae": mae, "r2": r2, "n": int(len(truth))}


def persistence(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Naive forecast: the value now, carried forward."""
    return df["us_aqi"]


def seasonal_naive(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Same hour, previous day. Captures the diurnal cycle for free."""
    return df["us_aqi"].shift(24 - (horizon % 24)) if horizon % 24 else df["aqi_lag_24"]


def rolling_mean_naive(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Last 24 hours averaged. Smoother than persistence."""
    return df["aqi_roll_24"]


BASELINES = {
    "Persistence": persistence,
    "Seasonal naive (24h)": seasonal_naive,
    "Rolling mean (24h)": rolling_mean_naive,
}


def evaluate(df: pd.DataFrame, test_only: bool = True,
             train_fraction: float = 0.8) -> pd.DataFrame:
    """Score every baseline on every horizon.

    Evaluated on the test split only, so the numbers are directly
    comparable with the trained models.
    """
    work = df.copy().sort_index()

    if test_only:
        split = int(len(work) * train_fraction)
        work = work.iloc[split:]
        print(f"Evaluating on the newest {len(work)} rows "
              f"({work.index.min().date()} -> {work.index.max().date()})")
    else:
        print(f"Evaluating on all {len(work)} rows")

    rows = []
    for horizon in config.HORIZONS:
        target = config.TARGET_COLUMNS[horizon]
        for name, function in BASELINES.items():
            prediction = function(work, horizon)
            result = metrics(work[target], prediction, name)
            result["horizon"] = horizon
            rows.append(result)

    return pd.DataFrame(rows)[["horizon", "model", "rmse", "mae", "r2", "n"]]


def report(results: pd.DataFrame) -> None:
    """Print the baseline table, horizon by horizon."""
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    for horizon in sorted(results["horizon"].unique()):
        subset = results[results["horizon"] == horizon].sort_values("rmse")
        print(f"\n  +{horizon}h")
        print(f"    {'model':<24} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
        print(f"    {'-' * 24} {'-' * 8} {'-' * 8} {'-' * 8}")
        for _, row in subset.iterrows():
            print(f"    {row['model']:<24} {row['rmse']:>8.2f} "
                  f"{row['mae']:>8.2f} {row['r2']:>8.3f}")

    print("\n" + "-" * 60)
    print("Targets to beat (persistence):")
    persistence_rows = results[results["model"] == "Persistence"]
    for _, row in persistence_rows.sort_values("horizon").iterrows():
        print(f"  +{int(row['horizon'])}h   RMSE < {row['rmse']:.2f}   "
              f"R2 > {row['r2']:.3f}")
    print("-" * 60)


def targets(results: pd.DataFrame) -> dict[int, dict]:
    """Persistence scores keyed by horizon, for later comparison."""
    out = {}
    for _, row in results[results["model"] == "Persistence"].iterrows():
        out[int(row["horizon"])] = {
            "rmse": float(row["rmse"]),
            "mae": float(row["mae"]),
            "r2": float(row["r2"]),
        }
    return out


if __name__ == "__main__":
    features = pd.read_csv(
        "data/processed/features.csv",
        index_col="timestamp",
        parse_dates=["timestamp"],
    )
    features.index = pd.to_datetime(features.index, utc=True)

    results = evaluate(features)
    report(results)
    results.to_csv("data/processed/baseline_metrics.csv", index=False)
    print("\nSaved to data/processed/baseline_metrics.csv")