"""Evaluation utilities: metrics, chronological splits, walk-forward CV.

A single 80/20 split on two years of Lahore data puts the entire test set in
one season. With a 110-point swing between January (220) and April (110),
that makes any single-split score unreliable on its own. Walk-forward CV
with an expanding window is the honest alternative: each fold trains on
everything before a cut point and tests on the block immediately after.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config

N_FOLDS = 5
TRAIN_FRACTION = 0.8


def score(y_true, y_pred, label: str = "") -> dict:
    """RMSE, MAE and R2, ignoring rows where either side is missing."""
    truth = pd.Series(np.asarray(y_true, dtype=float))
    pred = pd.Series(np.asarray(y_pred, dtype=float))
    mask = truth.notna() & pred.notna()
    truth, pred = truth[mask], pred[mask]

    return {
        "model": label,
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "mae": float(mean_absolute_error(truth, pred)),
        "r2": float(r2_score(truth, pred)),
        "n": int(len(truth)),
    }


def chronological_split(df: pd.DataFrame, features: list[str], target: str,
                        train_fraction: float = TRAIN_FRACTION):
    """Oldest rows train, newest rows test. Never shuffle a time series."""
    usable = df[features + [target]].dropna()
    cut = int(len(usable) * train_fraction)
    return (
        usable[features].iloc[:cut],
        usable[target].iloc[:cut],
        usable[features].iloc[cut:],
        usable[target].iloc[cut:],
    )


def walk_forward_folds(n_rows: int, n_folds: int = N_FOLDS,
                       min_train_fraction: float = 0.4):
    """Expanding-window fold boundaries.

    Fold k trains on rows [0, train_end) and tests on [train_end, test_end).
    The training window grows each fold; test blocks never overlap.
    """
    start = int(n_rows * min_train_fraction)
    block = (n_rows - start) // n_folds

    folds = []
    for k in range(n_folds):
        train_end = start + k * block
        test_end = train_end + block if k < n_folds - 1 else n_rows
        folds.append((train_end, test_end))
    return folds


def walk_forward_validate(model_factory, df: pd.DataFrame,
                          features: list[str], target: str,
                          n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Expanding-window CV. model_factory() must return an unfitted model."""
    usable = df[features + [target]].dropna()
    X, y = usable[features], usable[target]

    rows = []
    for fold, (train_end, test_end) in enumerate(
        walk_forward_folds(len(usable), n_folds), start=1
    ):
        model = model_factory()
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        predictions = model.predict(X.iloc[train_end:test_end])

        result = score(y.iloc[train_end:test_end], predictions, f"fold {fold}")
        result["fold"] = fold
        result["train_rows"] = train_end
        result["test_rows"] = test_end - train_end
        result["test_start"] = usable.index[train_end]
        result["test_end"] = usable.index[test_end - 1]
        rows.append(result)

    return pd.DataFrame(rows)

def summarise_cv(cv: pd.DataFrame, label: str) -> dict:
    """Collapse fold results into mean, spread and worst case."""
    return {
        "model": label,
        "cv_rmse_mean": float(cv["rmse"].mean()),
        "cv_rmse_std": float(cv["rmse"].std()),
        "cv_mae_mean": float(cv["mae"].mean()),
        "cv_r2_mean": float(cv["r2"].mean()),
        "cv_r2_std": float(cv["r2"].std()),
        "cv_r2_worst": float(cv["r2"].min()),
        "n_folds": int(len(cv)),
    }


def persistence_scores(df: pd.DataFrame,
                       train_fraction: float = TRAIN_FRACTION) -> dict:
    """Persistence baseline on the same test split the models use."""
    out = {}
    for horizon in config.HORIZONS:
        target = config.TARGET_COLUMNS[horizon]
        usable = df[["us_aqi", target]].dropna()
        cut = int(len(usable) * train_fraction)
        test = usable.iloc[cut:]
        out[horizon] = score(test[target], test["us_aqi"], "Persistence")
    return out


def comparison_table(results: list[dict], baseline: dict) -> pd.DataFrame:
    """Assemble the model-vs-baseline table used in the report."""
    frame = pd.DataFrame(results)
    frame["baseline_rmse"] = frame["horizon"].map(lambda h: baseline[h]["rmse"])
    frame["rmse_improvement_pct"] = (
        (frame["baseline_rmse"] - frame["rmse"]) / frame["baseline_rmse"] * 100
    )
    frame["beats_baseline"] = frame["rmse_improvement_pct"] > 0
    return frame.sort_values(["horizon", "rmse"])


def print_comparison(frame: pd.DataFrame, baseline: dict) -> None:
    """Print the results table, horizon by horizon."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    for horizon in sorted(frame["horizon"].unique()):
        subset = frame[frame["horizon"] == horizon].sort_values("rmse")
        reference = baseline[horizon]

        print(f"\n  +{horizon}h   (persistence: RMSE {reference['rmse']:.2f}, "
              f"R2 {reference['r2']:.3f})")
        print(f"    {'model':<20} {'RMSE':>8} {'MAE':>8} {'R2':>8} "
              f"{'vs base':>9} {'CV R2':>9}")
        print(f"    {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} "
              f"{'-' * 9} {'-' * 9}")

        for _, row in subset.iterrows():
            cv_value = row.get("cv_r2_mean")
            cv_display = f"{cv_value:.3f}" if pd.notna(cv_value) else "-"
            marker = "*" if row["beats_baseline"] else " "
            print(f"  {marker} {row['model']:<20} {row['rmse']:>8.2f} "
                  f"{row['mae']:>8.2f} {row['r2']:>8.3f} "
                  f"{row['rmse_improvement_pct']:>8.1f}% {cv_display:>9}")

    print("\n  * beats the persistence baseline")


def print_cv_detail(cv: pd.DataFrame, label: str) -> None:
    """Print per-fold results so seasonal variation is visible."""
    print(f"\n  Walk-forward folds - {label}")
    print(f"    {'fold':<6} {'test period':<26} {'rows':>7} "
          f"{'RMSE':>8} {'R2':>8}")
    for _, row in cv.iterrows():
        period = (f"{pd.Timestamp(row['test_start']).date()} to "
                  f"{pd.Timestamp(row['test_end']).date()}")
        print(f"    {int(row['fold']):<6} {period:<26} "
              f"{int(row['test_rows']):>7} {row['rmse']:>8.2f} "
              f"{row['r2']:>8.3f}")