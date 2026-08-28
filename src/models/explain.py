"""SHAP explainability for the selected models.

Computed on a 1,000-row sample of the test set, not a single row. A summary
plot built from one sample conveys nothing about feature importance in
general; it shows only that one prediction's decomposition.

    python -m src.models.explain
    python -m src.models.explain --sample 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.features import build_features as bf
from src.models import evaluate
from src.utils import db_client

warnings.filterwarnings("ignore")

FIGDIR = "reports/figures"
MODEL_DIR = "models"
SAMPLE_SIZE = 1000


def load_model(horizon: int):
    """Load the selected model and its metadata for one horizon."""
    meta_path = f"{MODEL_DIR}/meta_t{horizon}.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Run the training pipeline first."
        )
    with open(meta_path) as handle:
        meta = json.load(handle)

    if meta["model"] == "LSTM":
        raise ValueError(
            f"+{horizon}h selected an LSTM. KernelExplainer on a sequence "
            "model is impractically slow; skipping."
        )

    model = joblib.load(f"{MODEL_DIR}/best_t{horizon}.pkl")
    return model, meta


def explainer_for(model, background: pd.DataFrame):
    """Pick the right SHAP explainer for the model type."""
    name = type(model).__name__

    if name in ("XGBRegressor", "RandomForestRegressor"):
        return shap.TreeExplainer(model), "tree"

    # Pipeline(StandardScaler -> Ridge): explain the linear step directly,
    # feeding it scaled inputs so the coefficients are on the right scale.
    if name == "Pipeline":
        scaler = model.named_steps["standardscaler"]
        linear = model.named_steps["ridge"]
        scaled = pd.DataFrame(
            scaler.transform(background),
            columns=background.columns,
            index=background.index,
        )
        masker = shap.maskers.Independent(scaled, max_samples=len(scaled))
        return shap.LinearExplainer(linear, masker), "linear"

    raise ValueError(f"No SHAP strategy for {name}")

def compute_shap(horizon: int, df: pd.DataFrame, features: list[str],
                 sample_size: int = SAMPLE_SIZE) -> pd.Series:
    """Compute SHAP values for one horizon and write the plots."""
    target = config.TARGET_COLUMNS[horizon]
    model, meta = load_model(horizon)

    _, _, X_test, _ = evaluate.chronological_split(df, features, target)

    n = min(sample_size, len(X_test))
    sample = X_test.iloc[-n:]
    print(f"\n+{horizon}h  model={meta['model']}  "
          f"explaining {n} rows "
          f"({sample.index.min().date()} -> {sample.index.max().date()})")

    explainer, kind = explainer_for(model, sample)

    if kind == "linear":
        scaler = model.named_steps["standardscaler"]
        scaled = pd.DataFrame(
            scaler.transform(sample), columns=sample.columns, index=sample.index
        )
        shap_values = explainer.shap_values(scaled)
        display_data = scaled
    else:
        shap_values = explainer.shap_values(sample)
        display_data = sample

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=features
    ).sort_values(ascending=False)

    print(f"  top 15 features by mean |SHAP|:")
    for name, value in importance.head(15).items():
        print(f"    {name:<30} {value:8.3f}")

    os.makedirs(FIGDIR, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, display_data, max_display=20, show=False)
    plt.title(f"SHAP feature importance  +{horizon}h  ({meta['model']})",
              fontsize=11)
    plt.tight_layout()
    path = f"{FIGDIR}/08_shap_beeswarm_t{horizon}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"    saved {path}")

    plt.figure()
    shap.summary_plot(shap_values, display_data, plot_type="bar",
                      max_display=20, show=False)
    plt.title(f"Mean |SHAP| by feature  +{horizon}h  ({meta['model']})",
              fontsize=11)
    plt.tight_layout()
    path = f"{FIGDIR}/09_shap_bar_t{horizon}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"    saved {path}")

    return importance


def group_importance(importance: pd.Series) -> pd.Series:
    """Aggregate feature importance into the groups from the feature spec."""
    groups = {
        "current pollutants": 0.0,
        "current weather": 0.0,
        "future weather (perfect prog)": 0.0,
        "AQI lags": 0.0,
        "AQI rolling": 0.0,
        "EPA sub-indices": 0.0,
        "time (cyclic)": 0.0,
        "derived": 0.0,
        "other": 0.0,
    }

    for name, value in importance.items():
        if any(name.startswith(v + "_t") for v in bf.FUTURE_WEATHER_VARS):
            groups["future weather (perfect prog)"] += value
        elif "_lag_" in name:
            groups["AQI lags"] += value
        elif name.startswith(("aqi_roll", "aqi_std")):
            groups["AQI rolling"] += value
        elif name in config.POLLUTANT_VARS:
            groups["current pollutants"] += value
        elif name in config.WEATHER_VARS:
            groups["current weather"] += value
        elif name.endswith("_aqi") or name in ("us_aqi", "computed_aqi"):
            groups["EPA sub-indices"] += value
        elif name.endswith(("_sin", "_cos")) or name == "is_weekend":
            groups["time (cyclic)"] += value
        elif name in ("aqi_change_rate", "aqi_accel", "wind_u", "wind_v"):
            groups["derived"] += value
        else:
            groups["other"] += value

    series = pd.Series(groups)
    return (series / series.sum() * 100).sort_values(ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="SHAP explanations.")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--from-csv", action="store_true", default=True)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("SHAP EXPLAINABILITY")
    print("=" * 70)

    df = pd.read_csv("data/processed/features.csv",
                     index_col="timestamp", parse_dates=["timestamp"])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    features = bf.feature_columns(df)
    print(f"{len(df):,} rows, {len(features)} features")

    summary = {}
    for horizon in config.HORIZONS:
        try:
            importance = compute_shap(horizon, df, features, args.sample)
            grouped = group_importance(importance)

            print(f"\n  contribution by feature group (+{horizon}h):")
            for group, percent in grouped.items():
                if percent > 0.05:
                    print(f"    {group:<32} {percent:5.1f}%")

            summary[horizon] = {
                "top_features": importance.head(20).round(4).to_dict(),
                "group_percentages": grouped.round(2).to_dict(),
            }
        except Exception as exc:
            print(f"\n+{horizon}h  skipped: {exc}")

    if summary:
        path = f"{MODEL_DIR}/shap_summary.json"
        with open(path, "w") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print(f"\nWrote {path}")

        if not args.no_upload:
            try:
                db_client.upload_json(summary, "shap_summary.json")
            except Exception as exc:
                print(f"  registry upload skipped: {exc}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())