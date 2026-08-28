"""Training pipeline: all models, all horizons, artifacts to the registry.

Model roster, spanning statistical through deep learning as the guidelines
require:

    Persistence     baseline every model must clear
    SARIMAX         classical statistical time series
    Ridge           linear, fast sanity check
    Random Forest   bagged trees
    XGBoost         gradient-boosted trees
    LSTM            sequence model, real 24-step lookback

Every model is fitted independently per horizon (+24h, +48h, +72h). There is
no recursion and no fabricated future input: each prediction is the direct
output of a model trained for that specific horizon.

    python -m src.models.train_model
    python -m src.models.train_model --skip-lstm
    python -m src.models.train_model --no-upload
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src import config
from src.features import build_features as bf
from src.models import evaluate
from src.utils import db_client

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MODEL_DIR = "models"
LSTM_LOOKBACK = 24
LSTM_EPOCHS = 30
LSTM_BATCH = 64


def ridge_factory():
    """Linear model with standardised inputs."""
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def random_forest_factory():
    """Bagged trees. Depth capped to keep the artifact small enough to upload."""
    return RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=5,
        max_features=0.5,
        random_state=42,
        n_jobs=-1,
    )


def xgboost_factory():
    """Gradient boosting. Usually the strongest model on tabular data."""
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


SKLEARN_MODELS = {
    "Ridge": ridge_factory,
    "Random Forest": random_forest_factory,
    "XGBoost": xgboost_factory,
}

def fit_sarimax(aqi_series: pd.Series, train_index, test_index,
                horizon: int, stride: int = 12):
    """SARIMAX with a rolling forecast origin.

    At each evaluation point t the model has been filtered through all
    observations up to t, then forecasts exactly `horizon` steps ahead. This
    matches what the ML models do. A single long-range forecast instead
    would converge to the series mean and score meaninglessly badly.

    Evaluation points are strided to keep runtime sane; the state is still
    updated with every observation in between, so no information is skipped.

    Returns (fitted_model, predictions_series).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    history = aqi_series.loc[train_index].iloc[-1500:]
    history = history.asfreq("h") if history.index.freq is None else history

    fitted = SARIMAX(
        history,
        order=(3, 0, 1),
        seasonal_order=(1, 0, 0, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=200, method="lbfgs")

    state = fitted
    predictions = {}

    for start in range(0, len(test_index), stride):
        block = test_index[start : start + stride]
        origin = test_index[start]

        target_time = origin + pd.Timedelta(hours=horizon)
        if target_time <= aqi_series.index[-1]:
            forecast = state.forecast(steps=horizon)
            predictions[origin] = float(forecast.iloc[-1])

        observed = aqi_series.loc[block]
        state = state.append(observed, refit=False)

    return fitted, pd.Series(predictions).sort_index()


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Reshape flat rows into (samples, lookback, features) for the LSTM.

    Sample i contains rows [i, i+lookback) and predicts y at i+lookback-1.
    This is the step the reference project got wrong: reshaping to
    (n, 1, features) gives a sequence length of one, which is a dense layer
    with extra machinery and no temporal memory at all.
    """
    n_samples = len(X) - lookback + 1
    sequences = np.zeros((n_samples, lookback, X.shape[1]), dtype=np.float32)
    for i in range(n_samples):
        sequences[i] = X[i : i + lookback]
    return sequences, y[lookback - 1 :]


def fit_lstm(X_train, y_train, X_test, y_test, lookback: int = LSTM_LOOKBACK):
    """Two-layer LSTM over a real 24-hour lookback window."""
    import tensorflow as tf

    tf.random.set_seed(42)
    tf.get_logger().setLevel("ERROR")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(
        y_train.values.reshape(-1, 1)
    ).astype(np.float32).ravel()

    seq_train, target_train = make_sequences(
        X_train_scaled, y_train_scaled, lookback
    )
    seq_test, _ = make_sequences(
        X_test_scaled, np.zeros(len(X_test_scaled)), lookback
    )

    split = int(len(seq_train) * 0.85)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, X_train_scaled.shape[1])),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse"
    )
    model.fit(
        seq_train[:split], target_train[:split],
        validation_data=(seq_train[split:], target_train[split:]),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=0,
    )

    scaled_predictions = model.predict(seq_test, verbose=0)
    predictions = y_scaler.inverse_transform(scaled_predictions).ravel()

    # The first lookback-1 test rows have no full window behind them.
    padded = np.full(len(X_test), np.nan)
    padded[lookback - 1 :] = predictions

    return model, padded, x_scaler, y_scaler

def train_horizon(df: pd.DataFrame, features: list[str], horizon: int,
                  run_cv: bool = True, skip_lstm: bool = False,
                  skip_sarimax: bool = False) -> tuple:
    """Fit every model for one horizon. Returns (results, fitted_artifacts)."""
    target = config.TARGET_COLUMNS[horizon]

    X_train, y_train, X_test, y_test = evaluate.chronological_split(
        df, features, target
    )
    print(f"\n{'=' * 74}")
    print(f"HORIZON +{horizon}h   target={target}")
    print(f"{'=' * 74}")
    print(f"  train {len(X_train):,} rows  "
          f"({X_train.index.min().date()} -> {X_train.index.max().date()})")
    print(f"  test  {len(X_test):,} rows  "
          f"({X_test.index.min().date()} -> {X_test.index.max().date()})")

    results, artifacts = [], {}

    # Persistence baseline on the identical test rows.
    baseline_prediction = df.loc[X_test.index, "us_aqi"]
    baseline_result = evaluate.score(y_test, baseline_prediction, "Persistence")
    baseline_result["horizon"] = horizon
    results.append(baseline_result)
    print(f"\n  Persistence      RMSE {baseline_result['rmse']:7.2f}  "
          f"R2 {baseline_result['r2']:7.3f}")

    # Scikit-learn style models.
    for name, factory in SKLEARN_MODELS.items():
        started = time.time()
        model = factory()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        result = evaluate.score(y_test, predictions, name)
        result["horizon"] = horizon
        result["fit_seconds"] = round(time.time() - started, 1)

        if run_cv:
            cv = evaluate.walk_forward_validate(factory, df, features, target)
            result.update(evaluate.summarise_cv(cv, name))
            result["model"] = name
            evaluate.print_cv_detail(cv, name)

        results.append(result)
        artifacts[name] = model
        print(f"  {name:<16} RMSE {result['rmse']:7.2f}  "
              f"R2 {result['r2']:7.3f}  ({result['fit_seconds']}s)")

    # SARIMAX: statistical model on the target series alone.
    if not skip_sarimax:
        try:
            started = time.time()
            sarimax_model, sarimax_prediction = fit_sarimax(
                df["us_aqi"], X_train.index, X_test.index, horizon
            )
            aligned = y_test.reindex(sarimax_prediction.index)
            result = evaluate.score(aligned, sarimax_prediction, "SARIMAX")
            result["horizon"] = horizon
            result["fit_seconds"] = round(time.time() - started, 1)
            results.append(result)
            artifacts["SARIMAX"] = sarimax_model
            print(f"  {'SARIMAX':<16} RMSE {result['rmse']:7.2f}  "
                f"R2 {result['r2']:7.3f}  ({result['fit_seconds']}s, "
                f"n={result['n']})")
        except Exception as exc:
            print(f"  SARIMAX failed: {exc}")

    # LSTM: sequence model with a genuine 24-step lookback.
    if not skip_lstm:
        try:
            started = time.time()
            lstm_model, lstm_prediction, x_scaler, y_scaler = fit_lstm(
                X_train, y_train, X_test, y_test
            )
            result = evaluate.score(y_test, lstm_prediction, "LSTM")
            result["horizon"] = horizon
            result["fit_seconds"] = round(time.time() - started, 1)
            results.append(result)
            artifacts["LSTM"] = (lstm_model, x_scaler, y_scaler)
            print(f"  {'LSTM':<16} RMSE {result['rmse']:7.2f}  "
                  f"R2 {result['r2']:7.3f}  ({result['fit_seconds']}s)")
        except Exception as exc:
            print(f"  LSTM failed: {exc}")

    return results, artifacts


def select_best(results: list[dict]) -> dict:
    """Lowest test RMSE among the trained models, excluding the baseline."""
    candidates = [r for r in results if r["model"] != "Persistence"]
    return min(candidates, key=lambda r: r["rmse"])

def save_artifacts(artifacts: dict, horizon: int, best_name: str,
                   features: list[str], upload: bool) -> list[str]:
    """Write the winning model for this horizon to disk and the registry."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    written = []

    if best_name == "LSTM":
        model, x_scaler, y_scaler = artifacts["LSTM"]
        model_path = f"{MODEL_DIR}/best_t{horizon}.keras"
        model.save(model_path)
        joblib.dump(x_scaler, f"{MODEL_DIR}/xscaler_t{horizon}.pkl")
        joblib.dump(y_scaler, f"{MODEL_DIR}/yscaler_t{horizon}.pkl")
        written = [
            model_path,
            f"{MODEL_DIR}/xscaler_t{horizon}.pkl",
            f"{MODEL_DIR}/yscaler_t{horizon}.pkl",
        ]
    elif best_name == "SARIMAX":
        model_path = f"{MODEL_DIR}/best_t{horizon}.pkl"
        joblib.dump(artifacts["SARIMAX"], model_path, compress=3)
        written = [model_path]
    else:
        model_path = f"{MODEL_DIR}/best_t{horizon}.pkl"
        joblib.dump(artifacts[best_name], model_path, compress=3)
        written = [model_path]

    meta = {
        "horizon": horizon,
        "model": best_name,
        "features": features,
        "n_features": len(features),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = f"{MODEL_DIR}/meta_t{horizon}.json"
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    written.append(meta_path)

    for path in written:
        size_mb = os.path.getsize(path) / 1e6
        print(f"    {os.path.basename(path):<28} {size_mb:6.2f} MB")
        if upload:
            db_client.upload_artifact(path, os.path.basename(path))

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all AQI models.")
    parser.add_argument("--skip-sarimax", action="store_true",
                        help="Skip SARIMAX (slow: rolling Kalman filter)")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip the LSTM (avoids the TensorFlow import)")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip walk-forward CV (much faster)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Do not push artifacts to Supabase")
    parser.add_argument("--from-csv", action="store_true",
                        help="Read features from local CSV instead of Supabase")
    args = parser.parse_args()

    print("=" * 74)
    print(f"TRAINING PIPELINE  {config.CITY}")
    print("=" * 74)

    if args.from_csv:
        print("Loading features from data/processed/features.csv ...")
        df = pd.read_csv("data/processed/features.csv",
                         index_col="timestamp", parse_dates=["timestamp"])
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        config.validate_credentials()
        print("Loading features from Supabase ...")
        df = db_client.fetch_features()

    if df.empty:
        print("Feature store is empty. Run the feature pipeline first.")
        return 1

    df = df.sort_index()
    features = bf.feature_columns(df)
    print(f"  {len(df):,} rows, {len(features)} features")

    all_results, best_per_horizon = [], {}

    for horizon in config.HORIZONS:
        results, artifacts = train_horizon(
            df, features, horizon,
            run_cv=not args.skip_cv,
            skip_lstm=args.skip_lstm,
            skip_sarimax=args.skip_sarimax,
        )
        all_results.extend(results)

        best = select_best(results)
        best_per_horizon[horizon] = best
        print(f"\n  WINNER +{horizon}h: {best['model']} "
              f"(RMSE {best['rmse']:.2f}, R2 {best['r2']:.3f})")

        print("  Saving artifacts:")
        save_artifacts(artifacts, horizon, best["model"], features,
                       upload=not args.no_upload)

    baseline = {
        h: next(r for r in all_results
                if r["horizon"] == h and r["model"] == "Persistence")
        for h in config.HORIZONS
    }
    table = evaluate.comparison_table(
        [r for r in all_results if r["model"] != "Persistence"], baseline
    )
    evaluate.print_comparison(table, baseline)

    os.makedirs(MODEL_DIR, exist_ok=True)
    metrics_path = f"{MODEL_DIR}/metrics.json"
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "city": config.CITY,
        "n_rows": int(len(df)),
        "n_features": len(features),
        "results": all_results,
        "best_per_horizon": {str(k): v for k, v in best_per_horizon.items()},
    }
    with open(metrics_path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"\nWrote {metrics_path}")

    if not args.no_upload:
        db_client.upload_json(payload, "metrics.json")

        history = db_client.download_json("training_history.json") or []
        history.append({
            "trained_at": payload["trained_at"],
            "best_per_horizon": {
                str(h): {"model": b["model"], "rmse": b["rmse"], "r2": b["r2"]}
                for h, b in best_per_horizon.items()
            },
        })
        db_client.upload_json(history[-60:], "training_history.json")
        print("Registry updated.")

    table.to_csv("data/processed/model_comparison.csv", index=False)
    print("Wrote data/processed/model_comparison.csv")
    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())