"""Go/no-go gate: can a linear model beat persistence?

Ridge on well-built features is the cheapest possible test of whether the
feature set carries real signal. If it loses to persistence, the problem is
the features, not the model, and no amount of XGBoost will rescue it.

Chronological split. Never train_test_split().
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import build_features as bf
from src.models import baseline

print("Loading features...")
df = pd.read_csv(
    "data/processed/features.csv", index_col="timestamp", parse_dates=["timestamp"]
)
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()
print(f"  {len(df)} rows\n")

features = bf.feature_columns(df)
print(f"Using {len(features)} features")

baseline_results = baseline.evaluate(df)
bar = baseline.targets(baseline_results)

print("\n" + "=" * 66)
print(f"{'horizon':<9} {'model':<14} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'vs base':>10}")
print("=" * 66)

for horizon in config.HORIZONS:
    target = config.TARGET_COLUMNS[horizon]

    usable = df[features + [target]].dropna()
    split = int(len(usable) * 0.8)

    X_train = usable[features].iloc[:split]
    y_train = usable[target].iloc[:split]
    X_test = usable[features].iloc[split:]
    y_test = usable[target].iloc[split:]

    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(X_test), index=X_test.index)

    scored = baseline.metrics(y_test, predictions, "Ridge")
    reference = bar[horizon]
    improvement = (reference["rmse"] - scored["rmse"]) / reference["rmse"] * 100

    print(f"+{horizon}h{'':<5} {'Persistence':<14} {reference['rmse']:>8.2f} "
          f"{reference['mae']:>8.2f} {reference['r2']:>8.3f} {'':>10}")
    print(f"{'':<9} {'Ridge':<14} {scored['rmse']:>8.2f} "
          f"{scored['mae']:>8.2f} {scored['r2']:>8.3f} "
          f"{improvement:>9.1f}%")

    verdict = "BEATS baseline" if improvement > 0 else "LOSES to baseline"
    print(f"{'':<9} -> {verdict}")
    print("-" * 66)

    coefficients = pd.Series(
        model.named_steps["ridge"].coef_, index=features
    ).abs().sort_values(ascending=False)
    print(f"{'':<9} top features: "
          + ", ".join(coefficients.head(5).index.tolist()))
    print()