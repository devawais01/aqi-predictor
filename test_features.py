"""Build the feature set locally and inspect it before writing to Supabase."""
import pandas as pd

from src.features import build_features as bf

print("Loading raw observations...")
df = pd.read_csv(
    "data/raw/observations.csv", index_col="timestamp", parse_dates=["timestamp"]
)
df.index = pd.to_datetime(df.index, utc=True)
print(f"  {len(df)} rows\n")

features = bf.build(df)
bf.summarise(features)

print("\n--- Sanity: does the target actually look 24h ahead? ---")
sample = features[["us_aqi", "aqi_t24"]].dropna().head(3)
for timestamp, row in sample.iterrows():
    future = features.loc[timestamp + pd.Timedelta(hours=24), "us_aqi"]
    match = "OK" if abs(row["aqi_t24"] - future) < 0.01 else "MISMATCH"
    print(f"  {timestamp}  aqi={row['us_aqi']:.0f}  "
          f"aqi_t24={row['aqi_t24']:.0f}  actual at +24h={future:.0f}  {match}")

print("\n--- Correlation of key features with each target ---")
for target in ["aqi_t24", "aqi_t48", "aqi_t72"]:
    print(f"\n  {target}:")
    correlations = (
        features[bf.feature_columns(features) + [target]]
        .corr()[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
    )
    for name, value in correlations.head(8).items():
        print(f"    {name:<28} {value:.3f}")

features.to_csv("data/processed/features.csv")
print("\nSaved to data/processed/features.csv")