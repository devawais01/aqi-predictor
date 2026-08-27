"""Validate our EPA AQI computation against Open-Meteo's us_aqi.

These are two independent implementations of the same standard. They should
agree closely. Systematic divergence means a unit or averaging-window bug.
"""
import pandas as pd

from src.utils import aqi_calculator as calc

print("Loading raw observations from CSV...")
df = pd.read_csv(
    "data/raw/observations.csv", index_col="timestamp", parse_dates=["timestamp"]
)
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()
print(f"  {len(df)} rows\n")

print("--- Unit conversion spot check (first row) ---")
first = df.iloc[[0]]
converted = calc.to_epa_units(first)
for col in ["pm2_5", "pm10", "ozone", "carbon_monoxide",
            "sulphur_dioxide", "nitrogen_dioxide"]:
    unit = ("ug/m3" if col.startswith("pm")
            else "ppm" if col in ("ozone", "carbon_monoxide")
            else "ppb")
    print(f"  {col:<20} {first[col].iloc[0]:>10.3f} ug/m3 "
          f"-> {converted[col].iloc[0]:>10.4f} {unit}")

print("\n--- Computing AQI for all rows ---")
result = calc.compute(df)
combined = df[["us_aqi"]].join(result)

# Drop the 24-hour warm-up where rolling windows are still partial.
combined = combined.iloc[24:]

print(f"  computed_aqi   mean {combined['computed_aqi'].mean():6.1f}  "
      f"min {combined['computed_aqi'].min():5.0f}  "
      f"max {combined['computed_aqi'].max():5.0f}")
print(f"  us_aqi         mean {combined['us_aqi'].mean():6.1f}  "
      f"min {combined['us_aqi'].min():5.0f}  "
      f"max {combined['us_aqi'].max():5.0f}")

diff = combined["computed_aqi"] - combined["us_aqi"]
correlation = combined["computed_aqi"].corr(combined["us_aqi"])

print(f"\n  correlation        {correlation:.4f}")
print(f"  mean difference    {diff.mean():+.1f}")
print(f"  median abs diff    {diff.abs().median():.1f}")
print(f"  90th pct abs diff  {diff.abs().quantile(0.90):.1f}")
print(f"  within +/-10 AQI   {(diff.abs() <= 10).mean():.1%}")
print(f"  within +/-25 AQI   {(diff.abs() <= 25).mean():.1%}")

print("\n--- Dominant pollutant frequency ---")
counts = combined["dominant_pollutant"].value_counts()
for name, count in counts.items():
    print(f"  {name:<8} {count:>6}  ({count / len(combined):.1%})")

print("\n--- Mean sub-index by pollutant ---")
for col in sorted(c for c in combined.columns if c.endswith("_aqi")
                  and c not in ("us_aqi", "computed_aqi")):
    print(f"  {col:<26} {combined[col].mean():6.1f}")

print("\n--- Verdict ---")
if correlation > 0.90 and diff.abs().median() < 20:
    print("  PASS - implementations agree closely.")
elif correlation > 0.75:
    print("  ACCEPTABLE - correlated but with offset; check dominant pollutant.")
else:
    print("  FAIL - likely a unit or averaging-window bug.")