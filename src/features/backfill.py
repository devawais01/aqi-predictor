"""Backfill two years of hourly Lahore observations into Supabase.

Run once. Validation gates abort before writing if the data looks wrong,
so a bad pull never silently poisons the feature store.

    python -m src.features.backfill
    python -m src.features.backfill --years 1
    python -m src.features.backfill --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from src import config
from src.features import fetch_data
from src.utils import db_client


def validate(df: pd.DataFrame, min_rows: int) -> bool:
    """Run the Day 1 data-quality gates. Returns True if the data is usable."""
    print("\n--- Validation ---")
    ok = True

    rows = len(df)
    print(f"Rows: {rows}")
    if rows < min_rows:
        print(f"  FAIL: expected at least {min_rows}")
        ok = False
    else:
        print("  OK")

    duplicates = int(df.index.duplicated().sum())
    print(f"Duplicate timestamps: {duplicates}")
    if duplicates:
        print("  FAIL: timestamps must be unique")
        ok = False
    else:
        print("  OK")

    if not df.index.is_monotonic_increasing:
        print("Index not sorted ascending")
        print("  FAIL")
        ok = False

    span_hours = int((df.index.max() - df.index.min()).total_seconds() // 3600) + 1
    gaps = span_hours - rows
    print(f"Coverage: {rows}/{span_hours} hours ({gaps} missing)")

    print("\nNull fraction by column:")
    for column in df.columns:
        fraction = float(df[column].isna().mean())
        flag = "" if fraction <= config.MAX_NULL_FRACTION else "  <-- EXCEEDS LIMIT"
        print(f"  {column:<24} {fraction:6.2%}{flag}")
        if fraction > config.MAX_NULL_FRACTION:
            ok = False

    if "us_aqi" in df.columns:
        aqi = df["us_aqi"].dropna()
        if len(aqi):
            print(
                f"\nus_aqi range: {aqi.min():.0f} - {aqi.max():.0f} "
                f"(mean {aqi.mean():.1f}, median {aqi.median():.1f})"
            )
            if aqi.min() < 0:
                print("  FAIL: negative AQI values present")
                ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill historical AQI data.")
    parser.add_argument(
        "--years", type=float, default=config.BACKFILL_YEARS,
        help="Years of history to fetch (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and validate but do not write to Supabase",
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Also write a local copy to data/raw/",
    )
    args = parser.parse_args()

    config.validate_credentials()

    end = date.today() - timedelta(days=config.ARCHIVE_LAG_DAYS)
    start = end - timedelta(days=int(365 * args.years))
    expected_rows = int(365 * args.years * 24 * 0.9)

    print("=" * 62)
    print(f"BACKFILL  {config.CITY}  ({config.LATITUDE}, {config.LONGITUDE})")
    print(f"Range     {start} -> {end}  ({args.years} years)")
    print("=" * 62)

    df = fetch_data.fetch_observations(start, end)

    if not validate(df, expected_rows):
        print("\nValidation FAILED. Nothing written.")
        return 1

    print("\nValidation PASSED.")

    if args.save_csv:
        path = "data/raw/observations.csv"
        df.to_csv(path)
        print(f"Saved local copy to {path}")

    if args.dry_run:
        print("Dry run - skipping Supabase write.")
        return 0

    print()
    db_client.push_raw(df)
    total = db_client.raw_row_count()
    print(f"\nraw_observations now holds {total} rows.")
    print("Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())