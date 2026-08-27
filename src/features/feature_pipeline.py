"""Feature pipeline: raw observations -> engineered features -> feature store.

Two modes:

  full         Rebuild every feature row from all raw history. Run once
               after backfill, or whenever the feature spec changes.

  incremental  Fetch recent observations, append to raw, then rebuild only
               the tail. This is what GitHub Actions runs hourly.

Incremental still recomputes features over a trailing window rather than a
single row, because lag_168 and roll_168 need a week of history behind them.

    python -m src.features.feature_pipeline --mode full
    python -m src.features.feature_pipeline
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import config
from src.features import build_features as bf
from src.features import fetch_data
from src.utils import db_client

# Hours of history pulled back when rebuilding the tail. Must exceed the
# longest lag (168h) plus the longest target horizon (72h) with margin.
INCREMENTAL_WINDOW = 24 * 21


def run_full() -> int:
    """Rebuild the entire feature store from raw_observations."""
    print("Reading raw observations from Supabase...")
    raw = db_client.fetch_raw()
    if raw.empty:
        print("raw_observations is empty. Run the backfill first.")
        return 1
    print(f"  {len(raw)} rows, {raw.index.min()} -> {raw.index.max()}")

    features = bf.build(raw)
    columns = bf.feature_columns(features)

    print()
    db_client.push_features(features, columns)
    print("\nFeature store rebuilt.")
    return 0


def run_incremental() -> int:
    """Fetch recent data, update raw, rebuild the feature tail."""
    print("Fetching recent observations from Open-Meteo...")
    recent = fetch_data.fetch_recent(days_back=7)
    print(f"  {len(recent)} rows, {recent.index.min()} -> {recent.index.max()}")

    if recent.empty:
        print("No recent data returned.")
        return 1

    print()
    db_client.push_raw(recent)

    print("\nReading raw history for the rebuild window...")
    raw = db_client.fetch_raw()
    window = raw.iloc[-INCREMENTAL_WINDOW:]
    print(f"  rebuilding from {len(window)} rows "
          f"({window.index.min()} -> {window.index.max()})")

    features = bf.build(window)
    columns = bf.feature_columns(features)

    print()
    db_client.push_features(features, columns)
    print("\nIncremental update complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the feature pipeline.")
    parser.add_argument(
        "--mode", choices=["full", "incremental"], default="incremental"
    )
    args = parser.parse_args()

    config.validate_credentials()

    print("=" * 60)
    print(f"FEATURE PIPELINE  ({args.mode})  {config.CITY}")
    print("=" * 60)

    if args.mode == "full":
        return run_full()
    return run_incremental()


if __name__ == "__main__":
    sys.exit(main())