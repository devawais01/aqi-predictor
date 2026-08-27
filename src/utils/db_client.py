"""Supabase access layer: feature store (Postgres) + model registry (Storage).

Handles the two things that silently break naive implementations:
  1. PostgREST caps responses at 1000 rows, so every read paginates.
  2. JSON has no NaN, so every write sanitises NaN/Inf to None.
"""
import io
import json
import math
import time
from typing import Any, Optional

import pandas as pd
from supabase import Client, ClientOptions, create_client

from src import config

_client: Optional[Client] = None
PAGE_SIZE = 1000


def get_client() -> Client:
    """Return a cached Supabase client, creating it on first use."""
    global _client
    if _client is None:
        config.validate_credentials()
        _client = create_client(
            config.SUPABASE_URL,
            config.SUPABASE_KEY,
            options=ClientOptions(
                postgrest_client_timeout=600,
                storage_client_timeout=600,
            ),
        )
    return _client


# --------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------
def _clean_value(value: Any) -> Any:
    """Convert a single value into something json.dumps can handle."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):          # numpy scalar
        value = value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    return value


def _to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records with an ISO timestamp column."""
    out = df.copy()
    if out.index.name == "timestamp":
        out = out.reset_index()
    if "timestamp" not in out.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column or index.")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = out["timestamp"].apply(lambda t: t.isoformat())
    return [
        {k: _clean_value(v) for k, v in row.items()}
        for row in out.to_dict(orient="records")
    ]


def _upsert(table: str, records: list[dict], chunk_size: int = 500,
            max_retries: int = 5) -> int:
    """Upsert records in chunks, keyed on timestamp.

    Each chunk is retried with exponential backoff. Because the upsert is
    idempotent, a retry after a partial failure is always safe: rows already
    written are simply overwritten with identical values.
    """
    client = get_client()
    written = 0
    total = len(records)

    for start in range(0, total, chunk_size):
        chunk = records[start : start + chunk_size]
        delay = 2.0
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                client.table(table).upsert(chunk, on_conflict="timestamp").execute()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    print(f"    chunk at {start} failed (attempt {attempt}): "
                          f"{type(exc).__name__}; retrying in {delay:.0f}s")
                    time.sleep(delay)
                    delay *= 2

        if last_error is not None:
            raise RuntimeError(
                f"Upsert failed at row {start} after {max_retries} attempts: "
                f"{last_error}"
            ) from last_error

        written += len(chunk)
        print(f"  uploaded {written}/{total} rows to {table}")

    return written


def _fetch_all(table: str, columns: str = "*") -> pd.DataFrame:
    """Read an entire table, paginating past the 1000-row PostgREST cap."""
    client = get_client()
    frames, offset = [], 0
    while True:
        response = (
            client.table(table)
            .select(columns)
            .order("timestamp", desc=False)
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


# --------------------------------------------------------------------------
# Raw observations
# --------------------------------------------------------------------------
def push_raw(df: pd.DataFrame) -> int:
    """Write raw weather + pollutant observations."""
    records = _to_records(df)
    print(f"Writing {len(records)} raw rows to {config.RAW_TABLE}...")
    return _upsert(config.RAW_TABLE, records)


def fetch_raw() -> pd.DataFrame:
    """Read every raw observation, oldest first, indexed by UTC timestamp."""
    df = _fetch_all(config.RAW_TABLE)
    if not df.empty and "ingested_at" in df.columns:
        df = df.drop(columns=["ingested_at"])
    return df


def latest_raw_timestamp() -> Optional[pd.Timestamp]:
    """Most recent timestamp present in raw_observations, or None if empty."""
    client = get_client()
    response = (
        client.table(config.RAW_TABLE)
        .select("timestamp")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    return pd.to_datetime(response.data[0]["timestamp"], utc=True)


def raw_row_count() -> int:
    """Number of rows currently in raw_observations."""
    client = get_client()
    response = (
        client.table(config.RAW_TABLE)
        .select("timestamp", count="exact")
        .limit(1)
        .execute()
    )
    return response.count or 0


# --------------------------------------------------------------------------
# Feature store
# --------------------------------------------------------------------------
def push_features(df: pd.DataFrame, feature_columns: list[str]) -> int:
    """Write engineered features as jsonb plus the three target columns."""
    work = df.copy()
    if work.index.name == "timestamp":
        work = work.reset_index()

    records = []
    for row in work.to_dict(orient="records"):
        payload = {c: _clean_value(row.get(c)) for c in feature_columns}
        records.append(
            {
                "timestamp": pd.to_datetime(row["timestamp"], utc=True).isoformat(),
                "features": payload,
                "aqi_t24": _clean_value(row.get("aqi_t24")),
                "aqi_t48": _clean_value(row.get("aqi_t48")),
                "aqi_t72": _clean_value(row.get("aqi_t72")),
            }
        )

    print(f"Writing {len(records)} feature rows to {config.FEATURE_TABLE}...")
    return _upsert(config.FEATURE_TABLE, records)


def fetch_features() -> pd.DataFrame:
    """Read the feature store back into a flat, model-ready DataFrame."""
    df = _fetch_all(config.FEATURE_TABLE)
    if df.empty:
        return df
    expanded = pd.json_normalize(df["features"])
    expanded.index = df.index
    for target in config.TARGET_COLUMNS.values():
        if target in df.columns:
            expanded[target] = df[target]
    return expanded.sort_index()


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
def upload_artifact(local_path: str, remote_name: str) -> None:
    """Upload a model artifact to the registry bucket, overwriting if present."""
    client = get_client()
    with open(local_path, "rb") as handle:
        client.storage.from_(config.MODEL_BUCKET).upload(
            path=remote_name,
            file=handle.read(),
            file_options={"cache-control": "3600", "upsert": "true"},
        )
    print(f"  uploaded {remote_name}")


def download_artifact(remote_name: str, local_path: str) -> Optional[str]:
    """Download an artifact. Returns the local path, or None if absent."""
    client = get_client()
    try:
        data = client.storage.from_(config.MODEL_BUCKET).download(remote_name)
    except Exception as exc:
        print(f"  could not download {remote_name}: {exc}")
        return None
    with open(local_path, "wb") as handle:
        handle.write(data)
    return local_path


def upload_json(obj: Any, remote_name: str) -> None:
    """Serialise an object to JSON and upload it to the registry."""
    client = get_client()
    payload = json.dumps(obj, indent=2, default=str).encode("utf-8")
    client.storage.from_(config.MODEL_BUCKET).upload(
        path=remote_name,
        file=payload,
        file_options={"content-type": "application/json", "upsert": "true"},
    )
    print(f"  uploaded {remote_name}")


def download_json(remote_name: str) -> Optional[Any]:
    """Download and parse a JSON artifact. Returns None if absent."""
    client = get_client()
    try:
        data = client.storage.from_(config.MODEL_BUCKET).download(remote_name)
    except Exception:
        return None
    return json.loads(data.decode("utf-8"))