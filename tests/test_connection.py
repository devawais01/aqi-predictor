"""One-off sanity check: can we reach Supabase and write to it?"""
import pandas as pd

from src.utils import db_client

print("1. Building client...")
db_client.get_client()
print("   OK")

print("2. Counting existing raw rows...")
print(f"   {db_client.raw_row_count()} rows")

print("3. Writing a test row...")
probe = pd.DataFrame(
    [{"timestamp": pd.Timestamp("2000-01-01T00:00:00Z"), "us_aqi": 42.0}]
)
db_client.push_raw(probe)

print("4. Reading it back...")
found = db_client.fetch_raw()
print(f"   {len(found)} rows, latest = {db_client.latest_raw_timestamp()}")

print("5. Deleting the test row...")
db_client.get_client().table("raw_observations").delete().eq(
    "timestamp", "2000-01-01T00:00:00+00:00"
).execute()
print(f"   {db_client.raw_row_count()} rows remain")

print("\nConnection OK.")