"""Central configuration for the Pearls AQI Predictor.

Every constant the pipeline depends on lives here so that no magic numbers
are scattered across modules.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Location
# --------------------------------------------------------------------------
CITY = "Lahore"
COUNTRY = "Pakistan"
LATITUDE = 31.5204
LONGITUDE = 74.3587
TIMEZONE = "Asia/Karachi"          # UTC+5, no daylight saving

# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------
def _secret(name: str) -> str | None:
    """Resolve a credential from the environment, then Streamlit secrets.

    Locally and in GitHub Actions the value comes from the environment.
    On Streamlit Community Cloud there is no .env file, so it falls back to
    st.secrets. The import is guarded because streamlit is not installed in
    the training workflow.
    """
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


SUPABASE_URL = _secret("SUPABASE_URL")
SUPABASE_KEY = _secret("SUPABASE_KEY")
OPENWEATHER_API_KEY = _secret("OPENWEATHER_API_KEY")

# --------------------------------------------------------------------------
# Supabase objects
# --------------------------------------------------------------------------
RAW_TABLE = "raw_observations"
FEATURE_TABLE = "feature_store"
MODEL_BUCKET = "model-registry"

# --------------------------------------------------------------------------
# Open-Meteo endpoints (free, no API key, CC-BY 4.0)
# --------------------------------------------------------------------------
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# --------------------------------------------------------------------------
# Variables requested from each API
# --------------------------------------------------------------------------
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
    "boundary_layer_height",
]

POLLUTANT_VARS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]

AIR_QUALITY_VARS = POLLUTANT_VARS + ["us_aqi"]

# Every column stored in raw_observations, in table order
RAW_COLUMNS = ["timestamp"] + WEATHER_VARS + POLLUTANT_VARS + ["us_aqi"]

# --------------------------------------------------------------------------
# Backfill
# --------------------------------------------------------------------------
BACKFILL_YEARS = 2
CHUNK_DAYS = 180                   # request window per API call
ARCHIVE_LAG_DAYS = 5               # archive API trails real time by ~5 days

# --------------------------------------------------------------------------
# Forecast horizons (hours ahead)
# --------------------------------------------------------------------------
HORIZONS = [24, 48, 72]
TARGET_COLUMNS = {h: f"aqi_t{h}" for h in HORIZONS}

# --------------------------------------------------------------------------
# Validation gates — backfill aborts if these fail
# --------------------------------------------------------------------------
MAX_NULL_FRACTION = 0.05           # 5% nulls per column tolerated
MIN_EXPECTED_ROWS = 15000          # 2 years hourly, allowing for API gaps

# --------------------------------------------------------------------------
# EPA AQI categories: (lower, upper, label, hex colour)
# --------------------------------------------------------------------------
AQI_CATEGORIES = [
    (0, 50, "Good", "#00E400"),
    (51, 100, "Moderate", "#FFFF00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF7E00"),
    (151, 200, "Unhealthy", "#FF0000"),
    (201, 300, "Very Unhealthy", "#8F3F97"),
    (301, 500, "Hazardous", "#7E0023"),
]

ALERT_THRESHOLDS = {
    "sensitive": 101,
    "unhealthy": 151,
    "very_unhealthy": 201,
    "hazardous": 301,
}


def validate_credentials() -> None:
    """Fail loudly and early if the environment is not configured."""
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Check your .env file against .env.example."
        )
    if SUPABASE_URL.rstrip("/").endswith("/rest/v1"):
        raise EnvironmentError(
            "SUPABASE_URL must be the base project URL "
            "(https://xxxx.supabase.co) with no /rest/v1 suffix."
        )