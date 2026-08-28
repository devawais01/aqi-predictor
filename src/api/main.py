"""FastAPI service for the Pearls AQI Predictor.

Endpoints:
    GET /                 service metadata
    GET /health           liveness + dependency checks
    GET /predict          live 3-day forecast
    GET /current          current conditions only
    GET /historical       recent observed AQI
    GET /metrics          model performance from the registry
    GET /models           which model serves each horizon
    GET /alerts           active alerts over the forecast window

Run locally:
    uvicorn src.api.main:app --reload --port 8000
    open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.models import predict as predictor
from src.utils import db_client

app = FastAPI(
    title="Pearls AQI Predictor",
    description=(
        "Three-day Air Quality Index forecasting for Lahore, Pakistan. "
        "Direct multi-horizon models (+24h, +48h, +72h) trained on two years "
        "of hourly Open-Meteo data. Weather data by Open-Meteo (CC-BY 4.0)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Short-lived cache: the upstream data is hourly, so recomputing a forecast
# on every request wastes API quota and adds latency for no benefit.
_cache: dict = {}
CACHE_SECONDS = 900


def _cached(key: str, producer, ttl: int = CACHE_SECONDS):
    """Return a cached value, recomputing when it has expired."""
    now = datetime.now(timezone.utc).timestamp()
    entry = _cache.get(key)
    if entry and now - entry["at"] < ttl:
        return entry["value"]
    value = producer()
    _cache[key] = {"at": now, "value": value}
    return value


@app.get("/")
def root() -> dict:
    """Service metadata and endpoint index."""
    return {
        "service": "Pearls AQI Predictor",
        "version": "1.0.0",
        "city": config.CITY,
        "coordinates": {"lat": config.LATITUDE, "lon": config.LONGITUDE},
        "horizons_hours": config.HORIZONS,
        "endpoints": [
            "/health", "/predict", "/current", "/historical",
            "/metrics", "/models", "/alerts", "/docs",
        ],
        "data_source": "Open-Meteo (CC-BY 4.0)",
    }


@app.get("/health")
def health() -> dict:
    """Liveness plus a check on each downstream dependency."""
    checks = {}

    try:
        rows = db_client.raw_row_count()
        checks["supabase"] = {"ok": True, "raw_rows": rows}
    except Exception as exc:
        checks["supabase"] = {"ok": False, "error": str(exc)[:200]}

    missing = [
        h for h in config.HORIZONS
        if not os.path.exists(f"models/meta_t{h}.json")
    ]
    checks["models"] = {"ok": not missing, "missing_horizons": missing}

    healthy = all(c.get("ok") for c in checks.values())
    return {
        "status": "healthy" if healthy else "degraded",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }

@app.get("/predict")
def get_forecast() -> dict:
    """Live three-day AQI forecast.

    Each horizon is the direct output of a model trained for that horizon.
    Future-weather features are filled from the live Open-Meteo forecast
    (perfect prognosis).
    """
    try:
        return _cached("forecast", predictor.forecast)
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"Forecast unavailable: {exc}")


@app.get("/current")
def get_current() -> dict:
    """Current observed conditions, without the forecast."""
    try:
        result = _cached("forecast", predictor.forecast)
        return {
            "observation_time_local": result["observation_time_local"],
            "city": result["city"],
            **result["current"],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/historical")
def get_historical(
    hours: int = Query(168, ge=24, le=720,
                       description="Hours of history to return"),
) -> dict:
    """Recent observed AQI and pollutant values."""
    try:
        frame = _cached(f"history_{hours}",
                        lambda: predictor.recent_history(hours))
        records = []
        for index, row in frame.iterrows():
            value = row["us_aqi"]
            records.append({
                "timestamp": index.isoformat(),
                "us_aqi": None if value != value else round(float(value), 1),
                "pm2_5": round(float(row["pm2_5"]), 1),
                "pm10": round(float(row["pm10"]), 1),
            })
        return {"hours": len(records), "observations": records}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/metrics")
def get_metrics() -> dict:
    """Model performance from the registry, including the baseline comparison."""
    def load():
        payload = db_client.download_json("metrics.json")
        if payload is None and os.path.exists("models/metrics.json"):
            with open("models/metrics.json") as handle:
                payload = json.load(handle)
        return payload

    payload = _cached("metrics", load, ttl=3600)
    if payload is None:
        raise HTTPException(status_code=404, detail="No metrics available.")
    return payload


@app.get("/models")
def get_models() -> dict:
    """Which model serves each horizon, and why it was selected."""
    out = {}
    for horizon in config.HORIZONS:
        path = f"models/meta_t{horizon}.json"
        if os.path.exists(path):
            with open(path) as handle:
                meta = json.load(handle)
            out[f"+{horizon}h"] = {
                "model": meta["model"],
                "n_features": meta["n_features"],
                "trained_at": meta["trained_at"],
            }
    return {
        "selected": out,
        "selection_criterion": (
            "Highest mean walk-forward CV R2 across 5 expanding folds. A "
            "single 80/20 split on two years of seasonal data places the "
            "whole test set in one season and misleads: at +72h Random "
            "Forest wins the split (R2 0.307) but scores a negative mean "
            "CV R2 (-0.001), while Ridge stays positive (+0.166)."
        ),
    }


@app.get("/alerts")
def get_alerts() -> dict:
    """Active health alerts across the current forecast window."""
    try:
        result = _cached("forecast", predictor.forecast)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    alerts = []
    current = result["current"]
    if current["alert"]["severity"] > 0:
        alerts.append({
            "when": "now",
            "aqi": current["aqi"],
            **current["alert"],
        })
    for entry in result["forecast"]:
        if entry["alert"]["severity"] > 0:
            alerts.append({
                "when": f"+{entry['horizon_hours']}h",
                "valid_at_local": entry["valid_at_local"],
                "aqi": entry["aqi"],
                **entry["alert"],
            })

    return {
        "city": config.CITY,
        "generated_at": result["generated_at_utc"],
        "alert_count": len(alerts),
        "max_severity": max([a["severity"] for a in alerts], default=0),
        "alerts": alerts,
        "thresholds": config.ALERT_THRESHOLDS,
    }