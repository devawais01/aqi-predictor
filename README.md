# Pearls AQI Predictor

**Three-day Air Quality Index forecasting for Lahore, Pakistan — on a fully serverless stack.**

[![Live Dashboard](https://img.shields.io/badge/dashboard-live-brightgreen)](https://lahore-aqi-forecast.streamlit.app)
[![Feature Pipeline](https://github.com/devawais01/aqi-predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/devawais01/aqi-predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/devawais01/aqi-predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/devawais01/aqi-predictor/actions/workflows/training_pipeline.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Data: Open-Meteo](https://img.shields.io/badge/data-Open--Meteo%20CC--BY%204.0-orange)](https://open-meteo.com)

**Live app → [lahore-aqi-forecast.streamlit.app](https://lahore-aqi-forecast.streamlit.app)**

> If the dashboard shows a "Zzzz — this app has gone to sleep" screen, click **"Yes, get this app back up!"**. Streamlit Community Cloud sleeps free apps after ~12 hours without traffic; it wakes in about 30 seconds. The hourly feature pipeline pings the URL to keep it warm, but a long gap between scheduled runs can still let it sleep.

---

## Contents

- [What this is](#what-this-is)
- [Results](#results)
- [Architecture](#architecture)
- [Why the numbers look the way they do](#why-the-numbers-look-the-way-they-do)
- [Key design decisions](#key-design-decisions)
- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [API reference](#api-reference)
- [Automation](#automation)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Attribution](#attribution)

---

## What this is

An end-to-end machine learning system that forecasts the **US Air Quality Index for Lahore at +24h, +48h and +72h**. It collects data hourly, retrains daily, and serves predictions through a public dashboard and a REST API — all without a server, and entirely within free service tiers.

Lahore's air is persistently hazardous. Across 17,376 hourly observations spanning two years, the AQI **never once** reached the EPA "Good" band (0–50). The minimum recorded value was 56, and 55.8% of all hours were "Unhealthy" or worse.

**What's in the box**

- Automated hourly ingestion from Open-Meteo into a Supabase feature store
- Independent cross-validation against OpenWeather, scored with the same EPA implementation
- Two years of backfilled history (17,675 raw observations)
- 70 engineered features with a **programmatically enforced leakage audit**
- Five model families — persistence, SARIMAX, Ridge, Random Forest, XGBoost, LSTM
- Model selection by **walk-forward cross-validation**, not a single split
- Versioned model registry in Supabase Storage
- SHAP explanations computed over 1,000 test rows
- FastAPI service with 8 endpoints and auto-generated OpenAPI docs
- Streamlit dashboard with live forecasts, EDA, model telemetry and health alerts

---

## Results

Trained on 17,507 rows. Chronological 80/20 split — never `train_test_split()`.

| Horizon | Deployed model | RMSE | MAE | R² | Baseline R² | vs baseline | CV R² |
|---------|----------------|------|-----|-----|-------------|-------------|-------|
| **+24h** | XGBoost | 25.24 | 17.70 | **0.604** | 0.310 | **+24.3%** | 0.557 |
| **+48h** | Ridge Regression | 32.47 | 22.29 | **0.342** | 0.043 | **+17.1%** | 0.245 |
| **+72h** | Ridge Regression | 33.17 | 23.12 | **0.309** | −0.001 | **+16.9%** | 0.190 |

**Every model beats the persistence baseline at every horizon.**

### Why the baseline is the only metric that matters

Current AQI is one of the model inputs. A model can therefore score respectably by simply echoing it. The persistence baseline — *"three days from now will look like right now"* — isolates that effect.

At **+72h the baseline scores R² = −0.001**: worse than predicting the long-run mean. Three days out, the assumption that conditions persist carries no usable information about Lahore's air. The deployed model reaches 0.309. **That entire margin comes from feature engineering.**

### Full model comparison

<details>
<summary>Click to expand all models at all horizons</summary>

**+24h** — persistence: RMSE 33.33, R² 0.310

| Model | RMSE | MAE | R² | vs base | CV R² |
|---|---|---|---|---|---|
| Ridge | 25.11 | 18.16 | 0.608 | +24.6% | 0.536 |
| Random Forest | 25.17 | 18.09 | 0.606 | +24.5% | 0.534 |
| **XGBoost** ✅ | 25.24 | 17.70 | 0.604 | +24.3% | **0.557** |
| LSTM | 30.54 | 20.35 | 0.419 | +8.4% | — |

**+48h** — persistence: RMSE 39.17, R² 0.043

| Model | RMSE | MAE | R² | vs base | CV R² |
|---|---|---|---|---|---|
| LSTM | 31.59 | 21.94 | 0.375 | +19.4% | — |
| Random Forest | 32.15 | 23.35 | 0.355 | +17.9% | 0.171 |
| **Ridge** ✅ | 32.47 | 22.29 | 0.342 | +17.1% | **0.245** |
| XGBoost | 33.30 | 23.96 | 0.308 | +15.0% | 0.073 |

**+72h** — persistence: RMSE 39.92, R² −0.001

| Model | RMSE | MAE | R² | vs base | CV R² |
|---|---|---|---|---|---|
| Random Forest | 32.53 | 24.11 | 0.336 | +18.5% | 0.097 |
| XGBoost | 32.87 | 23.81 | 0.322 | +17.7% | 0.040 |
| **Ridge** ✅ | 33.17 | 23.12 | 0.309 | +16.9% | **0.190** |
| LSTM | 35.00 | 25.30 | 0.231 | +12.3% | — |

SARIMAX (evaluated separately with a rolling forecast origin): R² 0.222 / −0.195 / −0.251.

</details>

---

## Architecture

```
Open-Meteo APIs  (weather archive · air quality · live forecast)
        │
        ▼
GitHub Actions — feature_pipeline.yml          [hourly, cron 5 * * * *]
        │   fetch → validate → engineer → LEAKAGE AUDIT → store
        ▼
Supabase Postgres
   ├── raw_observations   17,675 rows · immutable observation log
   └── feature_store      17,507 rows × 70 features (jsonb) + 3 targets
        │
        ▼
GitHub Actions — training_pipeline.yml         [daily, cron 30 0 * * *]
        │   5 models × 3 horizons → walk-forward CV → select → SHAP
        ▼
Supabase Storage  "model-registry"
   └── best_t{24,48,72}.pkl · meta_t*.json · metrics.json · shap_summary.json
        │
        ├──────────────────────────┐
        ▼                          ▼
   FastAPI service          Streamlit dashboard
   8 REST endpoints         5 tabs · public URL
```

| Layer | Technology | Why |
|-------|-----------|-----|
| Data | Open-Meteo archive + air-quality APIs | Free, no API key, hourly, decades of archive |
| Cross-validation | OpenWeather Air Pollution API | Independent second source for current conditions |
| Feature store | Supabase Postgres (jsonb) | Stable free tier; direct SQL for validation and EDA |
| Model registry | Supabase Storage | Versioned artifacts, same credentials |
| Orchestration | GitHub Actions | Unlimited minutes on public repos, no server |
| Training | scikit-learn · XGBoost · statsmodels · TensorFlow | Statistical → deep learning |
| API | FastAPI + Uvicorn | Auto-generated OpenAPI docs |
| Dashboard | Streamlit + Plotly | Free public hosting |

Total storage: **39 MB** against the 500 MB free-tier ceiling (8%).

---

## Why the numbers look the way they do

**R² of 0.60 / 0.34 / 0.31 is modest in absolute terms — and that's the honest result.**

Three things frame it:

1. **The information ceiling is measurable.** AQI autocorrelation decays from r = 0.992 at 1 hour to **0.768 / 0.615 / 0.552** at the three forecast horizons. There's a hard limit on what any model can extract from recent history.

2. **These figures match published operational systems**, which typically report R² in the 0.3–0.6 range at 24–72h.

3. **A high R² here would be suspicious, not impressive.** Forecasting one hour ahead while retaining features that contain the current value yields R² ≈ 0.85 trivially — but that measures autocorrelation, not skill.

### Counter-intuitive finding: summer is harder than winter

Walk-forward CV at +24h:

| Fold | Test period | Ridge | RF | XGBoost |
|---|---|---|---|---|
| 1 | Jun–Sep 2025 | 0.216 | 0.313 | 0.362 |
| 2 | Sep–Dec 2025 | **0.799** | 0.764 | 0.765 |
| 3 | Dec–Feb 2026 | 0.771 | 0.663 | 0.686 |
| 4 | Feb–May 2026 | 0.544 | 0.572 | 0.588 |
| 5 | May–Aug 2026 | 0.416 | 0.395 | 0.405 |

R² measures explained *variance*. Summer AQI is flat (June–August: 135–150, narrow spread), so there's little variance to explain. Autumn and winter carry large weather-driven swings that **are** predictable from temperature, pressure and boundary-layer height.

Consequence: the headline 0.604 is measured on the **hardest** window (late March–August), making it conservative rather than flattering.

---

## Key design decisions

### 1. Direct multi-horizon, not recursive

Three independent targets, three independent model sets:

```python
aqi_t24 = us_aqi.shift(-24)
aqi_t48 = us_aqi.shift(-48)
aqi_t72 = us_aqi.shift(-72)
```

No recursive forecasting. No fabricated future pollutant values. **Every number displayed is the direct output of a model trained for that specific horizon.**

### 2. Current AQI is a feature — and this is not leakage

At prediction time *t*, `us_aqi[t]` and `pm2_5[t]` are real, published, observable values. Using them to forecast *t+24* is what forecasting *is*.

Leakage means using information **unavailable** at *t*. Every rolling and differencing feature is explicitly `.shift(1)`-ed:

```python
df['aqi_roll_24']     = df['us_aqi'].rolling(24).mean().shift(1)
df['aqi_change_rate'] = df['us_aqi'].diff().shift(1)
```

### 3. The leakage audit is enforced, not intended

`audit_leakage()` runs inside `build_features()` and raises on any violation, blocking the write. **It also runs inside the hourly GitHub Action** — a future change that breaks the shift discipline fails the pipeline rather than silently poisoning the feature store.

```
--- Leakage audit ---
  6 rolling means shifted: OK
  rolling std shifted: OK
  aqi_change_rate shifted: OK
  8 AQI lags aligned: OK
  future columns limited to perfect-prog weather: OK
  Audit passed.
```

### 4. Perfect prognosis — forecasting with future weather

Weather at time *t* cannot explain air quality three days later. Weather at *t+72* can.

- **Training** uses *actual* archived weather at t+24/48/72
- **Inference** substitutes the *live Open-Meteo forecast*

SHAP quantifies the payoff — future weather rises from **13.4%** of total importance at +24h to **25.2%** at +72h, becoming the largest single feature group:

| Feature group | +24h | +48h | +72h |
|---|---|---|---|
| EPA sub-indices | **32.4%** | 15.0% | 18.4% |
| Current pollutants | 19.9% | 15.3% | 14.3% |
| **Future weather (perfect prog)** | **13.4%** | **24.1%** | **25.2%** |
| AQI lags | 13.4% | 13.5% | 11.3% |
| Current weather | 8.1% | 13.9% | 12.2% |
| AQI rolling | 5.3% | 11.8% | 13.2% |
| Time (cyclic) | 5.9% | 4.8% | 4.4% |
| Derived | 1.6% | 1.5% | 1.0% |

At +72h the single highest-ranked feature is `relative_humidity_2m_t72` — a **forecast**, not an observation.

> ⚠️ **Honest caveat:** real forecasts carry error that archived actuals do not, so live accuracy will be modestly below reported test accuracy. This train/serve mismatch is inherent to perfect prognosis and is documented rather than hidden.

### 5. Selection by cross-validated robustness

Models are selected by **highest mean walk-forward CV R²**, not lowest single-split RMSE. The two criteria **disagreed at all three horizons**:

| Horizon | Lowest RMSE | Best CV R² | Selected |
|---|---|---|---|
| +24h | Ridge (25.11, CV 0.536) | XGBoost (25.24, CV 0.557) | **XGBoost** |
| +48h | LSTM (31.59, no CV) | Ridge (32.47, CV 0.245) | **Ridge** |
| +72h | Random Forest (32.53, CV 0.097) | Ridge (33.17, CV 0.190) | **Ridge** |

In an earlier run at +72h, Random Forest won the single split while scoring a **negative** mean CV R² (−0.001) — worse than the mean, averaged across five seasonal folds. Ridge lost that split by 0.05 RMSE and stayed positive throughout. That trade is only visible because CV was run.

### 6. Two years of data, not ninety days

Lahore's seasonal swing is **110 AQI points** (January 220 → April 110).

Fold-level results prove this empirically. Fold 1 trains on ~9.5 months — one monsoon, **zero complete winters** — and Random Forest scores **R² = −0.945** at +72h. By fold 3, with a full annual cycle in the training window, the same model reaches 0.351.

| Fold | Training history | RF R² @ +72h |
|---|---|---|
| 1 | ~9.5 months | **−0.945** |
| 2 | ~12 months | 0.240 |
| 3 | ~15 months | 0.351 |

### 6b. Independent cross-validation against a second provider

Open-Meteo supplies every value the models use. OpenWeather is queried separately for the same location and hour, and **its concentrations are scored with the same EPA implementation** — so any difference reflects the measurements, not the index definition.

| Source | PM2.5 | PM10 | AQI | Basis |
|---|---|---|---|---|
| Open-Meteo (primary) | 51.3 µg/m³ | 75.7 | 152 | 24-hour rolling mean |
| OpenWeather (independent) | 48.1 µg/m³ | 113.5 | 132 | instantaneous |

**Agreement is judged on concentration, not on AQI.** The two index values rest on different averaging windows: EPA defines the PM2.5 sub-index against a 24-hour rolling mean, which Open-Meteo applies, whereas OpenWeather returns a single instantaneous reading. Comparing them directly would overstate the disagreement.

On the fair comparison the sources agree to within **3.2 µg/m³, about 6%** — good agreement between independent sensor networks. The 20-point AQI gap is arithmetic, not disagreement: the index climbs roughly 2.5 points per µg/m³ in this band, so a 3 µg/m³ difference alone accounts for about 8 points, and the averaging-window mismatch supplies the rest.

This is the same trap described in the AQI computation section below, encountered from the opposite direction.

### 7. EPA AQI computed from first principles

Open-Meteo returns `us_aqi` directly (used as the target), but the index is also computed independently to derive the **dominant pollutant** and to verify correctness.

Two traps that silently corrupt results:

| Pollutant | EPA unit | Averaging window | MW |
|---|---|---|---|
| PM2.5 | µg/m³ | 24-hour | — |
| PM10 | µg/m³ | 24-hour | — |
| CO | **ppm** | 8-hour | 28.01 |
| O₃ | **ppm** | 8-hour | 48.00 |
| SO₂ | ppb | 1-hour | 64.06 |
| NO₂ | ppb | 1-hour | 46.01 |

Conversion at 25 °C / 1 atm: `ppb = µg/m³ × 24.45 / MW`.

**Verification:** correlation **0.9908** with Open-Meteo's independent implementation, median absolute difference 1.0 AQI, 91.2% within ±10.

The mean CO sub-index of **13.0** is the proof the unit conversion is right. Skip it and that column reads near 500, making CO falsely dominant every hour — with no error message.

---

## Quickstart

### Prerequisites

- Python 3.11+
- A free [Supabase](https://supabase.com) project

### 1. Clone and install

```bash
git clone https://github.com/devawais01/aqi-predictor
cd aqi-predictor

python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1

pip install -r requirements-train.txt
```

`requirements.txt` holds runtime dependencies only (no TensorFlow, to stay under Streamlit Cloud's ~1 GB memory ceiling). `requirements-train.txt` adds the full training stack.

### 2. Set up Supabase

Run in the SQL Editor:

```sql
create table if not exists raw_observations (
    timestamp timestamptz primary key,
    temperature_2m double precision,
    relative_humidity_2m double precision,
    surface_pressure double precision,
    wind_speed_10m double precision,
    wind_direction_10m double precision,
    cloud_cover double precision,
    precipitation double precision,
    boundary_layer_height double precision,
    pm2_5 double precision,
    pm10 double precision,
    carbon_monoxide double precision,
    nitrogen_dioxide double precision,
    sulphur_dioxide double precision,
    ozone double precision,
    dust double precision,
    us_aqi double precision,
    ingested_at timestamptz default now()
);

create table if not exists feature_store (
    timestamp timestamptz primary key,
    features jsonb not null,
    aqi_t24 double precision,
    aqi_t48 double precision,
    aqi_t72 double precision,
    created_at timestamptz default now()
);

create index if not exists idx_raw_ts on raw_observations (timestamp desc);
create index if not exists idx_fs_ts  on feature_store    (timestamp desc);
```

Then create a **public** storage bucket named `model-registry`.

### 3. Configure credentials

```bash
cp .env.example .env
```

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-publishable-key
```

### 4. Run the pipeline

```bash
# Verify connectivity
python test_connection.py

# Backfill 2 years of raw observations (~90 s)
python -m src.features.backfill --save-csv

# Build all features and write to the feature store
python -m src.features.feature_pipeline --mode full

# Train every model at every horizon, select by CV, upload artifacts
python -m src.models.train_model

# Compute SHAP attribution over 1,000 test rows
python -m src.models.explain

# Generate a live forecast
python -m src.models.predict
```

### 5. Serve

```bash
uvicorn src.api.main:app --reload --port 8000    # → http://127.0.0.1:8000/docs
streamlit run src/app/dashboard.py               # → http://localhost:8501
```

### Useful flags

```bash
python -m src.models.train_model --skip-sarimax   # skip the slow rolling-origin fit
python -m src.models.train_model --skip-lstm      # skip TensorFlow entirely
python -m src.models.train_model --from-csv       # local CSV instead of Supabase
python -m src.models.train_model --no-upload      # don't touch the registry
python -m src.features.backfill --dry-run         # validate without writing
```

---

## Project structure

```
aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml        # hourly · cron 5 * * * *
│   └── training_pipeline.yml       # daily  · cron 30 0 * * *
├── notebooks/
│   ├── 01_eda.ipynb                # executed EDA notebook
│   └── eda.py                      # jupytext source
├── reports/
│   ├── FINDINGS.md                 # running results log
│   └── figures/                    # 13 generated figures
├── src/
│   ├── config.py                   # constants, credentials, thresholds
│   ├── utils/
│   │   ├── db_client.py            # feature store + model registry
│   │   └── aqi_calculator.py       # EPA breakpoints, units, dominant pollutant
│   ├── features/
│   │   ├── fetch_data.py           # Open-Meteo clients with backoff
│   │   ├── build_features.py       # 70 features + leakage audit
│   │   ├── backfill.py             # 2-year historical load
│   │   └── feature_pipeline.py     # full and incremental modes
│   ├── models/
│   │   ├── baseline.py             # persistence + naive baselines
│   │   ├── evaluate.py             # metrics, splits, walk-forward CV
│   │   ├── train_model.py          # 5 models × 3 horizons
│   │   ├── explain.py              # SHAP over 1,000 rows
│   │   └── predict.py              # live inference, perfect prognosis
│   ├── api/main.py                 # FastAPI, 8 endpoints
│   └── app/dashboard.py            # Streamlit, 5 tabs
├── requirements.txt                # runtime only
└── requirements-train.txt          # full training stack
```

---

## API reference

Base URL when running locally: `http://127.0.0.1:8000`

| Endpoint | Description |
|----------|-------------|
| `GET /` | Service metadata and endpoint index |
| `GET /health` | Liveness plus per-dependency checks |
| `GET /predict` | Live three-day forecast with alerts |
| `GET /current` | Current observed conditions |
| `GET /historical?hours=168` | Recent observations (24–720 hours) |
| `GET /metrics` | Model performance from the registry |
| `GET /models` | Selected model per horizon **and the selection rationale** |
| `GET /alerts` | Active health alerts across the forecast window |
| `GET /crosscheck` | Compare current conditions against an independent second provider |
| `GET /docs` | Interactive OpenAPI documentation |

<details>
<summary>Sample response — <code>GET /predict</code></summary>

```json
{
  "city": "Lahore",
  "observation_time_local": "2026-08-28T15:00:00+05:00",
  "feature_source": "feature_store",
  "current": {
    "aqi": 170.0,
    "category": "Unhealthy",
    "dominant_pollutant": "O3",
    "pm2_5": 60.3,
    "pm10": 121.5,
    "alert": { "level": "unhealthy", "severity": 2 }
  },
  "forecast": [
    { "horizon_hours": 24, "aqi": 170.7, "category": "Unhealthy", "model": "XGBoost" },
    { "horizon_hours": 48, "aqi": 182.3, "category": "Unhealthy", "model": "Ridge" },
    { "horizon_hours": 72, "aqi": 165.8, "category": "Unhealthy", "model": "Ridge" }
  ],
  "peak_forecast_aqi": 182.3
}
```

</details>

### Health alert tiers

| Tier | AQI | Guidance |
|---|---|---|
| Sensitive groups | 101+ | Children, elderly and those with respiratory or cardiac conditions should reduce outdoor exertion |
| Unhealthy | 151+ | Everyone may experience effects; limit outdoor activity |
| Very unhealthy | 201+ | Avoid all outdoor exertion; use air purification indoors |
| Hazardous | 301+ | Health emergency; everyone should remain indoors |

---

## Automation

| Workflow | Schedule | Does |
|---|---|---|
| `feature_pipeline.yml` | Hourly, `5 * * * *` | Fetch → upsert raw → rebuild feature window → **leakage audit** → write to store |
| `training_pipeline.yml` | Daily, `30 0 * * *` | Retrain all models → walk-forward CV → select → upload artifacts → recompute SHAP |

Both support `workflow_dispatch` for manual runs. Concurrency groups prevent overlapping executions.

The feature workflow runs at **:05 past the hour**, not on the hour — GitHub's scheduler is heavily contended at `:00` and jobs get delayed.

### Scheduler reliability in practice

GitHub documents scheduled workflows as **best-effort** — they may be delayed or dropped under load, and high-frequency crons sit in the most contended tier. Observed over a 17-hour window:

| Run | Trigger | Time (UTC) | Rows written |
|---|---|---|---|
| #1 | Manual | 2026-08-28 10:52 | 131 |
| #2 | **Scheduled** | 2026-08-28 21:19 | 11 |
| #3 | **Scheduled** | 2026-08-29 03:33 | 6 |

Two scheduled runs, not seventeen.

**This produces no data gaps.** Each run fetches a **seven-day trailing window** and upserts on `timestamp`, so a missed hour is backfilled by the next successful run. Run #3 wrote 6 rows covering hours run #2 hadn't seen. `raw_observations` grew from 17,675 to 17,692 with no intervention, and the dashboard stayed on its primary path (`feature_source: feature_store`).

The idempotent-upsert design was adopted for network resilience during development; it turned out to also provide tolerance to scheduler unreliability. A pipeline ingesting only the most recent hour would have left permanent holes.

**Mitigation** — a coarser fallback schedule on each workflow, since GitHub schedules less-frequent crons more dependably:

| Workflow | Primary | Fallback |
|---|---|---|
| `feature_pipeline.yml` | `5 * * * *` | `25 */3 * * *` |
| `training_pipeline.yml` | `30 0 * * *` | `45 12 * * *` |

**Required GitHub Secrets:** `SUPABASE_URL`, `SUPABASE_KEY`

Verified run — 46 seconds, 131 new hourly observations ingested and engineered without intervention:

```
Fetching recent observations from Open-Meteo...
  179 rows, 2026-08-21 00:00 -> 2026-08-28 10:00
  uploaded 179/179 rows to raw_observations
--- Leakage audit ---  Audit passed.
  70 feature columns, 3 targets, 336 rows
  uploaded 336/336 rows to feature_store
raw_observations rows: 17675
```

---

## Deployment

### Streamlit Community Cloud

1. Push to a **public** GitHub repository
2. At [share.streamlit.io](https://share.streamlit.io) → **Create app**
3. Main file path: `src/app/dashboard.py`
4. Advanced settings → Python **3.11**
5. Secrets (TOML format — quotes required):

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-publishable-key"
```

> **Note:** `requirements.txt` deliberately excludes TensorFlow. Streamlit Cloud allocates ~1 GB of memory, and importing TensorFlow consumes a large share before any data loads. The deployed models are XGBoost and Ridge; neither needs it.

---

## Limitations

Stated plainly, because a report that lists only successes is less useful than one that states its boundaries.

| # | Limitation | Detail |
|---|---|---|
| 1 | **Train/serve mismatch** | Perfect prognosis trains on archived actuals but serves on live forecasts. Live accuracy will be modestly below reported test accuracy. Magnitude unmeasured. |
| 2 | **Seasonal variance** | Walk-forward CV R² ranges from −0.14 to 0.80 across folds. A single headline number conceals this. |
| 3 | **Reanalysis ≠ ground truth** | Open-Meteo serves gap-filled reanalysis, not raw station readings. The system learns to predict the reanalysis product. Validation against Punjab EPA stations would strengthen this. |
| 4 | **Asymmetric evaluation** | Walk-forward CV was implemented only for scikit-learn models. SARIMAX and LSTM were scored on the single split alone. |
| 5 | **LSTM instability** | Same data, same seed, four runs → R² of 0.382, 0.310, 0.007, 0.563 at +48h. Range **0.556**. Caused by oneDNN non-determinism plus early-stopping amplification. Not deployed at any horizon. |
| 6 | **Security posture** | RLS disabled, storage bucket has open write policies. Acceptable for public environmental data on a student project; production would restrict writes to the service role. |
| 7 | **Single location** | One coordinate pair. Lahore has considerable spatial variation in air quality. |
| 8 | **Computed AQI clips at 500** | Open-Meteo extrapolates beyond the formal EPA scale maximum (reaching 538 in one event). |

### Future work

- Validate against Punjab EPA ground stations
- Log live predictions and score them retrospectively to quantify the perfect-prognosis mismatch
- Extend walk-forward CV to SARIMAX and the LSTM
- Multi-seed averaging for the LSTM
- Prediction intervals via quantile regression
- Seasonally stratified SHAP analysis
- Spatial extension to multiple monitoring points

---

## Attribution

Weather and air quality data by **[Open-Meteo](https://open-meteo.com)**, licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). The free tier is for non-commercial use; this is an unpaid academic internship project.

AQI methodology follows US EPA technical guidance, including the PM2.5 breakpoints revised in May 2024.

---

## Author

**Muhammad Awais**
University of Central Punjab, Lahore Campus — Data Sciences
10Pearls Internship Programme · September 2026

- Live dashboard — [lahore-aqi-forecast.streamlit.app](https://lahore-aqi-forecast.streamlit.app)
- Full report — [`reports/`](reports/)
- Results log — [`reports/FINDINGS.md`](reports/FINDINGS.md)
