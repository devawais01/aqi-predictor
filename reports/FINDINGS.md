# Pearls AQI Predictor — Results Log

> Running record of every measured result. Written as runs complete so the
> final report is assembled from recorded numbers, not recalled ones.
> City: Lahore (31.5204, 74.3587). Period: 2024-08-22 to 2026-08-22.

---

## 1. Data Acquisition (Day 1)

| Metric | Value |
|---|---|
| Raw rows ingested | 17,544 |
| Period | 2024-08-22 00:00 UTC to 2026-08-22 23:00 UTC |
| Duplicate timestamps | 0 |
| Missing hours | 0 (17,544 of 17,544 expected) |
| Null fraction, every column | 0.00% |
| us_aqi mean / median | 155.2 / 154 |
| us_aqi min / max | 56 / 538 |

Sources: Open-Meteo ERA5 archive (weather), Open-Meteo air-quality API
(pollutants + us_aqi). Both free, no API key, CC-BY 4.0.

Feature store size: 25 MB. Raw table: 4.1 MB. Database total: 39 MB
(8% of the Supabase 500 MB free-tier cap).

---

## 2. EPA AQI Calculator Validation (Day 2)

Our EPA-breakpoint implementation vs Open-Meteo's independent `us_aqi`:

| Metric | Value |
|---|---|
| Correlation | **0.9908** |
| Mean difference | +0.3 |
| Median absolute difference | 1.0 |
| 90th percentile absolute difference | 9.0 |
| Within ±10 AQI | 91.2% |
| Within ±25 AQI | 98.3% |

**Mean sub-index by pollutant** (the unit-conversion check):

| Pollutant | Mean sub-index |
|---|---|
| PM2.5 | 150.3 |
| PM10 | 72.5 |
| O3 | 58.6 |
| NO2 | 20.0 |
| CO | **13.0** |
| SO2 | 8.4 |

CO at 13.0 is the evidence the µg/m³ → ppm conversion is correct. A failed
conversion would place CO near 500 and make it falsely dominant every hour.

Known divergence: our values clip at 500 (the formal EPA scale maximum);
Open-Meteo extrapolates beyond it, reaching 538. Expected, not a defect.

---

## 3. EDA Findings (Day 2)

Built on 17,376 feature rows (17,544 raw minus 168 hours of lag warm-up).

### 3.1 EPA category distribution

| Category | Hours | Share |
|---|---|---|
| Good (0–50) | **0** | 0.0% |
| Moderate (51–100) | 2,289 | 13.2% |
| Unhealthy for Sensitive Groups | 5,394 | 31.0% |
| Unhealthy | 7,204 | 41.5% |
| Very Unhealthy | 2,287 | 13.2% |
| Hazardous | 202 | 1.2% |

**55.8% of all hours were Unhealthy or worse. Not one hour in two years
reached the EPA "Good" band.**

### 3.2 Seasonality

| Month | Mean | Median | SD | Max |
|---|---|---|---|---|
| Jan | **219.7** | 215 | 51.2 | 339 |
| Feb | 159.9 | 163 | 32.7 | 236 |
| Mar | 118.1 | 114 | 29.1 | 209 |
| Apr | **109.7** | 101 | 33.0 | 213 |
| May | 147.8 | 145 | 59.3 | **538** |
| Jun | 149.6 | 151 | 39.3 | 397 |
| Jul | 148.7 | 152 | 37.4 | 364 |
| Aug | 134.8 | 140 | 30.2 | 229 |
| Sep | 128.9 | 127 | 30.1 | 210 |
| Oct | 155.4 | 157 | 22.7 | 204 |
| Nov | 192.4 | 189 | 41.6 | 368 |
| Dec | 197.0 | 192 | 40.4 | 308 |

**Seasonal swing: 110 AQI points (Jan 220 → Apr 110).**
This is the justification for a two-year backfill: a model trained on one
season has never observed the others.

Caveat: May's SD of 59.3 is inflated by a single spike to 538 (median 145).
Likely a dust event, not a persistent pattern.

### 3.3 Autocorrelation decay

| Lag | r |
|---|---|
| 1h | 0.992 |
| 3h | 0.949 |
| 6h | 0.888 |
| 12h | 0.822 |
| **24h** | **0.768** |
| **48h** | **0.615** |
| **72h** | **0.552** |
| 168h | 0.509 |

This is the mechanism behind persistence scoring R² = −0.002 at +72h.

### 3.4 Diurnal cycle

Peak 18:00 (173), trough 12:00 (151). **Range only 22 points** against a
110-point seasonal swing — Lahore's air is governed by season, not hour.

### 3.5 Correlation with us_aqi

| Variable | r |
|---|---|
| pm2_5 | +0.728 |
| sulphur_dioxide | +0.597 |
| pm10 | +0.561 |
| carbon_monoxide | +0.445 |
| **surface_pressure** | **+0.357** |
| nitrogen_dioxide | +0.310 |
| relative_humidity_2m | +0.226 |
| dust | +0.094 |
| cloud_cover | +0.007 |
| wind_direction_10m | −0.014 |
| ozone | −0.027 |
| precipitation | −0.077 |
| **boundary_layer_height** | **−0.131** |
| **wind_speed_10m** | **−0.229** |
| **temperature_2m** | **−0.361** |

The four bolded weather terms describe a classic winter temperature
inversion: cold, high-pressure, low-wind air with a shallow mixing layer
traps pollution near the surface.

### 3.6 Dominant pollutant

| Pollutant | Hours | Share |
|---|---|---|
| PM2.5 | 15,094 | 86.9% |
| O3 | 2,025 | 11.7% |
| PM10 | 257 | 1.5% |

Combustion, not dust, is Lahore's problem. Ozone's near-zero correlation
with AQI (−0.027) despite dominating 11.7% of hours is consistent: O3 peaks
on hot, sunny, well-mixed afternoons when particulates disperse.

Figures: `reports/figures/01` through `07`.

---

## 4. Feature Set (Day 2)

70 features, 3 targets, 17,376 rows.

| Group | Count |
|---|---|
| Weather (current) | 8 |
| Weather (future, perfect-prog) | 15 |
| Pollutants | 7 |
| AQI + EPA sub-indices | 8 |
| Lags | 14 |
| Rolling | 7 |
| Time (cyclic) | 7 |
| Derived | 4 |

Target availability: aqi_t24 17,352 / aqi_t48 17,328 / aqi_t72 17,304 —
exactly 24/48/72 short of 17,376, confirming the shifts are correct.

**Leakage audit passes**: 6 rolling means shifted, rolling std shifted,
aqi_change_rate shifted, 8 AQI lags aligned, future columns restricted to
perfect-prognosis weather only.

### 4.1 Feature correlation with each target

The horizon story — momentum gives way to regime:

| Rank | aqi_t24 | aqi_t48 | aqi_t72 |
|---|---|---|---|
| 1 | computed_aqi 0.774 | pm2_5_aqi 0.623 | **aqi_roll_168 0.587** |
| 2 | pm2_5_aqi 0.769 | computed_aqi 0.615 | pm2_5_aqi 0.556 |
| 3 | us_aqi 0.768 | us_aqi 0.615 | aqi_roll_72 0.552 |
| 4 | aqi_lag_1 0.754 | aqi_roll_168 0.608 | us_aqi 0.552 |
| 5 | pm2_5 0.739 | aqi_lag_1 0.607 | computed_aqi 0.548 |

At +24h current state dominates. By +72h the weekly rolling mean outranks
everything — short horizons are momentum, long horizons are regime.

---

## 5. Baseline Results (Day 2)

Test split: newest 3,476 rows, 2026-03-31 to 2026-08-22.

| Horizon | Baseline | RMSE | MAE | R² |
|---|---|---|---|---|
| +24h | **Persistence** | **33.54** | 20.84 | **0.305** |
| +24h | Rolling mean 24h | 37.53 | 25.47 | 0.130 |
| +24h | Seasonal naive | 39.39 | 25.67 | 0.041 |
| +48h | **Persistence** | **39.50** | 25.74 | **0.041** |
| +48h | Rolling mean 24h | 39.85 | 28.12 | 0.025 |
| +48h | Seasonal naive | 40.31 | 27.79 | 0.002 |
| +72h | **Persistence** | **40.43** | 27.90 | **−0.005** |
| +72h | Rolling mean 24h | 40.53 | 29.94 | −0.009 |
| +72h | Seasonal naive | 41.48 | 29.65 | −0.057 |

**At +72h, persistence scores below zero — worse than predicting the mean.**
"Tomorrow is like today" carries no information three days out in Lahore.

---

## 6. Model Results (Day 3)

Test split: 3,461 rows, 2026-03-28 to 2026-08-19.
Train: 13,843 rows, 2024-08-29 to 2026-03-28. Chronological, never shuffled.

### +24h (persistence: RMSE 33.40, R² 0.309)

| Model | RMSE | MAE | R² | vs baseline |
|---|---|---|---|---|
| **Random Forest** | **24.92** | 18.00 | **0.615** | +25.4% |
| Ridge | 25.16 | 18.23 | 0.608 | +24.7% |
| XGBoost | 25.35 | 17.65 | 0.602 | +24.1% |
| LSTM | 29.52 | — | 0.460 | +11.6% |
| SARIMAX | 33.15 | — | 0.222 | +0.7% |

### +48h (persistence: RMSE 39.42, R² 0.039)

| Model | RMSE | MAE | R² | vs baseline |
|---|---|---|---|---|
| **Random Forest** | **32.58** | 23.73 | **0.343** | +17.3% |
| Ridge | 32.73 | 22.70 | 0.337 | +17.0% |
| XGBoost | 34.09 | 24.82 | 0.281 | +13.5% |

### +72h (persistence: RMSE 40.25, R² −0.002)

| Model | RMSE | MAE | R² | vs baseline |
|---|---|---|---|---|
| **Random Forest** | **33.34** | 24.92 | **0.313** | +17.2% |
| Ridge | 33.43 | 23.55 | 0.309 | +17.0% |
| XGBoost | 33.95 | 25.17 | 0.287 | +15.6% |

### 6.1 Walk-forward CV, +24h (5 expanding folds)

| Fold | Test period | Ridge R² | RF R² | XGB R² |
|---|---|---|---|---|
| 1 | Jun–Sep 2025 | 0.216 | 0.284 | 0.299 |
| 2 | Sep–Dec 2025 | **0.799** | 0.760 | 0.766 |
| 3 | Dec–Feb 2026 | 0.771 | 0.661 | 0.693 |
| 4 | Feb–May 2026 | 0.544 | 0.564 | 0.579 |
| 5 | May–Aug 2026 | 0.416 | 0.404 | 0.420 |
| **Mean** | | **0.549** | **0.535** | **0.551** |

**Counter-intuitive and important: summer is the hardest season, not winter.**
R² measures explained variance. Summer AQI is flat (Jun–Aug 135–150, narrow
spread) so there is little variance to explain. Autumn and winter have large
weather-driven swings that the model predicts well from temperature,
pressure and boundary-layer height.

Two consequences:
1. The headline single-split R² of 0.615 is measured on the *hardest*
   window (Mar–Aug), so it is conservative rather than flattered.
2. **Selection criterion changes the winner.** Random Forest wins the single
   split (0.615); XGBoost wins on mean CV R² (0.551 vs 0.535).

### 6.2 Model notes

- **SARIMAX** decays 0.222 → −0.006 → −0.054 across horizons, tracking the
  raw autocorrelation. As a univariate model it cannot see PM2.5, wind, or
  forecast weather. The ~0.37 R² gap to Random Forest at +72h is a direct
  measurement of what the exogenous features contribute.
- **LSTM** at R² 0.460 uses a genuine 24-timestep lookback,
  shape `(n, 24, 70)`. The reference project reshaped to `(n, 1, features)`
  — sequence length 1, no temporal memory — and scored R² = −7.07.
- **Ridge within 0.3 RMSE of Random Forest at every horizon.** A linear
  model nearly matching the ensemble means the signal is largely linear and
  the performance comes from feature engineering, not model complexity.

### 6.3 Top Ridge coefficients (absolute)

| Horizon | Top 5 |
|---|---|
| +24h | pm2_5, temperature_2m, temperature_2m_t48, temperature_2m_t24, relative_humidity_2m |
| +48h | temperature_2m_t48, temperature_2m, relative_humidity_2m, relative_humidity_2m_t48, aqi_roll_72 |
| +72h | temperature_2m, temperature_2m_t72, relative_humidity_2m_t72, aqi_roll_72, us_aqi |

**Future weather features rank first at +48h and +72h.** This is direct
evidence the perfect-prognosis design earns its place.

### 6.4 Runtime

| Model | Fit time |
|---|---|
| Ridge | 0.1s |
| XGBoost | ~7s |
| Random Forest | ~12s |
| LSTM | ~60s |
| SARIMAX (rolling origin) | ~137s |

Artifacts: best_t24.pkl 5.05 MB, best_t48.pkl 5.17 MB, best_t72.pkl 5.23 MB.

---

## 7. Open Items

- [ ] Full run with registry upload (blocked by storage RLS, now fixed)
- [ ] SHAP on the winning model, 500+ sample rows
- [ ] Dashboard, API, CI/CD, final report

---

# PART II — Full Run Results and Interpretation

## 8. Final Model Results (Day 3, complete run)

Registry upload succeeded. All artifacts in Supabase `model-registry`.

### 8.1 Single-split results, all models

**+24h** (persistence RMSE 33.40, R² 0.309)

| Model | RMSE | MAE | R² | vs base | Mean CV R² |
|---|---|---|---|---|---|
| **Random Forest** | **24.92** | 18.00 | **0.615** | +25.4% | 0.534 |
| Ridge | 25.16 | 18.23 | 0.608 | +24.7% | 0.549 |
| XGBoost | 25.35 | 17.65 | 0.602 | +24.1% | **0.551** |
| LSTM | 28.51 | 19.85 | 0.496 | +14.6% | n/a |
| SARIMAX | 33.15 | 21.13 | 0.222 | +0.8% | n/a |

**+48h** (persistence RMSE 39.42, R² 0.039)

| Model | RMSE | MAE | R² | vs base | Mean CV R² |
|---|---|---|---|---|---|
| **LSTM** | **31.66** | 22.59 | **0.382** | +19.7% | n/a |
| Random Forest | 32.58 | 23.73 | 0.343 | +17.3% | 0.126 |
| Ridge | 32.73 | 22.70 | 0.337 | +17.0% | **0.242** |
| XGBoost | 34.09 | 24.82 | 0.281 | +13.5% | 0.046 |
| SARIMAX | 41.08 | 26.89 | −0.195 | −4.2% | n/a |

**+72h** (persistence RMSE 40.25, R² −0.002)

| Model | RMSE | MAE | R² | vs base | Mean CV R² |
|---|---|---|---|---|---|
| **LSTM** | **32.54** | 24.39 | **0.348** | +19.2% | n/a |
| Random Forest | 33.34 | 24.92 | 0.313 | +17.2% | **−0.045** |
| Ridge | 33.43 | 23.55 | 0.309 | +17.0% | **+0.166** |
| XGBoost | 33.95 | 25.17 | 0.287 | +15.6% | **−0.069** |
| SARIMAX | 42.06 | 29.01 | −0.251 | −4.5% | n/a |

### 8.2 Walk-forward CV, +48h

| Fold | Test period | Ridge | RF | XGB |
|---|---|---|---|---|
| 1 | Jun–Sep 2025 | −0.135 | −0.351 | −0.630 |
| 2 | Sep–Dec 2025 | 0.612 | 0.377 | 0.375 |
| 3 | Dec–Feb 2026 | 0.582 | 0.369 | 0.377 |
| 4 | Feb–May 2026 | 0.207 | 0.219 | 0.075 |
| 5 | May–Aug 2026 | −0.057 | 0.017 | 0.033 |
| **Mean** | | **0.242** | 0.126 | 0.046 |

### 8.3 Walk-forward CV, +72h

| Fold | Test period | Ridge | RF | XGB |
|---|---|---|---|---|
| 1 | Jun–Sep 2025 | −0.075 | **−0.945** | **−1.050** |
| 2 | Sep–Dec 2025 | 0.435 | 0.240 | 0.239 |
| 3 | Dec–Feb 2026 | 0.492 | 0.351 | 0.456 |
| 4 | Feb–May 2026 | 0.063 | 0.070 | −0.042 |
| 5 | May–Aug 2026 | −0.087 | 0.059 | 0.053 |
| **Mean** | | **+0.166** | **−0.045** | **−0.069** |

### 8.4 THE CENTRAL FINDING: single-split selection is misleading

At +72h the tree models have **negative mean CV R²** — averaged across five
seasonal folds they are worse than predicting the mean — while Ridge stays
positive at +0.166. Yet on the single 80/20 split, Random Forest (0.313)
appears to beat Ridge (0.309).

The failure is concentrated in fold 1 (Jun–Sep 2025): RF −0.945,
XGB −1.050, Ridge −0.075. Fold 1 has the smallest training window
(~9.5 months) and tests on flat summer data. With 70 heavily collinear
features the trees memorise the training regime and extrapolate badly;
Ridge's L2 penalty forces it to degrade gracefully instead.

**Implication for deployment: at +48h and +72h, Ridge is the more
defensible production model despite losing the single split.** Robustness
across seasons matters more than a point estimate from one window.

### 8.5 A limitation of our own selection procedure

Walk-forward CV was run only for the scikit-learn models. LSTM and SARIMAX
were scored on the single split alone. LSTM therefore won +48h and +72h on
the *least* reliable criterion available. Its CV behaviour is unmeasured and
should not be assumed good. Stated openly as a limitation; with more time,
CV would be extended to all five models.

### 8.6 Selected models per horizon

| Horizon | Selected (by test RMSE) | Artifact |
|---|---|---|
| +24h | Random Forest | best_t24.pkl, 5.05 MB |
| +48h | LSTM | best_t48.keras + 2 scalers, 0.61 MB |
| +72h | LSTM | best_t72.keras + 2 scalers, 0.61 MB |

---

## 9. Interpretation and Discussion (report narrative)

### 9.1 Why persistence is the only metric that matters

Current AQI (`us_aqi[t]`) is a feature in every model. A model can therefore
score respectably by echoing its input. Reporting R² without comparing to
persistence is meaningless.

The reference project trained on `shift(-1)` — one hour ahead — while
retaining `aqi_change_rate` and `aqi_rolling_24h`, both of which contain
`aqi[t]`. Its R² of 0.85 measured autocorrelation, not forecasting skill.
Our design uses `shift(-24/-48/-72)`, where current state is legitimately
known at prediction time and genuinely informative rather than circular.

### 9.2 Why current AQI is a feature and not leakage

At prediction time t, `us_aqi[t]`, `pm2_5[t]` and their lags are real,
observed, available values. Using them to forecast t+24 is exactly what a
forecaster does. Leakage would be using information unavailable at t.

Every rolling and difference feature is `.shift(1)`ed so its window excludes
t itself. Since `us_aqi[t]` is present as its own column this costs no
information, and it keeps the pipeline safe if the horizon is ever shortened.
The leakage audit asserts this programmatically on every build.

### 9.3 Why summer is harder than winter (counter-intuitive)

R² measures explained variance. Summer AQI in Lahore is flat — June through
August averages 135–150 with a narrow spread — so there is little variance
to explain and what remains is largely noise. Autumn and winter carry large,
weather-driven swings (November smog onset, January inversions) that are
genuinely predictable from temperature, pressure and boundary-layer height.

Consequence: the headline single-split R² of 0.615 is measured on the
*hardest* window (Mar–Aug), so it is conservative rather than flattered.

### 9.4 Why Random Forest beats XGBoost on the single split

RF wins by 0.4–1.5 RMSE at every horizon. With 70 features and heavy
multicollinearity among the lags, bagging's variance reduction does more
work here than boosting's bias reduction. Under CV the ordering reverses at
+24h (XGB 0.551 vs RF 0.534), which is further evidence that the two models'
relative merit depends on the evaluation window.

### 9.5 Why Ridge nearly matching the ensembles is significant

Ridge is within 0.3 RMSE of Random Forest at every horizon. A linear model
essentially matching an ensemble means the signal in these features is close
to linear, and that **the performance comes from feature engineering, not
model complexity.** This is the opposite of the usual "we applied gradient
boosting" narrative and is a stronger result: it is reproducible, cheap to
serve (0.1s fit time), and interpretable.

### 9.6 Why SARIMAX loses, and why that is informative

SARIMAX decays 0.222 → −0.195 → −0.251 across horizons, tracking the raw
autocorrelation (0.768 → 0.615 → 0.552). As a univariate model it can only
extrapolate the AQI series; it has no access to PM2.5, wind speed, or
forecast weather. The ~0.5 R² gap to the ML models at +72h is therefore a
direct measurement of what the exogenous feature set contributes.

Implementation note: SARIMAX is evaluated with a **rolling forecast origin**
— at each point the Kalman state is filtered through all observations up to
that time, then forecasts exactly h steps ahead. A single long-range
forecast over 3,461 hours converges to the series mean and scores
meaninglessly badly (our first attempt gave R² = −8.7, an artefact of the
evaluation protocol, not the model).

### 9.7 The LSTM, and the reshape mistake

Our LSTM uses input shape `(n, 24, 70)` — a genuine 24-hour lookback window.
It scored R² 0.496 / 0.382 / 0.348.

The reference project reshaped to `(n, 1, features)`. A sequence length of
one makes an LSTM a dense layer with extra machinery and no temporal memory
at all; it scored R² = −7.07. The distinction is worth stating because it is
a common and invisible error.

### 9.8 Perfect prognosis: the honest caveat

Weather at time t cannot explain air quality three days later; weather at
t+72 can. Open-Meteo publishes free forecasts for +24/48/72h.

- **Training** uses *actual* archived weather at t+h.
- **Inference** substitutes the *live forecast* for those same variables.

This is the standard perfect-prognosis method. There is a mild train/serve
mismatch: real forecasts carry error that archived actuals do not, so live
performance will be modestly worse than reported test performance. We state
this rather than conceal it.

Evidence the design works: the top Ridge coefficients at +48h and +72h are
`temperature_2m_t48`, `relative_humidity_2m_t72` and `temperature_2m_t72` —
future weather ranks first at exactly the horizons where current state has
decayed.

### 9.9 Ozone's apparent contradiction

Ozone correlates −0.027 with AQI yet is the dominant pollutant in 11.7% of
hours. Not a contradiction: O3 is anti-correlated with PM2.5 because it
peaks on hot, sunny, well-mixed afternoons when particulates disperse. On
the rare hours when PM2.5 falls far enough, ozone becomes the maximum
sub-index by default.

### 9.10 The horizon story

| Horizon | Strongest predictor | Interpretation |
|---|---|---|
| +24h | `computed_aqi`, `pm2_5` (r ≈ 0.77) | momentum |
| +48h | `pm2_5_aqi`, `aqi_roll_168` | transition |
| +72h | `aqi_roll_168` (r = 0.587) | regime |

Short-horizon forecasting is about momentum; long-horizon is about regime.
The weekly rolling mean outranks the most recent observation at +72h.

---

## 10. Architecture Decisions

### 10.1 Supabase instead of Hopsworks or Vertex AI

The brief names Hopsworks or Vertex AI. We used Supabase Postgres as the
feature store and Supabase Storage as the model registry, because:

- Hopsworks' free tier has reported limits that risk expiring mid-project.
- Vertex AI requires a GCP billing account even for free credits.
- Postgres gives direct SQL access, which made EDA and validation queries
  trivial.
- Storage buckets serve as a registry with versioned, addressable artifacts.

Usage: 25 MB feature store, 4.1 MB raw table, 39 MB total — 8% of the
500 MB free-tier cap.

### 10.2 Schema choice: jsonb for features

The 70 feature values are stored in a single `jsonb` column rather than 70
typed columns, so that changing the feature specification requires no
`ALTER TABLE`. Targets are stored as typed columns because they are queried
and filtered directly.

### 10.3 Row Level Security disabled

RLS is off on both tables, and the storage bucket has open insert/update
policies. For public environmental data on a student project this is an
acceptable trade; enabling RLS without policies would silently block every
write. In production, reads would be public and writes restricted to the
service role. Recorded as a limitation, not an oversight.

### 10.4 Retry logic

Both the Open-Meteo fetcher and the Supabase upsert use exponential backoff
(up to 4 and 5 attempts respectively). The upsert is keyed on `timestamp`
and therefore idempotent, so a retry after partial failure is always safe.
This proved necessary: three separate long uploads were interrupted by
network drops during development.

---

## 11. Data Quality Gates

Enforced before any write to Supabase; the backfill aborts on failure.

| Gate | Threshold | Actual |
|---|---|---|
| Row count | ≥ 15,000 | 17,544 |
| Duplicate timestamps | 0 | 0 |
| Missing hours | logged | 0 |
| Null fraction per column | ≤ 5% | 0.00% |
| Negative AQI values | 0 | 0 |
| Index monotonic ascending | required | yes |

---

## 12. Open Items

- [ ] SHAP on the winning model, 500+ sample rows
- [ ] FastAPI service
- [ ] Streamlit dashboard with alerts
- [ ] GitHub Actions: hourly feature, daily training
- [ ] Final report assembly

---

# PART III — Final Model Selection

## 13. CV-Aware Selection (supersedes §8.6)

Selection was changed from lowest single-split RMSE to highest mean
walk-forward CV R2, with RMSE as fallback for models lacking CV. The
criteria disagreed at **all three horizons**.

| Horizon | Lowest RMSE | Best CV R2 | **Selected** |
|---|---|---|---|
| +24h | Random Forest (24.92, CV 0.534) | XGBoost (25.35, CV 0.551) | **XGBoost** |
| +48h | Random Forest (32.58, CV 0.126) | Ridge (32.73, CV 0.242) | **Ridge** |
| +72h | Random Forest (33.34, CV **−0.045**) | Ridge (33.43, CV **+0.166**) | **Ridge** |

At +72h Random Forest wins the single split by 0.09 RMSE while scoring a
**negative** mean CV R2 — across five seasonal folds it is on average worse
than predicting the mean. Ridge loses the split by that same 0.09 and stays
positive throughout. Trading 0.09 RMSE for seasonal robustness is an easy
decision, and it is only visible because CV was run.

### 13.1 Artifact sizes after reselection

| Horizon | Model | Size |
|---|---|---|
| +24h | XGBoost | 0.53 MB |
| +48h | Ridge | < 0.01 MB |
| +72h | Ridge | < 0.01 MB |

Ridge serialises to a scaler plus 70 coefficients. The more robust model is
also roughly 10,000x smaller than the Random Forest it replaced, which
matters for the 1 GB memory ceiling on Streamlit Community Cloud.

## 14. Why fold 1 fails: a training-data-volume result

Fold boundaries, +72h target:

| Fold | Trains through | Tests | History available |
|---|---|---|---|
| 1 | 2025-06-13 | Jun–Sep 2025 | ~9.5 months |
| 2 | 2025-09-07 | Sep–Dec 2025 | ~12 months |
| 3 | 2025-12-03 | Dec–Feb 2026 | ~15 months |
| 4 | 2026-02-27 | Feb–May 2026 | ~18 months |
| 5 | 2026-05-25 | May–Aug 2026 | ~21 months |

Fold 1 has seen one monsoon and **zero complete winters**. Tree R2 at +72h:
−0.945 (fold 1) → 0.240 (fold 2) → 0.351 (fold 3). Performance stabilises
once a full annual cycle is in the training window.

**This independently validates the two-year backfill decision.** A project
using 30 or 90 days of data would be permanently in fold 1's regime.

## 15. LSTM run-to-run instability

The LSTM produced materially different scores across two runs on identical
data with the same seed:

| Horizon | Run 1 R2 | Run 2 R2 |
|---|---|---|
| +24h | 0.496 | 0.522 |
| +48h | 0.382 | 0.310 |
| +72h | 0.348 | 0.153 |

Sources: oneDNN's non-deterministic floating-point operation ordering, and
early stopping halting at different epochs. A swing of 0.20 R2 at +72h means
a single LSTM measurement is not a reliable basis for model selection — a
further argument for the CV-based criterion, and a limitation to state
plainly rather than hide by reporting only the favourable run.

## 16. Final production configuration

| Horizon | Model | Test RMSE | Test R2 | CV R2 | vs persistence |
|---|---|---|---|---|---|
| +24h | XGBoost | 25.35 | 0.602 | 0.551 | +24.1% |
| +48h | Ridge | 32.73 | 0.337 | 0.242 | +17.0% |
| +72h | Ridge | 33.43 | 0.309 | 0.166 | +17.0% |

All three beat persistence. At +72h the baseline scores R2 = −0.002 while
the deployed model reaches 0.309 on the test split and remains positive
across every seasonal fold on average.


---

# PART IV — SHAP Explainability

Computed on **1,000 test rows** (2026-07-09 to 2026-08-19), not a single
observation. TreeExplainer for XGBoost, LinearExplainer for Ridge — both
exact methods for their model class.

## 17. Feature-group contribution by horizon

Percentage of total mean |SHAP| attributable to each design group:

| Group | +24h | +48h | +72h |
|---|---|---|---|
| EPA sub-indices | **32.4%** | 14.9% | 18.4% |
| Current pollutants | 19.9% | 15.2% | 14.3% |
| **Future weather (perfect prog)** | **13.4%** | **24.1%** | **25.2%** |
| AQI lags | 13.4% | 13.5% | 11.3% |
| Current weather | 8.1% | 14.0% | 12.2% |
| AQI rolling | 5.3% | 11.9% | 13.1% |
| Time (cyclic) | 5.9% | 4.9% | 4.4% |
| Derived | 1.6% | 1.5% | 1.0% |

### 17.1 The headline result

Current pollutant state (sub-indices + pollutants combined) falls from
**52.3% at +24h to 32.7% at +72h**, while future weather rises from
**13.4% to 25.2%** — it nearly doubles and becomes the largest single group
at both longer horizons.

This is the perfect-prognosis design measured rather than asserted. Without
the forecast-weather features the +48h and +72h models would lose roughly a
quarter of their explanatory basis, which is consistent with SARIMAX (which
has no exogenous inputs) scoring R2 = −0.251 at +72h.

## 18. Top features by horizon

**+24h (XGBoost)** — dominated by present conditions:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | us_aqi | 9.82 |
| 2 | pm2_5 | 8.13 |
| 3 | computed_aqi | 5.36 |
| 4 | pm2_5_aqi | 4.57 |
| 5 | surface_pressure | 2.59 |
| 6 | pm10 | 2.50 |
| 7 | temperature_2m_t24 | 2.26 |

**+48h (Ridge)** — future weather takes the top slot:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | **temperature_2m_t48** | 12.68 |
| 2 | relative_humidity_2m | 11.81 |
| 3 | **relative_humidity_2m_t48** | 10.04 |
| 4 | temperature_2m | 9.34 |
| 5 | aqi_roll_72 | 7.11 |

**+72h (Ridge)** — a weather forecast is the single most important input:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | **relative_humidity_2m_t72** | 11.93 |
| 2 | pm10 | 10.92 |
| 3 | us_aqi | 9.88 |
| 4 | relative_humidity_2m | 9.71 |
| 5 | temperature_2m | 9.66 |
| 6 | dust | 9.39 |
| 7 | aqi_roll_72 | 9.19 |
| 8 | **temperature_2m_t72** | 9.17 |

At +72h the top-ranked feature is a *forecast* value, not an observation.

## 19. Physical interpretation

The SHAP rankings match the correlation structure found in EDA and are
physically coherent:

- **Humidity and temperature at the target hour** dominate long horizons.
  Cold, humid, stagnant conditions are exactly the winter-inversion regime
  identified in §3.5 (temperature r = −0.361, pressure r = +0.357).
- **`surface_pressure` ranks 5th at +24h** despite modest raw correlation —
  high pressure signals stagnation and suppressed dispersion.
- **`aqi_roll_72` and `aqi_roll_168` rise with horizon** (5.3% → 13.1% of
  total importance), confirming §4.1: short horizons are momentum, long
  horizons are regime.
- **`dust` ranks 6th at +72h** but is negligible at +24h. Dust intrusions
  are synoptic-scale and develop over days, so they carry more information
  about conditions three days out than one day out.

## 20. Limitation of this analysis

The 1,000-row sample covers 2026-07-09 to 2026-08-19 — six weeks of summer,
chosen as the most recent contiguous block of the test set. Feature
importance in January, when AQI averages 220 and PM2.5 dominates 87% of
hours, would likely weight current pollutants more heavily and future
weather less. A seasonally stratified SHAP analysis is future work.

Figures: `08_shap_beeswarm_t{24,48,72}.png`,
`09_shap_bar_t{24,48,72}.png`. Machine-readable summary in
`models/shap_summary.json`, also uploaded to the Supabase registry.

---

# PART V — Feature Store Verification and Final Selection

## 21. Training verified against the Supabase feature store

All earlier runs used `--from-csv` for speed. The pipeline was re-run reading
from the Supabase `feature_store` table (17,376 rows, 70 features from the
jsonb column). Results matched the CSV path to within run-to-run noise,
confirming the two paths are equivalent and that the guideline requirement
— training reads from the feature store — is met.

## 22. Final selection (CV-based, from the feature store)

| Horizon | Selected | RMSE | R2 | CV R2 | Rejected alternative |
|---|---|---|---|---|---|
| +24h | **XGBoost** | 25.72 | 0.590 | **0.561** | RF: RMSE 24.90, CV 0.541 |
| +48h | **Ridge** | 32.73 | 0.337 | **0.242** | LSTM: RMSE 26.64, CV n/a |
| +72h | **Ridge** | 33.43 | 0.309 | **+0.166** | RF: CV −0.001, XGB: CV −0.047 |

At +48h the LSTM produced the lowest RMSE of any model (26.64, R2 0.563) and
was still rejected, because it has no cross-validation score and its
single-run results are not stable (§23).

A guard was added so that `--skip-cv` combined with CV-based selection now
raises rather than silently falling back to RMSE — which on one occasion
selected Random Forest at +48h, a model with CV R2 0.156 against Ridge's
0.242.

## 23. LSTM run-to-run instability (quantified)

The same LSTM, same data, same seed, at the +48h horizon across four runs:

| Run | R2 | RMSE |
|---|---|---|
| 1 | 0.382 | 31.66 |
| 2 | 0.310 | 33.47 |
| 3 | 0.007 | 40.15 |
| 4 | 0.563 | 26.64 |

**Range: 0.556 R2.** Causes:

1. **oneDNN non-determinism.** TensorFlow parallelises floating-point
   operations across threads; summation order varies between runs, and small
   numerical differences accumulate over 30 epochs. `tf.random.set_seed`
   fixes initialisation but not this.
2. **Early stopping amplification.** With `patience=5` and
   `restore_best_weights=True`, a small numerical difference shifts which
   epoch records the best validation loss, so genuinely different weights
   are restored.

Mitigation applied: `TF_ENABLE_ONEDNN_OPTS=0` and
`tf.config.experimental.enable_op_determinism()`, which trade some speed for
reproducibility.

**This instability is itself a result.** A model whose measured performance
varies by 0.556 R2 across identical runs cannot be selected on a single
measurement. It is the strongest argument in this project for cross-
validated selection, and the reason the LSTM was not deployed at any horizon
despite winning one single-split comparison.

Full mitigation — training 5–10 seeds per horizon and averaging, plus
extending walk-forward CV to the LSTM — was scoped at roughly 30 minutes per
horizon and deferred as future work. The deep-learning requirement is
satisfied by a correctly-implemented sequence model
(input shape `(n, 24, 70)`); it is simply not the selected model.

## 24. Live inference verified

`python -m src.models.predict` output, 2026-08-28 09:00 UTC:


Reading latest features from Supabase feature store...
17376 rows, latest 2026-08-22 23:00 (130.2h old)
stale (> 3h); falling back to live computation

Current: AQI 160 (Unhealthy), dominant O3
+24h AQI 155.1 Unhealthy [XGBoost]
+48h AQI 177.2 Unhealthy [Ridge]
+72h AQI 159.8 Unhealthy [Ridge]



Behaviour confirmed:

- Feature store is the **primary** source; the staleness check triggered
  correctly because the hourly GitHub Action does not yet exist.
- Fallback to live computation is reported in the `feature_source` field
  rather than hidden.
- Future-weather columns are filled from the live Open-Meteo forecast
  (perfect prognosis), since archived actuals for t+24/48/72 do not exist.
- Feature names and **order** are read from the training metadata, not
  recomputed at inference — a mismatch here caused a hard failure that was
  caught and fixed.

Every value shown is the direct output of a model trained for that specific
horizon. Nothing is interpolated, tiled, noise-injected or bias-anchored.

---

# PART VI — Serving Layer

## 25. FastAPI service

Eight endpoints, all verified returning HTTP 200 against a live Supabase
backend on 2026-08-28.

| Endpoint | Purpose | Verified response |
|---|---|---|
| `GET /` | Service metadata, endpoint index | 200 |
| `GET /health` | Liveness + dependency checks | `status: healthy`, `raw_rows: 17544`, `missing_horizons: []` |
| `GET /predict` | Live 3-day forecast | 200 |
| `GET /current` | Current conditions only | AQI 160, Unhealthy, dominant O3 |
| `GET /historical` | Recent observations, 24–720h | 500h returned, 41,220 bytes |
| `GET /metrics` | Model performance from registry | 200 |
| `GET /models` | Selected model per horizon + rationale | 598 bytes |
| `GET /alerts` | Active health alerts | 4 alerts, max severity 2 |

Interactive OpenAPI documentation is auto-generated at `/docs`.

### 25.1 `/health` — dependency verification

```json
{
  "status": "healthy",
  "checks": {
    "supabase": { "ok": true, "raw_rows": 17544 },
    "models": { "ok": true, "missing_horizons": [] }
  }
}
```

The health check queries Supabase directly and confirms a model artifact
exists for every horizon, so a degraded deployment is detectable rather
than failing silently at request time.

### 25.2 `/models` — selection transparency

```json
{
  "selected": {
    "+24h": { "model": "XGBoost", "n_features": 70 },
    "+48h": { "model": "Ridge",   "n_features": 70 },
    "+72h": { "model": "Ridge",   "n_features": 70 }
  },
  "selection_criterion": "Highest mean walk-forward CV R2 across 5 expanding folds..."
}
```

The API states *why* each model was chosen, not merely which. Selection
logic is inspectable by any consumer.

### 25.3 `/alerts` — health guidance

Live response, 2026-08-28 09:37 UTC:

| When | Valid at (local) | AQI | Level | Severity |
|---|---|---|---|---|
| now | 2026-08-28 14:00 | 160.0 | unhealthy | 2 |
| +24h | 2026-08-29 14:00 | 155.1 | unhealthy | 2 |
| +48h | 2026-08-30 14:00 | 177.2 | unhealthy | 2 |
| +72h | 2026-08-31 14:00 | 159.8 | unhealthy | 2 |

Four active alerts, maximum severity 2. Tiers follow EPA breakpoints:
101 sensitive, 151 unhealthy, 201 very unhealthy, 301 hazardous.

### 25.4 Design notes

- **15-minute response cache.** Upstream data is hourly; recomputing a
  forecast per request would waste Open-Meteo quota and add latency for no
  benefit. `/metrics` uses a 1-hour TTL.
- **CORS enabled, GET only.** Allows the Streamlit dashboard to call the
  API directly from the browser while restricting write methods.
- **Errors return 503 with detail**, not a stack trace, so an upstream
  outage is distinguishable from a bug.

### 25.5 Live cross-check of the AQI calculator

`/current` at 14:00 local reported `us_aqi` 160 against our
`computed_aqi` of 164 — a 4-point difference, consistent with the 0.9908
correlation measured in §2.

The dominant pollutant was **O3**, not PM2.5. This is not an error: §3.6
established that ozone dominates 11.7% of hours, concentrated on hot sunny
afternoons when photochemical production peaks and particulates disperse.
A mid-afternoon August reading is exactly when this is expected, and it
confirms the dominant-pollutant logic responds to conditions rather than
returning a constant.

---

# PART VII — Automation and End-to-End Verification

## 26. GitHub Actions workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| `feature_pipeline.yml` | `5 * * * *` (hourly) | Fetch observations, upsert raw, rebuild feature tail |
| `training_pipeline.yml` | `30 0 * * *` (daily) | Retrain all models, select by CV, update registry |

Both support `workflow_dispatch` for manual runs. `concurrency` groups
prevent overlapping executions from causing duplicate upserts. The feature
workflow runs at five past the hour rather than on the hour, because
GitHub's scheduler is heavily contended at `:00` and jobs are delayed.

### 26.1 First manual run — verified success

Run 33164957197, 2026-08-28 10:52 UTC, duration **46 seconds**:

Fetching recent observations from Open-Meteo...
179 rows, 2026-08-21 00:00 -> 2026-08-28 10:00
uploaded 179/179 rows to raw_observations

Reading raw history for the rebuild window...
rebuilding from 504 rows (2026-08-07 11:00 -> 2026-08-28 10:00)

--- Leakage audit ---
6 rolling means shifted: OK
rolling std shifted: OK
aqi_change_rate shifted: OK
8 AQI lags aligned: OK
future columns limited to perfect-prog weather: OK
Audit passed.

70 feature columns, 3 targets, 336 rows
uploaded 336/336 rows to feature_store

raw_observations rows: 17675
latest raw timestamp : 2026-08-28 10:00:00+00:00


Row count rose from 17,544 to **17,675** — 131 genuinely new hourly
observations ingested and engineered without manual intervention.

### 26.2 The leakage audit runs in CI

`audit_leakage()` executes inside the hourly workflow, so a future change
that breaks the `.shift(1)` discipline fails the pipeline rather than
silently writing compromised features. This is the difference between a
one-off check and an enforced invariant.

## 27. End-to-end verification

With the hourly workflow running, `predict.py` uses the **primary** path:

Reading latest features from Supabase feature store...
17507 rows, latest 2026-08-28 10:00:00+00:00 (1.0h old)
Fetching live weather forecast...
120 forecast hours, to 2026-09-01 23:00:00+00:00

Current: AQI 170 (Unhealthy), dominant O3
+24h AQI 170.7 Unhealthy [XGBoost]
+48h AQI 182.3 Unhealthy [Ridge]
+72h AQI 165.8 Unhealthy [Ridge]


No staleness warning, no fallback. The complete chain is operational:

**GitHub Actions (hourly) → Supabase feature store → model registry →
inference → FastAPI / Streamlit.**

## 28. A bug only the full pipeline could expose

Switching from the fallback path to the feature-store path immediately
raised `KeyError: 'dominant_pollutant'`.

**Cause.** `dominant_pollutant` is a string. `feature_columns()` filters to
numeric dtypes, so the column was silently excluded from the jsonb payload
written to the feature store. The live fallback path computed it fresh and
therefore had it; the store path did not. The two paths were producing
subtly different frames, and until the hourly workflow made the store
current, the store path was never exercised.

**Fix.** Recompute `dominant_pollutant` at inference from the EPA sub-index
columns (`pm2_5_aqi`, `ozone_aqi`, …), which are numeric and therefore *are*
persisted, via `argmax` over the sub-indices.

**Lesson for the report.** A dual-path design — primary store plus live
fallback — improves availability but creates a class of bug where the less-
travelled path silently diverges. The fallback was exercised on every run
during development precisely because the store was stale, which masked the
defect in the path that matters. Divergence between paths should be tested
directly, not discovered when the primary path finally activates.