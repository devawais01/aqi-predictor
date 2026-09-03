# Verification scripts

Run from the repository root with the virtual environment active.

| Script | Verifies |
|---|---|
| `test_connection.py` | Supabase credentials, write, read and delete round-trip |
| `test_aqi_calculator.py` | Our EPA implementation against Open-Meteo's independent `us_aqi`. Reports correlation 0.9908, median absolute difference 1.0 AQI |
| `test_features.py` | Builds all 70 features, runs the leakage audit, confirms `aqi_t24` equals the AQI actually observed 24 h later |
| `test_ridge.py` | The go/no-go gate: does Ridge on these features beat the persistence baseline at every horizon? |

```bash
python -m tests.test_connection
python -m tests.test_aqi_calculator
python -m tests.test_features
python -m tests.test_ridge
```

Results from these scripts are recorded in [`../reports/FINDINGS.md`](../reports/FINDINGS.md).