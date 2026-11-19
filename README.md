# PV adoption forecasting

Monthly PV forecasts for 1,015 utility–ZIP pairs in PG&E and SDG&E, 2023–2025.

Requires Python 3.9–3.12.

```sh
pip install -r requirements.txt
python run.py          # Check saved forecasts and replay Bandit
python run.py --refit  # Refit from prepared training cases
```

Outputs are saved to `results/`. Prepared counts, training cases and fixed background paths are in `data/`; see [sources](data/sources.json).

Historical counts are revised records; the evaluation assumes year-end availability.
