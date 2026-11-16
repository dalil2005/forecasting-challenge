# Cloud Workload Forecasting: Solution

Solution to the Atomity AI/ML take-home on a curated subset of the Microsoft Azure Public Dataset V2 (10,000 VMs, 134 five-minute readings each).

**Task:** forecast `active_vm_count` and `aggregate_cpu_utilization` for the final 24 timestamps (two hours), using only the first 110 timestamps.

**Full write-up:** [`reports/final_report.md`](reports/final_report.md)

## Results in brief

- `active_vm_count` is exactly 10,000 at every timestamp, because the dataset only keeps VMs with complete traces. The pipeline detects a constant training series and predicts that constant.
- `aggregate_cpu_utilization` is forecast with an AR(1) process that decays from the last value toward the training mean.
- Rolling-origin validation on the training window only (27 origins, 24 steps ahead):

| Model | MAE | RMSE | MAE vs last value |
|---|---|---|---|
| last value | 0.2168 | 0.2709 | n/a |
| training mean | 0.1716 | 0.1991 | -20.9% |
| **AR(1) to mean (final)** | **0.1630** | **0.1910** | **-24.8%** |
| ridge AR (p=3) | 0.1625 | 0.1921 | -25.1% |

- Known limitation: the model over-forecasts after a drop in load. Skill beyond the first few steps is small. Details are in the report.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Linux/macOS:         source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev,plots]"
python -m pytest -q
```

## Data

The dataset is not stored in this repository. Download the `data-v1` release asset and extract it into `challenge_data/`:

```powershell
# Windows PowerShell
curl.exe -L -o data-v1.tar.gz https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz
tar -xzf data-v1.tar.gz -C challenge_data
```

```bash
# Linux/macOS
mkdir -p challenge_data
curl -L -o data-v1.tar.gz https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz
tar -xzf data-v1.tar.gz -C challenge_data
sha256sum -c challenge_data/SHA256SUMS   # run from inside challenge_data/
```

Expected files: `vm_cpu_readings.csv.gz`, `vm_metadata.csv.gz`, `DATA_CARD.md`, `SHA256SUMS`.

## Reproduce

Generate the submission file (one command):

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

Output: 24 rows with columns `timestamp,active_vm_count,aggregate_cpu_utilization`.

Reproduce the validation results and report figures:

```bash
python -m cwfc.evaluate --data challenge_data     # writes reports/validation_results.csv
python -m cwfc.figures --data challenge_data      # writes reports/figures/*.png
```

## No use of the holdout

The final 24 target observations are never used for fitting, feature selection, tuning or model selection.

- `cwfc.data.split_holdout` returns the training rows and only the *timestamps* of the holdout, never their target values.
- `cwfc.evaluate` runs on the training window only.
- `tests/test_pipeline.py` contains a leakage test: it overwrites the target values of the last 24 timestamps with different numbers and asserts that the predictions are identical.

## Repository layout

```
src/cwfc/
  data.py        build the two target series, split off the holdout timestamps
  models.py      baselines, AR(1) to mean, ridge AR, count forecast
  evaluate.py    rolling-origin validation and model comparison
  predict.py     writes predictions.csv
  figures.py     report figures
  metrics.py     starter MAE / RMSE helpers (from the challenge repo)
tests/           starter tests plus pipeline and leakage tests
notebooks/       exploratory analysis (01_eda.ipynb)
reports/         final report, validation table, figures
predictions.csv  submission file
```

`src/cwfc/baselines.py` and `src/cwfc/validation.py` are the challenge's starter utilities. They are unused here: the starter validation helper assumes datetime timestamps and a 7-day minimum training window, which does not fit an 110-point series.

## Notes

- If a reviewer replaces `challenge_data/vm_cpu_readings.csv.gz` with a smaller file of the same format, the pipeline still runs. The holdout is always the last `--horizon` timestamps (default 24).
- AI assistance: I used an AI assistant (Claude) for guidance, code scaffolding and drafting. I ran everything and made the modeling decisions based on the validation results.
