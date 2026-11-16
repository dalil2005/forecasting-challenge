from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .data import load_targets, split_holdout
from .metrics import mae, rmse
from .models import (ar1_to_mean, hist_mean, last_value, moving_average, ridge_ar)


def rolling_origin(forecast_fn, series, first_origin: int = 60, h: int = 24) -> np.ndarray:
    """Forecast h steps from every origin t using ONLY series[:t]; return errors (origins x steps)."""
    series = np.asarray(series, dtype=float)
    errs = []
    for t in range(first_origin, len(series) - h + 1):
        pred = forecast_fn(series[:t], h)
        errs.append(series[t:t + h] - pred)
    if not errs:
        raise ValueError("series too short for this first_origin and horizon")
    return np.array(errs)


def candidates() -> dict:
    c = {"last_value": last_value}
    for k in (3, 6, 12, 24):
        c[f"moving_average_k{k}"] = moving_average(k)
    c["hist_mean"] = hist_mean
    c["ar1_to_mean"] = ar1_to_mean
    for p in (2, 3, 6):
        c[f"ridge_ar_p{p}"] = ridge_ar(p)
    return c


def evaluate(series, first_origin: int = 60, h: int = 24) -> pd.DataFrame:
    series = np.asarray(series, dtype=float)
    rows, base = [], None
    for name, fn in candidates().items():
        e = rolling_origin(fn, series, first_origin, h)
        row = {"model": name, "mae": mae(e, np.zeros_like(e)), "rmse": rmse(e, np.zeros_like(e)),
               "origins": e.shape[0]}
        for step in (1, 8, 24):
            row[f"mae_step{step}"] = float(np.abs(e[:, step - 1]).mean())
        base = row["mae"] if name == "last_value" else base
        rows.append(row)
    out = pd.DataFrame(rows)
    out["mae_vs_last_value_pct"] = (out["mae"] / base - 1) * 100
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Rolling-origin validation on the TRAINING window only")
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--first-origin", type=int, default=60)
    ap.add_argument("--output", default="reports/validation_results.csv")
    a = ap.parse_args(argv)

    train, _ = split_holdout(load_targets(a.data), a.horizon)   # holdout values are never loaded here
    table = evaluate(train.aggregate_cpu_utilization.to_numpy(), a.first_origin, a.horizon)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.output, index=False)
    print(f"\nsaved {a.output}")


if __name__ == "__main__":
    main()