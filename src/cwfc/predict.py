from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from .data import load_targets, split_holdout
from .models import ar1_to_mean, forecast_count


def build_predictions(data_dir: str, horizon: int = 24) -> pd.DataFrame:
    y = load_targets(data_dir)
    train, future_ts = split_holdout(y, horizon)
    out = pd.DataFrame({
        "timestamp": future_ts,
        "active_vm_count": forecast_count(train.active_vm_count.to_numpy(), horizon),
        "aggregate_cpu_utilization": np.clip(
            ar1_to_mean(train.aggregate_cpu_utilization.to_numpy(), horizon), 0, 100),
    })
    assert len(out) == horizon and np.isfinite(out.drop(columns="timestamp").to_numpy()).all()
    assert out.timestamp.is_monotonic_increasing and not out.timestamp.duplicated().any()
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Write predictions.csv")
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--horizon", type=int, default=24)
    a = ap.parse_args(argv)
    build_predictions(a.data, a.horizon).to_csv(a.output, index=False)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()