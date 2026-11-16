from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_targets(data_dir: str | Path) -> pd.DataFrame:
    """Build the two aggregate series, one row per timestamp, sorted in time."""
    path = Path(data_dir) / "vm_cpu_readings.csv.gz"
    r = pd.read_csv(path, usecols=["timestamp", "vm_id", "avg_cpu"])
    r = r.dropna(subset=["avg_cpu"]).drop_duplicates(["timestamp", "vm_id"])  # a VM counts once per timestamp
    y = (r.groupby("timestamp")
           .agg(active_vm_count=("vm_id", "nunique"),
                aggregate_cpu_utilization=("avg_cpu", "mean"))
           .sort_index())
    return y


def split_holdout(y: pd.DataFrame, horizon: int = 24):
    """Return (train, holdout_timestamps). Holdout TARGET values are never returned."""
    if len(y) <= horizon:
        raise ValueError("series is shorter than the forecast horizon")
    return y.iloc[:-horizon], y.index[-horizon:]