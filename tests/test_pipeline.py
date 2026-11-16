import numpy as np
import pandas as pd

from cwfc.predict import build_predictions


def make_data(tmp_path, n_ts=40, n_vm=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_ts):
        for v in range(n_vm):
            rows.append((t * 300, f"vm{v}", 0.0, 0.0, rng.uniform(0, 100)))
    df = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    df.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    return df


def test_output_format(tmp_path):
    make_data(tmp_path)
    out = build_predictions(str(tmp_path), horizon=24)
    assert len(out) == 24
    assert list(out.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert not out.timestamp.duplicated().any()
    assert out.aggregate_cpu_utilization.between(0, 100).all()
    assert (out.active_vm_count >= 0).all()


def test_no_leakage_from_holdout(tmp_path):
    df = make_data(tmp_path)
    a = build_predictions(str(tmp_path), horizon=24)

    # overwrite the target values of the LAST 24 timestamps with different numbers
    df2 = df.copy()
    last_ts = sorted(df2.timestamp.unique())[-24:]
    mask = df2.timestamp.isin(last_ts)
    df2.loc[mask, "avg_cpu"] = np.random.default_rng(99).uniform(0, 100, mask.sum())
    df2.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)

    b = build_predictions(str(tmp_path), horizon=24)
    pd.testing.assert_frame_equal(a, b)   # predictions must be identical


def test_tiny_dataset_runs(tmp_path):
    make_data(tmp_path, n_ts=12, n_vm=5)
    out = build_predictions(str(tmp_path), horizon=4)
    assert len(out) == 4