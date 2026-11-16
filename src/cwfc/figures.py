from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_targets, split_holdout
from .evaluate import rolling_origin
from .models import ar1_to_mean, hist_mean, last_value


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Save report figures (training window only)")
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--outdir", default="reports/figures")
    a = ap.parse_args(argv)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    train, _ = split_holdout(load_targets(a.data), 24)
    s = train.aggregate_cpu_utilization.to_numpy()
    t = train.index.to_numpy()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, s)
    ax.axvspan(15000, 21000, alpha=0.15, label="bump (15,000-21,000 s)")
    ax.set_title("Aggregate CPU utilization (training window)")
    ax.set_xlabel("timestamp (s)")
    ax.set_ylabel("%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_series.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for name, fn in [("last_value", last_value), ("hist_mean", hist_mean), ("ar1_to_mean", ar1_to_mean)]:
        e = rolling_origin(fn, s)
        ax.plot(range(1, 25), np.abs(e).mean(axis=0), label=name)
    ax.set_title("Rolling-origin MAE by forecast step")
    ax.set_xlabel("steps ahead (5 min each)")
    ax.set_ylabel("MAE (% points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_mae_by_step.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharey=True)
    for ax, o in zip(axes.ravel(), [60, 66, 74, 86]):
        ax.plot(range(len(s)), s, color="lightgray")
        ax.plot(range(o, o + 24), s[o:o + 24], label="actual")
        ax.plot(range(o, o + 24), ar1_to_mean(s[:o], 24), label="AR(1) forecast")
        ax.axvline(o, ls="--", color="k")
        ax.set_title(f"origin {o}")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "fig3_examples.png", dpi=150)
    plt.close(fig)
    print(f"saved figures to {out}")


if __name__ == "__main__":
    main()