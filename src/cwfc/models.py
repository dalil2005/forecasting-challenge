from __future__ import annotations
import numpy as np


def ar1_to_mean(history, h: int = 24) -> np.ndarray:
    """Forecast decays geometrically from the last value toward the history mean."""
    x = np.asarray(history, dtype=float)
    if len(x) < 3:
        return np.repeat(x[-1], h)
    m = x.mean()
    z = x - m
    denom = (z[:-1] ** 2).sum()
    phi = 0.0 if denom == 0 else float(np.clip((z[1:] * z[:-1]).sum() / denom, 0.0, 0.99))
    return m + z[-1] * phi ** np.arange(1, h + 1)


def forecast_count(history, h: int = 24) -> np.ndarray:
    """If the count never changes in training, predict that constant; else AR(1), rounded and >= 0."""
    x = np.asarray(history, dtype=float)
    if np.ptp(x) == 0:
        return np.repeat(x[-1], h)
    return np.maximum(np.round(ar1_to_mean(x, h)), 0)