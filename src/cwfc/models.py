from __future__ import annotations
import numpy as np


def last_value(history, h: int = 24) -> np.ndarray:
    return np.repeat(float(np.asarray(history)[-1]), h)


def moving_average(k: int):
    def f(history, h: int = 24) -> np.ndarray:
        return np.repeat(float(np.asarray(history, dtype=float)[-k:].mean()), h)
    f.__name__ = f"moving_average_k{k}"
    return f


def hist_mean(history, h: int = 24) -> np.ndarray:
    return np.repeat(float(np.asarray(history, dtype=float).mean()), h)


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


def ridge_ar(p: int = 3, alpha: float = 1.0):
    """Ridge autoregression on the last p values (closed form, no extra dependency)."""
    def f(history, h: int = 24) -> np.ndarray:
        x = np.asarray(history, dtype=float)
        if len(x) <= p + 2:
            return np.repeat(x[-1], h)
        m = x.mean()
        z = x - m
        X = np.column_stack([z[i:len(z) - p + i] for i in range(p)])  # oldest ... newest
        y = z[p:]
        xb, yb = X.mean(axis=0), y.mean()
        Xc, yc = X - xb, y - yb
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
        b = yb - xb @ w
        buf = list(z[-p:])
        out = []
        for _ in range(h):
            nxt = float(np.array(buf[-p:]) @ w + b)
            out.append(nxt)
            buf.append(nxt)
        return m + np.array(out)
    f.__name__ = f"ridge_ar_p{p}"
    return f


def forecast_count(history, h: int = 24) -> np.ndarray:
    """If the count never changes in training, predict that constant; else AR(1), rounded and >= 0."""
    x = np.asarray(history, dtype=float)
    if np.ptp(x) == 0:
        return np.repeat(x[-1], h)
    return np.maximum(np.round(ar1_to_mean(x, h)), 0)