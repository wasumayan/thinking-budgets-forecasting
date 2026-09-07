"""Metrics and aggregation (SPEC §6). Definitions match fev 0.10.0 / gluonts 0.17.0 conventions
(see docs/verify-models.md §C). Implemented in numpy so the tests pin the exact definitions.

Per-window:
  mase(y, yhat, context, m)      seasonal error from the window's own 96-point context
  wql(y, q, levels)              factor-2 pinball, normalized by sum|y| (Chronos/fev WQL)
  crps_samples(y, samples)       fair ensemble estimator
Aggregation:
  rel_mase_by_series(...)        per-series mean MASE / per-series mean MASE of seasonal naive
  geo_mean(...)                  geometric mean over series (overall or per domain)
  bootstrap_ci(...)              percentile CI over series, n=1000, seed=0
  paired_bootstrap(...)          CI of mean per-series difference in log rel-MASE
"""
from __future__ import annotations

import numpy as np

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def seasonal_error(context: np.ndarray, m: int) -> float:
    """Mean |y[t] - y[t-m]| over the context. NaN if len(context) <= m or if the error is 0."""
    c = np.asarray(context, dtype=float)
    if len(c) <= m:
        return float("nan")
    se = float(np.nanmean(np.abs(c[m:] - c[:-m])))
    return se if se > 0 else float("nan")


def mase(y: np.ndarray, yhat: np.ndarray, context: np.ndarray, m: int) -> float:
    """Per-window MASE = mean_h |y_h - yhat_h| / seasonal_error(context, m). NaN if seasonal error undefined."""
    se = seasonal_error(context, m)
    if not np.isfinite(se):
        return float("nan")
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)) / se)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def quantile_loss(y: np.ndarray, q: np.ndarray, levels=QUANTILE_LEVELS) -> np.ndarray:
    """2 * |(y - q) * (1[y <= q] - alpha)|; y [H], q [H, Q] -> [H, Q]."""
    y = np.asarray(y, float)[:, None]
    q = np.asarray(q, float)
    a = np.asarray(levels, float)[None, :]
    return 2.0 * np.abs((y - q) * ((y <= q).astype(float) - a))


def wql(y: np.ndarray, q: np.ndarray, levels=QUANTILE_LEVELS) -> float:
    """Per-window WQL: mean over levels of sum_h QL / sum_h |y|. NaN if sum|y| == 0."""
    denom = float(np.sum(np.abs(np.asarray(y, float))))
    if denom == 0:
        return float("nan")
    ql = quantile_loss(y, q, levels)  # [H, Q]
    return float(np.mean(ql.sum(axis=0) / denom))


def samples_to_quantiles(samples: np.ndarray, levels=QUANTILE_LEVELS) -> np.ndarray:
    """samples [S, H] -> quantiles [H, Q] (linear interpolation, numpy default)."""
    s = np.asarray(samples, float)
    return np.quantile(s, np.asarray(levels), axis=0).T


def crps_samples(y: np.ndarray, samples: np.ndarray) -> float:
    """Fair ensemble CRPS averaged over the horizon. y [H], samples [S, H]."""
    y = np.asarray(y, float)
    s = np.asarray(samples, float)
    S = s.shape[0]
    if S < 2:
        return float(np.mean(np.abs(s[0] - y)))
    t1 = np.abs(s - y[None, :]).mean(axis=0)  # [H]
    t2 = np.abs(s[:, None, :] - s[None, :, :]).sum(axis=(0, 1)) / (2.0 * S * (S - 1))  # [H]
    return float(np.mean(t1 - t2))


def seasonal_naive_forecast(context: np.ndarray, H: int, m: int) -> np.ndarray:
    """Repeat the last season. m=1 -> last value repeated."""
    c = np.asarray(context, float)
    last = c[-m:]
    reps = int(np.ceil(H / m))
    return np.tile(last, reps)[:H]


# ---------------------------------------------------------------------------------------
# Aggregation over a long-format table of per-window metrics.
# Expected columns: config_id, series_id, domain, window_id, mase (float, NaN allowed).
# ---------------------------------------------------------------------------------------

def per_series_mean(df, value: str = "mase"):
    """DataFrame -> DataFrame[config_id, series_id, domain, value_mean, n_windows]. NaNs skipped."""
    g = df.groupby(["config_id", "series_id", "domain"], as_index=False)
    out = g.agg(**{value: (value, "mean"), "n_windows": (value, "count")})
    return out


def rel_mase_by_series(per_series, baseline_config_id: str):
    """Attach rel_mase = mase / mase(baseline) per series. Rows whose baseline is NaN are dropped."""
    base = per_series[per_series.config_id == baseline_config_id][["series_id", "mase"]].rename(columns={"mase": "mase_base"})
    m = per_series.merge(base, on="series_id", how="left")
    m["rel_mase"] = m["mase"] / m["mase_base"]
    return m[np.isfinite(m["rel_mase"])]


def geo_mean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, seed: int = 0, stat=geo_mean):
    """Percentile 95% CI of stat over series-level values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_resamples)])
    return stat(v), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap(a_by_series: dict, b_by_series: dict, n_resamples: int = 1000, seed: int = 0):
    """CI of mean over series of (log rel_mase_a - log rel_mase_b) on the common series.
    Negative => a better than b. Returns (delta, lo, hi, n_series)."""
    common = sorted(set(a_by_series) & set(b_by_series))
    d = np.array([np.log(a_by_series[s]) - np.log(b_by_series[s]) for s in common], float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_resamples)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(len(d))


def win_rate(a_by_series: dict, b_by_series: dict) -> float:
    """Fraction of common series where a has lower rel_mase than b (ties count 0.5)."""
    common = [s for s in a_by_series if s in b_by_series]
    if not common:
        return float("nan")
    wins = sum(1.0 if a_by_series[s] < b_by_series[s] else 0.5 if a_by_series[s] == b_by_series[s] else 0.0 for s in common)
    return wins / len(common)
