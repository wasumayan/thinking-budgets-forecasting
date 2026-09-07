"""TIER 3. Context is Key (CiK) loader + RCRPS (SPEC §2.2; docs/verify-datasets.md for the HF schema).

load_windows() -> list[dict] in the same window dict shape used by prompts.render_messages, with
  dataset="cik", window_id=f"cik__{task_name}__{instance}", context/target arrays, metadata = the CiK
  background + scenario/constraints text as provided (this IS the context; no calendar/events sections —
  render only ## Series with the CiK text, ## History, ## Task), seasonality from the task's seasonal period
  (m=1 if unknown), precision=4, domain="cik". History/horizon lengths are task-specific (not 96/12): the Task
  line says "Forecast the next {H} values"; parse.parse_forecast must therefore accept H from the window
  (add an `h` argument; default 12).
rcrps(rows) -> per-window RCRPS using the bundled compute_rcrps_with_hf_dataset.py logic (samples required).

Implementation: reads the HF dataset ServiceNow/context-is-key (split "test", 355 rows) via `datasets`
(honours HF_HOME / HF_HUB_OFFLINE). Each window also carries the fields RCRPS needs (metric_scaling,
region_of_interest, constraint_*, weight). Metadata = "Background:\n..\n\nConstraints:\n..\n\nScenario:\n.."
(the official baseline prompt layout), empty parts omitted.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from io import StringIO

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
HF_ID = "ServiceNow/context-is-key"
_cache: list[dict] | None = None


def _freq_desc(index: pd.Index) -> str:
    if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
        return "regular steps"
    step = pd.Series(index).diff().dropna().median()
    if step == pd.Timedelta(hours=1):
        return "hourly"
    if step == pd.Timedelta(days=1):
        return "daily"
    if step == pd.Timedelta(days=7):
        return "weekly"
    if pd.Timedelta(days=28) <= step <= pd.Timedelta(days=31):
        return "monthly"
    if step < pd.Timedelta(hours=1):
        return f"every {int(step.total_seconds() // 60)} minutes"
    return f"every {step}"


def _ts_strings(index: pd.Index) -> list[str]:
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    if isinstance(index, pd.DatetimeIndex):
        if (index == index.normalize()).all():
            return [t.strftime("%Y-%m-%d") for t in index]
        return [t.strftime("%Y-%m-%d %H:%M") for t in index]
    return [str(x) for x in index]


def _metadata(e: dict) -> str:
    parts = []
    for label in ("background", "constraints", "scenario"):
        txt = (e.get(label) or "").strip()
        if txt:
            parts.append(f"{label.capitalize()}:\n{txt}")
    return "\n\n".join(parts)


def load_windows() -> list[dict]:
    global _cache
    if _cache is not None:
        return _cache
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split="test")
    out = []
    for e in ds:
        past = pd.read_json(StringIO(e["past_time"]))
        future = pd.read_json(StringIO(e["future_time"]))
        target_col = future.columns[-1]
        ctx = past[target_col].to_numpy(dtype=float)
        tgt = future[target_col].to_numpy(dtype=float)
        sp = int(e.get("seasonal_period") or 0)
        m = sp if sp > 0 and sp < len(ctx) else 1
        origin = past.index[-1]
        origin_ts = pd.Timestamp(origin).tz_localize("UTC") if isinstance(origin, pd.Timestamp) and origin.tzinfo is None else origin
        out.append({
            "window_id": f"cik__{e['name']}__{int(e['seed'])}", "series_id": f"cik__{e['name']}", "dataset": "cik",
            "domain": "cik", "cluster": None, "freq": _freq_desc(past.index), "seasonality": m, "precision": 4,
            "title": e["name"], "units": str(target_col), "origin_ts": str(origin_ts),
            "context_ts": _ts_strings(past.index), "context": [float(x) for x in ctx],
            "target_ts": _ts_strings(future.index), "target": [float(x) for x in tgt], "n_ffill": 0,
            "metadata": _metadata(e), "calendar": "None", "events": [], "reports": [], "n_events": 0,
            "text_available_at_max": None, "strict_2026": False,
            # RCRPS inputs
            "weight": str(e.get("weight") or "1"), "metric_scaling": float(e.get("metric_scaling") or 1.0),
            "region_of_interest": list(e.get("region_of_interest") or []),
            "constraint_min": e.get("constraint_min"), "constraint_max": e.get("constraint_max"),
            "constraint_variable_max_index": list(e.get("constraint_variable_max_index") or []),
            "constraint_variable_max_values": list(e.get("constraint_variable_max_values") or []),
            "context_sources": list(e.get("context_sources") or []),
        })
    log.info("CiK: %d instances, %d tasks", len(out), len({w["series_id"] for w in out}))
    _cache = out
    return out


# ------------------------------------------------------------------------------------------------------------
# RCRPS (docs/verify-datasets.md §3.4; mirrors compute_rcrps_with_hf_dataset.py)
# ------------------------------------------------------------------------------------------------------------

def crps_pwm(target: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Per-timestep CRPS by the probability-weighted-moment estimator. target [H], samples [S, H] -> [H]."""
    x = np.sort(np.asarray(samples, float), axis=0)
    y = np.asarray(target, float)[None, :]
    S = x.shape[0]
    beta0 = x.mean(axis=0)
    i = np.arange(S, dtype=float)[:, None]
    beta1 = (i * x).sum(axis=0) / (S * (S - 1)) if S > 1 else beta0
    return np.abs(x - y).mean(axis=0) + beta0 - 2.0 * beta1


def rcrps_one(w: dict, samples: np.ndarray, roi_weight: float = 0.5, violation_factor: float = 10.0) -> dict:
    samples = np.asarray(samples, float)
    target = np.asarray(w["target"], float)
    crps = crps_pwm(target, samples)
    roi = list(w.get("region_of_interest") or [])
    if roi:
        mask = np.zeros(len(target), bool)
        mask[roi] = True
        crps_value = roi_weight * crps[mask].mean() + (1 - roi_weight) * crps[~mask].mean() if (~mask).any() else crps[mask].mean()
    else:
        crps_value = float(crps.mean())
    scale = float(w.get("metric_scaling") or 1.0)
    viol = np.zeros(samples.shape[0])
    cmin, cmax = w.get("constraint_min"), w.get("constraint_max")
    if cmin is not None and not (isinstance(cmin, float) and np.isnan(cmin)):
        viol += np.clip(scale * cmin - scale * samples, 0, None).mean(axis=1)
    if cmax is not None and not (isinstance(cmax, float) and np.isnan(cmax)):
        viol += np.clip(scale * samples - scale * cmax, 0, None).mean(axis=1)
    idx, vmax = list(w.get("constraint_variable_max_index") or []), list(w.get("constraint_variable_max_values") or [])
    if idx:
        viol += np.clip(scale * samples[:, idx] - scale * np.asarray(vmax, float)[None, :], 0, None).mean(axis=1)
    violation_crps = float(crps_pwm(np.zeros(1), (violation_factor * viol)[:, None])[0])
    metric = scale * float(crps_value) + violation_crps
    return {"metric": metric, "crps": float(crps_value), "violation_crps": violation_crps, "scaling": scale}


def rcrps(rows) -> dict:
    """rows: results rows (dicts) with `samples` for CiK windows. Returns
    {"per_window": {window_id: metric}, "mean_rcrps": weighted mean of min(metric, 5), "n": count}."""
    wins = {w["window_id"]: w for w in load_windows()}
    per, num, den = {}, 0.0, 0.0
    for r in rows:
        w = wins.get(r["window_id"])
        if w is None or not r.get("samples"):
            continue
        m = rcrps_one(w, np.asarray(r["samples"], float))["metric"]
        per[r["window_id"]] = m
        wgt = float(Fraction(w["weight"]))
        num += wgt * min(m, 5.0)
        den += wgt
    return {"per_window": per, "mean_rcrps": num / den if den else float("nan"), "n": len(per)}
