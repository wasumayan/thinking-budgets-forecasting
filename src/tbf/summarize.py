"""results/*.jsonl -> results/summary.csv + results/paired.csv (SPEC §6.2–6.4, §7.3).

CLI: python -m tbf.summarize --results-glob "results/*.jsonl" [--dataset freshts26] --out results/summary.csv --paired-out results/paired.csv

1. Load all rows; join windows (context, target, seasonality, domain, n_events, strict_2026) by window_id.
2. Per-window metrics: mase (metrics.mase with the window's context and seasonality), mae, wql when
   quantiles or samples (>=5) exist.
3. Derived configs (grid.yaml:derived): avgens_* = mean of chronos2 point and the named LLM config's point,
   per window; computed here, never run on GPU. config_id = "{dataset}__{model}__avgens_{off|b2048}".
4. Aggregate per (config_id, domain in {overall, web, health, energy, macro}, stratum in
   {all, event, no_event, strict_2026, valid_only}): per-series mean MASE -> rel_mase vs "{dataset}__snaive"
   -> geo-mean over series; bootstrap CI over series (n=1000, seed=0); win_rate vs snaive; valid_rate;
   mean_thinking_tokens; mean_answer_tokens; budget_hit_rate; n_windows; n_series. Strata with < 50 windows
   are written with NaN metrics and n_windows so the figure code can skip them.
   Also a window-weighted variant with domain suffix "_ww" for the appendix.
5. paired.csv: one row per pre-registered comparison (SPEC §1 H1–H5) computed with metrics.paired_bootstrap
   on per-series rel_mase (stratum=all unless the hypothesis names a stratum):
     H1: {model}×{setup}: on@2048 vs off; also on@512 vs off, on@8192 vs off
     H2: {model}×direct: (on@2048 vs off) within event stratum and within no_event stratum
     H3: {model}×{setup}: on@<closest budget by mean tokens> vs SC-5 (n5 off)
     H4: {model}: avgens_off vs best LLM config (min rel_mase among that model's configs); also avgens_off vs avgens_b2048
     H5: {model}×{setup}: (shuffled − full) on vs off  -> report delta of deltas as two rows
   Columns: hypothesis, dataset, model, setup, stratum, a_config_id, b_config_id, delta_log_rel_mase, lo, hi, n_series.
Seeds: configs differing only in seed are additionally aggregated as "{config_id with s*}" rows with seed_sd.

Implementation notes (see IMPLEMENTATION_NOTES.md):
- The seasonal-naive baseline for a cell is restricted to the SAME windows as the cell (so `valid_only` and the
  strata compare like with like).
- The 50-window minimum applies to the strata event / no_event / strict_2026 / valid_only; `all` is always reported.
- Window-weighted ("_ww") relative MASE = mean over windows of MASE(config) / mean over windows of MASE(snaive),
  bootstrap over windows.
- paired.csv carries an extra `domain` column (overall + each domain) so Fig 2 can group by domain, plus `label`.
"""
from __future__ import annotations

import argparse
import glob
import logging
import re
import sys

import numpy as np
import pandas as pd

from . import metrics as M
from .config import RunConfig, load_grid
from .data.windows import load_windows, read_results

log = logging.getLogger(__name__)
STRATA = ("all", "event", "no_event", "strict_2026", "valid_only")
MIN_STRATUM_WINDOWS = 50
N_BOOT = 1000
CANON_DOMAINS = ("web", "health", "energy", "macro")


# ----------------------------------------------------------------------------------------------------------
# 1–3. Load, per-window metrics, derived configs
# ----------------------------------------------------------------------------------------------------------

def _config_kind(config_id: str) -> str:
    if "__avgens_" in config_id:
        return "derived"
    if len(config_id.split("__")) == 2:
        return "baseline"
    return "llm"


def _config_fields(config_id: str) -> dict:
    parts = config_id.split("__")
    kind = _config_kind(config_id)
    out = {"kind": kind, "dataset": parts[0], "model": parts[1], "setup": None, "think": None, "budget": np.nan,
           "context_mode": None, "n_samples": np.nan, "seed": np.nan, "prior": None}
    if kind == "baseline":
        out.update(setup="baseline", think="n/a", budget=0, context_mode="numeric", n_samples=0, seed=0)
    elif kind == "derived":
        out.update(setup="avgens", think=parts[2].replace("avgens_", ""), context_mode="full", n_samples=1, seed=0)
        out["budget"] = 0 if out["think"] == "off" else int(out["think"][1:])
    else:
        try:
            c = RunConfig.parse(config_id)
            out.update(setup=c.setup, think=c.think_tag, budget=c.budget, context_mode=c.context_mode,
                       n_samples=c.n_samples, seed=c.seed, prior=c.prior if c.setup == "reviser" else None)
        except Exception:  # noqa: BLE001 — unknown id shape; keep the raw parts
            pass
    return out


def load_rows(results_glob: str, dataset: str | None) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(results_glob)):
        for r in read_results(path):
            if dataset and r.get("dataset") != dataset:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"no result rows found for {results_glob!r} (dataset={dataset})")
    df = pd.DataFrame(rows)
    # de-duplicate (resumed runs never duplicate, but be safe): keep the last row per (config, window)
    df = df.drop_duplicates(subset=["config_id", "window_id"], keep="last").reset_index(drop=True)
    for col in ("thinking_tokens_used", "answer_tokens", "budget_hit", "valid", "n_values_mismatch"):
        if col not in df:
            df[col] = np.nan
    if "samples" not in df:
        df["samples"] = None
    if "quantiles" not in df:
        df["quantiles"] = None
    return df


def _window_table(datasets) -> dict[str, dict[str, dict]]:
    out = {}
    for ds in datasets:
        try:
            out[ds] = {w["window_id"]: w for w in load_windows(ds)}
        except Exception as e:  # noqa: BLE001
            log.error("cannot load windows for dataset %s: %s", ds, e)
            out[ds] = {}
    return out


def add_derived(df: pd.DataFrame, grid: dict) -> pd.DataFrame:
    """avgens_* rows: arithmetic mean of the chronos2 point and the named LLM config's point, per window."""
    extra = []
    for ds in sorted(df.dataset.unique()):
        d = df[df.dataset == ds]
        chronos = d[d.config_id == f"{ds}__chronos2"].set_index("window_id")
        if chronos.empty:
            continue
        llm_models = sorted({m for m in d.model.unique() if len(d[(d.model == m) & (d.config_id.str.count("__") >= 5)]) > 0})
        for spec in grid.get("derived", []):
            name, comps = spec["name"], spec["components"]
            for model in llm_models:
                llm_id = f"{ds}__" + comps[1].format(model=model)
                llm = d[d.config_id == llm_id].set_index("window_id")
                if llm.empty:
                    continue
                common = llm.index.intersection(chronos.index)
                for wid in common:
                    a, b = np.asarray(chronos.at[wid, "point"], float), np.asarray(llm.at[wid, "point"], float)
                    if len(a) != len(b):
                        continue
                    extra.append({
                        "config_id": f"{ds}__{model}__{name}", "window_id": wid, "dataset": ds, "model": model,
                        "setup": "avgens", "thinking": bool(llm.at[wid, "thinking"]), "budget": int(llm.at[wid, "budget"]),
                        "context_mode": llm.at[wid, "context_mode"], "n_samples": 1, "seed": int(llm.at[wid, "seed"]),
                        "prior": "chronos2", "point": ((a + b) / 2).tolist(), "samples": None, "quantiles": None,
                        "valid": bool(llm.at[wid, "valid"]), "n_values_mismatch": False, "raw_answer": "",
                        "thinking_tokens_used": float(llm.at[wid, "thinking_tokens_used"]),
                        "answer_tokens": float(llm.at[wid, "answer_tokens"]), "budget_hit": bool(llm.at[wid, "budget_hit"]),
                        "finished": True, "wall_s": 0.0, "prompt_tokens": 0,
                    })
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        log.info("added %d derived (avgens) rows", len(extra))
    return df


def per_window_metrics(df: pd.DataFrame, wins: dict[str, dict[str, dict]]) -> pd.DataFrame:
    recs = []
    missing = 0
    for r in df.itertuples(index=False):
        w = wins.get(r.dataset, {}).get(r.window_id)
        if w is None:
            missing += 1
            continue
        y = np.asarray(w["target"], float)
        yhat = np.asarray(r.point, float)
        if len(yhat) != len(y):
            missing += 1
            continue
        ctx, m = np.asarray(w["context"], float), int(w["seasonality"])
        rec = {
            "config_id": r.config_id, "window_id": r.window_id, "dataset": r.dataset, "series_id": w["series_id"],
            "domain": w["domain"], "mase": M.mase(y, yhat, ctx, m), "mae": M.mae(y, yhat), "wql": np.nan,
            "valid": bool(r.valid) if r.valid == r.valid else True,
            "thinking_tokens_used": float(r.thinking_tokens_used) if r.thinking_tokens_used == r.thinking_tokens_used else 0.0,
            "answer_tokens": float(r.answer_tokens) if r.answer_tokens == r.answer_tokens else 0.0,
            "budget_hit": bool(r.budget_hit) if r.budget_hit == r.budget_hit else False,
            "n_events": int(w.get("n_events") or 0), "strict_2026": bool(w.get("strict_2026", False)),
        }
        if r.dataset == "cik" and isinstance(r.samples, list) and r.samples and "metric_scaling" in w:
            from .data.cik import rcrps_one
            rec["rcrps"] = rcrps_one(w, np.asarray(r.samples, float))["metric"]
            rec["rcrps_weight"] = float(__import__("fractions").Fraction(str(w.get("weight") or "1")))
        q = None
        if isinstance(r.quantiles, list) and r.quantiles:
            q = np.asarray(r.quantiles, float)  # [H, 9]
        elif isinstance(r.samples, list) and len(r.samples) >= 5:
            q = M.samples_to_quantiles(np.asarray(r.samples, float))
        if q is not None and q.shape == (len(y), len(M.QUANTILE_LEVELS)):
            rec["wql"] = M.wql(y, q)
        recs.append(rec)
    if missing:
        log.warning("%d result rows had no matching window (or wrong horizon) and were skipped", missing)
    df = pd.DataFrame(recs)
    for col in ("rcrps", "rcrps_weight"):
        if col not in df:
            df[col] = np.nan
    return df


# ----------------------------------------------------------------------------------------------------------
# 4. Aggregation
# ----------------------------------------------------------------------------------------------------------

def _stratum_mask(pw: pd.DataFrame, stratum: str) -> pd.Series:
    if stratum == "all":
        return pd.Series(True, index=pw.index)
    if stratum == "event":
        return pw.n_events >= 1
    if stratum == "no_event":
        return pw.n_events == 0
    if stratum == "strict_2026":
        return pw.strict_2026.astype(bool)
    if stratum == "valid_only":
        return pw.valid.astype(bool)
    raise ValueError(stratum)


def _series_rel(cell: pd.DataFrame, base: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Per-series relative MASE of `cell` vs the baseline restricted to the same windows."""
    b = base[base.window_id.isin(cell.window_id)]
    ps = cell.groupby("series_id")["mase"].mean()
    pb = b.groupby("series_id")["mase"].mean()
    rel = (ps / pb).replace([np.inf, -np.inf], np.nan).dropna()
    return rel.to_dict(), b


def aggregate_cell(cell: pd.DataFrame, base: pd.DataFrame | None, window_weighted: bool = False) -> dict:
    out = {"rel_mase": np.nan, "rel_mase_lo": np.nan, "rel_mase_hi": np.nan, "win_rate_vs_snaive": np.nan,
           "mase": float(cell.mase.mean()) if cell.mase.notna().any() else np.nan,
           "wql": float(cell.wql.mean()) if cell.wql.notna().any() else np.nan,
           "valid_rate": float(cell.valid.mean()), "mean_thinking_tokens": float(cell.thinking_tokens_used.mean()),
           "mean_answer_tokens": float(cell.answer_tokens.mean()), "budget_hit_rate": float(cell.budget_hit.mean()),
           "n_windows": int(len(cell)), "n_series": int(cell.series_id.nunique()),
           "n_mase_nan": int(cell.mase.isna().sum()), "rcrps": np.nan}
    if cell.rcrps.notna().any():  # CiK: weighted mean of min(RCRPS, 5) (docs/verify-datasets.md §3.4)
        c = cell[cell.rcrps.notna()]
        out["rcrps"] = float((c.rcrps.clip(upper=5.0) * c.rcrps_weight).sum() / c.rcrps_weight.sum())
    if base is None or base.empty:
        return out
    if not window_weighted:
        rel, _ = _series_rel(cell, base)
        vals = np.array(list(rel.values()), float)
        if len(vals):
            stat, lo, hi = M.bootstrap_ci(vals, N_BOOT, 0, M.geo_mean)
            out.update(rel_mase=stat, rel_mase_lo=lo, rel_mase_hi=hi,
                       win_rate_vs_snaive=M.win_rate(rel, {s: 1.0 for s in rel}))
    else:
        b = base[base.window_id.isin(cell.window_id)].set_index("window_id")["mase"]
        c = cell.set_index("window_id")["mase"]
        both = pd.concat([c.rename("c"), b.rename("b")], axis=1).dropna()
        if len(both):
            def ratio(idx_vals):  # idx_vals = 2-col array
                return float(np.mean(idx_vals[:, 0]) / np.mean(idx_vals[:, 1]))
            arr = both.to_numpy()
            rng = np.random.default_rng(0)
            boots = np.array([ratio(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(N_BOOT)])
            out.update(rel_mase=ratio(arr), rel_mase_lo=float(np.percentile(boots, 2.5)),
                       rel_mase_hi=float(np.percentile(boots, 97.5)),
                       win_rate_vs_snaive=float(np.mean(arr[:, 0] < arr[:, 1]) + 0.5 * np.mean(arr[:, 0] == arr[:, 1])))
    return out


def summarize(pw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (summary DataFrame, per-series rel_mase dict keyed by (config_id, domain, stratum))."""
    rows, rel_store = [], {}
    for ds in sorted(pw.dataset.unique()):
        d = pw[pw.dataset == ds]
        base_all = d[d.config_id == f"{ds}__snaive"]
        if base_all.empty:
            log.warning("no seasonal-naive baseline for dataset %s: rel_mase will be NaN", ds)
        domains = ["overall"] + [x for x in CANON_DOMAINS if x in set(d.domain)] + \
                  sorted(x for x in set(d.domain) if x not in CANON_DOMAINS)
        for cid in sorted(d.config_id.unique()):
            c_all = d[d.config_id == cid]
            fields = _config_fields(cid)
            for dom in domains:
                c_dom = c_all if dom == "overall" else c_all[c_all.domain == dom]
                for stratum in STRATA:
                    cell = c_dom[_stratum_mask(c_dom, stratum)]
                    for ww in (False, True):
                        rec = {"config_id": cid, "domain": dom + ("_ww" if ww else ""), "stratum": stratum, **fields}
                        if cell.empty:
                            continue
                        if stratum != "all" and len(cell) < MIN_STRATUM_WINDOWS:
                            rec.update(aggregate_cell(cell, None))
                            for k in ("mase", "wql", "rel_mase", "rel_mase_lo", "rel_mase_hi", "win_rate_vs_snaive"):
                                rec[k] = np.nan
                            rows.append(rec)
                            continue
                        rec.update(aggregate_cell(cell, base_all if not base_all.empty else None, ww))
                        rows.append(rec)
                        if not ww and not base_all.empty:
                            rel_store[(cid, dom, stratum)], _ = _series_rel(cell, base_all)
    summ = pd.DataFrame(rows)
    summ["seed_sd"] = np.nan
    summ["n_seeds"] = 1
    summ = pd.concat([summ, seed_rows(summ)], ignore_index=True)
    cols = ["config_id", "domain", "stratum", "rel_mase", "rel_mase_lo", "rel_mase_hi", "mase", "wql", "rcrps",
            "win_rate_vs_snaive", "valid_rate", "mean_thinking_tokens", "mean_answer_tokens", "budget_hit_rate",
            "n_windows", "n_series", "n_mase_nan", "seed_sd", "n_seeds", "kind", "dataset", "model", "setup", "think",
            "budget", "context_mode", "n_samples", "seed", "prior"]
    return summ[cols], rel_store


def seed_rows(summ: pd.DataFrame) -> pd.DataFrame:
    """Configs differing only in seed -> one '{config_id with s*}' row: mean headline, seed_sd = SD across seeds."""
    llm = summ[(summ.kind == "llm") & summ.seed.notna()].copy()
    if llm.empty:
        return pd.DataFrame(columns=summ.columns)
    llm["group_id"] = llm.config_id.str.replace(r"__s\d+", "__s*", regex=True)
    out = []
    for (gid, dom, stratum), g in llm.groupby(["group_id", "domain", "stratum"]):
        if g.seed.nunique() < 2:
            continue
        rec = g.iloc[0].to_dict()
        rec.update(config_id=gid, seed=np.nan, n_seeds=int(g.seed.nunique()),
                   rel_mase=float(g.rel_mase.mean()), seed_sd=float(g.rel_mase.std(ddof=1)),
                   rel_mase_lo=np.nan, rel_mase_hi=np.nan, mase=float(g.mase.mean()),
                   valid_rate=float(g.valid_rate.mean()), mean_thinking_tokens=float(g.mean_thinking_tokens.mean()),
                   mean_answer_tokens=float(g.mean_answer_tokens.mean()), budget_hit_rate=float(g.budget_hit_rate.mean()),
                   n_windows=int(g.n_windows.sum()))
        rec.pop("group_id", None)
        out.append(rec)
    return pd.DataFrame(out, columns=summ.columns) if out else pd.DataFrame(columns=summ.columns)


# ----------------------------------------------------------------------------------------------------------
# 5. Paired comparisons
# ----------------------------------------------------------------------------------------------------------

def _paired_row(hyp, label, ds, model, setup, dom, stratum, a, b, rel_store) -> dict | None:
    ra, rb = rel_store.get((a, dom, stratum)), rel_store.get((b, dom, stratum))
    if not ra or not rb:
        return None
    delta, lo, hi, n = M.paired_bootstrap(ra, rb, N_BOOT, 0)
    if n == 0:
        return None
    return {"hypothesis": hyp, "label": label, "dataset": ds, "model": model, "setup": setup, "domain": dom,
            "stratum": stratum, "a_config_id": a, "b_config_id": b, "delta_log_rel_mase": delta, "lo": lo, "hi": hi,
            "n_series": n}


def _dd_row(hyp, label, ds, model, setup, dom, a1, a0, b1, b0, rel_store) -> dict | None:
    """Delta of deltas: (a1 - a0) - (b1 - b0) in log rel-MASE, paired over series."""
    r = [rel_store.get((c, dom, "all")) for c in (a1, a0, b1, b0)]
    if not all(r):
        return None
    common = set(r[0]) & set(r[1]) & set(r[2]) & set(r[3])
    if not common:
        return None
    a = {s: r[0][s] / r[1][s] for s in common}
    b = {s: r[2][s] / r[3][s] for s in common}
    delta, lo, hi, n = M.paired_bootstrap(a, b, N_BOOT, 0)
    return {"hypothesis": hyp, "label": label, "dataset": ds, "model": model, "setup": setup, "domain": dom,
            "stratum": "all", "a_config_id": f"({a1})-({a0})", "b_config_id": f"({b1})-({b0})",
            "delta_log_rel_mase": delta, "lo": lo, "hi": hi, "n_series": n}


def paired(summ: pd.DataFrame, rel_store: dict) -> pd.DataFrame:
    rows = []
    llm = summ[(summ.kind == "llm") & (summ.stratum == "all") & (summ.domain == "overall")]
    for ds in sorted(summ.dataset.unique()):
        domains = ["overall"] + sorted({d for d in summ[summ.dataset == ds].domain.unique()
                                        if d != "overall" and not d.endswith("_ww")})
        models = sorted(llm[llm.dataset == ds].model.unique())
        for model in models:
            base = f"{ds}__{model}"
            for setup in ("direct", "reviser"):
                off = f"{base}__{setup}__off__full__s0"
                # H1 (+ dose-response) over every domain, stratum all
                for tag in ("b2048", "b512", "b8192"):
                    on = f"{base}__{setup}__{tag}__full__s0"
                    for dom in domains:
                        r = _paired_row("H1", f"on@{tag[1:]} vs off", ds, model, setup, dom, "all", on, off, rel_store)
                        if r:
                            rows.append(r)
                # H2: strata event / no_event (direct primary; reviser reported too)
                on = f"{base}__{setup}__b2048__full__s0"
                for stratum in ("event", "no_event"):
                    for dom in domains:
                        r = _paired_row("H2", f"on@2048 vs off ({stratum})", ds, model, setup, dom, stratum, on, off, rel_store)
                        if r:
                            rows.append(r)
                # H3: SC-5 vs the thinking budget whose actual mean token spend is closest
                sc5 = f"{off}__n5"
                s_sc5 = llm[llm.config_id == sc5]
                if not s_sc5.empty:
                    spend_sc5 = 5.0 * float(s_sc5.mean_answer_tokens.iloc[0])
                    cands = []
                    for tag in ("b512", "b2048", "b8192"):
                        on = f"{base}__{setup}__{tag}__full__s0"
                        s_on = llm[llm.config_id == on]
                        if not s_on.empty:
                            spend = float(s_on.mean_thinking_tokens.iloc[0] + s_on.mean_answer_tokens.iloc[0])
                            cands.append((abs(spend - spend_sc5), on, spend))
                    if cands:
                        _, on, spend = min(cands)
                        r = _paired_row("H3", f"on (closest spend {spend:.0f} vs SC-5 {spend_sc5:.0f}) vs SC-5",
                                        ds, model, setup, "overall", "all", on, sc5, rel_store)
                        if r:
                            rows.append(r)
                # H5: context ablations (shuffled - full, none - full) for off and on@2048, plus delta of deltas
                for abl in ("shuffled", "none"):
                    for tag in ("off", "b2048"):
                        full = f"{base}__{setup}__{tag}__full__s0"
                        alt = f"{base}__{setup}__{tag}__{abl}__s0"
                        r = _paired_row("H5", f"{abl} - full ({tag})", ds, model, setup, "overall", "all", alt, full, rel_store)
                        if r:
                            rows.append(r)
                    r = _dd_row("H5_dd", f"({abl}-full)@on2048 - ({abl}-full)@off", ds, model, setup, "overall",
                                f"{base}__{setup}__b2048__{abl}__s0", f"{base}__{setup}__b2048__full__s0",
                                f"{base}__{setup}__off__{abl}__s0", f"{base}__{setup}__off__full__s0", rel_store)
                    if r:
                        rows.append(r)
            # H4: AvgEns(off) vs best LLM config of this model; AvgEns(off) vs AvgEns(b2048)
            ens_off, ens_on = f"{base}__avgens_off", f"{base}__avgens_b2048"
            mine = llm[(llm.dataset == ds) & (llm.model == model) & llm.rel_mase.notna()]
            if (ens_off, "overall", "all") in rel_store and not mine.empty:
                best = mine.sort_values("rel_mase").iloc[0].config_id
                r = _paired_row("H4", "avgens_off vs best LLM config", ds, model, "avgens", "overall", "all", ens_off, best, rel_store)
                if r:
                    rows.append(r)
                r = _paired_row("H4", "avgens_off vs avgens_b2048", ds, model, "avgens", "overall", "all", ens_off, ens_on, rel_store)
                if r:
                    rows.append(r)
    cols = ["hypothesis", "label", "dataset", "model", "setup", "domain", "stratum", "a_config_id", "b_config_id",
            "delta_log_rel_mase", "lo", "hi", "n_series"]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob", default="results/*.jsonl")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results/summary.csv")
    ap.add_argument("--paired-out", default="results/paired.csv")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    grid = load_grid()
    df = load_rows(a.results_glob, a.dataset)
    log.info("loaded %d rows from %d configs", len(df), df.config_id.nunique())
    df = add_derived(df, grid)
    wins = _window_table(sorted(df.dataset.unique()))
    pw = per_window_metrics(df, wins)
    if pw.empty:
        raise SystemExit("no per-window metrics could be computed")
    summ, rel_store = summarize(pw)
    summ.to_csv(a.out, index=False)
    pr = paired(summ, rel_store)
    pr.to_csv(a.paired_out, index=False)
    log.info("wrote %s (%d rows) and %s (%d rows)", a.out, len(summ), a.paired_out, len(pr))
    head = summ[(summ.domain == "overall") & (summ.stratum == "all")][["config_id", "rel_mase", "rel_mase_lo", "rel_mase_hi", "valid_rate", "mean_thinking_tokens", "n_windows"]]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        log.info("headline (overall, all):\n%s", head.to_string(index=False))


if __name__ == "__main__":
    main()
