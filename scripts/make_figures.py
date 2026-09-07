#!/usr/bin/env python
"""Figures from results/summary.csv and results/paired.csv ONLY (SPEC §7.4). Never read raw jsonl here.

Usage: python scripts/make_figures.py --summary results/summary.csv --paired results/paired.csv --out figures [--smoke]

fig1_budget_curves.{pdf,png}: x = mean_thinking_tokens (off at x=0, drawn on a symlog axis with linthresh=100),
  y = rel_mase (overall, stratum=all) with CI error bars; one line per model size (qwen3-1.7b/4b/8b); panels:
  freshts26 direct | freshts26 reviser | cik direct (only if present). Horizontal dashed reference lines: chronos2,
  timesfm25, tirex (labelled), and per-size AvgEns(off) as dotted lines in the size's colour; SC-5 as a hollow
  marker of the size's colour at its mean total tokens (answer_tokens*5).
fig2_strata.{pdf,png}: from paired.csv H1/H2 rows: delta_log_rel_mase with [lo,hi] as horizontal bars, grouped
  by domain (overall, web, health, energy, macro) and by stratum (event / no_event); panels per model size;
  direct vs reviser as two colours. Vertical line at 0.
fig3_context_sensitivity.{pdf,png}: from paired.csv H5 rows: (shuffled − full) and (none − full) for off vs
  on@2048, 8B and 4B, direct and reviser; dot-and-CI plot; vertical line at 0.
Style: Okabe-Ito palette; size lines: 1.7b=#E69F00, 4b=#56B4E9, 8b=#009E73; reference grey #999999; 300 dpi;
figure widths 6.75 in (fig1), 6.75 in (fig2), 3.3 in (fig3); font 8 pt; embed fonts (pdf.fonttype 42).
--smoke: tolerate missing configs; draw whatever exists; exit 0.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("make_figures")
SIZE_COLORS = {"qwen3-1.7b": "#E69F00", "qwen3-4b": "#56B4E9", "qwen3-8b": "#009E73"}
OTHER_COLORS = ["#CC79A7", "#D55E00", "#0072B2", "#F0E442", "#000000"]
GREY = "#999999"
DOMAINS = ["overall", "web", "health", "energy", "macro"]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.dpi": 100, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})


def color_for(model: str, extra: dict) -> str:
    if model in SIZE_COLORS:
        return SIZE_COLORS[model]
    if model not in extra:
        extra[model] = OTHER_COLORS[len(extra) % len(OTHER_COLORS)]
    return extra[model]


def save(fig, out: pathlib.Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s/%s.{pdf,png}", out, name)


def _headline(summ: pd.DataFrame) -> pd.DataFrame:
    return summ[(summ.domain == "overall") & (summ.stratum == "all")].copy()


# ------------------------------------------------------------------------------------------------------
def fig1(summ: pd.DataFrame, out: pathlib.Path) -> None:
    h = _headline(summ)
    h = h[h.rel_mase.notna()]
    panels = []
    for ds, setup in (("freshts26", "direct"), ("freshts26", "reviser"), ("cik", "direct")):
        sub = h[(h.dataset == ds) & (h.setup == setup) & (h.kind == "llm")]
        if not sub.empty:
            panels.append((ds, setup, sub))
    if not panels:  # smoke / partial: draw whatever LLM rows exist, one panel per (dataset, setup)
        for (ds, setup), sub in h[h.kind == "llm"].groupby(["dataset", "setup"]):
            panels.append((ds, setup, sub))
    if not panels:
        log.warning("fig1: no LLM configs in summary; skipping")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.75, 2.4), sharey=True, squeeze=False)
    extra = {}
    for ax, (ds, setup, sub) in zip(axes[0], panels):
        base = sub[(sub.context_mode == "full") & (sub.seed == 0) & (sub.n_samples == 1) & (sub.prior.isna() | (sub.prior == "chronos2"))]
        for model, g in base.groupby("model"):
            g = g.sort_values("mean_thinking_tokens")
            x = np.where(g.think == "off", 0.0, g.mean_thinking_tokens)
            c = color_for(model, extra)
            yerr = np.vstack([g.rel_mase - g.rel_mase_lo, g.rel_mase_hi - g.rel_mase]).clip(min=0)
            ax.errorbar(x, g.rel_mase, yerr=yerr, marker="o", ms=3, lw=1, capsize=2, color=c, label=model)
            for _, r in g.iterrows():
                if r.think != "off":
                    ax.annotate(f"@{int(r.budget)}", (r.mean_thinking_tokens, r.rel_mase), fontsize=5.5,
                                xytext=(2, 3), textcoords="offset points", color=c)
            ens = h[(h.dataset == ds) & (h.model == model) & (h.config_id == f"{ds}__{model}__avgens_off")]
            if not ens.empty:
                ax.axhline(float(ens.rel_mase.iloc[0]), ls=":", lw=1, color=c)
                ax.annotate("AvgEns(off)", (0.99, float(ens.rel_mase.iloc[0])), xycoords=("axes fraction", "data"),
                            fontsize=5.5, ha="right", va="bottom", color=c)
            ens2 = h[(h.dataset == ds) & (h.config_id == f"{ds}__{model}__avgens_b2048")]
            if not ens2.empty:
                ax.axhline(float(ens2.rel_mase.iloc[0]), ls=(0, (1, 3)), lw=1, color=c)
            sc5 = sub[(sub.model == model) & (sub.n_samples == 5) & (sub.think == "off") & (sub.context_mode == "full")]
            if not sc5.empty:
                r = sc5.iloc[0]
                ax.errorbar([5 * r.mean_answer_tokens], [r.rel_mase], yerr=[[max(r.rel_mase - r.rel_mase_lo, 0)], [max(r.rel_mase_hi - r.rel_mase, 0)]],
                            marker="D", ms=4, mfc="white", mec=c, ecolor=c, ls="none", capsize=2)
                ax.annotate("SC-5", (5 * r.mean_answer_tokens, r.rel_mase), fontsize=5.5, xytext=(2, -8), textcoords="offset points", color=c)
        for i, ref in enumerate(("chronos2", "timesfm25", "tirex")):
            rr = h[h.config_id == f"{ds}__{ref}"]
            if not rr.empty:
                y = float(rr.rel_mase.iloc[0])
                ax.axhline(y, ls="--", lw=0.8, color=GREY)
                ax.annotate(ref, (0.01, y), xycoords=("axes fraction", "data"), fontsize=5.5, va="bottom", color="#555555")
        ax.axhline(1.0, lw=0.6, color="black", alpha=0.4)
        ax.set_xscale("symlog", linthresh=100)
        xmax = float(np.nanmax(base.mean_thinking_tokens)) if len(base) else 100.0
        ax.set_xlim(-30, max(xmax * 2.5, 300))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("mean thinking tokens (0 = thinking off)")
        ax.set_title(f"{ds} · {setup}")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("relative MASE (seasonal naive = 1)")
    axes[0][0].legend(frameon=False, loc="best")
    save(fig, out, "fig1_budget_curves")


# ------------------------------------------------------------------------------------------------------
def fig2(paired: pd.DataFrame, out: pathlib.Path) -> None:
    p = paired[paired.hypothesis.isin(["H1", "H2"])].copy()
    p = p[(p.label.str.startswith("on@2048 vs off"))]
    if p.empty:
        log.warning("fig2: no H1/H2 on@2048-vs-off rows; skipping")
        return
    models = sorted(p.model.unique(), key=lambda m: list(SIZE_COLORS).index(m) if m in SIZE_COLORS else 99)
    groups = [("domain", d) for d in DOMAINS if d in set(p.domain)] + [("stratum", s) for s in ("event", "no_event") if s in set(p.stratum)]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6.75, 0.35 * len(groups) + 1.2), sharex=True, squeeze=False)
    colors = {"direct": "#0072B2", "reviser": "#D55E00"}
    for ax, model in zip(axes[0], models):
        for j, setup in enumerate(("direct", "reviser")):
            ys, xs, lo, hi = [], [], [], []
            for gi, (kind, val) in enumerate(groups):
                if kind == "domain":
                    r = p[(p.model == model) & (p.setup == setup) & (p.hypothesis == "H1") & (p.domain == val) & (p.stratum == "all")]
                else:
                    r = p[(p.model == model) & (p.setup == setup) & (p.hypothesis == "H2") & (p.domain == "overall") & (p.stratum == val)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ys.append(gi + (0.18 if setup == "reviser" else -0.18))
                xs.append(r.delta_log_rel_mase); lo.append(r.delta_log_rel_mase - r.lo); hi.append(r.hi - r.delta_log_rel_mase)
            if xs:
                ax.errorbar(xs, ys, xerr=np.vstack([lo, hi]).clip(min=0), fmt="o", ms=3, capsize=2, lw=1, color=colors[setup], label=setup)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([f"{k}: {v}" if k == "stratum" else v for k, v in groups])
        ax.invert_yaxis()
        ax.set_title(model)
        ax.set_xlabel("Δ log rel-MASE (on@2048 − off)")
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    axes[0][0].legend(frameon=False, loc="best")
    save(fig, out, "fig2_strata")


# ------------------------------------------------------------------------------------------------------
def fig3(paired: pd.DataFrame, out: pathlib.Path) -> None:
    p = paired[(paired.hypothesis == "H5")].copy()
    if p.empty:
        log.warning("fig3: no H5 rows; skipping")
        return
    p["ablation"] = p.label.str.extract(r"^(shuffled|none)")[0]
    p["think"] = p.label.str.extract(r"\((off|b2048)\)")[0]
    rows = [(m, s, a) for m in ("qwen3-8b", "qwen3-4b") for s in ("direct", "reviser") for a in ("shuffled", "none")]
    rows = [r for r in rows if not p[(p.model == r[0]) & (p.setup == r[1]) & (p.ablation == r[2])].empty]
    if not rows:
        rows = sorted({(r.model, r.setup, r.ablation) for r in p.itertuples()})
    fig, ax = plt.subplots(figsize=(3.3, 0.28 * len(rows) + 1.0))
    colors = {"off": "#0072B2", "b2048": "#D55E00"}
    for think in ("off", "b2048"):
        xs, ys, lo, hi = [], [], [], []
        for i, (m, s, a) in enumerate(rows):
            r = p[(p.model == m) & (p.setup == s) & (p.ablation == a) & (p.think == think)]
            if r.empty:
                continue
            r = r.iloc[0]
            ys.append(i + (0.15 if think == "b2048" else -0.15)); xs.append(r.delta_log_rel_mase)
            lo.append(r.delta_log_rel_mase - r.lo); hi.append(r.hi - r.delta_log_rel_mase)
        if xs:
            ax.errorbar(xs, ys, xerr=np.vstack([lo, hi]).clip(min=0), fmt="o", ms=3, capsize=2, lw=1, color=colors[think],
                        label="thinking off" if think == "off" else "thinking on@2048")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{m.replace('qwen3-', '')} {s} {a}−full" for m, s, a in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Δ log rel-MASE (ablation − full context)")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    save(fig, out, "fig3_context_sensitivity")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--paired", default="results/paired.csv")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")
    out = pathlib.Path(a.out)
    summ = pd.read_csv(a.summary)
    paired = pd.read_csv(a.paired) if pathlib.Path(a.paired).exists() else pd.DataFrame(columns=["hypothesis"])
    for fn, arg in ((fig1, summ), (fig2, paired), (fig3, paired)):
        try:
            fn(arg, out)
        except Exception:  # noqa: BLE001
            if not a.smoke:
                raise
            log.exception("%s failed (tolerated in --smoke)", fn.__name__)


if __name__ == "__main__":
    main()
