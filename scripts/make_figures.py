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


def main(argv=None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
