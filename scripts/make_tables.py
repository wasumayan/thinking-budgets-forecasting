#!/usr/bin/env python
"""LaTeX tables from summary.csv / paired.csv -> paper/tables/*.tex (booktabs).

table1_main.tex: tier-1 configs (rows) × [overall, web, health, energy, macro] rel-MASE "x.xx [lo, hi]" plus
  valid rate and mean thinking tokens; baselines first (snaive=1.00 by definition, chronos2, timesfm25, tirex),
  then per size: off, b512, b2048, b8192, SC-5, AvgEns(off), AvgEns(b2048), for direct then reviser. Best per column bold.
tableA_full.tex: every config in summary.csv (stratum=all). tableA_valid_only.tex, tableA_ww.tex,
  tableA_strict2026.tex, tableA_seeds.tex (mean ± seed SD), tableA_budgets.tex (nominal vs mean actual thinking
  tokens, budget_hit_rate), tableA_paired.tex (all paired.csv rows).
"""
from __future__ import annotations


def main(argv=None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
