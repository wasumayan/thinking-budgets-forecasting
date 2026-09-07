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

import argparse
import logging
import pathlib
import sys

import numpy as np
import pandas as pd

log = logging.getLogger("make_tables")
DOMAINS = ["overall", "web", "health", "energy", "macro"]
SIZES = ["qwen3-8b", "qwen3-4b", "qwen3-1.7b"]


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt_ci(r) -> str:
    if r is None or not np.isfinite(r.rel_mase):
        return "--"
    if np.isfinite(r.rel_mase_lo) and np.isfinite(r.rel_mase_hi):
        return f"{r.rel_mase:.3f} [{r.rel_mase_lo:.2f}, {r.rel_mase_hi:.2f}]"
    return f"{r.rel_mase:.3f}"


def fmt_num(x, nd=2) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def booktabs(header: list[str], rows: list[list[str]], caption: str, label: str, align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def bold_best(cells: list[str], values: list[float]) -> list[str]:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return cells
    best = min(finite)
    return [f"\\textbf{{{c}}}" if np.isfinite(v) and v == best else c for c, v in zip(cells, values)]


def _row(summ: pd.DataFrame, cid: str, domain: str, stratum: str = "all"):
    r = summ[(summ.config_id == cid) & (summ.domain == domain) & (summ.stratum == stratum)]
    return None if r.empty else r.iloc[0]


def table_grid(summ: pd.DataFrame, config_ids: list[tuple[str, str]], stratum: str, caption: str, label: str,
               domain_suffix: str = "") -> str:
    doms = [d + domain_suffix for d in DOMAINS if (d + domain_suffix) in set(summ.domain)]
    header = ["config"] + [tex_escape(d.replace("_ww", "")) for d in doms] + ["valid", "think tok"]
    rows, col_vals = [], {d: [] for d in doms}
    for name, cid in config_ids:
        cells = [tex_escape(name)]
        for d in doms:
            r = _row(summ, cid, d, stratum)
            cells.append(fmt_ci(r))
            col_vals[d].append(float(r.rel_mase) if r is not None else np.nan)
        r = _row(summ, cid, doms[0] if doms else "overall", stratum)
        cells += [fmt_num(r.valid_rate if r is not None else np.nan), fmt_num(r.mean_thinking_tokens if r is not None else np.nan, 0)]
        rows.append(cells)
    for j, d in enumerate(doms):
        col = bold_best([r[1 + j] for r in rows], col_vals[d])
        for r, c in zip(rows, col):
            r[1 + j] = c
    return booktabs(header, rows, caption, label)


def tier1_ids(summ: pd.DataFrame, ds: str) -> list[tuple[str, str]]:
    ids = [("seasonal naive", f"{ds}__snaive"), ("Chronos-2", f"{ds}__chronos2"), ("TimesFM-2.5", f"{ds}__timesfm25"), ("TiRex", f"{ds}__tirex")]
    present = set(summ.config_id)
    for setup in ("direct", "reviser"):
        for m in SIZES:
            short = m.replace("qwen3-", "")
            for think in ("off", "b512", "b2048", "b8192"):
                ids.append((f"{short} {setup} {think}", f"{ds}__{m}__{setup}__{think}__full__s0"))
            ids.append((f"{short} {setup} SC-5", f"{ds}__{m}__{setup}__off__full__s0__n5"))
            if setup == "direct":
                ids.append((f"{short} AvgEns(off)", f"{ds}__{m}__avgens_off"))
                ids.append((f"{short} AvgEns(b2048)", f"{ds}__{m}__avgens_b2048"))
    return [(n, c) for n, c in ids if c in present]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--paired", default="results/paired.csv")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(a.summary)
    paired = pd.read_csv(a.paired) if pathlib.Path(a.paired).exists() else pd.DataFrame()
    datasets = sorted(summ.dataset.unique())
    ds = "freshts26" if "freshts26" in datasets else datasets[0]

    all_ids = [(c, c) for c in sorted(summ[(summ.dataset == ds) & (summ.stratum == "all") & (summ.domain == "overall")].config_id.unique())
               if "__s*" not in c]
    tables = {
        "table1_main.tex": table_grid(summ, tier1_ids(summ, ds), "all", f"Tier-1 grid on {tex_escape(ds)}: relative MASE (seasonal naive = 1) with 95\\% bootstrap CIs over series, valid-output rate and mean thinking tokens.", "tab:main"),
        "tableA_full.tex": table_grid(summ, all_ids, "all", f"All configs on {tex_escape(ds)} (stratum = all).", "tab:full"),
        "tableA_valid_only.tex": table_grid(summ, all_ids, "valid_only", "Valid-output windows only.", "tab:valid_only"),
        "tableA_ww.tex": table_grid(summ, all_ids, "all", "Window-weighted relative MASE (ratio of window means).", "tab:ww", domain_suffix="_ww"),
        "tableA_strict2026.tex": table_grid(summ, all_ids, "strict_2026", "Windows with origin $\\geq$ 2026-07-01.", "tab:strict2026"),
    }
    # seeds
    seed_rows = summ[summ.config_id.str.contains(r"__s\*", regex=True) & (summ.domain == "overall") & (summ.stratum == "all")]
    rows = [[tex_escape(r.config_id), f"{r.rel_mase:.3f} $\\pm$ {r.seed_sd:.3f}", str(int(r.n_seeds))] for r in seed_rows.itertuples()]
    tables["tableA_seeds.tex"] = booktabs(["config", "rel-MASE (mean $\\pm$ SD over seeds)", "seeds"], rows or [["--", "--", "--"]],
                                          "Seed variance of the headline (overall relative MASE).", "tab:seeds")
    # budgets: nominal vs actual
    b = summ[(summ.kind == "llm") & (summ.think != "off") & (summ.domain == "overall") & (summ.stratum == "all") & ~summ.config_id.str.contains(r"__s\*", regex=True)]
    rows = [[tex_escape(r.config_id), str(int(r.budget)), fmt_num(r.mean_thinking_tokens, 0), fmt_num(r.mean_answer_tokens, 0),
             fmt_num(r.budget_hit_rate), fmt_num(r.valid_rate)] for r in b.sort_values(["model", "setup", "budget"]).itertuples()]
    tables["tableA_budgets.tex"] = booktabs(["config", "nominal", "mean thinking tok", "mean answer tok", "budget-hit rate", "valid"],
                                            rows or [["--"] * 6], "Nominal thinking budget vs. actual mean thinking tokens.", "tab:budgets")
    # paired
    rows = [[r.hypothesis, tex_escape(r.model), tex_escape(str(r.setup)), tex_escape(r.domain), tex_escape(r.stratum), tex_escape(r.label),
             f"{r.delta_log_rel_mase:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", str(int(r.n_series))] for r in paired.itertuples()] if not paired.empty else []
    tables["tableA_paired.tex"] = booktabs(["H", "model", "setup", "domain", "stratum", "comparison", "$\\Delta$ log rel-MASE [95\\% CI]", "n"],
                                           rows or [["--"] * 8], "All pre-registered paired comparisons (negative = first config better).", "tab:paired",
                                           align="llllllrr")
    for name, tex in tables.items():
        (out / name).write_text(tex)
        log.info("wrote %s", out / name)


if __name__ == "__main__":
    main()
