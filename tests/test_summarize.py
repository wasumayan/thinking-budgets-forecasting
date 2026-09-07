"""Drive summarize.py end to end on synthetic results for the 12 fixture windows: baselines, an LLM grid
(off / b512 / b2048 / b8192 / SC-5 / shuffled / none / seeds) so derived AvgEns rows and every paired
comparison path (H1–H5) are exercised, then make_figures / make_tables on the output."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tbf import summarize as S
from tbf.config import RunConfig
from tbf.data.fixtures import load_windows
from tbf.tsfm import make_row


def _llm_row(cfg: RunConfig, w, rng, noise, n_samples=1, think_tokens=0.0):
    tgt = np.asarray(w["target"], float)
    point = tgt * (1 + rng.normal(0, noise, len(tgt)))
    samples = [(tgt * (1 + rng.normal(0, noise, len(tgt)))).tolist() for _ in range(n_samples)] if n_samples > 1 else None
    return {"config_id": cfg.config_id, "window_id": w["window_id"], "dataset": "fixtures", "model": cfg.model, "setup": cfg.setup,
            "thinking": cfg.thinking, "budget": cfg.budget, "context_mode": cfg.context_mode, "n_samples": cfg.n_samples,
            "seed": cfg.seed, "prior": None, "point": point.tolist(), "samples": samples, "valid": bool(rng.random() > 0.1),
            "n_values_mismatch": False, "raw_answer": "", "thinking_tokens_used": think_tokens, "answer_tokens": 60.0,
            "budget_hit": cfg.thinking and rng.random() < 0.3, "finished": True, "wall_s": 1.0, "prompt_tokens": 2500}


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("results")
    rng = np.random.default_rng(0)
    wins = load_windows()
    rows = {}
    for w in wins:
        m = int(w["seasonality"])
        from tbf.metrics import seasonal_naive_forecast
        rows.setdefault("fixtures__snaive", []).append(make_row("fixtures", "snaive", w, seasonal_naive_forecast(w["context"], 12, m), None, 0.0))
        tgt = np.asarray(w["target"], float)
        q = np.stack([tgt * (1 + 0.05 * (lvl - 0.5)) for lvl in np.linspace(0.1, 0.9, 9)], axis=1)
        rows.setdefault("fixtures__chronos2", []).append(make_row("fixtures", "chronos2", w, tgt * 1.01, q, 0.0))
    grid = [("direct", "off", "full", 1, 0), ("direct", "b512", "full", 1, 0), ("direct", "b2048", "full", 1, 0), ("direct", "b8192", "full", 1, 0),
            ("direct", "off", "full", 5, 0), ("direct", "off", "shuffled", 1, 0), ("direct", "b2048", "shuffled", 1, 0),
            ("direct", "off", "none", 1, 0), ("direct", "b2048", "none", 1, 0), ("direct", "off", "full", 1, 1), ("direct", "b2048", "full", 1, 2),
            ("reviser", "off", "full", 1, 0), ("reviser", "b2048", "full", 1, 0)]
    budgets = {"b512": 512, "b2048": 2048, "b8192": 8192}
    for setup, think, ctx, n, seed in grid:
        cfg = RunConfig("fixtures", "qwen3-8b", setup, think != "off", budgets.get(think, 0), ctx, n, seed)
        tt = {"off": 0.0, "b512": 400.0, "b2048": 1500.0, "b8192": 5000.0}[think]
        for w in wins:
            rows.setdefault(cfg.config_id, []).append(_llm_row(cfg, w, rng, 0.05 if think == "off" else 0.04, n, tt))
    for cid, rs in rows.items():
        with open(d / f"{cid}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    return d


def test_summarize_end_to_end(results_dir, tmp_path):
    out, paired = tmp_path / "summary.csv", tmp_path / "paired.csv"
    S.main(["--results-glob", str(results_dir / "*.jsonl"), "--dataset", "fixtures", "--out", str(out), "--paired-out", str(paired)])
    summ, pr = pd.read_csv(out), pd.read_csv(paired)
    head = summ[(summ.domain == "overall") & (summ.stratum == "all")].set_index("config_id")
    assert head.loc["fixtures__snaive", "rel_mase"] == pytest.approx(1.0)
    assert 0 < head.loc["fixtures__chronos2", "rel_mase"] < 1 and np.isfinite(head.loc["fixtures__chronos2", "wql"])
    # derived AvgEns rows exist for both off and b2048
    assert "fixtures__qwen3-8b__avgens_off" in head.index and "fixtures__qwen3-8b__avgens_b2048" in head.index
    # SC-5 gets a WQL from samples; single-sample configs do not
    assert np.isfinite(head.loc["fixtures__qwen3-8b__direct__off__full__s0__n5", "wql"])
    assert np.isnan(head.loc["fixtures__qwen3-8b__direct__off__full__s0", "wql"])
    # seed aggregation row
    assert "fixtures__qwen3-8b__direct__off__full__s*" in head.index and head.loc["fixtures__qwen3-8b__direct__off__full__s*", "n_seeds"] == 2
    # strata below 50 windows are written with NaN rel_mase but n_windows
    ev = summ[(summ.config_id == "fixtures__qwen3-8b__direct__off__full__s0") & (summ.domain == "overall") & (summ.stratum == "event")]
    assert len(ev) == 1 and np.isnan(ev.rel_mase.iloc[0]) and ev.n_windows.iloc[0] == 12
    # window-weighted variant present
    assert "overall_ww" in set(summ.domain)
    # paired comparisons: every hypothesis path produced rows
    hyps = set(pr.hypothesis)
    assert {"H1", "H3", "H4", "H5", "H5_dd"} <= hyps, hyps
    h1 = pr[(pr.hypothesis == "H1") & (pr.label == "on@2048 vs off") & (pr.domain == "overall") & (pr.setup == "direct")]
    assert len(h1) == 1 and h1.n_series.iloc[0] == 12 and h1.lo.iloc[0] <= h1.delta_log_rel_mase.iloc[0] <= h1.hi.iloc[0]
    assert set(pr[pr.hypothesis == "H1"].domain) >= {"overall", "web", "health", "energy", "macro"}
    h3 = pr[pr.hypothesis == "H3"]
    assert len(h3) >= 1 and h3.b_config_id.iloc[0].endswith("__n5")
    assert any("avgens_off vs best" in str(x) for x in pr[pr.hypothesis == "H4"].label)
    assert len(pr[pr.hypothesis == "H5"]) == 4  # shuffled/none x off/b2048 for direct
    # figures and tables run on these files
    subprocess.run([sys.executable, "scripts/make_figures.py", "--summary", str(out), "--paired", str(paired), "--out", str(tmp_path / "figs")],
                   check=True, capture_output=True)
    assert (tmp_path / "figs" / "fig1_budget_curves.pdf").exists() and (tmp_path / "figs" / "fig3_context_sensitivity.png").exists()
    subprocess.run([sys.executable, "scripts/make_tables.py", "--summary", str(out), "--paired", str(paired), "--out", str(tmp_path / "tables")],
                   check=True, capture_output=True)
    t1 = (tmp_path / "tables" / "table1_main.tex").read_text()
    assert "\\toprule" in t1 and "8b direct b2048" in t1 and "AvgEns(off)" in t1
