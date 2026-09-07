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
"""
from __future__ import annotations


def main(argv=None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
