# Thinking Budgets for Forecasting

Does chain-of-thought help text-conditioned time-series forecasting on leakage-free data? A same-weights, budget-controlled study with open hybrid-reasoning LLMs (Qwen3), plus **FreshTS-26**, a refreshable text-conditioned forecasting evaluation set whose forecast origins all post-date the models' release.

Target: NeurIPS 2026 FMTS workshop. Pre-registration and full design: [`SPEC.md`](SPEC.md). Human steps: [`RUNBOOK.md`](RUNBOOK.md). Implementer instructions: [`CLAUDE.md`](CLAUDE.md) (Claude Code reads it automatically; `IMPLEMENTER_PROMPT.md` for other agents).

```
make test         # unit tests (CPU)
make smoke        # end-to-end on 12 fixture windows with Qwen3-0.6B on CPU, < 10 min
make preflight    # internet needed: one request per data endpoint + Qwen3-0.6B tokenizer, PASS/FAIL per item
make build-data   # build FreshTS-26 (needs internet; run on the cluster login node)
make summary      # results/*.jsonl -> results/summary.csv, results/paired.csv
make figures      # figures/*.pdf|png from summary.csv only
```

Environments: `requirements-llm.txt` (vLLM 0.28.0, GPU) and `requirements-tsfm.txt` (TSFM baselines, metrics, data build; CPU OK).

Data license: numbers CC0 / public domain (Wikimedia, FRED/Federal Reserve, EIA, CDC); text CC BY-SA 4.0 (Wikipedia) and public domain (CDC, EIA). Dataset release: CC BY-SA 4.0. Code: MIT.
