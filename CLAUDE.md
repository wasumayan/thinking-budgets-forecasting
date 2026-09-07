# CLAUDE.md — instructions for Claude Code working in this repo

This is a research codebase implemented from a **pre-registered specification**. The paper deadline is
2026-09-16 11:59 UTC. Read, in this order, before writing code: `SPEC.md` (authoritative), `configs/series.yaml`,
`configs/grid.yaml`, `configs/prompts/README.md`, then `docs/verify-models.md` (§A.3 budget forcing, §B TSFM APIs,
§C metric definitions — copy those snippets, do not reinvent), `docs/verify-fresh-data.md` (API endpoints),
`docs/verify-datasets.md` (CiK schema, fev usage).

## Non-negotiables
1. Implement `SPEC.md` exactly. Do not change the grid, prompts, metrics, seasonality, window rules, or hypotheses.
   If something is impossible or ambiguous, add the smallest change to `SPEC.md` under a heading
   `## Deviations (implementation)` with the reason, then implement it. Never silently diverge.
2. Keep the function signatures in `src/tbf/`; every stub's docstring is its contract. Add helpers freely.
3. `make test` and `make smoke` must pass on a CPU-only machine (Python 3.11, `requirements-tsfm.txt` +
   `transformers`). vLLM is imported lazily; never required for tests/smoke.
4. Results files are append-only, resumable JSONL per config (`SPEC.md` §7.2): skip `window_id`s already present.
5. Every network fetch caches its raw response under `data/raw/<source>/` with a fetch timestamp and is skipped if
   cached. Send the `User-Agent` from `series.yaml` with `<EMAIL>` replaced by env `TBF_CONTACT_EMAIL`.
   Retry with exponential backoff (max 5) on 429/5xx; ≥ 0.2 s between calls to the same host.
6. Dependencies: only what is in the two requirements files plus `pyyaml`, `tqdm`. No notebooks.
7. Figures read only `results/summary.csv` / `results/paired.csv`, never raw JSONL.
8. Logging via `logging`, INFO to stdout; each config run ends with a one-line summary (n windows, valid rate,
   mean thinking tokens, budget-hit rate, wall time).

## Order of work (definition of done = SPEC.md §8)
`prompts.py` + `tests/test_prompts.py` snapshots (run `pytest --snapshot-update` once, inspect the rendered
prompts, commit `tests/snapshots/`) → `llm_hf.py` (CPU smoke path, `Qwen/Qwen3-0.6B`) → `tsfm.py` (`snaive` first,
then chronos2 / timesfm25 / tirex) → `run.py` → `summarize.py` → `scripts/make_figures.py` + `make_tables.py` →
`make smoke` green → `llm_vllm.py::VLLMBackend.generate` (uses the already-tested `two_pass_budgeted`) →
`data/http.py` → fetchers → `build_freshts26.py` + `tests/test_leakage.py` → `data/cik.py` (tier 3) → Slurm scripts.

Already implemented and tested (do not rewrite): `config.py`, `metrics.py`, `parse.py`,
`llm_vllm.two_pass_budgeted`, `scripts/expand_grid.py`, `scripts/make_fixtures.py`, `data/fixtures/`.

## Commands
- `make test` — unit tests. `make smoke` — end-to-end on 12 fixture windows on CPU (< 10 min).
- `make build-data` — build FreshTS-26 (internet required; on the cluster this runs on the login node).
- `python -m tbf.run --config-id <id> --limit 50` — one config (go/no-go check on the cluster).
- `python scripts/expand_grid.py --tier 1 --with-time` — config ids for a Slurm array.

## When done
Write `IMPLEMENTATION_NOTES.md`: what was built, deviations (mirroring SPEC), `make smoke` runtime, and anything
the human must do on the cluster beyond `RUNBOOK.md`. Commit in small steps with descriptive messages.
