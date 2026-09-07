# Instructions for a non-Claude-Code implementer (e.g. Codex). Claude Code users: CLAUDE.md is read automatically.

Paste this whole file as the first message, from the repo root.

---

You are implementing a research codebase from a pre-registered specification. Read, in this order: `SPEC.md` (authoritative), `configs/series.yaml`, `configs/grid.yaml`, `configs/prompts/README.md`, then `docs/verify-models.md` (§A.3 budget forcing, §B TSFM APIs, §C metric definitions — copy those snippets, do not reinvent), `docs/verify-fresh-data.md` (API endpoints), `docs/verify-datasets.md` (CiK schema, fev usage).

Rules:
1. Implement `SPEC.md` exactly. Do not change the grid, prompts, metrics, seasonality, window rules, or hypotheses. If something in the spec is impossible or ambiguous, write the smallest change into `SPEC.md` under a new heading `## Deviations (implementation)` with the reason, then implement it. Never silently diverge.
2. Work through the stubs in `src/tbf/` — every function there has a docstring stating its contract; keep the signatures. Add helpers freely.
3. Definition of done is `SPEC.md` §8. Order of work: `metrics.py` + `tests/test_metrics.py` → `parse.py` + tests → `prompts.py` + snapshot test on `data/fixtures/` → `llm_hf.py` (CPU smoke path with `Qwen/Qwen3-0.6B`) → `tsfm.py` (`snaive` first, then the three TSFMs) → `summarize.py` → `scripts/make_figures.py` → `make smoke` green → `llm_vllm.py` (two-pass budget forcing; unit-test with a fake LLM) → data fetchers → `build_freshts26.py` + `tests/test_leakage.py` → Slurm scripts.
4. `make test` and `make smoke` must pass on a CPU-only machine with Python 3.11 and `requirements-tsfm.txt` + `transformers` installed (vLLM is not required for tests/smoke; import it lazily).
5. Results are append-only, resumable JSONL per config (`SPEC.md` §7.2). Never overwrite a results file; skip window_ids already present.
6. Every network fetch caches its raw response under `data/raw/<source>/...` with a fetch timestamp and is skipped if cached. Send the `User-Agent` from `series.yaml` (fill `<EMAIL>` from env `TBF_CONTACT_EMAIL`). Retry with exponential backoff (max 5) on 429/5xx.
7. Logging: `logging` module, INFO to stdout; each config run prints a one-line summary at the end (n windows, valid rate, mean thinking tokens, wall time).
8. Keep dependencies to what is in the two requirements files plus `pyyaml`, `tqdm`. No notebooks.
9. Commit in small steps with descriptive messages. When done, write `IMPLEMENTATION_NOTES.md`: what was built, what was deviated (mirroring SPEC deviations), how long `make smoke` takes, and anything the human must do on the cluster beyond `RUNBOOK.md`.

Start by running `make test` (expect failures), then implement until green.
