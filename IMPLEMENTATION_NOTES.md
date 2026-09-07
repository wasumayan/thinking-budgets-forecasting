# IMPLEMENTATION_NOTES — what was built, what deviates, what the human still has to do

Implemented 2026-09-07 from `SPEC.md` v1.0 per `CLAUDE.md`. `make test` (51 tests) and `make smoke` are green on a
CPU-only machine (Python 3.11, `requirements-tsfm.txt` + `transformers` + `pytest`).

## What was built (order of work as in CLAUDE.md)

| module | status | notes |
|---|---|---|
| `prompts.py` + `tests/snapshots/` | done | Byte-identical snapshots for `web__Influenza__2025-09-05` × {direct, reviser} × {full, none, shuffled}. |
| `llm_hf.py` (`HFBackend`) | done | Qwen's transformers reference logic; all three budget branches exercised (finished in budget, natural `</think>`, forced early stop). |
| `llm_vllm.py` (`VLLMBackend.generate`) | done, **not executed** (no GPU/vLLM here) | Built on the already-tested `two_pass_budgeted`; adds `render_prompt` (enable_thinking rules per model family), `sample_seeds` (= `seed*1000+i`), `early_stop_prefix_len`. |
| `tsfm.py` | done | `snaive` executed on fixtures. `chronos2` / `timesfm25` / `tirex` use the exact §B snippets and were checked against the installed package signatures (chronos-forecasting 2.3.1, timesfm 3.0.1, tirex-ts 1.4.2) but **could not be run** because huggingface.co is blocked from this sandbox. |
| `run.py` | done | Both CLI forms; resumable; batch 64; imputation (reviser → prior, direct → seasonal naive); one-line SUMMARY log; `--model-path` for local checkpoints. Extra columns beyond §7.2: `raw_thinking` (≤ 2000 chars), `n_valid_samples`, `parse_reason`. |
| `summarize.py` | done | Per-window MASE/MAE/WQL, derived AvgEns, strata, series- and window-weighted geo-means with bootstrap CIs, seed rows, `paired.csv` for H1–H5 (+ H5 delta-of-deltas). CiK rows with samples get an `rcrps` column (weighted mean of min(RCRPS, 5)). |
| `scripts/make_figures.py`, `make_tables.py` | done | Read only `summary.csv` / `paired.csv`. Tested on a synthetic full grid (`tests/test_summarize.py`). |
| `data/http.py` + fetchers | done, **network untested** | Every parser has an offline test with realistic inputs (`tests/test_fetchers.py`); the HTTP layer is tested with a mocked `requests` (cache hit, 503→429→200 retry, 404 not retried, User-Agent with `TBF_CONTACT_EMAIL`). |
| `data/build_freshts26.py` + `tests/test_leakage.py` | done, **network untested** | Windowing, forward-fill, NaN-target drop, calendar, event matching, tier-2 report attachment, shuffle map, leakage assertion and parquet round trip are unit-tested on synthetic series (`tests/test_build.py`). |
| `data/cik.py` | done, **not executed** | HF loader + RCRPS (PWM-CRPS, ROI weighting, constraint penalties) per `docs/verify-datasets.md` §3.4; RCRPS unit-tested on synthetic rows. |
| Slurm scripts | reviewed, unchanged | See "human steps" below for the array size. |
| `make preflight` (`scripts/preflight.py`) | new | One request per endpoint (Wikimedia pageviews, Wikipedia revisions/parse/summary, FRED CSV + series page, Delphi fluview, CDC FluView) + Qwen3-0.6B tokenizer (checks ids 151668/151645 and the `enable_thinking=False` template), PASS/FAIL per item, exit 1 on any failure. |
| `scripts/make_tiny_model.py` | new | Offline stand-in for `make smoke` (see below). |

## Environment caveat: huggingface.co was unreachable

The sandbox's egress policy denies `huggingface.co` (HTTP 403 on CONNECT), as well as every data endpoint
(wikimedia.org, en.wikipedia.org, fred.stlouisfed.org, delphi.cmu.edu). PyPI was reachable, so all packages
installed. Consequences:

- `make smoke` could not download `Qwen/Qwen3-0.6B`. To exercise the full CPU path, `scripts/make_tiny_model.py`
  builds a 2-layer `Qwen3ForCausalLM` (9.8 M params, same special-token ids, a Qwen3-style chat template with
  `enable_thinking`) with a character-level tokenizer and trains it for ~5 min on CPU to answer
  `{"forecast": [101, …, 112]}` (with a short `<think>` block when thinking is on). Run:
  `python scripts/make_tiny_model.py && TBF_MODEL_PATH=data/tiny-qwen3-standin make smoke`.
  It is **not** Qwen3-0.6B and is never used for paper numbers; on your laptop/cluster run plain `make smoke`.
- `make smoke` runtime with the stand-in: **30–32 s wall** (snaive on 12 windows, 4 windows thinking-off, 4 windows
  budget 64, summarize, fig1). Fixture valid-output rate: **1.000** in both LLM configs (8/8 rows), mean thinking
  tokens 9.0 at budget 64, budget-hit rate 0. With the real 0.6B model expect minutes rather than seconds and a
  lower valid rate; the pipeline (imputation, `valid_only` stratum) handles invalid outputs.
- Nothing network-facing has been run against a live endpoint. `make preflight` is the first thing to run on the
  login node; it prints exactly what each endpoint returned.

## Deviations (mirrors `SPEC.md` → "Deviations (implementation)")

1. CiK prompts substitute the actual history/horizon lengths for the hard-coded 96/12 and render no Calendar/Events
   sections; `parse_forecast` takes an optional `h`. FreshTS-26 prompts are verbatim (snapshot-tested).
2. CiK sub-daily timestamps render as `YYYY-MM-DD HH:MM`.
3. The seasonal-naive baseline of a summary cell is restricted to the cell's own windows; the 50-window minimum applies
   to the four non-`all` strata; `_ww` = ratio of window means bootstrapped over windows.
4. Current-events topic-heading bullets are skipped but their wikilinks are inherited by child bullets.
5. Multi-sample rows: mean token counts, `budget_hit` = any, `finished` = all, `samples` = valid samples only,
   `valid` = ≥ 1 valid sample.
6. `TBF_MODEL_PATH` for the smoke test on machines without Hub access.

Other implementation choices worth knowing (not deviations):

- `thinking_tokens_used` counts ids up to and including `</think>` (Qwen's reference `index`), minus the injected
  early-stop string when forced — exactly what `tests/test_budget_forcing.py` pins.
- H3 pairs SC-5 with the thinking budget whose mean *total* generated tokens (thinking + answer) is closest to
  5 × SC-5's mean answer tokens; the chosen budget and both spends are in the `label` column of `paired.csv`.
- `paired.csv` has extra `label` and `domain` columns (H1/H2 rows exist per domain for Fig 2).
- Web series match events by title link **or** cluster keyword with no category filter (SPEC §2.1); other domains
  require category ∈ `event_categories` **and** a keyword. Keywords of ≤ 3 characters use word boundaries; note that
  spec keywords like `who`, `fed`, `won`, `real`, `final` will also match ordinary English words.
- FRED metadata: title/units/frequency/seasonal adjustment/notes are scraped from the series page (or the API when
  `FRED_API_KEY` is set); if the scrape yields nothing the `series.yaml` title is used and the preflight reports it.
- Delphi fluview: one request for all 31 regions; if the API does not answer `result == 1` (row cap), it falls back
  to per-region requests automatically.
- `results/summary.csv`, `results/paired.csv` and `figures/fig1_budget_curves.*` currently hold the **stand-in smoke
  output**; they are overwritten by `make summary` / `make figures` on the cluster.

## Human steps on the cluster beyond RUNBOOK.md

1. `export TBF_CONTACT_EMAIL=<you>@princeton.edu` then `make preflight` on `della-gpu` **before** `make build-data`.
   If the FRED page scrape fails, either set `FRED_API_KEY` (free) or accept `series.yaml` titles.
2. `make build-data` writes `data/freshts26/BUILD.json`; check `dropped` and `web_summary_not_asof` (articles whose
   2025-07-31 revision could not be fetched and fell back to the current summary). Then
   `pytest tests/test_leakage.py` runs the real-file audit (≥ 1,200 windows, ≥ 100 series, ≥ 3 domains).
3. `scripts/slurm/llm_array.sbatch` has `--array=0-29%8` (tier 1 = 30 configs). For tier 2 (24) / tier 3 (16) pass
   `sbatch --array=0-23%8 …` / `--array=0-15%8` (extra indices exit 0 harmlessly, but they occupy the queue).
4. `run.py` uses vLLM's `language_model_only=True` for `gemma3-4b` (multimodal architecture); Llama-3.1 is gated
   (accept the licence + `hf auth login`).
5. Tier 2 reports: `python -m tbf.data.build_freshts26 --tier2` needs `pdftotext` (poppler) or `pip install pypdf`
   for the EIA WPSR PDFs.
6. CiK (tier 3): `hf download ServiceNow/context-is-key --repo-type dataset` on the login node; `tbf.data.cik`
   loads it through `datasets` with `HF_HUB_OFFLINE=1`.
7. Go/no-go: `python -m tbf.run --config-id freshts26__qwen3-8b__direct__off__full__s0 --limit 50` prints
   `SUMMARY config=… valid_rate=… mean_thinking_tokens=… budget_hit_rate=…`; the reviser configs need
   `results/freshts26__chronos2.jsonl` first (`make tsfm` or `tsfm.sbatch`).
