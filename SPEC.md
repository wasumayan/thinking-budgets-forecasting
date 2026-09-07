# SPEC — Thinking Budgets for Forecasting

**Paper working title:** *Thinking Budgets for Forecasting: Does Chain-of-Thought Help Text-Conditioned Time-Series Forecasting on Leakage-Free Data?*

**Target:** NeurIPS 2026 Workshop on Foundation Models for Temporal Systems (FMTS), Sydney. Deadline **2026-09-16 11:59 UTC (07:59 ET Wed)**. 4 content pages + unlimited references/appendix, NeurIPS 2026 template, double-blind, non-archival, OpenReview. Notification Sept 29.

**Spec version:** 1.0 (2026-09-06). Any change to the grid, prompts, metrics, or hypotheses must be written into this file *before* code changes. `SPEC.md` is the pre-registration.

**Companion documents (read before implementing):** `docs/verify-models.md` (verified model/tooling APIs, vLLM budget forcing, Della facts), `docs/verify-datasets.md` (dataset facts, CiK schema, fev metrics), `docs/verify-fresh-data.md` (data sources, endpoints, licensing), `docs/FMTS-sprint-report.md` (literature, gaps, workshop facts).

---

## 0. One-paragraph summary

Two 2026 papers disagree about whether LLM reasoning helps text-conditioned time-series forecasting: AlphaCast and Time-R1 report large gains from reasoning, but on classic datasets the models have memorized; TimesX (ICML 2026) finds that reasoning models give "no clear advantage" on leakage-free data and that a simple average of a time-series foundation model (TSFM) and an LLM beats agentic loops — but TimesX only tested closed models and did not isolate reasoning from everything else that differs between models. We run the clean experiment: **same weights, thinking on vs. off, with a controlled reasoning-token budget**, on data whose forecast targets all fall **after the models' release dates**, with a matched-compute control and a context ablation. Because no public leakage-free text-conditioned dataset exists (TimesX was never released; Time-MMD ends May 2024), we build and release one — **FreshTS-26** — from public, keyless APIs, with every text snippet timestamped so leakage is checkable.

## 1. Research questions and pre-registered hypotheses

**RQ1.** For hybrid-reasoning open LLMs (Qwen3), does enabling chain-of-thought — and scaling its token budget — reduce forecast error on leakage-free, text-conditioned forecasting, relative to the same weights with thinking off?

**RQ2.** If thinking helps, *where*: only on windows whose context contains a relevant event, or everywhere?

**RQ3.** Is any benefit of thinking just a benefit of spending more tokens? (Compare against self-consistency at matched token spend.)

**RQ4.** Does the TimesX finding — AvgEns(TSFM, LLM) beats all LLM-only and reviser configurations — replicate with open ≤8B models?

**RQ5 (exploratory).** Does thinking amplify *misuse* of context (larger degradation under shuffled context)?

Hypotheses. The *primary metric* is relative MASE (§6). "Better" means lower. CIs are 95% paired bootstrap over windows (§6.4).

- **H1 (primary, null-leaning).** On FreshTS-26, for each (model size, setup) pair, thinking-on at budget 2048 does *not* improve relative MASE over thinking-off: the paired-bootstrap 95% CI of Δ = MASE(on@2048) − MASE(off) includes 0 or is entirely > 0. *Falsified if* on@2048 beats off with a CI entirely < 0 for ≥ 2 of the 3 sizes in ≥ 1 setup (direct or reviser). Budgets 512 and 8192 are reported as the dose-response.
- **H2 (mechanism).** The effect of thinking (Δ as in H1) differs between windows with ≥ 1 matched event bullet (`n_events ≥ 1`) and windows with none: |Δ_event| > |Δ_no-event| with non-overlapping CIs for Qwen3-8B direct. *Falsified if* the two strata's CIs overlap for all sizes.
- **H3 (matched compute).** Self-consistency with thinking off (SC-5: median of 5 samples) is at least as good as thinking-on at the budget whose *actual* mean token spend is closest to SC-5's. *Falsified if* thinking-on beats SC-5 with a CI entirely < 0 for ≥ 2 of 3 sizes.
- **H4 (ensemble, replication).** AvgEns(Chronos-2, LLM-direct-off) has lower relative MASE than every LLM-only and reviser configuration for each size. *Falsified if* any LLM configuration beats AvgEns with a CI entirely < 0.
- **H5 (exploratory, no falsification threshold).** MASE(shuffled) − MASE(full) is larger with thinking on than off.

Reporting rule: every hypothesis gets one subsection in Results whose first sentence states the deciding number. Null results are reported as results.

## 2. Data

### 2.1 FreshTS-26 (primary; we build it)

A text-conditioned univariate forecasting evaluation set in which every forecast origin is **≥ 2025-08-01** (after the release of every LLM in the pool: Qwen3 2025-04-29, Qwen3-2507 2025-07-25, Gemma-3 2025-03, Llama-3.1 2024-07) and every text item carries an `available_at` timestamp ≤ the origin. Built on the Della login node (compute nodes have no internet). Numbers are public domain / CC0; text is CC BY-SA 4.0 (Wikipedia) and public domain (CDC/EIA/FRED). Release license: CC BY-SA 4.0.

**Series (`configs/series.yaml` is authoritative; ~125 series, 4 domains):**

| domain | series | source | freq | seasonality m |
|---|---|---|---|---|
| `web` | 60 English Wikipedia articles, 6 clusters × 10 (health, economy, energy, geopolitics, science-tech, sports-entertainment); daily user pageviews | Wikimedia REST `pageviews/per-article/en.wikipedia/all-access/user/<title>/daily/...` | D | 7 |
| `health` | ILINet `ili` (% outpatient visits for ILI) for `nat`, `hhs1`–`hhs10`, and 20 states | Delphi Epidata `fluview` | W | 1 |
| `energy` | 7 EIA weekly series mirrored on FRED: `GASREGW, GASDESW, WCESTUS1, WGTSTUS1, WDISTUS1, WCRFPUS2, WGFUPUS2` | FRED `fredgraph.csv?id=` (no key) | W | 1 |
| `macro` | 18 FRED business-daily series: 12 H.10 FX pairs (`DEXUSEU, DEXJPUS, DEXUSUK, DEXCAUS, DEXCHUS, DEXMXUS, DEXINUS, DEXBZUS, DEXKOUS, DEXSZUS, DEXUSAL, DEXSFUS`), 3 Treasury yields (`DGS2, DGS10, DGS30`), 3 energy prices (`DCOILWTICO, DCOILBRENTEU, DHHNGSP`); plus 2 weekly labor series (`ICSA, CCSA`) | FRED `fredgraph.csv?id=` | B / W | 1 |

Do **not** include S&P/Dow/Nasdaq/VIX (redistribution terms). Drop any series with > 5 % missing values in the evaluation range; log the drop.

**Windows.** Rolling windows per series with context length **T = 96** and horizon **H = 12** steps (TimesX convention) at the series' native frequency. Origin `o` = timestamp of the last context point. Constraints: `o ≥ 2025-08-01`; the full horizon must be observed as of the build date. Stride: 24 steps for daily/business-daily series, 4 steps for weekly. Missing values inside the context (FRED `.`) are forward-filled and flagged (`n_ffill`); windows with any missing target are dropped. Expected size: ≈ 950 (web) + ≈ 340 (health) + ≈ 80 (energy) + ≈ 260 (macro) ≈ **1,600 windows**. Also record the subset with `o ≥ 2026-07-01` (`strict_2026` flag) for future models with 2026 cutoffs.

**Evaluation is against final (revised) values** for ILI/ICSA (state this in the paper's limitations; vintages via Delphi `issues=` are a documented extension, not in scope).

**Text attached to each window** (all items must satisfy `available_at ≤ o`; a unit test asserts `text_available_at_max ≤ origin_ts` for every window):

1. `metadata` (str): what the series is. Web: Wikipedia `page/summary/<title>` `extract` **as of the revision current on 2025-07-31** (use the MediaWiki `revisions` API with `rvstart=2025-07-31T23:59:59Z&rvdir=older&rvlimit=1`, then `action=parse` on that `revid` to get the lead paragraph; fall back to the current summary only if the historical fetch fails, and flag it). FRED: series `title`, `units`, `frequency`, `seasonal_adjustment`, and the `notes` field (from the FRED series page or API). ILI: fixed template "Percentage of outpatient visits for influenza-like illness reported to ILINet, {region_name}; weekly; source CDC FluView via Delphi Epidata." Always append: "Frequency: {freq}. Prediction target period: {h_start} to {h_end}."
2. `calendar` (str): US federal holidays (python `holidays.US()`) falling inside the horizon, listed with dates; "None" if none.
3. `events` (list, K ≤ 10, newest first; each `{date, category, text, links}`): bullets from Wikipedia **Portal:Current events** daily pages for the 28 days ending at `o` (inclusive), selected by: for web series, any `[[wikilink]]` target equal to the article title (redirect-resolved) **or** any cluster keyword (from `series.yaml`) appearing case-insensitively in the bullet text; for other domains, category ∈ the series' `event_categories` **and** ≥ 1 series keyword match. Strip wiki markup and citation links; keep the plain sentence. `available_at` = end of that day UTC.
4. `reports` (list, ≤ 2 most recent; **tier 2, optional**): health → CDC FluView "Key Points" bullets from `cdc.gov/fluview/surveillance/<YYYY>-week-<WW>.html` with `available_at` = the Friday of week w+1 (publication day); energy → EIA WPSR highlights paragraph (`ir.eia.gov/wpsr/wpsrsummary.pdf` archive; `pdftotext`) with `available_at` = Wednesday 10:30 ET release. `[]` when not built or not applicable.

Serialized to `data/freshts26/windows.parquet` with one row per window (schema §7.1) plus `data/freshts26/series.parquet` (series metadata) and `data/freshts26/BUILD.json` (build timestamp, per-source fetch counts, drops). Raw API responses are cached under `data/raw/` with fetch timestamps so the build is reproducible offline.

### 2.2 CiK — Context is Key (contrast condition; tier 3)

HF dataset `ServiceNow/context-is-key` (355 rows = 71 tasks × 5 instances; Apache-2.0; no keys). Synthetic textual contexts over real series; contamination of the *numbers* is possible, so it is reported only as the "synthetic-context" contrast where TimesX predicts reasoning might look useful. Metric: the bundled RCRPS (`compute_rcrps_with_hf_dataset.py`, see `docs/verify-datasets.md`) plus MASE on the median. Requires sample forecasts → 5 samples per instance.

### 2.3 Time-MMD (appendix only; tier 3, optional)

Pre-cutoff; contamination-risky. If run, report in an appendix table flagged as such, for comparability with prior work only. Not needed for any hypothesis.

## 3. Models

### 3.1 LLMs (served with vLLM 0.28.0; bf16; one A100-80GB each)

| role | HF id | thinking control | notes |
|---|---|---|---|
| primary hybrid family | `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-1.7B` | `chat_template_kwargs={"enable_thinking": bool}`; budget via two-pass forcing (§3.3) | text-only dense transformers; V2 runner |
| different-weights thinking pair (tier 3) | `Qwen/Qwen3-4B-Thinking-2507` vs `Qwen/Qwen3-4B-Instruct-2507` | Thinking-2507 is thinking-only (template pre-fills `<think>\n`); Instruct-2507 is non-thinking-only | robustness check that the on/off effect is not a hybrid-training artifact |
| non-thinking controls (tier 3) | `google/gemma-3-4b-it`, `meta-llama/Llama-3.1-8B-Instruct` (gated; optional) | n/a | Gemma-3-4B is PostTime's base |
| stretch | `Qwen/Qwen3-30B-A3B` | same as primary | only if GPUs are idle after tier 2; MoE → two-pass forcing only |

Sampling (Qwen model card): thinking **on**: `temperature=0.6, top_p=0.95, top_k=20, min_p=0`; thinking **off**: `temperature=0.7, top_p=0.8, top_k=20, min_p=0`. Never greedy in thinking mode. `seed` set per run. `max_model_len=40960` for Qwen3. Answer-phase `max_tokens=1024`.

Knowledge cutoffs to state in the paper: Llama-3.1 Dec 2023 (official), Gemma-3 Aug 2024 (official), Qwen3 undocumented — we use *release dates* (2025-04-29; 2507 variants 2025-07-25) as the upper bound on training data and require all forecast origins ≥ 2025-08-01.

### 3.2 TSFM priors / baselines (env B)

| model | HF id | package | call | quantiles |
|---|---|---|---|---|
| Chronos-2 (**primary prior**) | `amazon/chronos-2` | `chronos-forecasting==2.3.1` | `Chronos2Pipeline.from_pretrained(...).predict_quantiles(contexts, prediction_length=12, quantile_levels=[0.1,…,0.9])` | 9 |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | `timesfm[torch]==3.0.1` | `TimesFM_2p5_200M_torch.from_pretrained(...)`, `compile(ForecastConfig(max_context=1024, max_horizon=12, normalize_inputs=True, use_continuous_quantile_head=True, fix_quantile_crossing=True))`, `forecast(horizon=12, inputs=[...])` | mean + 9 |
| TiRex | `NX-AI/TiRex` | `tirex-ts==1.4.2` (`backend="torch"`, no CUDA build) | `load_model("NX-AI/TiRex").forecast(context=..., prediction_length=12)` | 9 |
| Seasonal naive | — | numpy | last-season repeat with `m` from §2.1 | point only |

All TSFMs get exactly the 96-point context (no text). Exact snippets are in `docs/verify-models.md` §B. TSFM training-data cutoffs are undocumented except TimesFM (Nov 2023); state in limitations.

### 3.3 Thinking budget forcing (must be implemented exactly like this)

Qwen's official two-pass recipe (`docs/verify-models.md` §A.3), batched in vLLM offline mode:

1. Render the prompt with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)`.
2. Pass 1: generate with thinking-mode sampling and `max_tokens = budget`.
3. For each output: if `<|im_end|>` (id 151645) was generated → done. Else, if `</think>` (id 151668) was **not** generated, append the exact early-stopping string `"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"`; then pass 2: continue generation (prompt = prompt ids + generated ids [+ early-stop ids]) with `max_tokens = 1024` (answer phase).
4. Split at the last `</think>`; record `thinking_tokens_used` (tokens before the split, excluding the injected string), `answer_tokens`, `budget_hit` (bool), `finished` (bool).

Budget 0 ("off") uses the hard switch `enable_thinking=False` — never forcing. Do **not** use vLLM's native `thinking_token_budget` for the main grid (path differs between dense and MoE/hybrid models); it may be used only as a cross-check on Qwen3-8B and, if used, must be reported as such.

## 4. Prompts (full text in `configs/prompts/`; do not paraphrase)

**System (all conditions):** `configs/prompts/system.txt` — "You are a careful forecasting assistant. You will be given a univariate time series, information about what it measures, and possibly relevant recent events and reports. Produce a point forecast for the requested horizon. Answer with a single JSON object and nothing else after it."

**Direct forecasting user prompt** (`direct.txt`), sections in this order: `## Series` (metadata), `## Calendar` (calendar), `## Recent events (dated; newest first)` (events, or "None"), `## Reports` (reports, or omitted if tier 2 not built), `## History (96 values; date: value)`, `## Task` ("Forecast the next 12 values for {h_start}..{h_end}. Return exactly: {"forecast": [v1, ..., v12]} with 12 numbers.").

**Reviser user prompt** (`reviser.txt`): identical, plus a `## Statistical model forecast` section before `## Task` containing the Chronos-2 median forecast for the 12 target dates, and the Task text "A statistical model that saw only the numbers produced the forecast above. Using the context, revise it where the context warrants and keep it where it does not. Return exactly: {"forecast": [...]}."

**Context modes** (`context_mode`): `full` = all sections; `none` = `## Series` reduced to title + units only, no calendar/events/reports; `shuffled` = real metadata and calendar, but `events` (and `reports`) taken from a fixed random *other* window from a *different domain* (permutation seeded with 0, stored in `data/freshts26/shuffle_map.parquet`).

Number formatting: values printed with the series' native precision (integers for pageviews; 4 decimals for FX; 2 for yields/prices/ILI). Dates ISO `YYYY-MM-DD` (weekly: week-ending Saturday for ILI, FRED's own date for FRED).

Parsing: take the **last** `{...}` JSON object in the answer; require a list of 12 finite numbers (coerce ints/floats; accept 11–13 values by truncating/padding with last value and flag `n_values_mismatch`). Invalid → `valid=false`; the forecast is imputed with the Chronos-2 median (reviser) or seasonal naive (direct) **and** a `valid_only` variant of every table is produced in the appendix.

## 5. Conditions grid (`configs/grid.yaml` is authoritative)

Each **config** = (dataset, model, setup, thinking, budget, context_mode, n_samples, seed, prior). One Slurm array task per config. Tiers define cut order under time pressure (cut tier 3 first, then tier 2).

**Tier 1 (must ship):** FreshTS-26.
- Baselines (no LLM): seasonal naive; Chronos-2; TimesFM-2.5; TiRex.
- LLM: {Qwen3-8B, 4B, 1.7B} × {direct, reviser(Chronos-2)} × {off, on@512, on@2048, on@8192} × `full` × 1 sample × seed 0 → **24 configs**.
- Self-consistency: {8B, 4B, 1.7B} × {direct, reviser} × off × 5 samples (median as point) × seed 0 → **6 configs**.
- Derived (no compute): AvgEns(Chronos-2, LLM-direct-off) and AvgEns(Chronos-2, LLM-direct-on@2048) for each size; AvgEns = arithmetic mean of the two point forecasts.

**Tier 2:** FreshTS-26.
- Context ablations: {8B, 4B} × {direct, reviser} × {off, on@2048} × {`none`, `shuffled`} → **16 configs**.
- Seed replication: 8B × direct × {off, on@2048} × seeds {1, 2} → **4 configs**.
- Other priors: 8B × reviser × {off, on@2048} × prior ∈ {TimesFM-2.5, TiRex} → **4 configs**.

**Tier 3:**
- CiK: {8B, 4B} × direct × {off, on@2048, on@8192} × 5 samples → **6 configs** (RCRPS needs samples).
- 2507 pair: Qwen3-4B-Thinking-2507 @ {2048, 8192} and Qwen3-4B-Instruct-2507 (off), × {direct, reviser} → **6 configs**.
- Non-thinking controls: {gemma-3-4b-it, Llama-3.1-8B-Instruct} × {direct, reviser} × off → **4 configs**.
- Stretch: Qwen3-30B-A3B × direct × {off, on@2048} → 2 configs.

Total ≈ 72 configs. Estimated GPU time (A100-80GB, vLLM, 1,600 prompts/config): off ≈ 5 min; @512 ≈ 10 min; @2048 ≈ 25–40 min; @8192 ≈ 1–2 h (8B). **Tier 1 ≈ 12–15 GPU-h; all tiers ≈ 35–50 GPU-h.**

## 6. Metrics and statistics

### 6.1 Per-window metrics (computed for every config, every window)
- `mae`, `mase` — MASE uses the window's own 96-point context for the seasonal error: `se = mean_{t>m} |y_ctx[t] − y_ctx[t−m]|` with `m` from §2.1 (fev convention: D=7, W=1, B=1); windows with `se == 0` get `mase = NaN` and are excluded (count reported).
- `wql` — weighted quantile loss over levels {0.1,…,0.9} (fev/Chronos definition, factor-2 pinball, normalized by Σ|y|): TSFMs from their quantiles; LLM only in 5-sample configs (samples → empirical quantiles). Not defined for 1-sample LLM configs.
- Bookkeeping: `valid`, `thinking_tokens_used`, `answer_tokens`, `budget_hit`, `wall_s`, `n_events`, `has_reports`, `strict_2026`.

### 6.2 Aggregation (GIFT-Eval convention)
1. Per (config, series): mean MASE over that series' windows.
2. **Relative MASE** per (config, series) = MASE(config) / MASE(seasonal naive).
3. Per (config, domain) and overall: **geometric mean** of relative MASE over series. Overall = geo-mean over all series (each series weighted equally, so `web` does not dominate by window count — report the window-weighted variant in the appendix).

Headline number per config = overall relative MASE (1.0 = seasonal naive). Also report `fev.leaderboard`-style win rate vs. seasonal naive.

### 6.3 Reasoning-token axis
For Figure 1 the x-position of a thinking config is the **mean actual `thinking_tokens_used`** (not the nominal budget); off is plotted at x = 0 (log axis with a broken/linear segment, or x = 1). Report the nominal→actual table in the appendix.

### 6.4 Uncertainty
- CIs on any single config's headline: bootstrap over **series** (n = 1000 resamples, seed 0), percentile 2.5/97.5.
- CIs on **paired differences** (on vs off, SC-5 vs on, AvgEns vs best-LLM, shuffled vs full): bootstrap over series of the per-series mean of per-window differences in *log* relative MASE (equivalently ratio of geo-means), 1000 resamples.
- Seed variance: SD of the headline across seeds {0,1,2} for the 8B direct configs; reported next to the bootstrap CI.
- Strata (H2): `n_events ≥ 1` vs `= 0`; also by domain. Minimum stratum size 50 windows, else not reported.

## 7. Outputs

### 7.1 `data/freshts26/windows.parquet` schema
`window_id (str) · series_id · domain · cluster · freq · seasonality (int) · origin_ts (timestamp, UTC) · context_ts (list[timestamp], 96) · context (list[float], 96) · target_ts (list[timestamp], 12) · target (list[float], 12) · n_ffill (int) · metadata (str) · calendar (str) · events (list[struct{date, category, text, links, available_at}]) · reports (list[struct{date, source, text, available_at}]) · n_events (int) · text_available_at_max (timestamp) · strict_2026 (bool)`

### 7.2 `results/<config_id>.jsonl` — one row per window
`config_id · window_id · dataset · model · setup · thinking (bool) · budget (int) · context_mode · n_samples · seed · prior · point (list[float], 12) · samples (list[list[float]] or null) · valid (bool) · n_values_mismatch (bool) · raw_answer (str, truncated to 2000 chars) · thinking_tokens_used · answer_tokens · budget_hit · finished · wall_s · prompt_tokens`

Runs are **resumable**: on start, read existing rows for this `config_id` and skip done `window_id`s; append per batch with `flush`.

### 7.3 `results/summary.csv` — committed to git
One row per (config_id, domain ∈ {overall, web, health, energy, macro}, stratum ∈ {all, event, no_event, strict_2026, valid_only}) with columns: `rel_mase, rel_mase_lo, rel_mase_hi, mase, wql (or NaN), win_rate_vs_snaive, valid_rate, mean_thinking_tokens, mean_answer_tokens, budget_hit_rate, n_windows, n_series`. Plus `results/paired.csv`: one row per pre-registered comparison with `delta_log_rel_mase, lo, hi, n_series`.

### 7.4 Figures (`scripts/make_figures.py`; read **only** `summary.csv` / `paired.csv`; PDF + PNG to `figures/`)
- **Fig 1** `fig1_budget_curves`: relative MASE vs. mean thinking tokens; lines = model size; panels = FreshTS direct | FreshTS reviser | CiK direct (if tier 3 ran); horizontal reference lines: Chronos-2, TimesFM, TiRex, AvgEns(off), AvgEns(on@2048); SC-5 as a marker at its mean token spend. Error bars = bootstrap CI.
- **Fig 2** `fig2_strata`: Δ log rel-MASE (on@2048 − off) with CIs, grouped by domain and by event/no-event stratum, for direct and reviser; one panel per model size.
- **Fig 3** `fig3_context_sensitivity`: MASE(shuffled) − MASE(full) and MASE(none) − MASE(full), off vs on@2048, 8B and 4B, direct and reviser.
- **Table 1**: tier-1 grid (rows: configs; cols: overall + 4 domains rel-MASE with CI, valid rate, mean thinking tokens). Appendix: full grid, valid-only, window-weighted, strict_2026, seeds.

Style: follow `docs/dataviz-notes.md` if present; otherwise colorblind-safe palette (Okabe-Ito), 300 dpi, fonts embedded, 3.3 in wide single-column or 6.75 in double-column.

## 8. Repository layout and definition of done

```
thinking-budgets-forecasting/
  SPEC.md  README.md  RUNBOOK.md  CLAUDE.md  IMPLEMENTER_PROMPT.md  Makefile
  requirements-llm.txt      # env A: vllm==0.28.0 (+ transformers, pyarrow, pandas)
  requirements-tsfm.txt     # env B: chronos-forecasting==2.3.1 timesfm[torch]==3.0.1 tirex-ts==1.4.2 fev==0.10.0 pandas pyarrow holidays requests matplotlib scipy
  configs/series.yaml  configs/grid.yaml  configs/prompts/{system,direct,reviser}.txt
  src/tbf/
    data/fetch_wikimedia.py  fetch_fred.py  fetch_delphi.py  fetch_current_events.py  fetch_cdc_fluview.py (tier 2)  fetch_eia_wpsr.py (tier 2)
    data/build_freshts26.py   # windows + text attachment + leakage assertion
    data/cik.py               # HF loader + RCRPS wrapper
    prompts.py                # renders direct/reviser prompts for a window + context_mode
    llm_vllm.py               # two-pass budgeted generation, batching, resumable jsonl writer
    llm_hf.py                 # transformers fallback for CPU smoke tests (Qwen3-0.6B)
    tsfm.py                   # chronos2 / timesfm25 / tirex / seasonal_naive → results jsonl
    parse.py                  # JSON extraction + validation
    metrics.py                # mase, wql, crps_samples, aggregation, bootstrap
    summarize.py              # results/*.jsonl → summary.csv, paired.csv
    run.py                    # CLI: run one config_id from grid.yaml
  scripts/slurm/{build_data.sh, tsfm.sbatch, llm_array.sbatch, summarize.sbatch}
  scripts/make_figures.py  scripts/make_tables.py  scripts/expand_grid.py
  tests/test_metrics.py  test_parse.py  test_budget_forcing.py  test_leakage.py  test_prompts.py
  data/fixtures/            # 12 hand-made windows (3 per domain) + 5 CiK rows for smoke tests; committed
  results/summary.csv  results/paired.csv   # committed; raw jsonl gitignored
  figures/                  # committed
  paper/                    # NeurIPS 2026 template, main.tex, refs.bib
```

**Definition of done for implementation (before any cluster time):**
1. `make test` passes: metrics match the numpy reference in `docs/verify-models.md` §C on synthetic inputs; parser handles the 12 fixture answers in `tests/fixtures_answers.json`; budget-forcing unit test with a fake tokenizer/LLM covers the three branches (finished in budget, `</think>` emitted before budget, forced); leakage test fails on a deliberately corrupted fixture window.
2. `make smoke` runs on a laptop/CPU in < 10 min: builds prompts for the 12 fixture windows, runs `Qwen/Qwen3-0.6B` via transformers (`llm_hf.py`) with budget 64 on 4 windows in both modes, runs seasonal naive on all 12, writes `results/smoke_*.jsonl`, `results/summary.csv`, and `figures/fig1_budget_curves.png`.
3. `make build-data` (login node, internet) produces `windows.parquet` with ≥ 1,200 windows, ≥ 100 series, ≥ 3 domains, `BUILD.json`, and passes `tests/test_leakage.py` against the real file.
4. `python -m tbf.run --config-id <id> --limit 50` works for one LLM config on a MIG or A100 slice and reports valid rate ≥ 0.9 for Qwen3-8B off and on@2048 (**go/no-go check**).

## 9. Compute plan (Della)

- Envs on `/scratch/gpfs/$USER/envs/`: `tbf-llm` (py3.12, `pip install vllm==0.28.0`), `tbf-tsfm` (py3.11, `requirements-tsfm.txt`). Create on `della-gpu` login node.
- Downloads on login node with `HF_HOME=/scratch/gpfs/$USER/.cache/huggingface`: the 3 Qwen3 models (+ 2507 pair, Gemma, Llama if licensed, 0.6B for smoke), `amazon/chronos-2`, `google/timesfm-2.5-200m-pytorch`, `NX-AI/TiRex`, `ServiceNow/context-is-key`. Jobs set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- `scripts/slurm/build_data.sh`: runs on the login node (no sbatch), ≈ 700 HTTP calls, polite `User-Agent` with contact email, cached.
- `tsfm.sbatch`: 1 GPU, 1 h, `--constraint=gpu80`, runs all TSFM baselines + seasonal naive → `results/tsfm_*.jsonl`.
- `llm_array.sbatch`: `--array=0-N%8` over `scripts/expand_grid.py --tier 1` config ids; 1 GPU each, `--time` 3:00:00 (8192 configs) or 1:00:00 (others) — QOS gpu-short; `--mem=64G --cpus-per-task=8`. Each task: `python -m tbf.run --config-id $CFG`.
- `summarize.sbatch`: CPU only; `python -m tbf.summarize && python scripts/make_figures.py && python scripts/make_tables.py`.
- Reviser configs depend on `tsfm.sbatch` (`--dependency=afterok`).

## 10. Schedule (ET) and go/no-go

| day | work | owner |
|---|---|---|
| Sun 9/6 | spec + skeleton (this) | Claude |
| Mon 9/7 | Claude Code implements to definition-of-done items 1–2; OpenReview profile; create Della envs; `hf download` models | Claude Code / mayan |
| Tue 9/8 | `make build-data` on login node; `tsfm.sbatch`; `--limit 50` go/no-go on 8B; launch tier 1 array | mayan |
| **Tue 9/8 evening — go/no-go** | if valid rate < 0.9 or budget forcing misbehaves: fix parser/prompt first; if data build is short (< 800 windows): drop `energy`, keep going; if nothing runs: fall back to CiK + Time-MMD (contamination-flagged) as the datasets and keep the same grid | |
| Wed 9/9 | tier 1 done → `summary.csv` v1, Fig 1 draft; launch tier 2; Claude drafts Intro/Related/Setup from the sprint report | mayan / Claude |
| Thu 9/10 | tier 2 done; launch tier 3; Figs 1–3; Results drafted against partial numbers | |
| Fri 9/11 | tier 3 done; final `summary.csv`; full Results + Discussion | Claude |
| Sat 9/12 | complete draft; fact-check pass (every number traced to `summary.csv`); appendix tables | Claude / mayan |
| Sun 9/13 | polish, anonymize, page-limit check with `\usepackage[final]{neurips_2026}` | |
| Mon 9/14 | buffer: reruns, reviewer-anticipation ablations, internal read | |
| Tue 9/15 | **submit by 18:00 ET** (hard stop Wed 9/16 07:59 ET) | mayan |

## 11. Paper skeleton (4 pages)

1. Introduction (¾ page): the contradiction; the two confounds (contamination, model identity); our controlled design; contributions: (i) FreshTS-26, (ii) same-weights budget-controlled study, (iii) matched-compute + context-ablation controls, (iv) open-model replication of TimesX's ensemble finding.
2. Related work (½ page): TSFMs; LLM/text-conditioned forecasting (CiK, Time-MMD, TimesX, PostTime, AlphaCast, Time-R1); reasoning budgets / test-time compute; contamination in TS (Meyer et al.). ≥ 15 citations from the sprint report.
3. FreshTS-26 (½ page): sources, windows, text with `available_at`, leakage argument (release dates), license, refreshability.
4. Setup (½ page): models, thinking control, priors, conditions, metrics, CIs.
5. Results (1¼ pages): H1 (Fig 1 + Table 1), H2 (Fig 2), H3, H4, H5 (Fig 3), valid rates and token spend.
6. Discussion & limitations (¼ page): what a null says; TSFM cutoffs undocumented; final-value evaluation for ILI; single family (Qwen3) as primary; 4-page scope.
Appendix: full grids, prompts verbatim, dataset card, compute, seeds, nominal vs actual budgets, valid-only tables, CiK details.

## 12. Known risks and fixed responses

| risk | response (decided now) |
|---|---|
| Wikimedia/Wikipedia API blocked or rate-limited from login node | retry with backoff; `User-Agent` with contact; if still blocked, drop `web` domain and add NSSP `covidcast` weekly signals (3 × 51 states) to reach ≥ 1,200 windows |
| Small models emit malformed JSON | parser fallbacks + `valid_only` tables; if valid rate < 0.8 for 1.7B, keep 1.7B in the appendix only |
| Thinking helps a lot | report as the positive result; H1 falsified is still the headline; dose-response and strata become the story |
| Budget forcing produces empty answers after the injected phrase | count as `valid=false`; report `budget_hit_rate`; if > 20 % at 512, add a 1024 budget point for 8B |
| vLLM install friction on Della | Apptainer `docker://vllm/vllm-openai:v0.28.0` (see `docs/verify-models.md` §D) |
| Someone posts the same study before 9/16 | differentiate on FreshTS-26 + budget forcing + matched compute + context ablation; cite them |

## Deviations (implementation)

Recorded before the corresponding code was written (CLAUDE.md non-negotiable 1). None changes the grid, prompts
(for FreshTS-26), metrics, seasonality, window rules, or hypotheses.

1. **CiK prompt lengths (§2.2, §4).** `configs/prompts/{direct,reviser}.txt` hard-code "96 values" / "next 12 values" /
   "12 numbers" / `v12`. CiK histories and horizons are task-specific, so for windows whose lengths are not (96, 12)
   the renderer substitutes the actual lengths in exactly those four places; everything else is verbatim. CiK windows
   render only `## Series` (the CiK background/constraints/scenario text, then the frequency + target-period sentence),
   `## History` and `## Task` (no Calendar / Recent events / Reports sections), per `src/tbf/data/cik.py`.
   `parse.parse_forecast` gained an optional `h` argument (default 12) for the same reason. FreshTS-26 prompts are
   unchanged (snapshot-tested).
2. **Sub-daily timestamps (CiK only).** §4 says dates are ISO `YYYY-MM-DD`; CiK has hourly / 10-minute series, whose
   stamps are rendered as `YYYY-MM-DD HH:MM` so rows are distinguishable. FreshTS-26 dates are unchanged.
3. **Baseline for a stratum (§6.2, §7.3).** The seasonal-naive per-series MASE that normalises a summary cell is computed
   on the *same windows* as the cell (so `valid_only`, `event`, `no_event`, `strict_2026` compare like with like).
   The 50-window minimum (§6.4) is applied to those four strata; `all` is always reported. The window-weighted variant
   (`_ww` domains) is the ratio of window means, bootstrapped over windows.
4. **Current-events topic headings (§2.1 item 3).** A Portal bullet that is only a topic heading with deeper child bullets
   (e.g. `*[[Gaza war]]`) is not an event by itself; it is skipped, but its wikilinks are inherited by the child bullets
   so that title matching still works for events filed under a topic.
5. **Multi-sample rows (§7.2).** For `n_samples > 1`, `thinking_tokens_used` / `answer_tokens` are the mean over samples,
   `budget_hit` is "any sample", `finished` is "all samples", `samples` stores only the valid parsed samples and
   `n_valid_samples` is added; `valid` is true when at least one sample parsed.
6. **Smoke test model (§8 item 2).** `make smoke` uses `Qwen/Qwen3-0.6B` as specified; on a machine that cannot reach
   huggingface.co, `TBF_MODEL_PATH=<dir> make smoke` points the two LLM steps at a local checkpoint
   (`scripts/make_tiny_model.py` builds a tiny Qwen3-architecture stand-in trained to emit valid JSON). The stand-in
   is never used for any paper number.
