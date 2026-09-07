# Dataset verification notes for text-conditioned TSF loaders

Verified 2026-09-06. Written for an implementer WITHOUT internet access. Every claim is tagged
[VERIFIED: source] (seen directly in code/data/paper), [PAPER-SUMMARY] (extracted from the paper via a
summarising fetch; wording may be paraphrased), or [UNVERIFIED/ESTIMATE].

Environment note for whoever re-runs this: from the sandbox, `raw.githubusercontent.com`, `pypi.org` and
`files.pythonhosted.org` are reachable with curl; `arxiv.org`, `huggingface.co`, `api.github.com`,
`github.com` are blocked for curl and only partially reachable through the WebFetch tool (which truncates
long arXiv HTML pages, so the appendices of the TimesX / PostTime papers could not be read verbatim).

---

## 0. TL;DR of contradictions with the spec assumptions

| Assumption in spec | Finding |
|---|---|
| TimesX has a GitHub repo / HF dataset | **NOT FOUND.** No public repo or HF dataset exists as of 2026-09-06 (searched HF API `search=TimesX/timesx` -> empty; `github.com/google-research/TimesX` -> 404; `google-research/google-research/timesx` -> 404; AdityaLab org has no TimesX repo). The ICML paper contains no release URL; only "We plan to update the TimesX dataset every three months and version each release". Treat TimesX as **unavailable**; the loader must be a re-construction from the paper's recipe (Google Trends weekly + commodity/FX daily) or be stubbed. |
| TimesX: 190 variables | Correct for the ICML paper (19 domains x 10). But the "fine-tuning split 2018-2022, 88 ID / 11 OOD vars" is from the **PostTime** paper's *refreshed* TimesX (99 variables, 2022-2025, cutoff 2025-01-30), not the ICML TimesX. The ICML TimesX OOD set is 11 multilingual + 5 rare-disease variables (16), with training 2018-2022 and evaluation 2023-2025. Two different datasets are being conflated. |
| TimesX T=96, H=12 | Correct in both papers. Rolling stride 4 (weekly) / 12 (daily). |
| TimesX has post-2025 windows | Data span is "Jan 2018 to Oct 2025" [PAPER-SUMMARY]; the paper's own leakage cutoff is 2024-07-01 (~2.5K windows with horizon start >= 2024-07-01). Windows with horizon start >= 2025-01-01 / >= 2025-07-01 exist in principle but the count is **not determinable** without the data (estimates below). |
| TimesX "event-type labels" | The paper does not define an event taxonomy. Events are free-text, dated, source-cited sentences ("(1) On June 2, 2024, OPEC+ agreed ... Source: [1] [2] [3]; (2) ..."). No type field. |
| Time-MMD extends into 2025/26 | **No.** Latest numerical timestamps are 2024-03 .. 2024-05 (per-domain table below); text ends 2024-05. Repo has not been updated with new data (28 commits; CSVs unchanged). No official HF mirror. |
| Time-MMD text columns `fact, pred` | Actual column is **`preds`** (plus an `Unnamed: 0` index column). README says `pred`. |
| Time-MMD "9 domains" | Repo has **10 folders**: Health is split into `Health_US` and `Health_AFR`. |
| CiK `pip install` | Package name is `cik-benchmark` but it is **not on PyPI**; install is `pip install git+https://github.com/ServiceNow/context-is-key-forecasting.git` and it pulls heavy git deps (tactis, lag-llama, chronos, uni2ts...). Strongly recommend the HF dataset `ServiceNow/context-is-key` (355 rows = 71 tasks x 5 seeds) + the bundled `compute_rcrps_with_hf_dataset.py` instead. |
| CiK requires API keys | Only for the LLM *baselines*. Task instantiation from the repo downloads data from HF (`yatsbm/*`) and GluonTS/Monash; the HF dataset needs nothing but `datasets`. |
| GIFT-Eval MASE aggregation | Leaderboard: per-config gluonts `MASE[0.5]` (in-sample seasonal-error scaling), **geometric mean across the 97 configs**, then divided by seasonal_naive's aggregate; ranks are mean of per-config ranks. Verified from the leaderboard Space code. |
| `fev` package | `pip install fev` (v0.10.0, 2026-08-31). `fev.leaderboard(..., n_resamples=1000)` gives skill score (1 - gmean relative error vs baseline) and win rate with bootstrap 95% CIs. |
| PostTime releases reviser code/prompts | **No.** Paper says "The dataset, code, and models will be released upon acceptance of the paper." No repo found. Prompt template is in Appendix C.3 of the PDF, which could not be fetched; only the structural description is available (below). |

---

## 1. TimesX (arXiv 2607.06973, "Rethinking Multimodal Time-Series Forecasting Evaluation", ICML 2026)

Authors: Haoxin Liu, Yichen Zhou, Rajat Sen, B. Aditya Prakash, Abhimanyu Das (Georgia Tech + Google Research). [VERIFIED: arXiv HTML]

### 1.1 Availability
- No GitHub / HuggingFace / GCS URL anywhere in the paper text that could be fetched; no "code availability" section; Impact statement is boilerplate. [PAPER-SUMMARY, 3 independent fetches]
- Probed and 404/empty: `github.com/google-research/TimesX`, `raw.githubusercontent.com/google-research/google-research/master/{timesx,TimesX,times_x}/README.md`, `raw.githubusercontent.com/{google-research,AdityaLab,haoxinliu}/TimesX/main/README.md`, HF `api/datasets?search=TimesX`, `?search=timesx`, `?author=google&search=time`. [VERIFIED]
- AdityaLab GitHub org repositories (2026-09): pinnsformer, Time-MMD, MM-TSFlib, MM4TSA, lstprompt, FOIL, Samay, Hiereinterpret, ShifTS, TimeRecipe, ... -- no TimesX/PostTime. [VERIFIED: WebFetch of github.com/AdityaLab]
- Paper: "TimesX is designed to be automatically refreshable over time" and "We plan to update the TimesX dataset every three months and version each release." [PAPER-SUMMARY]
- License stated in paper: CC BY 4.0. [PAPER-SUMMARY]
- Cost figure: "$0.7 per variable per three-month block"; ">312,000 independent LLM inferences". [PAPER-SUMMARY]

**Implication:** the implementer cannot write a real TimesX loader. Options: (a) define a `TimesXLike` interface and populate it later; (b) reconstruct a TimesX-style set from Google Trends (pytrends) + commodity/FX APIs following the recipe below.

### 1.2 Composition (ICML version)
- "TimesX contains time-series obtained from 19 domains. Each domain has 10 variables, resulting in a total of 190 variables." [PAPER-SUMMARY, quoted]
- Sources: "Weekly series from Google Search Trends across 12 domains"; "Daily series including (i) commodity prices from a market data API and (ii) major USD exchange rates from a currency rates API". [PAPER-SUMMARY, quoted]
- 19 domains (Table 6 abbreviations -> full names) [PAPER-SUMMARY]:
  - Weekly / Google Trends (12): A&E = Arts & Entertainment; C&E = Commodities & Energy; Econ = Economics; E. Tech = Emerging Technologies; Fin = Finance; P&A = Politics & Affairs; Pub. H. = Public Health; PPG = Personal & Professional Growth; Sci = Science; Shop = Shopping; SSSG = Sports, Society & Social Good; Traf = Traffic.
  - Daily / commodities (6): Crops; Energy (energy markets); Lvstk. = Livestock; RMC = Raw Materials & Construction; SAM = Specialty & Advanced Materials; SHVM = Strategic & High-Value Materials.
  - Daily / FX (1): Curr = Currency (major USD exchange rates).
  - So 120 weekly variables + 70 daily variables. (Appendix B refers to "the 121 variables in the SearchTrend subset" while Table 2's caption says "Search Trend subset (120 variables)" -- off-by-one in the paper.)
- Geographic coverage: North America, Asia, Europe, South America, Africa. [PAPER-SUMMARY]
- Whether daily series are calendar-day or trading-day: NOT STATED in what could be fetched. [UNVERIFIED] (Commodity/FX APIs usually give trading days; assume 5/week unless data says otherwise.)

### 1.3 Date range and splits
- "The core TimesX dataset now contains 190 variables spanning Jan 2018 to Oct 2025 to support both training (2018-2022) and evaluation (2023-2025)." [PAPER-SUMMARY, quoted]
- A second fetch rendered the same passage as "training (2018-2023) and testing (2023-2025)". Treat the boundary as: train windows entirely before 2023-01-01; eval windows with horizon in 2023-2025. [PAPER-SUMMARY, ambiguous]
- Leakage-controlled evaluation: "we construct evaluation examples whose forecast horizon begins after the pretraining cutoff of all involved models", "2024-07-01 as the cutoff, i.e., evaluating only the subset whose horizon start time is 2024-07-01 or later." [PAPER-SUMMARY, quoted]
- "This setup yields about 2.5K samples" (evaluation windows used in the main tables, i.e. horizon start >= 2024-07-01). [PAPER-SUMMARY, quoted]
- Table 2 (leakage validation on the Search-Trend subset): MASE before 2024-06 vs after 2024-06: Gemini-2.0-Flash 0.514 -> 0.594 (+14.81%); DeepSeek-V3 0.606 -> 0.681 (+12.38%); TimesFM-2.5 0.563 -> 0.573 (+1.78%); Moirai-2.0 0.691 -> 0.696 (+0.72%). [PAPER-SUMMARY]

### 1.4 Windows
- "For each variable in TimesX and its context corpus, the look-back window and the forecast horizon are set to 96 and 12, respectively. The rolling window is set to 4 for weekly data and 12 for daily data" (stride in time steps). [PAPER-SUMMARY, quoted; identical statement in PostTime]
- Events attached per window: "K most recent textual events whose announcement dates strictly precede the first prediction horizon", K = 10. [PAPER-SUMMARY, quoted]

### 1.5 Post-LLM-cutoff window counts  [ESTIMATE ONLY -- data not public]
Assumptions: data through 2025-10-31; weekly stride 4 wk, daily stride 12 steps; horizon (12 steps) must fit inside the data.
- Weekly (120 vars): horizon start >= 2025-01-01 -> ~7-8 windows/var -> **~840-960**; >= 2025-07-01 -> 1-2 windows/var -> **~120-240**.
- Daily (70 vars), calendar days: >= 2025-01-01 -> ~24/var -> ~1,700; >= 2025-07-01 -> ~9/var -> ~630. Trading days (5/wk): ~17/var -> ~1,200; ~7/var -> ~470.
- Totals: **>= 2025-01-01: roughly 2.0-2.7K windows; >= 2025-07-01: roughly 0.6-0.9K.**
- Cross-check: the paper's "~2.5K samples" for horizon start >= 2024-07-01 (~16 months) implies ~1.5K for >= 2025-01-01 (~10 months) and ~0.5K for >= 2025-07-01 (~3.5 months) if the 2.5K figure was computed on the Oct-2025 refresh; it may instead have been computed on an earlier snapshot. Either way: **hundreds, not thousands, of windows with horizon start >= 2025-07-01, and none after ~2025-08 (weekly) / ~2025-10-20 (daily).**

### 1.6 Text-context fields (four types) [PAPER-SUMMARY, quotes from Sec 3.2.3 / App. C-E]
1. **Metadata** (str): "descriptive high level summaries of the forecasting variable". Template: `This time series records {variable} ({unit}) in the {domain} domain, with {frequency} frequency. Prediction target period: from {start} to {end}.` Example: "This time series records gasoline price (USD/GAL) in the Commodity Price domain, with daily frequency. Prediction target period: from 2024-09-01 to 2024-09-15."
2. **Calendar** (str): holidays in the prediction window from the Python `holidays` library. Example: "Upcoming holidays in the prediction window: Labor Day (2024-09-02)."
3. **Covariates** (str): "textual information around other related time series" -- max/min with dates, mean, median, trend direction. Example: "Brent Crude Oil (USD/BBL): The maximum value was 87.43, occurring on July 4, the minimum ..., showing an overall downward trend."
4. **Time-stamped events** (list of str, K<=10, announcement date < horizon start): "Textual description of related events aligned with time-windows". Example: "(1) On June 2, 2024, OPEC+ agreed to extend deep oil output cuts [1,3], the cut of 2.2 million bpd would be extended until September 2024, after which it would be gradually phased out [2,3]. Source: [1] [2] [3]; (2) ..."
   - Built by a multi-agent pipeline (Hypothesizer -> Verifier -> Enricher -> Synthesizer) with timestamp verification; "A manual audit of 50 samples shows 94% exact matches; 4% conservative (later) offsets; and 2% earlier due to date ambiguity".
   - **No event-type/category labels are defined.** Each event carries: date, description, source URLs, and a timestamp-accuracy flag (exact vs bounded +/- k days). [PAPER-SUMMARY]
- History is serialised as `(timestamp, value)` pairs, e.g. "(2024-06-02, 2.4116), (2024-06-03, 2.3279), ..." [PAPER-SUMMARY, quoted]

Suggested loader schema (matches the paper; fill from data when it appears):
```python
@dataclass
class TimesXWindow:
    var_id: str; domain: str; freq: Literal["W","D"]
    hist_ts: list[str]; hist_y: list[float]          # len 96
    fut_ts: list[str]; fut_y: list[float]            # len 12
    metadata: str; calendar: str; covariates: str
    events: list[dict]  # {"date": "YYYY-MM-DD", "text": str, "sources": list[str], "date_exact": bool}
    split: Literal["train","eval"]; language: str = "en"; ood: bool = False
```

### 1.7 ID / OOD
- ICML TimesX: ID = the 190 English-context variables. OOD = (a) 11 multilingual variables (Afrikaans, French, German, Hindi, Japanese, Korean, Portuguese, Simplified Chinese, Spanish, Swahili, Turkish) and (b) 5 rare-disease variables (Chagas, Guinea worm, Huntington's, Marburg virus, Nipah virus). [PAPER-SUMMARY]
- PostTime (2605.29401) refreshed TimesX: "covering 99 variables from 2022 to 2025"; "We further randomly select 88 variables for training and in-domain (ID) evaluation, and hold out the remaining 11 variables for out-of-domain (OOD) evaluation."; cutoff **Jan 30, 2025**: "Our post-training uses samples whose prediction windows end before this cutoff, while evaluation uses samples whose prediction windows start after this cutoff." [PAPER-SUMMARY, quoted]
- **The spec's "fine-tuning split 2018-2022, 88 ID vars, 11 OOD vars" mixes the two**: 2018-2022 training is ICML-TimesX; 88/11 is PostTime's 99-variable refresh (2022-2025).

### 1.8 Metric
- Primary: normalized MASE, geometric mean. Appendix N: "we calculate the average MASE over all rolling windows of a variable and normalize that by the average MASE of a seasonal naive baseline. Then we take the Geometric Mean (GM), for robustness to normalization choice, of these normalized MASE ratios across all variables." [PAPER-SUMMARY, quoted]
  - i.e. `score = GM_v( mean_w MASE(model, v, w) / mean_w MASE(snaive, v, w) )`.
  - Seasonal period m for the MASE scaling and for the seasonal-naive baseline: **NOT STATED** in fetched text. [UNVERIFIED] Use gluonts defaults (W -> 1, D -> 1, i.e. plain naive) unless the data release says otherwise; PostTime's normaliser is "the seasonal naive forecast which repeats the last period in the history" (period again unspecified).
- Secondary: average rank across variables; CRPS for probabilistic models (App. U.1). [PAPER-SUMMARY]
- PostTime uses nMAE/nMSE instead: `nMAE = sum|y_t - yhat_t| / sum|y_t - yhat_t^{snaive}|` (same for nMSE), plus p50/p75/p90/p95/p99 percentiles. [PAPER-SUMMARY, quoted]

### 1.9 Loader code in repo
None (no repo).

---

## 2. Time-MMD (github.com/AdityaLab/Time-MMD, NeurIPS 2024 D&B, arXiv 2406.08627)

### 2.1 Repo layout [VERIFIED: raw.githubusercontent.com, all files downloaded 2026-09-06]
Default branch `main`. README file is literally `readme.MD` (capitalised extension). No `LICENSE` file (LICENSE, LICENSE.md, LICENSE.txt all 404); the paper does not state a license either. [VERIFIED]
```
readme.MD
DescriptionOfOT.png
VisualizationOfText.png
numerical/<Domain>/<Domain>.csv
textual/<Domain>/<Domain>_report.csv
textual/<Domain>/<Domain>_search.csv
Downstream_Tasks/Imputation/...     (forecasting examples live in the separate MM-TSFlib repo)
```
Domains (folder names, exact case): `Agriculture, Climate, Economy, Energy, Environment, Health_US, Health_AFR, SocialGood, Traffic, Security` (10 folders; paper counts Health as one domain -> "9 domains").

Raw URL pattern: `https://raw.githubusercontent.com/AdityaLab/Time-MMD/main/numerical/Agriculture/Agriculture.csv` etc.

### 2.2 Numerical CSVs [VERIFIED by reading the files]
Column layouts are NOT uniform; only `OT` and `end_date` are present everywhere, `start_date` is missing in Health_AFR. Always sort by date after loading (Climate and Economy are stored newest-first).

| Domain | rows | freq (median step) | first start | last start | last end_date | columns (verbatim) | notes |
|---|---|---|---|---|---|---|---|
| Agriculture | 532 | monthly (31 d) | 1980-01-01 | 2024-04-01 | 2024-04-30 | `Date, Wholesale broiler composite, OT, Retail-wholesale spread for broiler composite, date, start_date, end_date` | OT = retail broiler composite (cents/lb); other cols mostly NaN |
| Climate | 1272 | weekly (7 d) | 2000-01-04 | 2024-05-14 | 2024-05-20 | `MapDate, AreaOfInterest, OT, D0, D1, D2, D3, D4, ValidStart, ValidEnd, StatisticFormatID, date, start_date, end_date` | **stored newest-first**; paper says "Monthly" but file is weekly (US Drought Monitor) |
| Economy | 447 | monthly | 1987-01-01 | 2024-03-01 | 2024-03-31 | `Month, Exports, Imports, OT, start_date, date, end_date` | **stored newest-first**; OT = trade balance |
| Energy | 1622 | weekly | 1993-04-05 | 2024-04-29 | 2024-05-05 | `date, OT, Weekly East Coast ... , ... (8 regional gasoline price cols), start_date, end_date` | OT = US regular gasoline $/gal |
| Environment | 15979 | daily (1 d) | 1980-01-01 | 2023-09-30 | 2023-09-30 | `CBSA, CBSA Code, date, OT, Category, Defining Parameter, Defining Site, Number of Sites Reporting, start_date, end_date` | NYC AQI; ends 2023-09-30 |
| Health_US | 1389 | weekly (min step 0, max 14) | 1997-09-29 | 2024-05-06 | 2024-05-12 | `date, start_date, end_date, REGION TYPE, REGION, YEAR, WEEK, % WEIGHTED ILI, OT, AGE 0-4, AGE 25-49, AGE 25-64, AGE 5-24, AGE 50-64, AGE 65, ILITOTAL, NUM. OF PROVIDERS, TOTAL PATIENTS, YEAR_WEEK` | OT = %UNWEIGHTED ILI; some age cols contain 'X' strings |
| Health_AFR | 1461 | weekly | 1996-01-01 | 2024-04-29 | 2024-05-05 | `date, date.1, end_date, OT` | **no `start_date`** (use `date`); 3 leading NaN OT |
| SocialGood | 924 | monthly | 1948-01-01 | 2024-12-01 | 2024-12-31 | `date, start_date, end_date, OT` | OT = unemployment rate; **rows after 2024-04-01 are NaN** (pre-filled placeholders) |
| Traffic | 651 | monthly | 1970-01-01 | 2024-03-01 | 2024-03-31 | `OT, Date, start_date, date, end_date` | OT = travel volume |
| Security | 309 | monthly | 1998-09-01 | 2024-05-01 | 2024-05-31 | `date, OT, start_date, end_date` | OT = FEMA disaster grants $ |

**Latest usable observation across all domains: 2024-05 (Climate 2024-05-14, Health_US 2024-05-06, Security 2024-05-01).** Nothing in 2025 or 2026. Dataset has not been extended since the 2024 release (numerical files end 2024-03..05; search text ends 2024-05-05).

### 2.3 Textual CSVs [VERIFIED]
All 20 files share the header `Unnamed: 0, start_date, end_date, fact, preds` (README says `pred`; the real column is **`preds`**).
- `fact`: LLM (Llama3-70B) summary of objective statements from the source document(s) for that window.
- `preds`: LLM-extracted forward-looking statements; long-term and short-term predictions are joined by `;` (e.g. "...steady in the short-term.;Based on the current...") and may contain the literal string `NA`.
- `_report.csv` = curated domain reports (e.g. USDA broiler reports, CDC FluView); `_search.csv` = weekly Google-search results (one row per week, from 1979-12-31).

| Domain | report rows / span / non-null fact | search rows / span / non-null fact |
|---|---|---|
| Agriculture | 890 / 2018-11-21..2024-05-13 / 890 | 1653 / 1979-12-31..2024-04-29 / 1280 |
| Climate | 590 / 1999-01-01..2024-04-01 / 582 | 2261 / ..2024-05-13 / 1948 |
| Economy | 435 / 1995-01-01..2024-03-01 / 435 | 2314 / ..2024-04-29 / 2314 |
| Energy | 354 / 2011-07-25..2024-04-22 / 354 | 2314 / ..2024-04-29 / 2307 |
| Environment | 156 / 2021-05-15..2023-10-03 / 141 | 2312 / ..2024-04-29 / 2272 |
| Health_US | 864 / 1999-01-01..2019-12-30 / 489 (**reports stop at end-2019**) | 2087 / ..2024-04-29 / 1994 |
| Health_AFR | 482 / 2017-08-21..2024-05-02 / 248 | 1911 / 1980-01-14..2024-04-29 / 1818 |
| SocialGood | 371 / 1994-01-01..2024-04-01 / 347 | 4590 / ..2024-04-29 / 4513 (multiple rows per week) |
| Traffic | 367 / 2002-01-01..2024-03-01 / 367 | 2309 / ..2024-04-29 / 2292 |
| Security | 19569 / 1980-04-10..2024-04-08 / **0 (fact and preds all NaN)** | 2311 / ..2024-04-29 / 2289 |

Text-to-window alignment (paper, Sec 2.3 / App.): "Binary timestamps" `(start_date, end_date)` on both modalities; MM-TSFlib pre-matches "the most recent k text samples based on the timestamp of the numerical sample" (no future leakage). Rule for a loader: attach text rows with `end_date <= history_end` (strictly before horizon start); take the k most recent. [PAPER-SUMMARY + VERIFIED code in MM-TSFlib data_loader]

### 2.4 Standard forecasting protocol
- Paper Appendix K.4 [PAPER-SUMMARY, quoted]: Daily: lookback 96, horizons [48, 96, 192, 336]; Weekly: lookback 36, horizons [12, 24, 36, 48]; Monthly: lookback 8, horizons [6, 8, 10, 12]. Metric MSE/MAE (z-scored), aggregated with geometric mean over domains in later papers (e.g. "LLM as Forecasting Planner", 2607.24892: "Geometric-mean MSE and MAE over nine domains in StandardScaler space"; that paper uses 24-step conditioning, 50 windows per condition).
- MM-TSFlib (github.com/AdityaLab/MM-TSFlib, MIT license file inherited from THUML Time-Series-Library) `scripts/week_health.sh` [VERIFIED]: `--seq_len 24 --label_len 12 --pred_len {12,24,36,48} --features M --data custom --root_path ./data/Public_Health --data_path US_FLURATIO_Week.csv --text_len 4 --prompt_weight 0.1 --llm_model BERT`. Note `seq_len 24` in the script vs 36 in the paper table -> protocol is not consistent across sources; pick one and state it.
- MM-TSFlib `data_provider/data_loader.py::Dataset_Custom` [VERIFIED]: chronological split `num_train = int(0.7*N)`, `num_test = int(0.2*N)`, val = rest; borders `[0, num_train-seq_len, N-num_test-seq_len]`; StandardScaler fit on train; preprocessed CSV columns used: `['date'] + cols + [target] + ['prior_history_avg','start_date','end_date', text_col]` where `text_col = 'Final_Search_'+str(text_len)` (search) or `'Final_Output'` (report); text for a window is the text row at index `s_end` (end of look-back).

### 2.5 Loader code
No loader in Time-MMD itself; MM-TSFlib has the Time-Series-Library-style loader above and preprocessed CSVs under `./data/<Domain>/` (e.g. `Public_Health/US_FLURATIO_Week.csv`) -- directory listing could not be fetched (GitHub tree pages blocked), so treat MM-TSFlib preprocessed files as optional.

Minimal loader from raw files:
```python
import pandas as pd
RAW = "https://raw.githubusercontent.com/AdityaLab/Time-MMD/main"
def load_numeric(domain):
    df = pd.read_csv(f"{RAW}/numerical/{domain}/{domain}.csv")
    if "start_date" not in df: df["start_date"] = df["date"]          # Health_AFR
    df["start_date"] = pd.to_datetime(df["start_date"]); df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values("start_date").dropna(subset=["OT"]).reset_index(drop=True)   # SocialGood trailing NaNs, Climate/Economy reversed
    return df[["start_date","end_date","OT"]]
def load_text(domain, kind):   # kind in {"report","search"}
    t = pd.read_csv(f"{RAW}/textual/{domain}/{domain}_{kind}.csv").drop(columns=["Unnamed: 0"])
    t["start_date"] = pd.to_datetime(t["start_date"]); t["end_date"] = pd.to_datetime(t["end_date"])
    return t.dropna(subset=["fact"]).sort_values("end_date")        # columns: start_date,end_date,fact,preds
```

---

## 3. CiK -- Context is Key (github.com/ServiceNow/context-is-key-forecasting; ICML 2025; arXiv 2410.18959)

### 3.1 Install / package
- `pyproject.toml` [VERIFIED]: `name = "cik-benchmark"`, `license = {text = "Apache-2.0"}`, `requires-python = ">3.9"`, `version = attr cik_benchmark.__version__` (= "0.0.1"), dependencies from `requirements.txt`.
- **Not on PyPI** (`pypi.org/pypi/cik-benchmark/json` -> no `info`). Install: `pip install git+https://github.com/ServiceNow/context-is-key-forecasting.git` (tag `v1.0.0` = ICML release).
- `requirements.txt` [VERIFIED] pins git deps: `tactis @ git+.../ServiceNow/TACTiS.git@tactis-2`, `lag_llama`, `llm_processes`, `chronos_forecasting`, `uni2ts[notebook]`, `timellm`, `unitime`, plus `gluonts[torch]>=0.14.4, gradio, nixtla, numpy, openai, pandas, statsmodels, huggingface_hub, datasets, causalchamber==0.1.1, termcolor, tenacity, h5py, transformers>4.4.1, sentencepiece, lm-format-enforcer`. Task modules import `tactis.gluon.dataset.get_dataset`, so even task instantiation needs TACTiS installed.
- **Recommended path: HuggingFace dataset `ServiceNow/context-is-key`** (license apache-2.0, lastModified 2025-07-24, parquet). Files: `data/test-00000-of-00001.parquet` (current, with post-ICML fixes), `data/ICML2025-00000-of-00001.parquet` (as published), `compute_rcrps_with_hf_dataset.py`. Split `test` has **355 rows = 71 tasks x seeds 1..5**. [VERIFIED: HF API + card]

### 3.2 Enumerating the 71 tasks
Repo: `cik_benchmark/__init__.py::ALL_TASKS` [VERIFIED] = ELECTRICITY(7) + NN5(6) + PRED_CHANGE(1) + PREDICTABLE_CONSTRAINT(2) + SENSOR_MAINTENANCE(4) + CAUSAL_CHAMBERS(3) + CATEGORICAL_CAUSAL(3) + SOLAR(6) + TRAFFIC(4) + NSRDB(6) + MONTREAL_FIRE(2 analogy + 16 causal + 8 short_history = 26) + FRED_COUNTY(3) = **71**. (PREDICTABLE_GROCER_SHOCKS (40) and PEMS (18) are defined but commented out of ALL_TASKS.) Task names are class names, e.g. `ElectricityIncreaseInPredictionTask`, `MontrealFireFieldFireExplicitShortHistoryTask`.
```python
import cik_benchmark
for task_cls in cik_benchmark.ALL_TASKS:
    for seed in range(1, 6):
        task = task_cls(seed=seed)
```
HF: `sorted(set(ds["name"]))` gives the same 71 names; `weight` column (string Fraction) = `cik_benchmark.TASK_NAME_TO_WEIGHT` (weight of the task's cluster / #tasks in cluster).

### 3.3 Task object API (`cik_benchmark/base.py`) [VERIFIED]
```python
class BaseTask(ABC):
    _context_sources: list  # subset of ["c_h","c_i","c_f","c_cov","c_causal"]
    _skills: list
    def __init__(self, seed: int = None, fixed_config: dict | None = None)
    past_time: pd.DataFrame      # history; DatetimeIndex (or PeriodIndex); LAST column is the target
    future_time: pd.DataFrame    # ground truth over the horizon; same columns
    background: str | None       # static context
    scenario: str | None         # instance-specific context
    constraints: str | None      # textual constraints
    @property name -> str        # class name
    @property seasonal_period -> int   # statsmodels freq_to_period(past_time.index.freq); negative => none/too short; overridden per task
    def evaluate(self, samples: np.ndarray) -> dict   # samples (n_samples, n_time[, n_dim]); uses last dim 0
    def plot(self)
class UnivariateCRPSTask(BaseTask):
    region_of_interest: None | int | list[int] | slice | bool-mask
    roi_weight: float = 0.5
    metric_constraint: Constraint | None
```
`evaluate` -> `threshold_weighted_crps(target=future_time[last_col], forecast=samples, scaling=DefaultScalingCache(cls), region_of_interest, roi_weight, constraint, compute_variance=COMPUTE_METRIC_VARIANCE)` returning dict with keys `metric, raw_metric, scaling, crps, roi_crps, non_roi_crps, standard_crps, num_roi_timesteps, num_non_roi_timesteps, violation_mean, violation_crps, variance`.

Prompt inputs a model may see (HF card): `background, scenario, constraints, seasonal_period, past_time, future_time` timestamps only. The full prompt used in official baselines = background + scenario + constraints (see `evaluation.save_context`: "Background:\n{background}\n\nConstraints:\n{constraints}\n\nScenario:\n{scenario}").

HF row fields [VERIFIED: HF API]: `name (str), seed (int64 1..5), weight (str Fraction), context_sources (list), skills (list), background (str), scenario (str), constraints (str), seasonal_period (int64), past_time (str = JSON of DataFrame), future_time (str = JSON), metric_scaling (float64), region_of_interest (list[int]), constraint_min (float), constraint_max (float), constraint_variable_max_index (list[int]), constraint_variable_max_values (list[float])`.
```python
from datasets import load_dataset; import pandas as pd; from io import StringIO
ds = load_dataset("ServiceNow/context-is-key", split="test")   # 355 rows
e = ds[0]
past = pd.read_json(StringIO(e["past_time"])); future = pd.read_json(StringIO(e["future_time"]))
H = len(future); target_col = future.columns[-1]
```

### 3.4 RCRPS metric [VERIFIED: `cik_benchmark/metrics/roi_metric.py` and HF `compute_rcrps_with_hf_dataset.py`]
Per instance, with `forecast` shape `(n_samples, H)` (official n_samples = 50, `config.DEFAULT_N_SAMPLES`; HF example uses 25), `target` = last column of `future_time`:
1. CRPS per timestep by the exact probability-weighted-moment estimator (`crps()`; Taillardat et al.): `crps = mean|X - y| + beta0 - 2*beta1`, with sorted samples, `beta0 = mean(X)`, `beta1 = sum(i * X_(i)) / (N(N-1))`.
2. If `region_of_interest` is non-empty: `crps_value = 0.5 * mean(crps[roi]) + 0.5 * mean(crps[~roi])` (roi_weight = 0.5); else plain mean over horizon.
3. Constraint violation per sample: `viol = mean_t clip(scale*cmin - scale*x, 0) + mean_t clip(scale*x - scale*cmax, 0) + mean_j clip(scale*x[idx_j] - scale*vmax_j, 0)`; `violation_crps = crps(target=0, samples=10.0 * viol)` (violation_factor 10, linear).
4. `metric = metric_scaling * crps_value + violation_crps` where `metric_scaling` = `1 / mean_{seeds 1001..1025}(max(future) - min(future))` for that task class (`scaling_cache.inverse_mean_forecast_range`; precomputed in the HF column).
5. Aggregate: `mean_RCRPS = sum_i w_i * min(metric_i, 5.0) / sum_i w_i` with `w_i = Fraction(entry["weight"])`; std error `sqrt(sum w_i^2 var_i)/sum w_i` (variance via `weighted_sum_crps_variance`; set to 0 when clipped at 5). Reported headline numbers = this weighted mean over 355 instances; per-context-type columns use the task's `context_sources`.
`compute_all_rcprs(dataset, forecasts)` expects `forecasts = [{"name", "seed", "forecast": np.ndarray (n_samples, H)}]` aligned in dataset order.

### 3.5 First-use downloads / keys
- Repo tasks download on instantiation: GluonTS/Monash datasets via `tactis.gluon.dataset.get_dataset(...)` (electricity_hourly, nn5, traffic, solar_10_minutes) into `CIK_DATA_STORE` (default `./data`); HF Hub files `yatsbm/NSRDB_extract`, `yatsbm/FRED`, `yatsbm/TrafficFresh` (PEMS, not official); `causalchamber==0.1.1` package data; Montreal fire data (bundled/HF). Env vars: `CIK_DATA_STORE, CIK_MODEL_STORE, CIK_RESULT_CACHE (./inference_cache), CIK_METRIC_SCALING_CACHE (./metric_scaling_cache), CIK_METRIC_COMPUTE_VARIANCE, HF_HOME`. [VERIFIED: config.py, task files]
- API keys (`CIK_OPENAI_API_KEY`, `CIK_LLAMA31_405B_URL/_API_KEY`, `CIK_NIXTLA_*`) are only for the baselines. [VERIFIED: README/config.py]
- The HF dataset needs no keys and no extra downloads.
- CHANGELOG (post-ICML): 2025-07-11 fixed X_0 covariate scaling in `FullCausalContextImplicitEquationBivarLinSVAR` / `FullCausalContextExplicitEquationBivarLinSVAR`; official results table "holds for the version of the benchmark as of July 11th, 2025". [VERIFIED]
- License: Apache-2.0 (code and HF dataset); underlying series CC-BY-4.0 / public domain. [VERIFIED]
- Official RCRPS reference points (README, 2025-07-11 version): Llama-3.1-405B-Instruct direct prompt 0.143 +/- 0.006; GPT-4o 0.191; Chronos-Large 0.326; Lag-Llama 0.327; Moirai-Large 0.520; ETS 0.530. [VERIFIED]

---

## 4. GIFT-Eval-style MASE conventions and `fev`

### 4.1 GIFT-Eval (github.com/SalesforceAIResearch/gift-eval; HF Space `Salesforce/GIFT-Eval`)
- Per-config evaluation uses `gluonts.model.evaluate_model(predictor, test_data=dataset.test_data, metrics=metrics, batch_size=512, axis=None, mask_invalid_label=True, allow_nan_forecast=False, seasonality=season_length)` with `season_length = gluonts.time_feature.get_seasonality(dataset.freq)`. [VERIFIED: notebooks/naive.ipynb + README]
- gluonts `MASE` (`gluonts/ev/metrics.py`, `forecast_type="0.5"`) = `absolute_error / seasonal_error`, where `seasonal_error` (`gluonts/ev/ts_stats.py`) = `mean_t |y_t - y_{t-m}|` over the **in-sample history (the input/context of each test instance)**, with `m = seasonality`, and `m -> 1` if `m > len(history)`. `axis=None` -> one number per config averaging over all items and timesteps. [VERIFIED: gluonts 0.16.0 wheel]
- gluonts `DEFAULT_SEASONALITIES`: `S/s: 3600, T/min: 1440, H/h: 24, D: 1, W: 1, M/ME: 12, B: 5, Q/QE: 4`; `get_seasonality("2h") = 12` (base // n, else 1). **Weekly and daily default to m = 1 (plain naive scaling), monthly m = 12.** [VERIFIED]
- Leaderboard aggregation (`src/utils.py` in the Space) [VERIFIED]: all `results/<model>/all_results.csv` concatenated (97 configs `dataset/freq/term`); per config `rank` by `mean_weighted_sum_quantile_loss` (CRPS) and `Rank_MASE` by `MASE[0.5]` (`rank(method="first")`); overall = `groupby("model")[["MASE[0.5]","mean_weighted_sum_quantile_loss"]].agg(scipy.stats.gmean)` and mean of per-config ranks; `norm_sNavie(df)` then divides each model's aggregate by the `seasonal_naive` row. Because gmean is multiplicative, this equals `gmean_configs(MASE_model / MASE_snaive)`.
- Recipe to reproduce for our datasets: per (dataset, model) compute mean MASE with in-sample seasonal-error scaling; `score = exp(mean_d log(MASE_d(model)/MASE_d(snaive)))`; rank per dataset and average ranks.

### 4.2 `fev` (github.com/autogluon/fev; Apache-2.0) [VERIFIED: PyPI + installed 0.10.0]
- `pip install fev` (0.10.0, uploaded 2026-08-31; deps `datasets<5,>=2.15, numpy, pydantic~=2, scipy`).
- MASE: `fev.metrics.MASE` scales `|y_true - y_pred|` by per-item in-sample seasonal error `mean|y_t - y_{t-seasonality}|` over the past window (items with undefined seasonal error are excluded); `Task(seasonality=...)` default 1.
- Aggregation API:
```python
import fev
lb = fev.leaderboard(summaries, metric_column="test_error", baseline_model="seasonal_naive",
                     n_resamples=1000, seed=123)          # columns: win_rate[_lower/_upper], skill_score[_lower/_upper], ...
pw = fev.pairwise_comparison(summaries, n_resamples=1000)  # per model pair: skill score + win rate with 95% CI
```
  `summaries` = DataFrame/list of dicts/CSV path with at least `model_name, task_name, test_error` (as produced by `task.evaluation_summary(preds, model_name=...)`).
- Definitions (`fev/analysis.py`) [VERIFIED]: relative error = `err_model / err_baseline` clipped to `[0.01, 100]`; `skill_score = 1 - gmean_tasks(relative_error)`; `win_rate` = mean over other models of `P(err_A < err_B) + 0.5 P(err_A == err_B)` across tasks; `bootstrap(errors[n_tasks, n_models], statistic, n_resamples=1000, alpha=0.05, seed=123)` resamples **tasks with replacement** and reports the 2.5/97.5% quantiles as `*_lower/*_upper`. `missing_strategy` in {"error","drop","impute"}.

---

## 5. PostTime (arXiv 2605.29401, "Rethinking Post-Training Recipes for Multimodal Time-Series Forecasting")

Authors: Haoxin Liu, Yichen Zhou, Rajat Sen, B. Aditya Prakash, Abhimanyu Das (same group). [PAPER-SUMMARY]
- **Code/prompts released?** No. "The dataset, code, and models will be released upon acceptance of the paper." No GitHub/HF URL in the paper or findable by search as of 2026-09-06. [PAPER-SUMMARY + searches]
- Setup: LLM = context-guided **reviser** of a TSFM prior: `(tau, yhat_{1:H}) = pi_theta(x_{1:T}, c, ytilde_{1:H})` where `x` = history (T=96), `c` = text context (metadata, calendar, covariates, events), `ytilde` = TimesFM-2.5 forecast (H=12), `tau` = reasoning trace. "for both strategies the LLM is prompted to output some reasoning trace before a final forecast." The LLM may revise, preserve, or ignore the prior; if the output cannot be parsed the system falls back to the TSFM forecast (App. C.8 "Inference-Time Fallback Behavior Metrics"). [PAPER-SUMMARY]
- Zero-shot reviser prompt: template is Appendix C.3 "Prompt and Output Format" (with C.2 "Example TimesX Forecasting Instance"); **the verbatim text could not be retrieved** (arXiv HTML fetch truncated before the appendices; PDF endpoints blocked). Structure per the main text: [history as (timestamp, value) pairs] + [metadata, calendar, covariates, up to 10 dated events] + [initial TSFM forecast for the 12 horizon steps] -> reasoning trace -> final list of H=12 numbers. [PAPER-SUMMARY]
- Training recipes: CoT-SFT (5 traces/example from Gemini-3.1-Flash-Lite, keep top-3 that improve MAE over the prior, plus fallback supervision) then RLVR with reward `R = clip_[0,1](0.5 + 0.5 * ImpRatio)`; base LLM Gemma-3-4B. Metrics nMAE/nMSE vs seasonal naive. [PAPER-SUMMARY]
- Related but different: "LLM as Forecasting Planner" (arXiv 2607.24892, Deakin Univ.) uses LLM as ranker/judge over TSFM-sampled trajectories (MCTS), evaluated on CiK (RCRPS) and Time-MMD (gmean MSE/MAE, 24-step context, 50 windows/condition); prompts in its Appendix K; no code URL found. [PAPER-SUMMARY]

---

## 6. Sources
- TimesX paper HTML: https://arxiv.org/html/2607.06973v1 (abs: https://arxiv.org/abs/2607.06973); ICML page: https://icml.cc/virtual/2026/poster/63015 (no code link); OpenReview https://openreview.net/forum?id=Z1TMV4bGuu (blocked by bot check).
- PostTime paper HTML: https://arxiv.org/html/2605.29401v1 (appendix C.2/C.3 not retrievable).
- LLM as Forecasting Planner: https://arxiv.org/html/2607.24892v1.
- Negative probes for TimesX release: HF `https://huggingface.co/api/datasets?search=TimesX` (empty), `github.com/google-research/TimesX` (404), raw `google-research/google-research/master/timesx/README.md` (404), `github.com/AdityaLab` org listing.
- Time-MMD: https://github.com/AdityaLab/Time-MMD ; raw files `https://raw.githubusercontent.com/AdityaLab/Time-MMD/main/{readme.MD,numerical/<D>/<D>.csv,textual/<D>/<D>_{report,search}.csv}` (all 30 CSVs downloaded and profiled); paper https://arxiv.org/html/2406.08627 ; MM-TSFlib https://github.com/AdityaLab/MM-TSFlib (`README.md`, `scripts/week_health.sh`, `data_provider/data_loader.py`, `LICENSE`).
- CiK: https://github.com/ServiceNow/context-is-key-forecasting (`README.md`, `pyproject.toml`, `requirements.txt`, `CHANGELOG`, `cik_benchmark/{__init__,base,config,evaluation}.py`, `cik_benchmark/metrics/{roi_metric,crps,scaling_cache}.py`, `cik_benchmark/tasks/*.py`, `cik_benchmark/tasks/montreal_fire/*.py`, `cik_benchmark/data/pems.py`); HF https://huggingface.co/datasets/ServiceNow/context-is-key (+ `/raw/main/compute_rcrps_with_hf_dataset.py`, API metadata).
- GIFT-Eval: https://github.com/SalesforceAIResearch/gift-eval (`README.md`, `notebooks/naive.ipynb`); paper https://arxiv.org/html/2410.10393 ; leaderboard code https://huggingface.co/spaces/Salesforce/GIFT-Eval/raw/main/src/utils.py ; gluonts 0.16.0 wheel (`gluonts/ev/metrics.py`, `gluonts/ev/stats.py`, `gluonts/ev/ts_stats.py`, `gluonts/time_feature/seasonality.py`, `gluonts/model/evaluation.py`).
- fev: https://pypi.org/pypi/fev/json (0.10.0), https://github.com/autogluon/fev (`README.md`), installed package source (`fev/analysis.py`, `fev/metrics.py`).
