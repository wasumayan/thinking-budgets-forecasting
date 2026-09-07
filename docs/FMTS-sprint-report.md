# FMTS @ NeurIPS 2026 — Sprint Strategy Report

Prepared: Sunday, September 6, 2026
Target: *Foundation Models for Temporal Systems: From Forecasting to World Modeling* (FMTS), NeurIPS 2026 workshop, Sydney, Dec 11–12, 2026
Time to deadline: **~9.5 days** (submission closes Wed Sep 16, 11:59 UTC = 07:59 ET / 04:59 PT).

---

## 1. Workshop facts (verified against the official site)

Both the landing page and the CFP subpage were fetched directly on Sep 6, 2026.

| Item | Value | Source |
|---|---|---|
| Submission deadline | **September 16, 2026, 11:59 am UTC** (site also writes "Sep 15 AoE") | official CFP |
| Notification | September 29, 2026 (fixed by NeurIPS) | official CFP |
| Camera-ready | November 6, 2026 (tentative) | official CFP |
| Page limit | **Up to 4 pages, excluding references and appendices** | official CFP |
| Template | Official **NeurIPS 2026 LaTeX** style; no separate workshop style | official CFP |
| Review | **Double-blind**, 3 reviews per paper, 40+ reviewers; criteria: soundness, evaluation rigor, reproducibility, relevance/impact; reviewers prohibited from using LLMs | official CFP |
| Archival | **Non-archival**; papers posted on OpenReview; does not preclude later archival publication | official CFP |
| Dual submission | Concurrent NeurIPS main-track submission allowed; previously published work ineligible; anonymized arXiv preprints OK | official CFP |
| Portal | OpenReview: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/FMTS (JS-rendered; verify it accepts submissions when you first log in) | official site |
| Tiny-paper / findings track | **None listed.** Encouraged: datasets/benchmarks/simulators, **negative results and replication studies**, deployment reports | official CFP |
| Spotlight priority | Junior / early-career first authors prioritized for spotlights | official CFP |
| Desk-reject triggers | Over-length; improper anonymization (no self-identification anywhere, incl. acknowledgments) | official CFP |
| Organizers | B. Oreshkin, M. Jauhari, D. Maddix Robinson, O. Azencot, M. Jin, E. Eldele, C. Liu, N. B. Erichson | official site |
| Invited speakers | Rose Yu, M. W. Mahoney, A. F. Ansari (Chronos), A. Krishnapriyan, M. Zitnik, D. F. Schmidt, F. Salim, Junnan Li, W. Gilpin | official site |
| Contact | fmtsworkshop@gmail.com | official site |

Tracker cross-check (aiworkshoptracker.com, refreshed 2026-09-06): same deadline (Sep 16, 11:59 UTC), and it notes the deadline **was extended from Aug 30** — do not expect a second extension.

Timezone caution: "11:59 am UTC" and "Sep 15 AoE" (= Sep 16 11:59 UTC) are consistent. Treat **Sep 16, 07:59 ET** as the hard wall; plan to submit by Sep 15 evening ET.

**Four topic axes** (verbatim-ish from the CFP):
1. Forecasting & simulation: long-horizon, multimodal contextual forecasting, calibrated probabilistic forecasts, trajectory simulation, adaptive forecasting.
2. Temporal data & environments: time series with text/video/graphs/sensors/trajectories/actions/events; pretraining corpora, benchmarks, simulation suites, synthetic data.
3. Temporal models: irregular/event-based, hierarchical multi-timescale, state-space, generative temporal models, TSFMs, multimodal fusion.
4. Evaluation & reliability: **leakage-aware evaluation, robustness under shift, calibration, consistency, scaling laws, contamination audits, reproducibility.**

Note the two organizers/speakers whose recent papers are directly relevant to the recommended project: Chenghao Liu and Ming Jin are co-authors of the TIME benchmark (2602.12147); Abdul Fatir Ansari leads Chronos-2 / fev-bench. Axis 4 (evaluation & reliability) plus "negative results and replication studies" is the sweet spot for a 4-page, 5-day empirical paper.

---

## 2. Literature scan (2025–2026)

Grouped by theme. "→" = one-line finding; "⚠" = what it did NOT test / where it stops.

### 2a. Time-series foundation models (TSFMs) and benchmarks

1. **Chronos-2: From Univariate to Universal Forecasting** — Ansari et al., Oct 2025, arXiv:2510.15821. → Group-attention 120M model; zero-shot univariate/multivariate/covariate forecasting; SOTA on fev-bench, GIFT-Eval, Chronos Benchmark II; trained with synthetic multivariate coupling. ⚠ No text conditioning; no post-cutoff/leakage analysis.
2. **TiRex-2: Generalizing TiRex to Multivariate Data and Streaming** — Podest et al. (Hochreiter group), Jul 2026, arXiv:2607.01204. → xLSTM 38M(+44M) model; constant per-patch streaming cost; SOTA on GIFT-Eval and fev-bench. ⚠ No calibration/probabilistic deep dive; no text.
3. **fev-bench: A Realistic Benchmark for Time Series Forecasting** — Shchur et al., Sep 2025 (v4 Jun 2026), arXiv:2509.26468. → 100 tasks / 7 domains, 46 with covariates; bootstrapped CIs on win-rate and skill score. ⚠ No textual context; leakage handled by curation, not by hard temporal cutoff.
4. **It's TIME: Towards the Next Generation of TSF Benchmarks** — Qiao, …, Ming Jin, Chenghao Liu, Mar 2026, arXiv:2602.12147. → 50 fresh datasets / 98 tasks; GIFT-Eval "plateaued"; Chronos-2, TimesFM-2.5, TiRex top-3; rankings flip with aggregation level; **TSFMs on high-variation series emit "relatively constant lines" that still score well** (open question). ⚠ No text/multimodal; no calibration study.
5. **Rethinking Evaluation in the Era of TSFMs: (Un)known Information Leakage Challenges** — Meyer et al., Oct 2025 (v3 Feb 2026), arXiv:2510.13654. → Audits 22 TSFMs × 401 datasets: **only 6% of datasets were never used in pre-training/fine-tuning**; documents >50% score inflation in isolated direct-overlap cases and 37–43% gains from *temporal* (correlated-series) leakage. ⚠ Explicitly states effect size of direct test contamination "has only been shown in isolated cases" and "the magnitude of temporal leakage … remains largely unknown in real TSFM settings"; calls for "carefully designed TSFM training regimes" nobody has run.
6. **FETS Benchmark: Foundation Models Outperform Dataset-specific ML in Energy TSF** — Obermeier et al., 2026, arXiv:2604.22328. → 54 energy datasets; Chronos-2 (covariate mode) wins; context-length saturates ~2,000 steps; fine-tuning shows catastrophic-forgetting-like collapses. ⚠ Point forecasts only — "lacks full probabilistic assessment and tail-risk analysis."
7. **TSFMs as Strong Baselines in Transportation Forecasting** — Yanes-Pulido & Rodrigues, May 2026, arXiv:2602.24238. → Chronos-2 zero-shot ranks 1st on 9/10 datasets; 80% intervals roughly calibrated but **under-coverage on zero-heavy series**. ⚠ Single TSFM; calibration is a side note.
8. **Foundation vs. Specialized Models: Catastrophic Forgetting in Continual TSF** — Karaouli et al., Oct 2025 (rev Jun 2026), arXiv:2510.00809. → Fine-tuning TimesFM-2.0/Chronos-2 causes forgetting; DER mitigation levels small vs. large. ⚠ Energy only.
9. **MixFT: LoRA fine-tuning of TSFMs by Bayesian sub-domain mixtures** — Mar 2026, arXiv:2603.02840. → Sub-domain-aligned LoRA modules beat per-dataset fine-tuning. ⚠ Small-scale.
10. **Zero-shot forecasting of chaotic systems** — Zhang & Gilpin (FMTS invited speaker), arXiv:2409.15771 (+ follow-up "True Zero-Shot Inference of Dynamical Systems Preserving Long-Term Statistics", arXiv:2505.13192). → TSFMs (Chronos) competitive with trained models on 135 chaotic systems for short horizons; long-term statistics need special handling. ⚠ Pre-Chronos-2/TiRex; no text.
11. **Dissecting Chronos: SAEs Reveal Causal Feature Hierarchies in TSFMs** — Mishra, Mar 2026, arXiv:2603.10071. → SAEs on Chronos-T5-Large: mid-encoder change-detection features are causally critical (ΔCRPS up to 38.6); ablating deep-layer features can *improve* forecasts. ⚠ Single model/architecture; no Chronos-2/TimesFM/TiRex.
12. **Sparse probes and murky physics (continuum-dynamics FM interpretability)** — Jun 2026, arXiv:2606.11657. → Probing a physics FM is hard; sparse probes yield ambiguous physical concepts. (Context for interp gap.)

### 2b. LLMs + text-conditioned ("multimodal") forecasting

13. **Are Language Models Actually Useful for Time Series Forecasting?** — Tan et al., NeurIPS 2024, arXiv:2406.16964. → Removing/replacing the LLM in LLM-TSF methods does not hurt; LLMs add cost, not accuracy. (Baseline skepticism.)
14. **Re-assessing LLMs as zero-shot forecasters** — Jun 2025, arXiv:2506.00457. → GPT-4o/Llama-3.1-70B lose to DLinear/TimeMixer on Monash/Informer; LLMs are noise-sensitive. ⚠ No textual context; heavily contaminated datasets.
15. **Context is Key (CiK)** — Williams et al., ICML 2025, arXiv:2410.18959. → 71 tasks where text is *essential*; LLMs with context beat TSFMs. ⚠ Contexts are largely synthetic; leakage not controlled — TimesX shows rankings reverse on real data.
16. **Rethinking Multimodal TSF Evaluation (TimesX)** — Liu, Zhou, Sen, Prakash, Das (Google), ICML 2026, arXiv:2607.06973. → New leakage-free, auto-refreshable benchmark (190 vars / 19 domains, real events). **Key findings:** rankings reverse between synthetic (CiK) and real (TimesX) contexts; LLMs degrade ~13% post-knowledge-cutoff, TSFMs <2%; **simple AvgEns(TimesFM-2.5 + Gemini) is #1 (MASE 0.619) while agentic revision methods are worse (0.713–0.720)**; **reasoning LLMs (o1, R1, Gemini-2.5) show "no clear advantage"** — "mechanisms remain unexplored"; no fine-tuning study ("deferred to future work"). Only closed LLMs (Gemini, GPT-4o, DeepSeek-V3/R1) evaluated — **no open ≤8B model**. Code/data public (google-research/TimesX).
17. **PostTime: Rethinking Post-Training Recipes for Multimodal TSF** — same Google/GT group, May 2026, arXiv:2605.29401. → CoT-SFT + RLVR on **Gemma-3-4B** as a *reviser* of the TimesFM-2.5 prior; improvement-ratio reward beats accuracy reward; fitting ground-truth directly *hurts*; keeping hard cases matters for tail (p99). Zero-shot revising with GPT-5 is only marginal — post-training "essential." ⚠ **Single TSFM prior (TimesFM-2.5), single base LLM, H=12 only, traces from Gemini-3.1-Flash-Lite; RL-only (no SFT) not tested; no text-ablation (is the reviser using text or just correcting TSFM bias?).**
18. **LLM as Forecasting Planner (LAFP): training-free text conditioning for TSFMs** — Nguyen et al., Jul 2026, arXiv:2607.24892. → MCTS over TSFM sample trajectories with an LLM ranker/judge; gains on CiK and Time-MMD with Chronos/TimesFM. ⚠ Evaluated on the two benchmarks TimesX flags as synthetic/leaky; no post-cutoff control.
19. **Context-Aware Probabilistic Modeling with LLM for Multimodal TSF** — May 2025, arXiv:2505.10774. → LLM-guided probabilistic heads on Time-MMD. ⚠ Same contamination caveat.
20. **Time-R1 / "TSF as Reasoning: Slow-Thinking with Reinforced LLMs"** — Zhou et al., Jun 2025 (rev Jun 2026), arXiv:2506.10630. → SFT warm-up + GRIP RL on multi-objective reward "significantly improves" forecasting. ⚠ Evaluated on classic (contaminated) datasets; no thinking-on/off control.
21. **COUNTS: Chain-of-thought for Understanding Numerical TS** — Oct 2025, arXiv:2510.01116. → RVQ-VAE tokens + SFT + GRPO improves TS reasoning tasks. ⚠ Task suite, not forecasting under real text.
22. **VeriTime: process-verifiable thinking data for TS reasoning** — Feb 2026, arXiv:2602.07830. → 3B/4B models match proprietary on TS-reasoning QA after two-stage RL. ⚠ QA-style; "leveraging LLM reasoning for TS remains in its infancy."
23. **AlphaCast: interaction-driven agentic reasoning for TSF** — Zhang, …, Cheng, Liu (USTC), Mar 2026, arXiv:2511.08947. → Claims reasoning + reflection are *essential* (large MSE drops on ETT/EPF datasets); GPT-5 best backbone. ⚠ Evaluated on ETTh/ETTm/EPF (heavily contaminated); **directly contradicts TimesX on whether agentic/reasoning helps.** Same group's position paper (arXiv:2602.01776) argues for agentic TSF with no experiments.
24. **SCALER: Efficient Test-Time Scaling for LLM-based TSF** — Le et al., KDD 2026, arXiv:2608.08675. → Coarse-to-fine refinement gives test-time-scaling gains 7× cheaper; notes candidate-selection TTS causes "global-shape drift." ⚠ Classic datasets; frozen LLM as refiner, no text context, no thinking-budget axis.
25. **Test-Time Augmentation for LLMs: input diversity beats output diversity at matched compute** — Aug 2026, arXiv:2608.09351. (Method reference for matched-compute controls.)
26. **Scaling Open-Ended Reasoning To Predict the Future (OpenForecaster-8B)** — Chandak, Goel, Prabhu, Hardt, Geiping, Dec 2025, arXiv:2512.25070. → GRPO with accuracy+Brier reward on Qwen3-8B-thinking; matches larger proprietary models on open-ended event forecasting; calibration transfers OOD. ⚠ Text events, not numeric series; thinking-length effects not analyzed.
27. **MMTS-Bench** (Feb 2026, arXiv:2602.08588), **TSAQA** (2601.23204), **ARFBench** (2604.21199), **SenTSR-Bench** (2602.19455), **MTBench** (2503.16858). → TS-QA benchmarks: TS-LLMs lag general LLMs; CoT helps QA; backbone capability dominates encoder design. ⚠ QA accuracy, not forecasting error.

### 2c. World models / language world models (workshop's "world modeling" axis)

28. **Qwen-AgentWorld: Language World Models for General Agents** — Qwen Team, Jun 2026, arXiv:2606.24597. → 35B-A3B and 397B-A17B LWMs trained CPT→SFT→RL on 10M trajectories; used as RL simulators; AgentWorldBench. ⚠ No analysis of long-horizon error accumulation; sizes far above 8B.
29. **Bridging the Agent-World Gap: Text World Models for LLM-based Agents (survey)** — Li et al., Jun 2026, arXiv:2606.09032. → Structured envs saturate ~99% single-step accuracy at 20K trajectories; web tasks unsaturated at 70K; "world-model RL remains noticeably less mature than SFT"; hallucination drift under on-policy rollout unresolved.
30. **From World Models to World Action Models: A Concise Tutorial for Robotics** — Jul 2026, arXiv:2607.00836; NVIDIA blog "Pretrained to Imagine, Fine-Tuned to Act." → Video-pretrained WAMs; compute far beyond a 4-GPU week. **Excluded** from sprint candidates on compute grounds.
31. **LIVE: Long-horizon Interactive Video World Modeling** — Feb 2026, arXiv:2602.03747. (Same exclusion.)

### 2d. What is claimed vs. untested vs. contradictory (summary)

| Claim in literature | Status | Cheap test? |
|---|---|---|
| Reasoning/agentic LLM loops improve forecasting (AlphaCast, Time-R1, position paper 2602.01776) | **Contradicted** by TimesX on leakage-free data (AvgEns > agentic; reasoning models no advantage) | Yes — inference-only with open hybrid thinking models |
| LLM text-conditioning beats TSFMs (CiK) | Reverses on real contexts (TimesX) | Yes |
| Post-trained small reviser > zero-shot frontier LLM (PostTime) | Shown only for Gemma-3-4B on TimesFM-2.5 prior | Yes — transfer to Chronos-2/TiRex untested |
| Direct test-set contamination inflates TSFM scores (Meyer et al.) | Effect size "shown only in isolated cases"; no controlled training regime exists | Partly — needs small-scale TSFM pretraining |
| TSFMs are well-calibrated (transport paper) | Under-coverage on zero-heavy series; FETS/TIME lack probabilistic audit; TIME notes "constant line" outputs | Yes — inference-only |
| Mid-layer change-detection features are causal in Chronos-T5 (SAE paper) | Single architecture | Yes — replicate on 3 more TSFMs |
| Text world models: SFT saturates structured envs; RL immature (survey) | No controlled SFT-vs-RL multi-step-drift study at ≤8B | Yes, with more engineering |

Saturated: GIFT-Eval aggregate leaderboard (TIME paper: "plateaued"), classic ETT/Weather/Traffic long-horizon suites (contaminated, discouraged). Under-explored: leakage-free multimodal forecasting with open small models; calibration under shift on fresh data; controlled contamination effect sizes; small-model world-model RL.

---

## 3. Gaps

**Gap 1 — Does reasoning actually help text-conditioned forecasting? (Unresolved contradiction, no controlled study.)**
TimesX (2607.06973) reports reasoning models (o1, R1, Gemini-2.5) give "no clear advantage" and explicitly says the mechanism is unexplored; PostTime (2605.29401) shows zero-shot GPT-5 revision is marginal. AlphaCast (2511.08947) and Time-R1 (2506.10630) claim reasoning is essential — but on contaminated classic datasets. Nobody has done the clean experiment: **same weights, thinking on vs. off, with a controlled reasoning-token budget, on leakage-free real data**, with a matched-compute self-consistency control. All TimesX/PostTime LLM baselines are closed models; no ≤8B open model has been evaluated on TimesX at all. This is squarely Axis 4 ("evaluation & reliability") + Axis 1 ("multimodal contextual forecasting") and matches the CFP's call for negative results/replications.

**Gap 2 — Does a post-trained small reviser transfer across TSFM priors, and is it using the text?**
PostTime trains one reviser (Gemma-3-4B) against one prior (TimesFM-2.5) and lists "generalization to other TSFMs (Chronos, Moirai) mentioned but not empirically validated" and single base LLM as limitations. No ablation trains/evaluates with shuffled or removed text, so it is unknown whether the reviser learns text-grounded revision or merely learns TimesFM-2.5's systematic bias. RL-only (no CoT-SFT, no frontier trace generator) is untested — the cheapest, most reproducible recipe.

**Gap 3 — Effect size of direct and temporal contamination in TSFMs under a controlled training regime.**
Meyer et al. (2510.13654) audit 22 TSFMs and find 94% of benchmark datasets were in some pre-training corpus, but state the effect size "has only been shown to be significant in isolated cases" and call for "carefully designed TSFM training regimes." The TIME benchmark (Ming Jin & Chenghao Liu — FMTS organizers) motivates fresh datasets by "plateaued" leaderboards but does not measure leakage effect size either. No paper trains a TSFM with dose-controlled contamination (0 / 0.1 / 1 / 10% of test data in pretraining) and reports the score inflation curve.

**Gap 4 — Multi-step consistency of small (≤4B) text world models: SFT vs. RL.**
The TWM survey (2606.09032) says world-model RL is "noticeably less mature than SFT," that "hallucination drift" under on-policy rollouts is unresolved, and that structured environments saturate single-step accuracy at ~20K trajectories — implying single-step accuracy is the wrong metric. Qwen-AgentWorld (35B+) does not analyze error accumulation. No controlled study measures k-step free-running rollout fidelity and downstream lookahead-planning utility for ≤4B models across SFT (full-state vs. delta) vs. GRPO with behavioral-consistency rewards.

**Gap 5 — Probabilistic calibration of zero-shot TSFMs under shift on fresh (post-release) data.**
FETS explicitly lacks probabilistic/tail evaluation; the transport benchmark finds under-coverage on zero-heavy series; TIME observes "constant line" forecasts that score well on MASE. No paper reports reliability diagrams / PIT histograms / coverage-vs-horizon for Chronos-2, TimesFM-2.5, TiRex(-2), Moirai-2, Sundial, Toto on data strictly after each model's release, nor whether cheap post-hoc conformalization (2507.08858 is the only related work) closes the gap.

**Gap 6 — Cross-architecture replication of the "change-detection features are causal" interpretability result.**
The Chronos SAE paper (2603.10071) is single-author, single-model (Chronos-T5-Large). Whether the mid-layer causal hierarchy and the "ablate deep features → better forecasts" effect hold for Chronos-2 (group attention), TimesFM-2.5 (decoder-only), and TiRex (xLSTM) is untested. Cheap (inference + SAE training on activations) but lower impact for this workshop than Gaps 1–3.

---

## 4. Sprint projects for the top 3 gaps

### Project A (Gap 1): "Thinking Budgets for Forecasting: When Does Reasoning Help Text-Conditioned Time-Series Forecasting?"

**Research question.** For open hybrid-reasoning LLMs, does enabling chain-of-thought (and scaling its token budget) improve leakage-free, text-conditioned forecasting accuracy over (a) the same model with thinking off, (b) the TSFM prior alone, and (c) the simple TSFM+LLM average ensemble — and under which context types?

**Falsifiable hypotheses.**
- H1 (null-leaning): On leakage-free real data (TimesX post-cutoff, Time-MMD post-cutoff), thinking-on does not reduce normalized MASE vs. thinking-off by more than the 95% bootstrap CI width. Falsified if thinking-on wins on ≥2/3 datasets with non-overlapping CIs.
- H2 (mechanism): Any benefit of thinking concentrates on windows whose context contains a *scheduled/contemporaneous event* (TimesX event-type labels) and vanishes on metadata-only windows. Falsified if the event-stratified effect is flat.
- H3 (compute): At matched output-token budget, self-consistency (median of N thinking-off samples) ≥ thinking-on. Falsified if thinking-on beats N=5 self-consistency at equal tokens.
- H4 (ensemble): AvgEns(TSFM, LLM-off) ≥ any thinking configuration in MASE (replicates TimesX with open models).

**Models (HF names).**
- Hybrid thinking LLMs (same weights, `enable_thinking` toggle in chat template): `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-1.7B`; optional `Qwen/Qwen3-30B-A3B` (fits one A100-80GB in bf16 via vLLM) for a size-scaling point.
- Non-thinking controls: `google/gemma-3-4b-it` (PostTime's base), `meta-llama/Llama-3.1-8B-Instruct`.
- Thinking-only variants for a robustness check (different weights, so secondary): `Qwen/Qwen3-4B-Thinking-2507` vs `Qwen/Qwen3-4B-Instruct-2507`.
- TSFM priors: `google/timesfm-2.5-200m-pytorch`, `amazon/chronos-2`, `NX-AI/TiRex` (add `Salesforce/moirai-2.0-R-small` if time).

**Datasets.**
- TimesX (google-research/TimesX; 190 vars; use eval windows with horizon start ≥ 2025-01-01 so Qwen3/Gemma-3 knowledge cutoffs are respected; T=96, H=12; context types: metadata / calendar / covariates / events).
- Time-MMD (AdityaLab/Time-MMD; 9 domains) — restrict to windows after the LLM cutoff; report full-set numbers in appendix flagged as contamination-risky.
- CiK (ServiceNow/context-is-key-forecasting; 71 tasks, synthetic contexts) — included *as the contrast condition* where TimesX predicts reasoning might look useful.

**Conditions.**
1. TSFM alone (3 priors).
2. LLM direct forecasting with context, thinking off.
3. LLM direct, thinking on, budget ∈ {512, 2048, 8192} tokens (budget forcing: truncate `<think>` at budget, append `</think>`, generate answer).
4. LLM-as-reviser of TSFM prior (PostTime zero-shot prompt), thinking off / on at the same budgets.
5. Self-consistency control: thinking-off, N=5 samples (T=0.7), median — matched to ~budget 2048.
6. AvgEns(TSFM, LLM-off) and AvgEns(TSFM, LLM-on@2048).
7. Context ablation: shuffled-context (events from another variable) and no-context, thinking on/off — tests whether thinking amplifies context use or context-*misuse*.

**Metrics.** Normalized MASE (geo-mean over variables, GIFT-Eval convention), CRPS (LLM: 5 samples; TSFM: quantiles), valid-output rate, "context sensitivity" = MASE(shuffled) − MASE(real), reasoning tokens used, wall-clock. Bootstrap CIs over variables (fev-style).

**Experimental grid.** Core: 3 datasets × 3 Qwen sizes × {off, on@512, on@2048, on@8192} × {direct, reviser(TimesFM-2.5)} = 72 configs; + 2 non-thinking controls × 3 datasets × 2 setups = 12; + reviser with Chronos-2/TiRex priors for Qwen3-8B only (12); + ablations (shuffled/no-context, self-consistency) for Qwen3-8B/4B (≈24). ≈120 configs. TimesX post-2025 eval ≈ 190 vars × ~15 windows ≈ 3k prompts; Time-MMD ≈ 1–2k; CiK ≈ 71 tasks × 5 seeds.

**GPU-hours.** vLLM on A100-80GB: thinking-off ≈ 3k prompts × ~200 tokens → minutes; thinking-on@8192 ≈ 3k × ~6k tokens ≈ 18M tokens ≈ 1.5–2.5 h for 8B. Estimate **60–90 GPU-hours total**; runs in ~2 days on 2 A100s, or 1 day on 4. TSFM inference negligible.

**Headline figure.** Fig 1: x = reasoning tokens (log; 0 for off), y = normalized MASE relative to TSFM prior (1.0 = prior), one line per model size, 3 panels (TimesX / Time-MMD / CiK); horizontal bands for AvgEns and self-consistency control. The expected story: flat-or-worse on real data, improving on synthetic CiK. Fig 2: event-stratified Δ(MASE) bars (metadata-only vs. event windows) — "where thinking helps." Fig 3: context-sensitivity scatter (thinking on vs. off) or reliability/CRPS. Table 1: full grid with bootstrap CIs.

**Risks & fallback.** (i) Thinking helps substantially → still a clean positive result: "thinking helps *only* via events; budget X saturates" — report as is. (ii) Output-format failures at small sizes → JSON schema + fallback-to-prior, report valid-rate. (iii) TimesX post-2025 windows too few for stable rankings (paper says ≥20 variables needed; we have 190) → fine. (iv) Someone posts the same study before Sep 16 → differentiate via budget-forcing + matched-compute + context-ablation, which is unlikely to be duplicated together.

**Novelty claim (1 sentence).** First same-weights, budget-controlled study of whether chain-of-thought reasoning helps LLM text-conditioned forecasting on leakage-free real data, resolving the TimesX-vs-AlphaCast contradiction with open ≤8B models and a matched-compute control.

**Feasibility: 9/10.** Inference-only, all assets public, 4-page format suits a negative/replication result the CFP explicitly welcomes.

---

### Project B (Gap 2): "Cheap Revisers: RL-Only Post-Training of a 1.7–4B LLM to Revise TSFM Forecasts — Transfer Across Priors and Text-Ablation"

**Research question.** Can a ≤4B open LLM be post-trained *with RL only* (no CoT-SFT, no frontier trace generator) to revise a TSFM prior using text, and does the learned reviser (a) transfer to unseen TSFM priors, (b) depend on the text rather than on the prior's systematic bias?

**Hypotheses.**
- H1: GRPO with PostTime's improvement-ratio reward on Qwen3-4B yields ΔMASE ≥ 3% over TimesFM-2.5 prior on TimesX OOD (falsified if CI includes 0).
- H2 (transfer): A reviser trained on TimesFM-2.5 retains ≥50% of its gain when swapped to Chronos-2 / TiRex priors at test time (falsified if gain ≤ 0 on ≥2 priors).
- H3 (text use): A reviser trained with shuffled text has ≤ one-third the gain of one trained with real text (falsified if equal — meaning revision is bias-correction, not context use).
- H4: RL-only ≥ 80% of the SFT+RL gain reported by PostTime for the same base (Gemma-3-4B replication point).

**Models.** Policy: `Qwen/Qwen3-4B`, `Qwen/Qwen3-1.7B`, `google/gemma-3-4b-it` (replication). LoRA r=64 via TRL `GRPOTrainer` or verl; group size 8; reward = clip(0.5 + 0.5·ImpRatio) with ImpRatio = (MAE_prior − MAE_revised)/MAE_prior on the *training* horizon (PostTime's reward), + format reward. Priors: `google/timesfm-2.5-200m-pytorch` (train), `amazon/chronos-2`, `NX-AI/TiRex`, `Salesforce/moirai-2.0-R-small` (test-time swap).

**Datasets.** TimesX fine-tuning split (2018–2022, 88 ID vars) for RL; eval on TimesX 2023–2025 ID + 11 OOD vars; Time-MMD as second OOD set.

**Grid.** 3 bases × {RL real-text, RL shuffled-text, RL no-text} = 9 training runs (+1 seed repeat for the main config) ≈ 10 runs × 5–8 GPU-h (LoRA, 4B, ~600 steps, 8 rollouts × 256 tokens) ≈ 60–80 GPU-h; eval on 4 priors × 3 sets ≈ 10 GPU-h. **Total ≈ 80–100 GPU-h**; 2–3 days on 4 A100s.

**Metrics.** Normalized MASE/MSE, p99 nMSE (PostTime's tail metric), valid-window rate, defer-to-prior rate, CRPS.

**Headline figure.** Fig 1: 3×4 transfer heatmap (train prior × test prior) of ΔMASE vs. prior. Fig 2: bars of gain under real / shuffled / no text at train and test time (2×2). Fig 3: RL training curves of reward and valid-rate.

**Risks & fallback.** RL instability / reward hacking (e.g., model copies prior → ImpRatio 0 → reward 0.5; mitigate with format + small length penalty). If RL-only fails to beat prior, the paper becomes a negative-result/replication ("PostTime's CoT-SFT stage is necessary") — still in scope. Compute is tighter; drop Gemma replication first if behind schedule.

**Novelty claim.** First evidence on whether a text-grounded forecast reviser learned by RL alone transfers across TSFM priors, and a text-ablation that separates context-use from prior-bias-correction.

**Feasibility: 6.5/10.**

---

### Project C (Gap 4): "Rollout Drift in Small Text World Models: SFT vs. RL for Multi-Step Consistency"

**Research question.** For ≤4B LLM world models of a text environment, how does free-running k-step rollout fidelity decay under SFT (full-state vs. delta targets) vs. GRPO with a behavioral-consistency reward, and which yields better lookahead planning?

**Hypotheses.** H1: Single-step accuracy saturates (>97%) for all recipes at 20K transitions (survey claim), but k=10 free-running exact-state accuracy differs by >15 points between recipes. H2: GRPO with behavioral-consistency reward (does the predicted state induce the same next action from a fixed policy?) beats token-level SFT on k≥5 drift. H3: Planning success (1-step lookahead with the WM over 3 candidate actions) tracks k-step fidelity, not single-step accuracy.

**Models/envs.** `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` (LoRA). Environments: ALFWorld (TextWorld engine; deterministic, structured observations) and ScienceWorld or WebShop-lite as a second, noisier domain. Collect 20K (s, a, s') transitions with a scripted + random policy (CPU, hours). Fixed evaluation policy: `Qwen/Qwen3-8B` (thinking off) for behavioral-consistency reward and planning tests.

**Grid.** 2 sizes × 3 recipes × 2 envs = 12 runs (SFT ~1 GPU-h each; GRPO ~6 GPU-h each) ≈ 40–50 GPU-h + eval (k up to 20, 500 trajectories, ≈10 GPU-h). **Total ≈ 60 GPU-h.**

**Metrics.** Exact-match / token-F1 per step k under teacher forcing vs. free running; behavioral-consistency rate; hallucinated-object rate; planning success rate with WM lookahead vs. no lookahead.

**Headline figure.** Fig 1: fidelity vs. rollout step k (teacher-forced dashed, free-running solid), lines per recipe, panels per env. Fig 2: planning success vs. k-step fidelity scatter across the 12 models. Fig 3: hallucination-drift examples/rates.

**Risks & fallback.** Env plumbing eats a day; behavioral reward needs a policy call per rollout (use small policy; cache). If RL fails, report SFT full-state vs. delta drift curves + the "single-step saturates but multi-step doesn't" result. Fits Axis 1 ("trajectory simulation") and the "world modeling" half of the workshop title but is farther from the organizers' forecasting center of gravity.

**Novelty claim.** First controlled measurement of multi-step rollout drift in ≤4B text world models across SFT and RL recipes, showing single-step accuracy is a misleading proxy for planning utility.

**Feasibility: 6/10.**

---

### Honorable mention (Gap 3): controlled contamination dose-response in a small TSFM
Train `chronos-t5-tiny/mini`-class models (8–20M) from Chronos's public recipe on the public TSMixup+KernelSynth corpus (`autogluon/chronos_datasets`) with 0 / 0.1 / 1 / 10% of selected GIFT-Eval test series injected (direct leakage) or of temporally-overlapping correlated series (temporal leakage); evaluate on injected vs. held-out sets. ~8 runs × ~6–10 GPU-h at reduced steps ≈ 60–80 GPU-h. Directly answers Meyer et al.'s open call and the CFP's "contamination audits" bullet; organizers (TIME authors) care. Feasibility 5.5/10 — pretraining time and reduced-scale generalization risk make it a stretch in 9 days but an excellent follow-up.

---

## 5. Ranked recommendation

1. **Project A — Thinking budgets for forecasting (9/10).** Highest expected value per GPU-hour: inference-only, every asset public, it resolves a live contradiction between two 2026 papers (TimesX vs. AlphaCast/Time-R1), extends the ICML-2026 TimesX benchmark to open ≤8B models for the first time, and its most likely outcome (a well-controlled negative result with a mechanism stratification) is explicitly welcomed by the CFP. It intersects the researcher's interests in reasoning/post-training, model behavior, and small models. Schedule: Sep 6–7 pipeline (TimesX loader, vLLM harness, budget forcing, TSFM priors); Sep 8–10 core grid; Sep 11 ablations; Sep 12–14 write 4 pages + figures; Sep 15 buffer/submit. Add Project B's zero-shot reviser condition (already in the grid) so the paper connects to PostTime.
2. **Project B — RL-only reviser transfer (6.5/10).** Strong follow-on; the same harness from A provides the eval. Do it only if A's core grid finishes by Sep 10 with GPUs free, or as the next paper.
3. **Project C — Text world model rollout drift (6/10).** Best fit to the researcher's agents/world-model interests and the workshop's "world modeling" half, but more engineering and less aligned with the organizers' forecasting/evaluation focus; save for a later venue unless A is blocked.

Write-up guidance: use the NeurIPS 2026 template, 4 pages + appendix, anonymized; frame as "evaluation & reliability" with a replication component; report bootstrap CIs (fev convention); release code + prompts; state LLM knowledge cutoffs and the exact post-cutoff window selection in the main text (reviewers will look for it given TimesX's leakage finding).

---

## 6. Sources

Workshop
- https://fmts-workshop.github.io/
- https://fmts-workshop.github.io/cfp.html
- https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/FMTS
- https://aiworkshoptracker.com/workshop/neurips-2026-fmts/

TSFMs and benchmarks
- Chronos-2: https://arxiv.org/abs/2510.15821
- TiRex-2: https://arxiv.org/abs/2607.01204
- fev-bench: https://arxiv.org/abs/2509.26468
- TIME benchmark: https://arxiv.org/html/2602.12147v3
- Leakage in TSFM evaluation (Meyer et al.): https://arxiv.org/html/2510.13654v3
- FETS benchmark: https://arxiv.org/html/2604.22328
- TSFMs in transportation: https://arxiv.org/html/2602.24238
- Catastrophic forgetting in TSFMs: https://arxiv.org/abs/2510.00809
- MixFT: https://arxiv.org/abs/2603.02840
- Zero-shot chaotic systems: https://arxiv.org/abs/2409.15771 ; https://arxiv.org/abs/2505.13192
- Dissecting Chronos (SAEs): https://arxiv.org/abs/2603.10071
- Sparse probes and murky physics: https://arxiv.org/abs/2606.11657
- Moirai 2.0: https://arxiv.org/pdf/2511.11698
- GIFT-Eval: https://huggingface.co/spaces/Salesforce/GIFT-Eval ; https://github.com/SalesforceAIResearch/gift-eval
- Chronos code/datasets: https://github.com/amazon-science/chronos-forecasting

LLMs and multimodal forecasting
- Are LMs actually useful for TSF: https://arxiv.org/abs/2406.16964
- Re-assessing LLM zero-shot forecasting: https://arxiv.org/abs/2506.00457
- Context is Key: https://arxiv.org/abs/2410.18959
- TimesX (Rethinking Multimodal TSF Evaluation): https://arxiv.org/abs/2607.06973 ; code https://github.com/google-research/google-research/tree/master/TimesX/dataset_agent ; project https://haoxin1998.github.io/TimesX-project/
- PostTime: https://arxiv.org/html/2605.29401v1
- LAFP (LLM as Forecasting Planner): https://arxiv.org/abs/2607.24892
- Context-aware probabilistic modeling with LLM: https://arxiv.org/abs/2505.10774
- Time-R1: https://arxiv.org/abs/2506.10630
- COUNTS: https://arxiv.org/abs/2510.01116
- VeriTime: https://arxiv.org/abs/2602.07830
- AlphaCast: https://arxiv.org/html/2511.08947v3
- Position: Agentic TSF: https://arxiv.org/html/2602.01776v1
- SCALER (test-time scaling for LLM TSF): https://arxiv.org/abs/2608.08675
- Test-time augmentation at matched compute: https://arxiv.org/abs/2608.09351v1
- OpenForecaster (Scaling open-ended reasoning to predict the future): https://arxiv.org/html/2512.25070
- Noise injection for zero-shot LLM TSF: https://arxiv.org/html/2512.20140v1
- MMTS-Bench: https://arxiv.org/abs/2602.08588 ; TSAQA: https://arxiv.org/html/2601.23204 ; ARFBench: https://arxiv.org/pdf/2604.21199 ; SenTSR-Bench: https://arxiv.org/html/2602.19455v1 ; MTBench: https://arxiv.org/abs/2503.16858
- Qwen3 technical report (hybrid thinking toggle): https://arxiv.org/abs/2505.09388
- Conformal prediction with TSFMs: https://arxiv.org/abs/2507.08858v1

World models
- Qwen-AgentWorld: https://arxiv.org/html/2606.24597v1
- Text World Models for LLM-based Agents (survey): https://arxiv.org/html/2606.09032v1
- From Word to World (LLMs as implicit text world models): https://arxiv.org/abs/2512.18832
- World Models to World Action Models tutorial: https://arxiv.org/html/2607.00836v1
- NVIDIA world-action models blog: https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
- LIVE long-horizon video world modeling: https://arxiv.org/html/2602.03747v1

Contamination (LLM side, methodological reference)
- Detecting data contamination from RL post-training: https://arxiv.org/pdf/2510.09259
