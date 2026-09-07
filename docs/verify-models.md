# Verified model & tooling facts for the implementation spec

Verified 2026-09-06 against live model cards, PyPI metadata, package wheels (unzipped and read) and the
vLLM 0.28.0 source distribution. Every statement below is either quoted from a primary source or was read
directly from package source. Things I could NOT confirm are marked **UNVERIFIED**.

Implementer note: you have no internet. All HF ids, package names/versions and function signatures below
were checked; when in doubt, trust this file over memory.

---

## 0. Headline changes vs. the original plan (read first)

1. **Newer Qwen families exist and keep the `enable_thinking` toggle.** `Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B,35B-A3B,...}`
   (Feb 2026) and `Qwen/Qwen3.6-{27B,35B-A3B}` (Apr 2026). Both are hybrid-thinking (thinking ON by default,
   `chat_template_kwargs={"enable_thinking": False}` turns it off), both use `--reasoning-parser qwen3`.
   Caveats: they are **multimodal** (`Qwen3_5ForConditionalGeneration`, pipeline `image-text-to-text`) with a
   Gated-DeltaNet + Gated-Attention hybrid architecture (small ones are dense; 35B-A3B is MoE); use
   `--language-model-only` in vLLM. No `Qwen3.6` model below 27B exists. No `-Instruct`/`-Thinking` split
   variants exist for Qwen3.5 (only `-Base` and quantized variants).
2. **vLLM (current release 0.28.0, 2026-08-26) has a native thinking budget**: `SamplingParams(thinking_token_budget=N)`
   + `ReasoningConfig(reasoning_start_str="<think>", reasoning_end_str="...</think>")` (offline) or
   `--reasoning-config '{...}'` + `"thinking_token_budget": N` in the request (server). Shipped since v0.19.0
   (2026-04-03). **BUT** (read from 0.28.0 source) for MoE / hybrid models (Qwen3-30B-A3B, Qwen3.5/3.6) the
   legacy model-runner path detects a *natural* `</think>` by matching the *full* `reasoning_end_str`, so a
   transition phrase in `reasoning_end_str` can cause forced injection mid-answer. Safe choices: (a) the
   two-pass Qwen recipe (Section A.3, recommended, exactly reproducible), or (b) native budget with
   `reasoning_end_str="</think>"` (no phrase). Also: native budget was silently ignored with MTP/spec-decode
   before v0.21.0; don't enable spec decoding when using it.
3. **TimesFM 3.0 is out (Aug 2026)**: `google/timesfm-3.0-pytorch`, 0.3B params, native covariates, #1 on
   fev-bench/GIFT-Eval — but weights are under a **non-commercial license** (fine for research). The PyPI package
   `timesfm==3.0.1` (2026-09-02) contains BOTH the 2.5 API (`import timesfm`) and the 3.0 API (`import timesfm3`).
   `google/timesfm-2.5-200m-pytorch` still exists (Apache-2.0).
4. **TiRex**: the pip package is **`tirex-ts`** (NOT `tirex` — the PyPI `tirex` 0.0.2 is an unrelated 2022 squat).
   Pure-PyTorch sLSTM backend is the default (`backend="torch"`); custom CUDA kernels are optional
   (`pip install "tirex-ts[cuda]"`, compiled at first load via `xlstm`, needs nvcc + CC>=8.0). **No CUDA kernel
   build is required** on Della. A successor **TiRex-2** (`NX-AI/TiRex-2`, pip `tirex-2`, Apache-2.0, multivariate
   + covariates, July 2026) exists; its CUDA path *does* JIT-compile FlashRNN with nvcc, but `device="cpu"` works.
5. **Knowledge cutoffs**: only Gemma 3 (Aug 2024) and Llama 3.1 (Dec 2023) are officially documented. Qwen3 /
   Qwen3-2507 / Qwen3.5 / Qwen3.6 have **no official cutoff**; third-party (OpenRouter metadata) lists
   Qwen3 = 2025-03-31, Qwen3-2507 = 2025-06-30. Qwen3.5 released 2026-02, Qwen3.6 2026-04. If Qwen3.5/3.6 are in
   the pool, evaluation windows must start **no earlier than 2026-03-01** (prefer 2026-05-01+); if only
   Qwen3/2507 + Gemma3 + Llama3.1, windows starting **2025-08-01 or later** are safe.
6. **Moirai-2 via `uni2ts`** pins `torch<2.5`, `gluonts~=0.14.3`, `numpy~=1.26` — incompatible with the
   chronos/timesfm/tirex-2 environments. Put it in a separate conda env or drop it.
7. **Della**: compute nodes have NO internet; download on `della-gpu` login node (or della-vis1/2). 59 nodes x 4 x
   A100-80GB (`--constraint=gpu80`), 42 nodes x 8 x H100-80GB are PLI-only (`--partition=pli-c` / `--partition=pli`),
   QOS wall-times: gpu-short 24h, gpu-medium 72h, gpu-long 144h. Current module: `anaconda3/2026.7`. Apptainer available.

---

## A. Qwen3 hybrid thinking + budget forcing in vLLM

### A.1 HF ids (all confirmed to exist via the HF API on 2026-09-06)

| id | notes |
|---|---|
| `Qwen/Qwen3-8B` | hybrid, dense, Apr 2025. 32,768 native ctx (131,072 w/ YaRN). config `max_position_embeddings=40960`. |
| `Qwen/Qwen3-4B` | hybrid, dense |
| `Qwen/Qwen3-1.7B` | hybrid, dense |
| `Qwen/Qwen3-0.6B` | hybrid, dense |
| `Qwen/Qwen3-30B-A3B` | hybrid, MoE (30B total / 3B active). bf16 ≈ 60 GB → fits one A100-80GB with small KV; `Qwen/Qwen3-30B-A3B-FP8` exists. |
| `Qwen/Qwen3-4B-Thinking-2507` | **thinking-only**; card: "This model supports only thinking mode. Meanwhile, specifying `enable_thinking=True` is no longer required." The chat template pre-fills `<think>\n`, so generated text contains only `</think>`. Native ctx 262,144. lastModified 2025-08-06. |
| `Qwen/Qwen3-4B-Instruct-2507` | non-thinking only (no `<think>` blocks). |
| `Qwen/Qwen3-30B-A3B-Thinking-2507`, `Qwen/Qwen3-30B-A3B-Instruct-2507` | exist |
| `Qwen/Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B`, `-27B`, `-35B-A3B`, `-122B-A10B`, `-397B-A17B` | Feb 2026. Hybrid thinking (default ON). Multimodal (T+I+V). 262,144 native ctx (1,010,000 w/ YaRN). `-Base` and `-FP8`/`-GPTQ-Int4` (27B and up only) variants exist. **No `-Instruct`/`-Thinking` variants.** |
| `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B` | Apr 2026. Same toggle; adds `chat_template_kwargs={"preserve_thinking": True}`. Card: "vllm>=0.19.0 is recommended for Qwen3.6". **No small (<27B) Qwen3.6.** |
| `google/gemma-3-4b-it` | exists; vLLM arch `Gemma3ForConditionalGeneration` (text+image). 128K ctx. |
| `meta-llama/Llama-3.1-8B-Instruct` | exists (gated; needs HF token accepted on login node). 128K ctx. |

Qwen3.5-4B architecture (card, verbatim): "Hidden Layout: 8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))";
Qwen3.5-9B: 32 layers, hidden 4096, same layout. vLLM 0.28.0 supported-models table lists
`Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration` (both ✅). Qwen3.5 card install note at
release: "vLLM from the main branch ... is required" — superseded; 0.28.0 supports it.

**Recommendation:** keep `Qwen/Qwen3-8B` / `-4B` / `-1.7B` as the *primary* hybrid family (standard transformer,
text-only, best-documented budget-forcing behavior, V2 model runner in vLLM). Add `Qwen/Qwen3.5-4B` and
`Qwen/Qwen3.5-9B` as a "newer family" ablation only if evaluation windows can be pushed to ≥ 2026-03-01.

### A.2 Toggling thinking (verified syntax; vLLM 0.28.0)

Reasoning parser: `--reasoning-parser qwen3` (vLLM ≥ 0.9.0; the older `--enable-reasoning --reasoning-parser deepseek_r1`
still appears on some Qwen model cards but is legacy). vLLM docs (0.28.0): "The reasoning feature for the Qwen3
series is enabled by default. To disable it, you must pass `enable_thinking=False` in your `chat_template_kwargs`."
**Field rename:** vLLM now returns `message.reasoning` (docs: "`reasoning` used to be called `reasoning_content`.
To migrate, directly replace `reasoning_content` with `reasoning`.").

Offline (Qwen docs, verbatim; `LLM.chat` + `chat_template_kwargs` since vLLM 0.9.0):

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
llm = LLM(model="Qwen/Qwen3-8B")
messages = [{"role": "user", "content": "Give me a short introduction to large language models."}]
outputs = llm.chat(
    [messages],
    sampling_params,
    chat_template_kwargs={"enable_thinking": True},  # Set to False to strictly disable thinking
)
```

`LLM.chat` signature in 0.28.0 (read from `vllm/entrypoints/llm.py`):
`chat(messages, sampling_params=None, use_tqdm=True, lora_request=None, chat_template=None,
chat_template_content_format="auto", add_generation_prompt=True, continue_final_message=False, tools=None,
chat_template_kwargs=None, tokenization_kwargs=None, mm_processor_kwargs=None) -> list[RequestOutput]`.

Alternative offline route (also in Qwen docs): render the prompt yourself with
`tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)` and
call `llm.generate([text], sampling_params)`.

OpenAI-compatible server (Qwen docs, verbatim):

```bash
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3
```

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Give me a short introduction to large language models."}],
    max_tokens=8192,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

Server-wide default (vLLM docs): `vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'`
— request-level `chat_template_kwargs` always overrides. Qwen note: "passing `enable_thinking` is not OpenAI API
compatible." Soft switch: append `/think` or `/no_think` to the user turn (only works when `enable_thinking=True`).

Qwen3.5 specifics: `vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only`
(card, verbatim). Offline equivalent: `LLM(model="Qwen/Qwen3.5-4B", language_model_only=True, max_model_len=...)`
(`language_model_only: bool` is an `EngineArgs` field in 0.28.0; "If True, disables all multimodal inputs").
Thinking-off: same `chat_template_kwargs={"enable_thinking": False}`.

### A.3 Budget forcing — Qwen's official recipe (verbatim from QwenLM/Qwen3 docs, `quickstart.md` "Thinking Budget")

> "Qwen3 supports the configuration of thinking budget. It is achieved by ending the thinking process once the
> budget is reached and guiding the model to generate the "summary" with an early-stopping prompt. Since this
> feature involves customization specific to each model, it is currently not available in the open-source
> frameworks and only implemented by the Alibaba Cloud Model Studio API. However, with existing open-source
> frameworks, one can generate twice to implement this feature as follows:
> 1. For the first time, generate tokens up to the thinking budget and check if the thinking process is finished.
>    If the thinking process is not finished, append the early-stopping prompt.
> 2. For the second time, continue generation until the end of the content or the upper length limit is fulfilled."

Exact early-stopping string (verbatim from their code; 24 tokens):

```python
early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
```

Special token ids used by their reference code: `151645` = `<|im_end|>`, `151668` = `</think>`. Their note:
"`thinking_budget` should not be set to that low in practice. We recommend tuning `thinking_budget` based on the
latency users can accept and setting it higher than 1024 for meaningful improvements across tasks. If thinking is
not desired at all, developers should make use of the hard switch instead."

Their HF Transformers reference (verbatim, trimmed to the logic):

```python
thinking_budget = 16          # demo value only; use >= 1024 in practice
max_new_tokens = 32768
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
input_length = model_inputs.input_ids.size(-1)

generated_ids = model.generate(**model_inputs, max_new_tokens=thinking_budget)
output_ids = generated_ids[0][input_length:].tolist()

if 151645 not in output_ids:                       # not finished
    if 151668 not in output_ids:                   # thinking not finished
        early_stopping_ids = tokenizer([early_stopping_text], return_tensors="pt", return_attention_mask=False).input_ids.to(model.device)
        input_ids = torch.cat([generated_ids, early_stopping_ids], dim=-1)
    else:
        input_ids = generated_ids
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   max_new_tokens=input_length + max_new_tokens - input_ids.size(-1))
    output_ids = generated_ids[0][input_length:].tolist()

try:
    index = len(output_ids) - output_ids[::-1].index(151668)   # rindex of </think>
except ValueError:
    index = 0
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
```

**Two-pass port to vLLM offline (recommended primary implementation; batched; prefix caching makes pass 2 cheap):**

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

MODEL = "Qwen/Qwen3-8B"
IM_END, THINK_END = 151645, 151668
EARLY_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, max_model_len=40960, enable_prefix_caching=True)  # prefix caching is default-on in V1
early_ids = tok(EARLY_STOP, add_special_tokens=False).input_ids

def budgeted_generate(list_of_messages, budget, answer_max_tokens, seed=0):
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
               for m in list_of_messages]
    prompt_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]

    # pass 1: think up to `budget` tokens (thinking-mode sampling params)
    sp1 = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=budget, seed=seed)
    out1 = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp1)

    pass2_inputs, pass2_idx, results = [], [], [None] * len(prompts)
    for i, o in enumerate(out1):
        gen = list(o.outputs[0].token_ids)
        if IM_END in gen:                      # fully finished within budget
            results[i] = gen
            continue
        cont = prompt_ids[i] + gen + ([] if THINK_END in gen else early_ids)
        pass2_inputs.append(TokensPrompt(prompt_token_ids=cont)); pass2_idx.append((i, len(prompt_ids[i])))

    if pass2_inputs:
        sp2 = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=answer_max_tokens, seed=seed)
        out2 = llm.generate(pass2_inputs, sp2)
        for (i, plen), o in zip(pass2_idx, out2):
            full = list(o.prompt_token_ids[plen:]) + list(o.outputs[0].token_ids)
            results[i] = full

    parsed = []
    for ids in results:
        try:
            k = len(ids) - ids[::-1].index(THINK_END)
        except ValueError:
            k = 0
        parsed.append((tok.decode(ids[:k], skip_special_tokens=True).strip("\n"),
                       tok.decode(ids[k:], skip_special_tokens=True).strip("\n"),
                       len(ids[:k])))   # (thinking, answer, thinking_tokens_used)
    return parsed
```

Notes for the port: `TokensPrompt` is exported from `vllm` top-level in 0.28.0; `RequestOutput.outputs[0].token_ids`
gives generated ids and `RequestOutput.prompt_token_ids` the prompt ids. For `Qwen3-4B-Thinking-2507` the
template already pre-fills `<think>\n` so `THINK_END` alone marks the boundary (same code works). Budget = 0
("no thinking") should be run with `enable_thinking=False` (Qwen's "hard switch") rather than forcing.

**Native vLLM alternative (`thinking_token_budget`, shipped v0.19.0+, present in 0.28.0):**

Docs (verbatim): "Token counting starts from `reasoning_start_str`. Once the reasoning token count reaches the
configured `thinking_token_budget`, vLLM forces the model to produce `reasoning_end_str`, effectively terminating
the reasoning block. ... `--reasoning-parser` enables reasoning extraction. `--reasoning-config` defines the
reasoning boundary tokens (e.g., `reasoning_start_str`, `reasoning_end_str`). If not set, vLLM will attempt to
automatically initialize these tokens from the reasoning parser. `thinking_token_budget` (a sampling parameter)
sets the per-request reasoning token limit." Note: "`reasoning_end_str` can include a transition phrase before the
reasoning end token. For example, setting `reasoning_end_str` to `"I have to give the solution based on the
reasoning directly now.</think>"` ...".

```bash
vllm serve Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}'
# request body (top-level, not inside extra_body's chat_template_kwargs):  "thinking_token_budget": 10
```

```python
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
llm = LLM(model="Qwen/Qwen3-0.6B",
          reasoning_config=ReasoningConfig(reasoning_parser="qwen3",           # <- add this (see caveat)
                                           reasoning_start_str="<think>",
                                           reasoning_end_str="</think>"))     # <- no phrase (see caveat)
sampling_params = SamplingParams(thinking_token_budget=10)   # int >= 0, or -1 = unlimited (validator in sampling_params.py)
outputs = llm.chat(messages, sampling_params=sampling_params)
```

`ReasoningConfig` fields (0.28.0 `vllm/config/reasoning.py`): `reasoning_parser: str = ""`,
`reasoning_start_str: str = ""`, `reasoning_end_str: str = ""` ("String forced when the thinking budget is
exhausted"). Token ids derived by `initialize_token_ids`; when `reasoning_parser` is set, the parser supplies the
*natural* end marker (`</think>`), otherwise `natural_reasoning_end_str = reasoning_end_str`.

Caveats found by reading the 0.28.0 source:
- Two runner implementations exist. Dense Qwen3 (`Qwen3ForCausalLM`) defaults to the **V2 model runner**
  (`vllm/v1/worker/gpu/sample/thinking_budget.py`), which uses `natural_reasoning_end_token_ids` — correct with a
  transition phrase. MoE (`Qwen3MoeForCausalLM`, not in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`) and hybrid
  Qwen3.5/3.6 (`is_hybrid`) fall back to the **V1 runner** (`vllm/v1/sample/thinking_budget_state.py`), whose
  natural-end scan is `_find_last_sequence_index(output, self.think_end_token_ids)` where `think_end_token_ids =
  reasoning_config.reasoning_end_token_ids` (the FULL forced string). A bare `</think>` emitted naturally will not
  match → counting continues into the answer → forced injection mid-answer once the budget is hit. Mitigation:
  `reasoning_end_str="</think>"` (no phrase), or force `VLLM_USE_V2_MODEL_RUNNER=1` (unsupported for hybrids).
- Before v0.21.0 (2026-05-15), `thinking_token_budget` was silently ignored with MTP/speculative decoding
  (fixed by PR #34668). Do not combine with `--speculative-config`.
- Open issue #44676 (vLLM 0.22): with tool calling on Qwen3.5+, tight budgets inject the end string inside tool-call
  JSON. Plain chat is unaffected.
- The V1 holder counts tokens after a `<think>` already present in the prompt (Thinking-2507 templates), so
  budgets apply to those models too.

### A.4 Recommended sampling parameters (model cards, verbatim)

Qwen3 (Apr 2025; `Qwen/Qwen3-8B` card):
- Thinking mode (`enable_thinking=True`): `Temperature=0.6, TopP=0.95, TopK=20, MinP=0`. "DO NOT use greedy
  decoding, as it can lead to performance degradation and endless repetitions."
- Non-thinking mode (`enable_thinking=False`): `Temperature=0.7, TopP=0.8, TopK=20, MinP=0`.
- "presence_penalty ... between 0 and 2 to reduce endless repetitions" (Qwen's server example uses 1.5 for
  non-thinking). Recommended output length 32,768 (38,912 for hard math/code).

Qwen3-4B-Thinking-2507: `Temperature=0.6, TopP=0.95, TopK=20, MinP=0`, presence_penalty 0–2, output 32,768
(81,920 for hard problems). Qwen3-4B-Instruct-2507: same as Qwen3 non-thinking (T=0.7, top_p=0.8, top_k=20, min_p=0).

Qwen3.5 / Qwen3.6 cards:
- Thinking, general: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
- Thinking, precise coding: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`
- Non-thinking (instruct), general: `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5`
- Non-thinking, reasoning tasks (Qwen3.5 card): `temperature=1.0, top_p=1.0, top_k=40, min_p=0.0, presence_penalty=2.0`

vLLM `SamplingParams` accepts `temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, max_tokens,
seed, thinking_token_budget` (0.28.0). For numeric forecasting outputs, keep the card defaults but set `seed` and
run ≥3 seeds; do not use `temperature=0` in thinking mode.

### A.5 Knowledge cutoffs (for choosing evaluation windows)

| model | cutoff | source / status |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | **December 2023** | model card: "The pretraining data has a cutoff of December 2023." (official) |
| `google/gemma-3-4b-it` | **August 2024** | Gemma 3 model card (ai.google.dev): "The knowledge cutoff date for the training data was August 2024." (official; the HF card omits it) |
| `Qwen/Qwen3-*` (Apr 2025) | not stated | Tech report (arXiv 2505.09388) gives 36T tokens / 119 languages, no date. Third-party OpenRouter metadata: 2025-03-31. **UNVERIFIED** |
| `Qwen/Qwen3-*-2507` | not stated | third-party: 2025-06-30. **UNVERIFIED** |
| `Qwen/Qwen3.5-*` (Feb 2026) | not stated | DashScope system prompt says "2026" only. Assume ≤ 2026-01. **UNVERIFIED** |
| `Qwen/Qwen3.6-*` (Apr 2026) | not stated | assume ≤ 2026-03. **UNVERIFIED** |

Decision rule: with Qwen3 + 2507 + Gemma3 + Llama3.1 → eval windows with target dates ≥ **2025-08-01**
(2507 release date 2025-07-25 is the hard bound on their data). With Qwen3.5 → ≥ **2026-03-01**; with Qwen3.6 → ≥ **2026-05-01**.

---

## B. Time-series foundation models (PyTorch, pip-installable)

Summary table:

| model | HF id | pip (version, date) | quantiles | max ctx | covariates | license | data cutoff |
|---|---|---|---|---|---|---|---|
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | `timesfm[torch]` (3.0.1, 2026-09-02; also 2.0.2) | 10 channels = mean + 9 quantiles (0.1..0.9) via continuous quantile head | 16,384 (`max_context` you choose) | via XReg (`timesfm[xreg]`, sklearn ridge; not native) | Apache-2.0 | Wikimedia pageviews cutoff Nov 2023; Google Trends EoY 2022; GiftEvalPretrain |
| TimesFM 3.0 | `google/timesfm-3.0-pytorch` | `timesfm[torch]` 3.0.1 (`import timesfm3`) | 9 quantiles 0.1..0.9 | 15,360 (`_MAX_CONTEXT_LENGTH`) | native past-only & past+future | **timesfm-non-commercial-license-v1.0** | same as 2.5; "GiftEvalPretrain excluding the datasets that overlap with fev-bench" |
| Chronos-2 | `amazon/chronos-2` | `chronos-forecasting>=2.0` (2.3.1, 2026-07-02) | 21 quantiles {0.01,0.05,0.1,…,0.9,0.95,0.99}; `predict_quantiles` returns any subset | 8,192 | native past & future, numeric + categorical | Apache-2.0 | not stated (Chronos datasets + GIFT-Eval pretrain + synthetic) |
| TiRex | `NX-AI/TiRex` | `tirex-ts` (1.4.2, 2026-06-09) | 9 quantiles 0.1..0.9 (mean = median) | 2,016 default (`max_context`), model `train_ctx_len` (2048 per paper) | none (univariate only) | NXAI community license | not stated (GiftEvalPretrain + chronos_datasets) |
| TiRex-2 | `NX-AI/TiRex-2` | `tirex-2` (0.2.1, 2026-08-05; py>=3.11, torch>=2.8) | 9 quantiles | UNVERIFIED | native past & future | Apache-2.0 | not stated |
| Moirai-2 | `Salesforce/moirai-2.0-R-small` | `uni2ts` (2.0.0, 2025-11-04; pins torch<2.5, gluonts~=0.14.3, numpy~=1.26) | quantile loss model; gluonts predictor | example uses 1680 | `feat_dynamic_real_dim`, `past_feat_dynamic_real_dim` | cc-by-nc-4.0 | not stated (GIFT-Eval pretrain/train subsets, Chronos, KernelSynth, internal Salesforce) |

None of the TS models states a calendar cutoff except TimesFM (Wikimedia Nov 2023 / Trends EoY 2022). All were
trained on GIFT-Eval-Pretrain / Chronos corpora whose newest public series end ≈ 2023–2024; 2025+ targets are
out-of-training for all of them as far as documented. (Chronos-2 and Moirai-2 do not exclude the possibility of
newer data; treat as **UNVERIFIED** but very likely pre-2025.)

Environment advice: env A = `vllm==0.28.0` (needs `torch==2.13.0`, `transformers>=5.5.3`, py 3.10–3.14).
env B = `timesfm[torch]==3.0.1 chronos-forecasting==2.3.1 tirex-ts==1.4.2 fev==0.10.0 gluonts==0.17.0`
(all torch>=2.x compatible; `tirex-2` needs py>=3.11/torch>=2.8 and can live here too). env C (optional) = `uni2ts==2.0.0`.

### B.1 TimesFM 2.5 (`import timesfm`, package `timesfm==3.0.1` still ships the 2.5 module)

Model card (verbatim):

```python
import numpy as np
import timesfm
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=True)

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],  # Two dummy inputs
)
point_forecast.shape  # (2, 12)
quantile_forecast.shape  # (2, 12, 10): mean, then 10th to 90th quantiles.
```

`ForecastConfig` fields (read from `timesfm/configs.py`): `max_context: int = 0`, `max_horizon: int = 0`,
`normalize_inputs: bool = False`, `window_size: int = 0`, `per_core_batch_size: int = 1`,
`use_continuous_quantile_head: bool = False`, `force_flip_invariance: bool = True`, `infer_is_positive: bool = True`,
`fix_quantile_crossing: bool = False`, `return_backcast: bool = False`. Inputs shorter than `max_context` are
zero-padded, longer are truncated (docstring). README: "supports up to 16k context length"; "continuous quantile
forecast up to 1k horizon via an optional 30M quantile head". Inputs are a list of 1-D float arrays (ragged OK).
`quantile_forecast[..., 0]` is the mean; `[..., 1:10]` are quantiles 0.1..0.9. No text input. Covariates only via
the separate XReg helper (`timesfm/utils/xreg_lib.py`, extra `timesfm[xreg]` pulls `jax[cuda]` + scikit-learn) —
skip for this project.

### B.2 TimesFM 3.0 (`import timesfm3`; same wheel)

README (verbatim):

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

config = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=32, device="cuda")
forecaster = TimesFM3Evaluator(config)

ts1 = np.linspace(0, 1, 100).astype(np.float32)
ts2 = np.sin(np.linspace(0, 24, 72)).astype(np.float32)
outputs = list(forecaster.predict_batch([ts1, ts2], horizon=12, return_quantiles=True, use_symmetric_averaging=False))
print(outputs[0].forecast.shape)   # (12,)
print(outputs[0].quantiles.shape)  # (12, 9)
```

Covariates (README): pass `contexts=[target_2d]` with shape `(num_variates, context_len)`,
`past_only_covariates=[np.ndarray (k, context_len)]`, `past_future_covariates=[np.ndarray (m, context_len + horizon)]`.
`predict_batch` signature (read from `timesfm3/evaluator.py`):
`predict_batch(contexts, horizon, past_only_covariates=None, past_future_covariates=None, ts_ids=None,
return_quantiles=True, use_symmetric_averaging=True, make_positive=True, sort_quantiles=True, use_znorm=False,
padding_mode="none", univariate=False) -> Iterator[ForecastOutput]`; `ForecastOutput(ts_id, forecast, quantiles)`.
`ModelConfig` defaults: `quantiles=[0.1,...,0.9]`, `median_quantile_index=4`, `input_patch_length=32`,
`output_patch_length=64`, `cache_dir`, `local_files_only`, `token`, `revision` (HF download options).
`_MAX_CONTEXT_LENGTH = 15360`. License: non-commercial (research use OK; note it in the paper).

### B.3 Chronos-2 (`chronos-forecasting==2.3.1`)

Model card: `pip install "chronos-forecasting>=2.0"`. Deps: `torch>=2.2,<3`, `transformers>=4.41,<6`,
`accelerate`, `einops`, `pandas>=2`. 120M-param encoder-only; max context 8,192; max prediction length 1,024
(longer horizons auto-regress on quantiles).

```python
import torch, numpy as np
from chronos import Chronos2Pipeline

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")   # or torch_dtype=torch.bfloat16

# (a) plain univariate batch, ragged lengths OK (left-padded internally)
contexts = [np.asarray(x, dtype=np.float32) for x in list_of_1d_arrays]
quantiles, mean = pipe.predict_quantiles(
    contexts, prediction_length=H, quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)
# quantiles: list of tensors, each (n_variates=1, H, len(quantile_levels)); mean: list of (1, H)

# (b) with covariates: list of dicts sharing one schema
inputs = [{
    "target": y_hist,                                    # (T,) or (n_variates, T)
    "past_covariates": {"temp": x_hist, "dow": np.array([...], dtype=object)},   # each length T; str arrays = categorical
    "future_covariates": {"temp": x_fut, "dow": ...},    # each length H; keys subset of past_covariates
}]
quantiles, mean = pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
```

Signatures (read from `chronos/chronos2/pipeline.py`):
`predict(inputs, prediction_length=None, batch_size=256, context_length=None, cross_learning=False,
limit_prediction_length=False, **kwargs) -> list[Tensor (n_variates, n_quantiles=21, prediction_length)]`;
`predict_quantiles(inputs, prediction_length=None, quantile_levels=[0.1,...,0.9], **predict_kwargs)
-> (quantiles: list[Tensor (n_variates, H, Q)], mean: list[Tensor (n_variates, H)])`.
`context_length` > 8192 is reset to 8192 with a warning. Also `pipe.predict_df(context_df, future_df=..., prediction_length=..., quantile_levels=[...], id_column="id", timestamp_column="timestamp", target="target")` for pandas long format.

### B.4 TiRex (`tirex-ts==1.4.2`) — **no CUDA kernels required**

README quick start (verbatim):

```python
import torch
from tirex import load_model, ForecastModel

model: ForecastModel = load_model("NX-AI/TiRex")
data = torch.rand((5, 128))  # Sample Data (5 time series with length 128)
quantiles, mean = model.forecast(context=data, prediction_length=64)
```

`load_model(path, device=None, backend: Literal["torch","cuda"]="torch", compile=False, hf_kwargs=None,
ckp_kwargs=None)` (read from `tirex/base.py`). Downloads `model.ckpt` via `hf_hub_download(repo_id, filename="model.ckpt")`
(so `HF_HOME` / `HF_HUB_OFFLINE=1` apply). `device=None` → `"cuda:0"` if available else `"cpu"`. `backend="torch"`
is the pure-PyTorch sLSTM; `backend="cuda"` requires `pip install "tirex-ts[cuda]"` (`xlstm`, `ninja`) and
JIT-compiles kernels on first load (needs nvcc matching torch's CUDA, `TORCH_CUDA_ARCH_LIST="8.0;9.0"`, CC ≥ 8.0);
if `xlstm` is missing it silently falls back to torch with a warning. `TIREX_NO_CUDA=1` env var also forces torch.
`compile=True` applies `torch.compile` to the sLSTM cell (torch backend only) — use this on GPU for speed.
`model.forecast(context, output_type="torch"|"numpy"|"gluonts", batch_size=512, yield_per_batch=False,
resample_strategy=None, **predict_kwargs)`; `context` = 1-D/2-D tensor/ndarray or list of 1-D (ragged → padded per
batch). Returns `(quantiles [B, H, 9], mean [B, H])`; quantile levels 0.1..0.9; `mean` is the 0.5 quantile.
`max_context_length` defaults to 2016; contexts are truncated to the model's `train_ctx_len` and NaN-padded if shorter.
CPU works (slower). Extras: `tirex-ts[gluonts,hfdataset]`.

TiRex-2 (optional upgrade; `pip install tirex-2`, py ≥ 3.11, torch ≥ 2.8, deps `flashrnn>=1.0.5`, `xlstm~=2.0.3`):

```python
import torch
from tirex2 import TimeseriesType, load_model
model = load_model("NX-AI/TiRex-2", device="cpu")   # device="cuda" JIT-builds FlashRNN with nvcc on first forecast
ts = TimeseriesType(target=context.unsqueeze(0), past_covariates=None, future_covariates=None)  # target: (n_targets, T)
forecast = model.forecast([ts], prediction_length=32, output_type="numpy")[0]   # (n_targets, 9 quantiles, H)
```

README FAQ: CUDA needs "`nvcc` on PATH or reachable via CUDA_HOME", toolkit major matching torch's `cu12x`/`cu13x`,
CC ≥ 8.0 (A100/H100 OK). On Della: `module load cudatoolkit/12.x` before running with `device="cuda"`, or use CPU.

### B.5 Moirai-2 (optional; separate env)

`pip install uni2ts` (2.0.0). README (verbatim, trimmed):

```python
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
model = Moirai2Forecast(
    module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
    prediction_length=100, context_length=1680, target_dim=1,
    feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
)
predictor = model.create_predictor(batch_size=32)
forecasts = predictor.predict(test_data.input)     # gluonts TestData; yields gluonts Forecast objects (use .quantile(q))
```

11.4M params, cc-by-nc-4.0. Because of the `torch<2.5` / `gluonts~=0.14.3` / `numpy~=1.26` pins this cannot share an
env with chronos-2.3.1/timesfm-3.0.1. Recommend dropping unless a 4th TSFM is needed.

---

## C. Metrics: MASE and CRPS

### C.1 Definitions (match `fev` 0.10.0 and `gluonts` 0.17.0 exactly)

Notation: series i, horizon h = 1..H, truth y, point forecast ŷ (use the **median** for MASE), quantile forecast
q_α at levels α ∈ A, in-sample history y_past of length T, seasonality m. Conventions differ, so fix one and
state it: gluonts `get_seasonality` defaults (read from `gluonts/time_feature/seasonality.py`) are `H`=24, `D`=1,
`W`=1, `M`=12, `B`=5, `Q`=4; fev-bench / Chronos evaluations pass an explicit per-task `seasonality` (typically
daily=7, weekly=1, monthly=12, hourly=24). Recommend the fev convention (daily=7).

**MASE** (fev `MASE.compute`, gluonts `MASE`):

  seasonal_error_i = mean_{t=m+1..T} |y_past[t] − y_past[t−m]|      (per series; series with T ≤ m or error 0 are dropped/NaN)
  MASE = mean_{i,h} |y_{i,h} − ŷ_{i,h}| / seasonal_error_i

fev implementation: `scaled = np.abs(y_true - y_pred) / seasonal_error[:, None, :]` then `nanmean` over (N,H) then
mean over D. gluonts: `absolute_scaled_error = absolute_error / data["seasonal_error"]`.

**Quantile (pinball) loss** as used by both libraries — note the factor 2:

  QL_α(y, q) = 2 · |(y − q) · (1[y ≤ q] − α)|        (fev `_quantile_loss`; gluonts `quantile_loss` uses `(prediction >= label) - q`, identical)

**CRPS from quantiles** (a.k.a. weighted quantile loss / "mean_wQuantileLoss"; exact in the limit of dense quantiles):

  CRPS_Q(y) ≈ (1/|A|) Σ_{α∈A} QL_α(y, q_α) = (2/|A|) Σ_α pinball_α(y, q_α)

  (scoringrules `crps_quantile` docstring: "quantileCRPS = (2/|Q|) Σ_{q∈Q} PB_q", ref. Berrisch & Ziel 2023.)

Normalizations used on leaderboards:
- **WQL** (Chronos/fev-bench): per-dataset  Σ_{i,h} QL_α / Σ_{i,h} |y|, averaged over α (fev `WQL._per_quantile_level`:
  `nanmean(ql, axis=(0,1)) / nanmean(|y_true|)`). Scale-free; use A = {0.1,…,0.9}.
- **SQL** (scaled quantile loss, GIFT-Eval-style CRPS relative to seasonal naive): QL_α / seasonal_error_i (fev `SQL`).
- gluonts `MeanWeightedSumQuantileLoss` = mean over α of Σ QL_α / Σ |y|; gluonts `MeanScaledQuantileLoss` = SQL.

**CRPS from samples** (exact ensemble form; use when an LLM produces S sampled paths):

  CRPS(F, y) = E|X − y| − ½ E|X − X'|
  ≈ (1/S) Σ_s |x_s − y| − (1/(2S²)) Σ_s Σ_{s'} |x_s − x_{s'}|        ("nrg" estimator; biased-low for small S)
  fair estimator: replace 1/(2S²) by 1/(2S(S−1))  (scoringrules `estimator="fair"`).

  Equivalent closed form via sorted samples x_(1)≤…≤x_(S):  CRPS = (2/S²) Σ_{k=1}^{S} (x_(k) − y) · (S·1[y < x_(k)] − k + ½).

Either (i) compute this directly, or (ii) convert samples → quantiles with `np.quantile(samples, A, axis=0)` and use
the quantile formula (this is what gluonts `SampleForecast.quantile()` + `Evaluator` does; slightly different number).
For the paper, report **one** definition for everything: recommend WQL@{0.1..0.9} for LLMs *and* TSFMs (LLM
samples → quantiles), plus MASE on the median.

### C.2 Reference implementations (pip-installable, verified signatures)

```python
# fev==0.10.0  (pip install fev)  -- standalone metric classes; all arrays numpy
from fev.metrics import get_metric          # names: "MASE","WQL","SQL","MQL","MAE","RMSE","WAPE",...
m = get_metric("WQL")
score = m.compute(
    y_true=Y,            # [N, H, D]  D=1 for univariate
    y_pred=P,            # [N, H, D]  point (median) forecast
    y_past=Ypast,        # [sum(T_i), D]  concatenated histories (ragged)
    y_past_lengths=L,    # [N]
    q_pred=Q,            # [N, H, D, len(quantile_levels)]
    seasonality=m_season,
    quantile_levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
)
# get_metric("MASE").compute(...) same call. Items with undefined seasonal error are excluded (nanmean).
```

```python
# gluonts==0.17.0 -- if forecasts are gluonts Forecast objects (QuantileForecast / SampleForecast)
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss, MeanScaledQuantileLoss
df = evaluate_forecasts(forecasts, test_data=test_data,
                        metrics=[MASE(), MeanWeightedSumQuantileLoss(quantile_levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])],
                        seasonality=m_season)
```

```python
# scoringrules (0.11.0 needs Python>=3.12; 0.10.0 works on 3.10/3.11) -- CRPS from samples or quantiles
import scoringrules as sr
crps_s = sr.crps_ensemble(obs=y, fct=samples, m_axis=-1, estimator="fair")   # samples: (..., S)
crps_q = sr.crps_quantile(obs=y, fct=quantiles, alpha=np.array(levels), m_axis=-1)  # quantiles: (..., Q); NOTE: no factor 2 issue — returns (2/|Q|)Σ pinball
```

Self-contained numpy (no deps), matching fev conventions:

```python
import numpy as np
def quantile_loss(y, q, levels):            # y [N,H], q [N,H,Q]
    a = np.asarray(levels)[None, None, :]
    return 2 * np.abs((y[..., None] - q) * ((y[..., None] <= q) - a))       # [N,H,Q]
def wql(y, q, levels):                       # Chronos/fev WQL
    return float(np.nanmean(quantile_loss(y, q, levels), axis=(0, 1)).mean() / np.nanmean(np.abs(y)))
def mase(y, yhat, y_past_list, m):           # y_past_list: list of 1-D histories
    se = np.array([np.nanmean(np.abs(h[m:] - h[:-m])) if len(h) > m else np.nan for h in y_past_list])
    return float(np.nanmean(np.abs(y - yhat) / se[:, None]))
def crps_samples(y, samples):                # y [N,H], samples [N,H,S]; fair estimator
    S = samples.shape[-1]
    t1 = np.abs(samples - y[..., None]).mean(-1)
    t2 = np.abs(samples[..., :, None] - samples[..., None, :]).sum((-1, -2)) / (2 * S * (S - 1))
    return float(np.mean(t1 - t2))
```

---

## D. Princeton Della cluster facts (researchcomputing.princeton.edu, fetched 2026-09-06)

Hardware (Della page, verbatim where quoted):
- A100 80 GB: "There are 59 nodes with 4 GPUs per node. Each GPU has 80 GB of memory." Intel Ice Lake, 48 cores,
  1000 GB RAM/node. Request with `#SBATCH --constraint=gpu80`.
- A100 40 GB: "There are 120 GPUs with 40 GB of memory. To run a job using GPUs with 40 GB: #SBATCH --constraint=gpu40"
  (20 AMD EPYC Rome nodes × 2 GPUs, 128 cores, 768 GB).
- MIG slices (10 GB, ~1/7 A100): 10 nodes × 8; `--partition=mig` (use for smoke tests: `salloc --nodes=1 --ntasks=1 --time=60:00 --gres=gpu:1 --partition=mig`).
- H100 80 GB SXM: 42 nodes × 8 (Sapphire Rapids, 96 cores, 1 TB) — **PLI-only**: `--partition=pli-c` (PLI core members)
  or `--partition=pli --account=<ACCOUNT>` (needs a PLI allocation). H200 141 GB: 18 nodes × 8, `--partition=ailab` (AI Lab only).
  GH200: `--partition=grace` (experimental, ARM).
- CPU constraint flags: `--constraint=amd` / `--constraint=intel`.
- OS: RHEL 9. Login node for GPU work: `della-gpu.princeton.edu`. Visualization nodes `della-vis1` (A100-40GB) /
  `della-vis2` (4×P100): "Both nodes have internet access."

GPU QOS (auto-assigned by requested time):

| QOS | time limit | jobs/user | nodes/user | GPUs/user |
|---|---|---|---|---|
| gpu-test | 61 min | 2 | no limit | no limit |
| gpu-short | 24 h | 30 | 30 | 35 |
| gpu-medium | 72 h | 24 | 24 | 24 |
| gpu-long | 144 h (6 days) | 7 | 16 | 16 |

Max wall-time = 144 h. Plan jobs ≤ 24 h to land in gpu-short (best throughput).

Modules / env: `module load anaconda3/2026.7` (current version on the PyTorch KB page; the della-inference repo
used `anaconda3/2024.10`). CUDA: `cudatoolkit/12.x` (KB says 12.x or 13.2 preferred). pip-installed torch wheels bundle
their own CUDA runtime, so `cudatoolkit` is only needed for JIT kernels (TiRex CUDA backend, flash-attn builds, etc.).
Princeton's PyTorch recipe: `conda create --name torch-env "pytorch==2.9*=cuda12*" torchvision -c conda-forge -y` or
`pip3 install torch torchvision` inside a conda env. vLLM: create a py3.12 conda env and `pip install vllm==0.28.0`
(pulls `torch==2.13.0`); do this on `della-gpu` (login node has internet).

Internet: "The compute nodes do not have internet access so we must obtain the data while on the head node."
HF KB page: "You must download models and datasets on the login node before submitting jobs to Slurm since the
compute nodes, where the jobs run, do not have internet access."

HF cache (HF KB page, verbatim for Della): `export HF_HOME=/scratch/gpfs/<YourNetID>/.cache/huggingface/`
(put in `~/.bashrc`). Download on login node: `hf download Qwen/Qwen3-8B` (new CLI name; `huggingface-cli download`
also works) or `--local-dir /scratch/gpfs/$USER/models/Qwen3-8B`. In jobs set `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1` (and `VLLM_...` nothing extra; vLLM honours HF_HUB_OFFLINE) so nothing tries the network.
`/home` is small (quota); `/scratch/gpfs/$USER` is the large, fast, **not backed up** filesystem for caches, envs
(`conda create --prefix /scratch/gpfs/$USER/envs/...` is allowed) and outputs.

Apptainer: available on Della without a module (KB page); `apptainer pull docker://...` on login node,
`apptainer exec --nv ./image.sif python3 script.py` in jobs with `--gres=gpu:1`; set
`export APPTAINER_CACHEDIR=/scratch/gpfs/<SPONSOR>/$USER/APPTAINER_CACHE`. Official vLLM docker image
(`docker://vllm/vllm-openai:v0.28.0`) can be pulled this way if pip install is painful.

Template sbatch (assembled from the Princeton PyTorch example + Della directives):

```bash
#!/bin/bash
#SBATCH --job-name=qwen3-budget
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80        # A100 80GB; omit for 40GB; use --partition=pli-c/--partition=pli --account=... for H100
#SBATCH --time=23:59:00           # <= 24h -> gpu-short QOS
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=<NetID>@princeton.edu

module purge
module load anaconda3/2026.7
conda activate /scratch/gpfs/$USER/envs/vllm028
export HF_HOME=/scratch/gpfs/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python run_budget_eval.py --model Qwen/Qwen3-8B --budgets 0,256,512,1024,2048,4096
```

Memory sizing on A100-80GB: Qwen3-8B bf16 (~16 GB) + KV for 40,960 ctx fits easily; Qwen3-30B-A3B bf16 (~60 GB) fits
with `--max-model-len 16384 --gpu-memory-utilization 0.95` or use `Qwen/Qwen3-30B-A3B-FP8` (A100 has no FP8 tensor
cores; vLLM will use weight-only FP8→bf16 dequant, slower but fits). Qwen3.5-9B bf16 ≈ 18 GB.

---

## Sources

- Qwen3 docs quickstart (thinking budget, vLLM/SGLang examples): https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md and https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html
- Qwen3 vLLM deployment doc: https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md
- Model cards: https://huggingface.co/Qwen/Qwen3-8B , https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507 , https://huggingface.co/Qwen/Qwen3.5-4B , https://huggingface.co/Qwen/Qwen3.5-9B , https://huggingface.co/Qwen/Qwen3.6-35B-A3B , https://huggingface.co/google/gemma-3-4b-it , https://ai.google.dev/gemma/docs/core/model_card_3 , https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- HF API listings used for existence checks: https://huggingface.co/api/models?author=Qwen&search=Qwen3.5 , https://huggingface.co/api/models?author=Qwen&search=Qwen3-
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- Cutoff aggregators (third-party, unverified): https://metehan.ai/articles/llm-knowledge-cutoff-dates/ , https://github.com/QwenLM/Qwen3/discussions/1093
- vLLM reasoning outputs doc (thinking budget, default chat template kwargs, `reasoning` field): https://docs.vllm.ai/en/latest/features/reasoning_outputs.html (raw: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/reasoning_outputs.md ; same section present in tags v0.19.0…v0.28.0)
- vLLM 0.28.0 sources read: `vllm/sampling_params.py`, `vllm/config/reasoning.py`, `vllm/config/vllm.py`, `vllm/config/multimodal.py`, `vllm/engine/arg_utils.py`, `vllm/v1/sample/thinking_budget_state.py`, `vllm/v1/worker/gpu/sample/thinking_budget.py`, `vllm/entrypoints/llm.py`, `docs/models/supported_models.md` (sdist https://pypi.org/project/vllm/0.28.0/)
- vLLM thinking-budget PR/issues: https://github.com/vllm-project/vllm/pull/37112 , https://github.com/vllm-project/vllm/issues/44676 , https://www.neoteric.no/blog/your-vllm-thinking-budget-was-doing-nothing-with-mtp-on/ , https://yankee.dev/hands-on-vllm-thinking-token-budget
- PyPI metadata: https://pypi.org/pypi/vllm/json , /timesfm/ , /chronos-forecasting/ , /tirex-ts/ , /tirex-2/ , /uni2ts/ , /gluonts/ , /fev/ , /scoringrules/
- TimesFM: https://github.com/google-research/timesfm (README) , https://huggingface.co/google/timesfm-2.5-200m-pytorch , https://huggingface.co/google/timesfm-3.0-pytorch ; wheel `timesfm-3.0.1` inspected (`timesfm/configs.py`, `timesfm3/timesfm3_forecaster.py`, `timesfm3/evaluator.py`)
- Chronos-2: https://huggingface.co/amazon/chronos-2 , https://arxiv.org/abs/2510.15821 ; wheel `chronos_forecasting-2.3.1` inspected (`chronos/chronos2/pipeline.py`)
- TiRex: https://github.com/NX-AI/tirex (README) , https://huggingface.co/NX-AI/TiRex ; wheel `tirex_ts-1.4.2` inspected (`tirex/base.py`, `tirex/models/tirex.py`, `tirex/api_adapter/forecast.py`). TiRex-2: https://github.com/NX-AI/tirex-2 , https://arxiv.org/abs/2607.01204
- Moirai-2: https://huggingface.co/Salesforce/moirai-2.0-R-small , https://github.com/SalesforceAIResearch/uni2ts (README, `src/uni2ts/model/moirai2/__init__.py`)
- Metrics: wheel `fev-0.10.0` (`fev/metrics.py`), wheel `gluonts-0.17.0` (`gluonts/ev/stats.py`, `gluonts/ev/metrics.py`, `gluonts/model/evaluation.py`), wheel `scoringrules-0.10.0` (`scoringrules/_crps.py`)
- Della: https://researchcomputing.princeton.edu/systems/della , https://researchcomputing.princeton.edu/support/knowledge-base/pytorch , https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face , https://researchcomputing.princeton.edu/support/knowledge-base/apptainer , https://github.com/benediktstroebl/della-inference
