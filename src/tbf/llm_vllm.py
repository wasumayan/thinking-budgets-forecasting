"""vLLM backend with Qwen's two-pass thinking-budget forcing (SPEC §3.3; docs/verify-models.md §A.3).

Contract:
  VLLMBackend(model_hf, max_model_len, seed, gpu_memory_utilization=0.9, language_model_only=False)
  .generate(list_of_messages, thinking: bool, budget: int, n_samples: int, sampling: dict,
            answer_max_tokens: int, seed: int) -> list[list[Generation]]   # [prompt][sample]

Generation fields: thinking_text, answer_text, thinking_tokens_used, answer_tokens, budget_hit,
finished, prompt_tokens. thinking=False => hard switch enable_thinking=False, budget must be 0.
Sampling seeds: per-sample seed = seed * 1000 + sample_index so n_samples>1 gives distinct draws.

The core two-pass function is implemented below and unit-tested with a fake LLM in
tests/test_budget_forcing.py. vLLM is imported lazily so CPU machines can import this module.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Sequence

IM_END = 151645      # <|im_end|>
THINK_END = 151668   # </think>
EARLY_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"


@dataclasses.dataclass
class Generation:
    thinking_text: str
    answer_text: str
    thinking_tokens_used: int
    answer_tokens: int
    budget_hit: bool
    finished: bool
    prompt_tokens: int


def split_think(ids: Sequence[int], tok) -> tuple[str, str, int]:
    """Split generated ids at the LAST </think>. Returns (thinking_text, answer_text, n_thinking_ids)."""
    ids = list(ids)
    try:
        k = len(ids) - ids[::-1].index(THINK_END)
    except ValueError:
        k = 0
    thinking = tok.decode(ids[:k], skip_special_tokens=True).strip("\n")
    answer = tok.decode(ids[k:], skip_special_tokens=True).strip("\n")
    return thinking, answer, k


def two_pass_budgeted(llm, tok, prompt_ids: list[list[int]], budget: int, answer_max_tokens: int,
                      sampling: dict[str, Any], seeds: list[int]) -> list[Generation]:
    """Qwen's official recipe, batched.

    pass 1: think up to `budget` tokens. For each output:
      - contains IM_END  -> finished within budget (answer already produced).
      - else: continuation = prompt + generated (+ EARLY_STOP ids if THINK_END absent); pass 2 with answer_max_tokens.
    `llm.generate(list_of_TokensPrompt, SamplingParams)` and `SamplingParams(..., max_tokens, seed)` are the only
    vLLM APIs used, so a fake with the same shape can be injected in tests.
    """
    from vllm import SamplingParams, TokensPrompt  # lazy

    early_ids = tok(EARLY_STOP, add_special_tokens=False).input_ids
    sp1 = [SamplingParams(**sampling, max_tokens=budget, seed=s) for s in seeds]
    out1 = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp1)

    results: list[list[int] | None] = [None] * len(prompt_ids)
    budget_hit = [False] * len(prompt_ids)
    pass2_inputs, pass2_sp, pass2_idx = [], [], []
    for i, o in enumerate(out1):
        gen = list(o.outputs[0].token_ids)
        if IM_END in gen:
            results[i] = gen
            continue
        forced = THINK_END not in gen
        budget_hit[i] = forced
        cont = prompt_ids[i] + gen + (early_ids if forced else [])
        pass2_inputs.append(TokensPrompt(prompt_token_ids=cont))
        pass2_sp.append(SamplingParams(**sampling, max_tokens=answer_max_tokens, seed=seeds[i]))
        pass2_idx.append((i, len(prompt_ids[i])))
    if pass2_inputs:
        out2 = llm.generate(pass2_inputs, pass2_sp)
        for (i, plen), o in zip(pass2_idx, out2):
            results[i] = list(o.prompt_token_ids[plen:]) + list(o.outputs[0].token_ids)

    # tokens of the injected string up to and including </think> are not the model's own thinking
    early_prefix_len = early_ids.index(THINK_END) + 1 if THINK_END in early_ids else len(early_ids)
    gens = []
    for i, ids in enumerate(results):
        thinking, answer, k = split_think(ids, tok)
        n_think = k
        if budget_hit[i]:
            n_think = max(k - early_prefix_len, 0)
            thinking = tok.decode(ids[:n_think], skip_special_tokens=True).strip("\n")
        gens.append(Generation(thinking, answer, n_think, len(ids) - k, budget_hit[i],
                               IM_END in ids, len(prompt_ids[i])))
    return gens


class VLLMBackend:
    def __init__(self, model_hf: str, max_model_len: int, seed: int = 0, gpu_memory_utilization: float = 0.9,
                 language_model_only: bool = False):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tok = AutoTokenizer.from_pretrained(model_hf)
        kwargs = dict(model=model_hf, max_model_len=max_model_len, seed=seed,
                      gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True)
        if language_model_only:
            kwargs["language_model_only"] = True
        self.llm = LLM(**kwargs)

    def generate(self, list_of_messages, thinking: bool, budget: int, n_samples: int, sampling: dict,
                 answer_max_tokens: int, seed: int) -> list[list[Generation]]:
        """See module docstring. Implement: render prompts with apply_chat_template(enable_thinking=thinking),
        tokenize, expand each prompt n_samples times with distinct seeds, then either
        (a) thinking=False: one llm.generate with max_tokens=answer_max_tokens, or
        (b) thinking=True: two_pass_budgeted(...). Regroup [prompt][sample]."""
        raise NotImplementedError
