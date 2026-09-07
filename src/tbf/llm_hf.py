"""transformers backend for CPU smoke tests only (Qwen/Qwen3-0.6B, budget <= 64, <= 4 windows).

Same .generate(...) contract and Generation dataclass as llm_vllm.VLLMBackend, implemented with
model.generate following Qwen's HF reference (docs/verify-models.md §A.3, verbatim logic):
  pass 1 max_new_tokens=budget; if IM_END absent: append EARLY_STOP ids if THINK_END absent; pass 2.
Not used for any paper number.
"""
from __future__ import annotations

import logging

from .llm_vllm import (EARLY_STOP, IM_END, THINK_END, Generation, early_stop_prefix_len,  # noqa: F401
                       render_prompt, sample_seeds, split_think)

log = logging.getLogger(__name__)


class HFBackend:
    def __init__(self, model_hf: str, max_model_len: int = 8192, seed: int = 0, hybrid: bool = True,
                 thinking_only: bool = False, device: str | None = None, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_hf = model_hf
        self.max_model_len = max_model_len
        self.hybrid, self.thinking_only = hybrid, thinking_only
        self.tok = AutoTokenizer.from_pretrained(model_hf)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_hf, dtype=dtype).to(self.device).eval()
        self.early_ids = self.tok(EARLY_STOP, add_special_tokens=False).input_ids
        torch.manual_seed(seed)

    # -- helpers ---------------------------------------------------------------------------------------
    def _gen_kwargs(self, sampling: dict, max_new_tokens: int) -> dict:
        kw = dict(do_sample=True, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        for k in ("temperature", "top_p", "top_k", "min_p"):
            if k in sampling:
                kw[k] = sampling[k]
        if kw.get("temperature", 1.0) == 0:
            kw["do_sample"] = False
        return kw

    def _generate_ids(self, input_ids: list[int], sampling: dict, max_new_tokens: int, seed: int) -> list[int]:
        import torch

        torch.manual_seed(seed)
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(ids)
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=attn, **self._gen_kwargs(sampling, max_new_tokens))
        return out[0][len(input_ids):].tolist()

    def _one(self, prompt_ids: list[int], thinking: bool, budget: int, sampling: dict, answer_max_tokens: int,
             seed: int) -> Generation:
        if not thinking:
            gen = self._generate_ids(prompt_ids, sampling, answer_max_tokens, seed)
            _, answer, _ = split_think(gen, self.tok)
            return Generation("", answer, 0, len(gen), False, IM_END in gen, len(prompt_ids))

        # pass 1: think up to `budget` tokens (Qwen reference logic)
        gen = self._generate_ids(prompt_ids, sampling, budget, seed)
        budget_hit = False
        if IM_END not in gen:
            if THINK_END not in gen:
                budget_hit = True
                cont = prompt_ids + gen + self.early_ids
            else:
                cont = prompt_ids + gen
            gen = cont[len(prompt_ids):] + self._generate_ids(cont, sampling, answer_max_tokens, seed)
        thinking_text, answer, k = split_think(gen, self.tok)
        n_think = k
        if budget_hit:
            n_think = max(k - early_stop_prefix_len(self.early_ids), 0)
            thinking_text = self.tok.decode(gen[:n_think], skip_special_tokens=True).strip("\n")
        return Generation(thinking_text, answer, n_think, len(gen) - k, budget_hit, IM_END in gen, len(prompt_ids))

    # -- public contract -------------------------------------------------------------------------------
    def generate(self, list_of_messages, thinking: bool, budget: int, n_samples: int, sampling: dict,
                 answer_max_tokens: int, seed: int) -> list[list[Generation]]:
        if not thinking and budget != 0:
            raise ValueError("thinking=False requires budget=0 (hard switch)")
        if thinking and budget <= 0:
            raise ValueError("thinking=True requires budget > 0")
        results: list[list[Generation]] = []
        for msgs in list_of_messages:
            prompt = render_prompt(self.tok, msgs, thinking, self.hybrid, self.thinking_only)
            prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
            if len(prompt_ids) + max(budget, 0) + answer_max_tokens > self.max_model_len:
                log.warning("prompt (%d tokens) + budget + answer exceeds max_model_len=%d", len(prompt_ids), self.max_model_len)
            gens = [self._one(prompt_ids, thinking, budget, sampling, answer_max_tokens, s)
                    for s in sample_seeds(seed, n_samples)]
            results.append(gens)
        return results
