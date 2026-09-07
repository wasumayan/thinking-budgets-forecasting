"""transformers backend for CPU smoke tests only (Qwen/Qwen3-0.6B, budget <= 64, <= 4 windows).

Same .generate(...) contract and Generation dataclass as llm_vllm.VLLMBackend, implemented with
model.generate following Qwen's HF reference (docs/verify-models.md §A.3, verbatim logic):
  pass 1 max_new_tokens=budget; if IM_END absent: append EARLY_STOP ids if THINK_END absent; pass 2.
Not used for any paper number.
"""
from __future__ import annotations

from .llm_vllm import EARLY_STOP, IM_END, THINK_END, Generation, split_think  # noqa: F401


class HFBackend:
    def __init__(self, model_hf: str, max_model_len: int = 8192, seed: int = 0, **_):
        raise NotImplementedError

    def generate(self, list_of_messages, thinking: bool, budget: int, n_samples: int, sampling: dict,
                 answer_max_tokens: int, seed: int) -> list[list[Generation]]:
        raise NotImplementedError
