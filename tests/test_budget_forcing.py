"""Unit test for the two-pass budget forcing with a fake tokenizer and a fake LLM (no vLLM needed).
Covers: (a) finished within budget, (b) </think> emitted naturally before the budget, (c) forced early stop."""
import sys
import types

import pytest

from tbf import llm_vllm as L


class FakeTok:
    """Token id = ord(char) for ordinary chars; special ids for markers. Decoding drops special ids."""
    SPECIAL = {L.IM_END: "<|im_end|>", L.THINK_END: "</think>"}

    def __call__(self, text, add_special_tokens=False, **_):
        ids = []
        i = 0
        while i < len(text):
            if text.startswith("</think>", i):
                ids.append(L.THINK_END); i += len("</think>")
            else:
                ids.append(ord(text[i])); i += 1
        return types.SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(t) for t in ids if t not in self.SPECIAL)


class FakeOut:
    def __init__(self, prompt_ids, gen_ids):
        self.prompt_token_ids = list(prompt_ids)
        self.outputs = [types.SimpleNamespace(token_ids=list(gen_ids))]


class FakeLLM:
    """Scripted: pass-1 responses keyed by prompt; pass-2 always answers '{"forecast": [...]}' then IM_END."""
    def __init__(self, tok, pass1):
        self.tok, self.pass1, self.calls = tok, pass1, []

    def generate(self, prompts, sps):
        self.calls.append([sp.max_tokens for sp in sps])
        outs = []
        for p, sp in zip(prompts, sps):
            ids = list(p["prompt_token_ids"]) if isinstance(p, dict) else list(p.prompt_token_ids)
            key = self.tok.decode(ids[:5])
            if len(self.calls) == 1:  # pass 1
                gen = self.tok(self.pass1[key]).input_ids[: sp.max_tokens]
                if self.pass1[key].endswith("<END>"):
                    gen = self.tok(self.pass1[key][:-5]).input_ids[: sp.max_tokens] + [L.IM_END]
                outs.append(FakeOut(ids, gen))
            else:  # pass 2
                outs.append(FakeOut(ids, self.tok('{"forecast": [1]}').input_ids + [L.IM_END]))
        return outs


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, max_tokens=None, seed=None, **kw):
            self.max_tokens, self.seed, self.kw = max_tokens, seed, kw

    def TokensPrompt(prompt_token_ids):
        return {"prompt_token_ids": prompt_token_ids}

    mod.SamplingParams, mod.TokensPrompt = SamplingParams, TokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod


def test_three_branches(fake_vllm):
    tok = FakeTok()
    budget = 40
    prompts = {"AAAAA": "think</think>{\"forecast\": [7]}<END>",   # (a) finished within budget (23 tokens)
               "BBBBB": "short</think>",                             # (b) natural end, answer needs pass 2
               "CCCCC": "very long thinking that goes on and on and exceeds the forty-token budget"}  # (c) forced
    llm = FakeLLM(tok, prompts)
    prompt_ids = [tok(k + " prompt").input_ids for k in prompts]
    gens = L.two_pass_budgeted(llm, tok, prompt_ids, budget=budget, answer_max_tokens=50,
                               sampling={"temperature": 0.6}, seeds=[0, 1, 2])
    a, b, c = gens
    assert a.finished and not a.budget_hit and a.answer_text == '{"forecast": [7]}' and a.thinking_text == "think"
    assert b.finished and not b.budget_hit and b.answer_text == '{"forecast": [1]}' and b.thinking_text == "short"
    assert c.finished and c.budget_hit and c.answer_text == '{"forecast": [1]}'
    assert c.thinking_tokens_used == budget  # exactly the budget; the injected early-stop string is excluded
    assert llm.calls[0] == [budget] * 3 and llm.calls[1] == [50, 50]


def test_split_think_no_marker():
    tok = FakeTok()
    thinking, answer, k = L.split_think(tok("just an answer").input_ids, tok)
    assert thinking == "" and answer == "just an answer" and k == 0
