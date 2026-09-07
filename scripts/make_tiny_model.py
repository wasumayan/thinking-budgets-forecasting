#!/usr/bin/env python
"""Build an OFFLINE stand-in for Qwen/Qwen3-0.6B so `make smoke` can run where huggingface.co is unreachable.

NOT Qwen3-0.6B and never used for any paper number. It is a randomly initialised 2-layer Qwen3ForCausalLM
(same architecture class, same special-token ids 151643..151668, same chat-template semantics incl.
`enable_thinking`) with a character-level tokenizer, briefly trained on CPU to answer any chat prompt with
    <think>\nOkay.\n</think>\n\n{"forecast": [101, 102, ..., 112]}<|im_end|>          (thinking on)
    {"forecast": [101, 102, ..., 112]}<|im_end|>                                     (thinking off)
so the two-pass budget forcing, parsing, summary and figure code paths get real (if trivial) outputs.

Usage: python scripts/make_tiny_model.py --out data/tiny-qwen3-standin [--steps 400]
Then:  TBF_MODEL_PATH=data/tiny-qwen3-standin make smoke
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

IM_START, IM_END, THINK, THINK_END, EOT = 151644, 151645, 151667, 151668, 151643
VOCAB_SIZE = 151936
ANSWER = '{"forecast": [' + ", ".join(str(101 + i) for i in range(12)) + "]}"
THINKING = "Okay."

CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)


def build_tokenizer(out: pathlib.Path):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"] + list("–—’‘“”€£¥°±×÷…•·éèêàáâäçñöüßøåæ")
    vocab = {"<unk>": 0}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    for i in range(len(vocab), EOT):
        vocab[f"<|unused_{i}|>"] = i
    vocab["<|endoftext|>"] = EOT
    vocab["<|im_start|>"] = IM_START
    vocab["<|im_end|>"] = IM_END
    for i in range(IM_END + 1, THINK):
        vocab[f"<|unused_{i}|>"] = i
    vocab["<think>"] = THINK
    vocab["</think>"] = THINK_END
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    from tokenizers import AddedToken
    tok.add_special_tokens([AddedToken(t, special=True) for t in ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>")])
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<|im_end|>",
                                     pad_token="<|endoftext|>", additional_special_tokens=["<|im_start|>", "<think>", "</think>"])
    hf_tok.chat_template = CHAT_TEMPLATE
    hf_tok.save_pretrained(out)
    return hf_tok


def build_model(out: pathlib.Path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB_SIZE, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, head_dim=32, max_position_embeddings=8192,
                      tie_word_embeddings=True, bos_token_id=None, eos_token_id=IM_END, pad_token_id=EOT,
                      rope_theta=1000000.0)
    return Qwen3ForCausalLM(cfg)


def training_examples(tok, n_random: int = 64, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """(prompt_ids, target_ids) pairs: real fixture prompts (both modes) + random-junk prompts."""
    from tbf.data.fixtures import load_windows
    from tbf.prompts import render_messages

    rng = random.Random(seed)
    on_target = tok(f"<think>\n{THINKING}\n</think>\n\n{ANSWER}", add_special_tokens=False).input_ids + [IM_END]
    off_target = tok(ANSWER, add_special_tokens=False).input_ids + [IM_END]
    ex = []
    for w in load_windows():
        for setup in ("direct", "reviser"):
            prior = w["context"][-12:] if setup == "reviser" else None
            msgs = render_messages(w, setup, "full", prior_forecast=prior)
            for thinking, tgt in ((True, on_target), (False, off_target)):
                p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
                ex.append((tok(p, add_special_tokens=False).input_ids, tgt))
    alphabet = [chr(i) for i in range(32, 127)] + ["\n"]
    for _ in range(n_random):
        junk = "".join(rng.choice(alphabet) for _ in range(rng.randint(20, 1500)))
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": junk}]
        for thinking, tgt in ((True, on_target), (False, off_target)):
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            ex.append((tok(p, add_special_tokens=False).input_ids, tgt))
    # continuation examples for pass 2 after a forced early stop: "<think>\n" + junk thinking + EARLY_STOP -> answer
    from tbf.llm_vllm import EARLY_STOP
    early = tok(EARLY_STOP, add_special_tokens=False).input_ids
    for _ in range(n_random // 2):
        junk = "".join(rng.choice(alphabet) for _ in range(rng.randint(20, 400)))
        think_junk = "".join(rng.choice(alphabet[:-1]) for _ in range(rng.randint(64, 120)))
        msgs = [{"role": "user", "content": junk}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok(p, add_special_tokens=False).input_ids + tok("<think>\n" + think_junk, add_special_tokens=False).input_ids + early
        ex.append((ids, off_target))
    return ex


def train(model, examples, steps: int, lr: float = 3e-3, batch: int = 4, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    for step in range(steps):
        items = rng.sample(examples, batch)
        L = max(len(p) + len(t) for p, t in items)
        ids = torch.full((batch, L), EOT, dtype=torch.long)
        labels = torch.full((batch, L), -100, dtype=torch.long)
        attn = torch.zeros((batch, L), dtype=torch.long)
        for i, (p, t) in enumerate(items):
            seq = p + t
            ids[i, :len(seq)] = torch.tensor(seq)
            attn[i, :len(seq)] = 1
            labels[i, len(p):len(seq)] = torch.tensor(t)
        # logits only at target positions: full-vocab logits over the whole 3k-token sequence would need ~7 GB
        hidden = model.model(input_ids=ids, attention_mask=attn).last_hidden_state  # (B, L, H)
        pos = labels[:, 1:] != -100                                                  # predict token t from t-1
        logits = model.lm_head(hidden[:, :-1][pos])
        loss = torch.nn.functional.cross_entropy(logits.float(), labels[:, 1:][pos])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if step % 25 == 0 or step == steps - 1:
            print(f"step {step:4d} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()


def check(model, tok):
    import torch

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "Forecast please. " * 40}]
    for thinking in (True, False):
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        ids = torch.tensor([tok(p, add_special_tokens=False).input_ids])
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=120, do_sample=False,
                                 pad_token_id=EOT)
        gen = out[0][ids.shape[1]:].tolist()
        print(f"thinking={thinking}: {tok.decode(gen, skip_special_tokens=False)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tiny-qwen3-standin")
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(out)
    assert tok.convert_tokens_to_ids("<|im_end|>") == IM_END and tok.convert_tokens_to_ids("</think>") == THINK_END
    model = build_model(out)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    examples = training_examples(tok)
    print(f"{len(examples)} training examples")
    train(model, examples, a.steps)
    model.save_pretrained(out)
    (out / "STANDIN.json").write_text(json.dumps({"note": "offline stand-in for Qwen/Qwen3-0.6B; smoke tests only",
                                                  "steps": a.steps, "answer": ANSWER}, indent=1))
    check(model, tok)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
