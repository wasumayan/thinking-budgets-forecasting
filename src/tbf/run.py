"""Run one LLM config -> results/<config_id>.jsonl (SPEC §5, §7.2). Resumable, batched.

CLI (either form):
  python -m tbf.run --config-id freshts26__qwen3-8b__direct__b2048__full__s0 [--limit N] [--backend vllm|hf] [--out PATH]
  python -m tbf.run --dataset fixtures --model qwen3-0.6b --setup direct --think off|--budget 64 --context-mode full --n-samples 1 --seed 0 --backend hf --limit 4 --out PATH

Steps:
 1. cfg = RunConfig.parse(config_id) (or build from flags). Load grid.yaml for model spec + sampling.
 2. Load windows: data/freshts26/windows.parquet | data/fixtures/windows.jsonl | CiK via tbf.data.cik.
    Apply --limit (first N by window_id sort, stable) — used for the go/no-go check.
 3. If setup == reviser: priors = tsfm.load_prior(cfg.prior, cfg.dataset). Shuffled mode: load shuffle_map.
 4. Skip window_ids already present in the output file (resume).
 5. For batches of 64 windows: render messages (prompts.render_messages), backend.generate(...), parse
    each sample with parse.parse_forecast; point = single sample values, or per-horizon median across
    valid samples when n_samples > 1 (samples stored too); invalid -> impute (reviser: prior; direct:
    seasonal naive from the context) and valid=false. Append rows; flush per batch.
 6. Final one-line summary log: n, valid_rate, mean_thinking_tokens, budget_hit_rate, wall.
Backend selection: vllm (default) or hf (CPU smoke). thinking_only models (Thinking-2507): thinking=True is
implied and the template pre-fills <think>; thinking=False for them is an error. Non-hybrid non-thinking
models (Instruct-2507, gemma3, llama31): thinking must be False; do not pass enable_thinking to their templates.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import numpy as np

from .config import RunConfig, load_grid
from .data.windows import append_results, done_window_ids, load_shuffle_map, load_windows
from .metrics import seasonal_naive_forecast
from .parse import parse_forecast
from .prompts import render_messages
from .tsfm import load_prior

log = logging.getLogger(__name__)
BATCH_SIZE = {"vllm": 256, "hf": 4}


def evenly_spaced(items: list, n: int) -> list:
    if n is None or n >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, num=n).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx.tolist())]
RAW_MAX = 2000


def build_backend(cfg: RunConfig, backend: str, grid: dict, model_path: str | None = None):
    spec = cfg.model_spec(grid)
    hf = model_path or spec["hf"]
    kw = dict(model_hf=hf, max_model_len=int(spec.get("max_model_len", 40960)), seed=cfg.seed,
              hybrid=bool(spec.get("hybrid", False)), thinking_only=bool(spec.get("thinking_only", False)))
    if backend == "hf":
        from .llm_hf import HFBackend
        return HFBackend(**kw)
    if backend == "vllm":
        from .llm_vllm import VLLMBackend
        if "gpu_mem_util" in spec:
            kw["gpu_memory_utilization"] = float(spec["gpu_mem_util"])
        if spec.get("family") in ("gemma3",) or spec.get("language_model_only"):
            kw["language_model_only"] = True
        return VLLMBackend(**kw)
    raise ValueError(f"unknown backend {backend!r}")


def _check_model_mode(cfg: RunConfig, grid: dict) -> None:
    spec = cfg.model_spec(grid)
    if spec.get("thinking_only") and not cfg.thinking:
        raise ValueError(f"{cfg.model} is thinking-only; thinking=False is not allowed")
    if not spec.get("hybrid") and not spec.get("thinking_only") and cfg.thinking:
        raise ValueError(f"{cfg.model} is a non-thinking model; thinking=True is not allowed")


def _truncate(s: str | None) -> str:
    s = s or ""
    return s if len(s) <= RAW_MAX else s[:RAW_MAX]


def make_row(cfg: RunConfig, w: dict, gens, fallback: list[float], wall_s: float) -> dict:
    """Combine the n_samples generations for one window into one results row."""
    H = len(w["target"])
    parsed = [parse_forecast(g.answer_text, h=H) for g in gens]
    valid_vals = [p.values for p in parsed if p.valid]
    n_valid = len(valid_vals)
    if n_valid == 0:
        point, valid = [float(x) for x in fallback], False
    elif len(gens) == 1:
        point, valid = valid_vals[0], True
    else:
        point, valid = np.median(np.asarray(valid_vals, float), axis=0).tolist(), True
    return {
        "config_id": cfg.config_id, "window_id": w["window_id"], "dataset": cfg.dataset, "model": cfg.model,
        "setup": cfg.setup, "thinking": cfg.thinking, "budget": cfg.budget, "context_mode": cfg.context_mode,
        "n_samples": cfg.n_samples, "seed": cfg.seed, "prior": cfg.prior if cfg.setup == "reviser" else None,
        "point": [float(x) for x in point],
        "samples": valid_vals if len(gens) > 1 else None,
        "n_valid_samples": n_valid,
        "valid": valid,
        "n_values_mismatch": any(p.n_values_mismatch for p in parsed if p.valid),
        "parse_reason": ";".join(sorted({p.reason for p in parsed if p.reason})),
        "raw_answer": _truncate(gens[0].answer_text),
        "raw_thinking": _truncate(gens[0].thinking_text),
        "thinking_tokens_used": float(np.mean([g.thinking_tokens_used for g in gens])),
        "answer_tokens": float(np.mean([g.answer_tokens for g in gens])),
        "budget_hit": any(g.budget_hit for g in gens),
        "finished": all(g.finished for g in gens),
        "wall_s": wall_s,
        "prompt_tokens": int(gens[0].prompt_tokens),
    }


def run(cfg: RunConfig, backend_name: str = "vllm", limit: int | None = None, out: str | pathlib.Path | None = None,
        batch_size: int | None = None, model_path: str | None = None, results_dir: str = "results") -> dict:
    batch_size = batch_size or BATCH_SIZE[backend_name]
    grid = load_grid()
    _check_model_mode(cfg, grid)
    out = pathlib.Path(out) if out else pathlib.Path(results_dir) / f"{cfg.config_id}.jsonl"
    sampling = cfg.sampling_params(grid)
    answer_max_tokens = int(grid["sampling"]["answer_max_tokens"])

    windows = load_windows(cfg.dataset)
    if limit:
        windows = evenly_spaced(windows, limit)
    by_id = {w["window_id"]: w for w in windows}
    priors = load_prior(cfg.prior, cfg.dataset, results_dir) if cfg.setup == "reviser" else None
    smap = load_shuffle_map(cfg.dataset) if cfg.context_mode == "shuffled" else None
    all_by_id = by_id if smap is None else {w["window_id"]: w for w in load_windows(cfg.dataset)}

    done = done_window_ids(out)
    todo = [w for w in windows if w["window_id"] not in done]
    log.info("config %s: %d windows, %d done, %d to do -> %s", cfg.config_id, len(windows), len(done), len(todo), out)
    stats = {"n": 0, "valid": 0, "think": 0.0, "hit": 0, "wall": 0.0}
    if not todo:
        log.info("SUMMARY config=%s n=0 (nothing to do)", cfg.config_id)
        return stats

    backend = build_backend(cfg, backend_name, grid, model_path)
    t_all = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        msgs, fallbacks = [], []
        for w in batch:
            prior = priors.get(w["window_id"]) if priors is not None else None
            if cfg.setup == "reviser" and prior is None:
                raise KeyError(f"no {cfg.prior} prior forecast for window {w['window_id']}")
            src = all_by_id[smap[w["window_id"]]] if smap is not None else None
            msgs.append(render_messages(w, cfg.setup, cfg.context_mode, prior_forecast=prior, shuffle_source=src))
            fallbacks.append(prior if cfg.setup == "reviser" else
                             seasonal_naive_forecast(w["context"], len(w["target"]), int(w["seasonality"])).tolist())
        t0 = time.time()
        gens = backend.generate(msgs, cfg.thinking, cfg.budget, cfg.n_samples, sampling, answer_max_tokens, cfg.seed)
        wall = (time.time() - t0) / len(batch)
        rows = [make_row(cfg, w, g, fb, wall) for w, g, fb in zip(batch, gens, fallbacks)]
        append_results(out, rows)
        for r in rows:
            stats["n"] += 1
            stats["valid"] += int(r["valid"])
            stats["think"] += r["thinking_tokens_used"]
            stats["hit"] += int(r["budget_hit"])
        log.info("  batch %d: %d/%d windows, valid so far %.3f", i // batch_size, stats["n"], len(todo),
                 stats["valid"] / max(stats["n"], 1))
    stats["wall"] = time.time() - t_all
    n = max(stats["n"], 1)
    log.info("SUMMARY config=%s n=%d valid_rate=%.3f mean_thinking_tokens=%.1f budget_hit_rate=%.3f wall_s=%.1f",
             cfg.config_id, stats["n"], stats["valid"] / n, stats["think"] / n, stats["hit"] / n, stats["wall"])
    return stats


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one LLM config (SPEC §5)")
    ap.add_argument("--config-id")
    ap.add_argument("--dataset", choices=["freshts26", "cik", "fixtures"])
    ap.add_argument("--model")
    ap.add_argument("--setup", choices=["direct", "reviser"])
    ap.add_argument("--think", choices=["off", "on"], help="'off' = hard switch; 'on' requires --budget")
    ap.add_argument("--budget", type=int, default=None, help="thinking budget in tokens (implies thinking on)")
    ap.add_argument("--context-mode", default="full", choices=["full", "none", "shuffled"])
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", default="chronos2")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--model-path", default=None, help="local checkpoint dir overriding the HF id (smoke/offline)")
    return ap.parse_args(argv)


def config_from_args(a: argparse.Namespace) -> RunConfig:
    if a.config_id:
        return RunConfig.parse(a.config_id)
    missing = [k for k in ("dataset", "model", "setup") if getattr(a, k) is None]
    if missing:
        raise SystemExit(f"--config-id or all of --dataset/--model/--setup required (missing {missing})")
    if a.budget is not None and a.budget > 0:
        thinking, budget = True, a.budget
    elif a.think in (None, "off"):
        thinking, budget = False, 0
    else:
        raise SystemExit("--think on requires --budget N")
    return RunConfig(a.dataset, a.model, a.setup, thinking, budget, a.context_mode, a.n_samples, a.seed, a.prior)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    cfg = config_from_args(a)
    run(cfg, a.backend, a.limit, a.out, a.batch_size, a.model_path, a.results_dir)


if __name__ == "__main__":
    main()
