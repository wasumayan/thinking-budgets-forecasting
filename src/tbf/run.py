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


def main(argv=None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
