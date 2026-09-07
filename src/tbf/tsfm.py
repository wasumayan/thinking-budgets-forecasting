"""TSFM baselines and seasonal naive -> results JSONL (SPEC §3.2, §7.2).

CLI: python -m tbf.tsfm --dataset {freshts26|cik|fixtures} --model {snaive|chronos2|timesfm25|tirex} --out PATH
Row schema = SPEC §7.2 with model=<name>, setup="baseline", thinking=False, budget=0, context_mode="numeric",
n_samples=0, seed=0, prior=None, point=[12 floats] (median for TSFMs), quantiles=[[Q=9] x 12] (TSFMs only;
levels 0.1..0.9), valid=True, thinking_tokens_used=0, answer_tokens=0, wall_s per batch / batch size.

Exact API calls: docs/verify-models.md §B (copy verbatim):
  chronos2:  Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE).predict_quantiles(contexts, prediction_length=12, quantile_levels=[0.1..0.9])
  timesfm25: TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch"); compile(ForecastConfig(max_context=1024, max_horizon=12, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True)); forecast(horizon=12, inputs=[...]) -> quantile_forecast[..., 1:10]
  tirex:     load_model("NX-AI/TiRex", backend="torch", compile=True).forecast(context=..., prediction_length=12) -> (quantiles [B,12,9], mean [B,12])
Contexts are the 96 window values as float32; batch_size 256. Resumable: skip window_ids already in --out.
Also exposes load_prior(prior_name, dataset) -> dict[window_id, list[float]] (median) for the reviser setup,
read from results/<dataset>__<prior>.jsonl (raise a clear error if missing).
"""
from __future__ import annotations


def main(argv=None) -> None:
    raise NotImplementedError


def load_prior(prior: str, dataset: str, results_dir="results") -> dict[str, list[float]]:
    raise NotImplementedError


if __name__ == "__main__":
    main()
