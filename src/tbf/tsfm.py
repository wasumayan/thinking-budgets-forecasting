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

import argparse
import logging
import pathlib
import sys
import time

import numpy as np

from .config import load_grid
from .data.windows import append_results, done_window_ids, load_windows, read_results
from .metrics import QUANTILE_LEVELS, seasonal_naive_forecast

log = logging.getLogger(__name__)
MODELS = ("snaive", "chronos2", "timesfm25", "tirex")
BATCH_SIZE = 256


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class _Forecaster:
    """forecast(contexts: list[np.ndarray], H) -> (point [B,H], quantiles [B,H,9] or None)."""

    def __init__(self, name: str, seasonality_of: dict[str, int] | None = None):
        self.name = name
        if name == "snaive":
            return
        grid = load_grid()
        hf = grid["priors"][name]["hf"]
        if name == "chronos2":
            from chronos import Chronos2Pipeline
            self.pipe = Chronos2Pipeline.from_pretrained(hf, device_map=_device())
        elif name == "timesfm25":
            import timesfm
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(hf)
            self.model.compile(timesfm.ForecastConfig(
                max_context=1024, max_horizon=12, normalize_inputs=True, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        elif name == "tirex":
            from tirex import load_model
            self.model = load_model(hf, backend="torch", compile=_device() == "cuda")
        else:
            raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")

    def __call__(self, contexts: list[np.ndarray], H: int, seasonalities: list[int]):
        if self.name == "snaive":
            point = np.stack([seasonal_naive_forecast(c, H, m) for c, m in zip(contexts, seasonalities)])
            return point, None
        if self.name == "chronos2":
            quantiles, mean = self.pipe.predict_quantiles(
                [np.asarray(c, dtype=np.float32) for c in contexts], prediction_length=H,
                quantile_levels=list(QUANTILE_LEVELS))
            q = np.stack([np.asarray(t)[0] for t in quantiles])  # each (1, H, Q) -> (H, Q)
        elif self.name == "timesfm25":
            _, quantile_forecast = self.model.forecast(horizon=H, inputs=[np.asarray(c, dtype=np.float32) for c in contexts])
            q = np.asarray(quantile_forecast)[..., 1:10]  # (B, H, 9): drop the leading mean channel
        elif self.name == "tirex":
            import torch
            data = torch.tensor(np.stack([np.asarray(c, dtype=np.float32) for c in contexts]))
            quantiles, _mean = self.model.forecast(context=data, prediction_length=H)
            q = np.asarray(quantiles.cpu() if hasattr(quantiles, "cpu") else quantiles)  # (B, H, 9)
        else:
            raise AssertionError(self.name)
        q = q.astype(float)
        return q[..., 4], q  # median (level 0.5 is index 4), quantiles


def make_row(dataset: str, model: str, w: dict, point: np.ndarray, q, wall_s: float) -> dict:
    return {
        "config_id": f"{dataset}__{model}", "window_id": w["window_id"], "dataset": dataset, "model": model,
        "setup": "baseline", "thinking": False, "budget": 0, "context_mode": "numeric", "n_samples": 0, "seed": 0,
        "prior": None, "point": [float(x) for x in point], "samples": None,
        "quantiles": None if q is None else [[float(x) for x in row] for row in q],
        "valid": True, "n_values_mismatch": False, "raw_answer": "", "thinking_tokens_used": 0, "answer_tokens": 0,
        "budget_hit": False, "finished": True, "wall_s": wall_s, "prompt_tokens": 0,
    }


def run(dataset: str, model: str, out: str | pathlib.Path, limit: int | None = None, batch_size: int = BATCH_SIZE) -> int:
    windows = load_windows(dataset)
    if limit:
        windows = windows[:limit]
    done = done_window_ids(out)
    todo = [w for w in windows if w["window_id"] not in done]
    log.info("%s/%s: %d windows, %d already done, %d to do", dataset, model, len(windows), len(done), len(todo))
    if not todo:
        return 0
    fc = _Forecaster(model)
    t_all = time.time()
    n = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        # group by horizon so that ragged CiK horizons still batch correctly
        by_h: dict[int, list[dict]] = {}
        for w in batch:
            by_h.setdefault(len(w["target"]), []).append(w)
        rows = []
        for H, ws in by_h.items():
            t0 = time.time()
            contexts = [np.asarray(w["context"], dtype=np.float32) for w in ws]
            point, q = fc(contexts, H, [int(w["seasonality"]) for w in ws])
            wall = (time.time() - t0) / len(ws)
            for j, w in enumerate(ws):
                rows.append(make_row(dataset, model, w, point[j], None if q is None else q[j], wall))
        append_results(out, rows)
        n += len(rows)
        log.info("  wrote %d/%d", n, len(todo))
    log.info("SUMMARY config=%s__%s n=%d valid_rate=1.000 mean_thinking_tokens=0 budget_hit_rate=0.000 wall_s=%.1f",
             dataset, model, n, time.time() - t_all)
    return n


def load_prior(prior: str, dataset: str, results_dir="results") -> dict[str, list[float]]:
    path = pathlib.Path(results_dir) / f"{dataset}__{prior}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"prior forecasts {path} not found. Run: python -m tbf.tsfm --dataset {dataset} --model {prior} --out {path}")
    rows = read_results(path)
    if not rows:
        raise ValueError(f"prior file {path} is empty")
    return {r["window_id"]: [float(x) for x in r["point"]] for r in rows}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=["freshts26", "cik", "fixtures"])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(a.dataset, a.model, a.out, a.limit, a.batch_size)


if __name__ == "__main__":
    main()
