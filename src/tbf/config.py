"""Config loading and config-id conventions (SPEC §5, configs/grid.yaml).

config_id = "{dataset}__{model}__{setup}__{think}__{ctx}__s{seed}" + optional "__n{n}" + optional "__prior-{p}"
  think ∈ {off, b512, b2048, b8192}; ctx ∈ {full, none, shuffled}
Baselines (no LLM): "{dataset}__{model}" with model ∈ {snaive, chronos2, timesfm25, tirex}.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_series_config() -> dict[str, Any]:
    return load_yaml(CONFIGS / "series.yaml")


def load_grid() -> dict[str, Any]:
    return load_yaml(CONFIGS / "grid.yaml")


@dataclasses.dataclass(frozen=True)
class RunConfig:
    dataset: str            # freshts26 | cik | fixtures
    model: str              # key in grid.yaml:models
    setup: str              # direct | reviser
    thinking: bool
    budget: int             # 0 when thinking is False
    context_mode: str       # full | none | shuffled
    n_samples: int = 1
    seed: int = 0
    prior: str = "chronos2"

    @property
    def think_tag(self) -> str:
        return "off" if not self.thinking else f"b{self.budget}"

    @property
    def config_id(self) -> str:
        s = f"{self.dataset}__{self.model}__{self.setup}__{self.think_tag}__{self.context_mode}__s{self.seed}"
        if self.n_samples != 1:
            s += f"__n{self.n_samples}"
        if self.setup == "reviser" and self.prior != "chronos2":
            s += f"__prior-{self.prior}"
        return s

    @classmethod
    def parse(cls, config_id: str) -> "RunConfig":
        """Inverse of config_id. Must round-trip: RunConfig.parse(c.config_id) == c (tested)."""
        parts = config_id.split("__")
        dataset, model, setup, think, ctx, seed = parts[:6]
        n_samples, prior = 1, "chronos2"
        for extra in parts[6:]:
            if extra.startswith("n"):
                n_samples = int(extra[1:])
            elif extra.startswith("prior-"):
                prior = extra[len("prior-"):]
            else:
                raise ValueError(f"unknown config_id part {extra!r} in {config_id}")
        thinking = think != "off"
        budget = 0 if not thinking else int(think[1:])
        return cls(dataset, model, setup, thinking, budget, ctx, n_samples, int(seed[1:]), prior)

    def sampling_params(self, grid: dict[str, Any] | None = None) -> dict[str, float]:
        grid = grid or load_grid()
        return dict(grid["sampling"]["thinking_on" if self.thinking else "thinking_off"])

    def model_spec(self, grid: dict[str, Any] | None = None) -> dict[str, Any]:
        grid = grid or load_grid()
        return grid["models"][self.model]
