#!/usr/bin/env python
"""Expand configs/grid.yaml into config ids. Usage:
  python scripts/expand_grid.py --tier 1            # ids for tier 1, one per line
  python scripts/expand_grid.py --tier 1 --with-time # "config_id<TAB>HH:MM:SS" for sbatch --time
  python scripts/expand_grid.py --all
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbf.config import RunConfig, load_grid  # noqa: E402


def expand(grid: dict, tier: int, include_stretch: bool = False) -> list[RunConfig]:
    out: list[RunConfig] = []
    for block in grid["tiers"][tier]:
        if block.get("stretch") and not include_stretch:
            continue
        priors = block.get("priors", ["chronos2"])
        for model, setup, think, ctx, seed, prior in itertools.product(
                block["models"], block["setups"], block["think"], block["ctx"], block["seeds"], priors):
            thinking = think != "off"
            budget = grid["budgets"][think] if thinking else 0
            out.append(RunConfig(block["dataset"], model, setup, thinking, budget, ctx,
                                 block.get("n_samples", 1), seed, prior))
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for c in out:
        if c.config_id not in seen:
            seen.add(c.config_id)
            uniq.append(c)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stretch", action="store_true")
    ap.add_argument("--with-time", action="store_true")
    a = ap.parse_args()
    grid = load_grid()
    tiers = sorted(grid["tiers"]) if a.all else [a.tier]
    for t in tiers:
        for c in expand(grid, t, a.stretch):
            if a.with_time:
                h, m, s = (int(x) for x in grid["slurm_time"][c.think_tag].split(":"))
                mult = 3 if c.n_samples > 1 else 1
                secs = (h * 3600 + m * 60 + s) * mult
                print(f"{c.config_id}\t{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}")
            else:
                print(c.config_id)


if __name__ == "__main__":
    main()
