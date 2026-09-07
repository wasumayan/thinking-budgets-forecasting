"""Load data/fixtures/windows.jsonl (12 synthetic windows, 3 per domain) for tests and `make smoke`.
Same fields as windows.parquet (SPEC §7.1) plus title/units/precision/freq. Also data/fixtures/shuffle_map.json.
"""
from __future__ import annotations

import json
import pathlib

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / "data" / "fixtures"


def load_windows() -> list[dict]:
    with open(FIXTURES / "windows.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_shuffle_map() -> dict[str, str]:
    with open(FIXTURES / "shuffle_map.json") as f:
        return json.load(f)
