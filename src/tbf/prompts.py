"""Prompt rendering (SPEC §4, configs/prompts/README.md).

render_messages(window: dict, setup: str, context_mode: str, prior_forecast: list[float] | None,
                shuffle_source: dict | None) -> list[dict]   # [{"role":"system",...},{"role":"user",...}]

- window: one row of windows.parquet as a dict (schema SPEC §7.1) + "title", "units", "precision", "freq".
- setup="reviser" requires prior_forecast (12 floats) rendered with the series precision.
- context_mode: full | none | shuffled. In shuffled mode `shuffle_source` is the mapped window whose
  events/reports replace this window's (metadata and calendar stay real).
- Section omission rules are in configs/prompts/README.md. Templates are read from configs/prompts/*.txt
  at import time; never inline them.
- Deterministic: same inputs -> byte-identical output (snapshot-tested).

format_value(v: float, precision: int) -> str   # "12345" for precision 0, "1.0834" for 4, no thousands separators
"""
from __future__ import annotations

from .config import CONFIGS

PROMPT_DIR = CONFIGS / "prompts"
SYSTEM = (PROMPT_DIR / "system.txt").read_text().strip()
DIRECT = (PROMPT_DIR / "direct.txt").read_text()
REVISER = (PROMPT_DIR / "reviser.txt").read_text()

FREQ_DESC = {"D": "daily", "B": "business days", "W": "weekly"}


def format_value(v: float, precision: int) -> str:
    return f"{v:.{precision}f}"


def render_messages(window: dict, setup: str, context_mode: str, prior_forecast=None, shuffle_source=None) -> list[dict]:
    raise NotImplementedError
