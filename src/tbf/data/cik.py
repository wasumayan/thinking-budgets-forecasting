"""TIER 3. Context is Key (CiK) loader + RCRPS (SPEC §2.2; docs/verify-datasets.md for the HF schema).

load_windows() -> list[dict] in the same window dict shape used by prompts.render_messages, with
  dataset="cik", window_id=f"cik__{task_name}__{instance}", context/target arrays, metadata = the CiK
  background + scenario/constraints text as provided (this IS the context; no calendar/events sections —
  render only ## Series with the CiK text, ## History, ## Task), seasonality from the task's seasonal period
  (m=1 if unknown), precision=4, domain="cik". History/horizon lengths are task-specific (not 96/12): the Task
  line says "Forecast the next {H} values"; parse.parse_forecast must therefore accept H from the window
  (add an `h` argument; default 12).
rcrps(rows) -> per-window RCRPS using the bundled compute_rcrps_with_hf_dataset.py logic (samples required).
"""
from __future__ import annotations


def load_windows():
    raise NotImplementedError


def rcrps(rows):
    raise NotImplementedError
