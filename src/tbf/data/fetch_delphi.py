"""Delphi Epidata fluview (SPEC §2.1 health).

fluview(regions: list[str], epiweeks="202301-<current>") -> pd.DataFrame[region, epiweek, week_ending(date), ili, num_patients]
  One request for all regions (comma-separated). Field 'ili' (state-level available); week_ending = Saturday of the
  epiweek (MMWR week) — implement epiweek->date conversion and unit-test it against 202634 -> 2026-08-29.
  Final values (no issues= parameter). Drop regions with > 5 % missing in 2025-08..now and log them.
"""
from __future__ import annotations


def fluview(regions: list[str], epiweeks: str | None = None):
    raise NotImplementedError


def epiweek_to_saturday(epiweek: int):
    """202634 -> date(2026, 8, 29). MMWR weeks end on Saturday; week 1 contains Jan 4."""
    raise NotImplementedError
