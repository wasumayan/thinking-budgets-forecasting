"""FRED no-key CSV + notes (SPEC §2.1 energy/macro).

fred_series(series_id) -> pd.Series indexed by observation_date, float, with '.' -> NaN (kept; the builder
  forward-fills inside windows and drops windows with NaN targets). URL: series.yaml:sources.fred_csv.url.
fred_notes(series_id) -> dict(title, units, frequency, seasonal_adjustment, notes)
  Scrape the series page (notes_url) — the 'Notes' section and the header fields; if FRED_API_KEY is set,
  use https://api.stlouisfed.org/fred/series?series_id=...&api_key=...&file_type=json instead. Cache both.
"""
from __future__ import annotations


def fred_series(series_id: str):
    raise NotImplementedError


def fred_notes(series_id: str) -> dict:
    raise NotImplementedError
