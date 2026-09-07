"""Wikimedia pageviews + Wikipedia summaries as-of 2025-07-31 (SPEC §2.1 web domain).

pageviews(title, start="20240101", end=<yesterday UTC>) -> pd.Series indexed by date (daily, complete; missing days
  filled with 0 and counted in n_missing_days). Endpoint pattern in series.yaml:sources.wikimedia_pageviews.
  Titles are URL-encoded with underscores; resolve redirects first via
  https://en.wikipedia.org/w/api.php?action=query&titles={t}&redirects=1&format=json (record canonical title).
summary_asof(title, asof="2025-07-31T23:59:59Z") -> (text, revid, fetched_asof: bool)
  1) revisions API (series.yaml) -> revid; 2) parse API oldid=revid, prop=text, section=0 -> HTML -> strip tags,
  keep the first 1–2 paragraphs (<= 1200 chars); 3) on failure use page/summary extract and set fetched_asof=False.
"""
from __future__ import annotations


def pageviews(title: str, start: str = "20240101", end: str | None = None):
    raise NotImplementedError


def summary_asof(title: str, asof: str = "2025-07-31T23:59:59Z"):
    raise NotImplementedError
