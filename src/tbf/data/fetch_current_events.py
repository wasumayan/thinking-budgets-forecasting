"""Wikipedia Portal:Current events daily pages -> event bullets (SPEC §2.1 item 3).

daily_events(date) -> list[dict(date, category, text, links, available_at)]
  Fetch wikitext (series.yaml:sources.current_events.parse_api; month_name e.g. 'August'). Parse: lines starting
  with ';' set the current category (must be one of defaults.all_event_categories, else 'Other'); lines starting
  with '*' (any depth) are bullets. From each bullet: links = all [[Target|label]] targets (underscored,
  first-letter-capitalized); text = bullet with wiki markup stripped ([[a|b]] -> b, [[a]] -> a, '''x''' -> x,
  templates {{...}} removed, external links [http... label] -> label, <ref>...</ref> removed, trailing
  '(source)' citation parentheses removed). available_at = date 23:59:59 UTC. Skip empty/heading-only lines.
all_events(start="2025-06-01", end=<today>) -> list[dict]  (one fetch per day, cached)
match_events(events, window_origin, lookback_days, title=None, keywords=(), categories=None, k=10) -> list[dict]
  Filter events with origin-lookback < date <= origin; keep if (title and title in links) or any keyword
  (case-insensitive, word-boundary for tokens of length <= 3 like 'eu', 'ai', 'ev', 'fed', 'f1'; substring otherwise)
  in text, and category in categories when categories is given; newest first; truncate to k.
"""
from __future__ import annotations


def daily_events(date):
    raise NotImplementedError


def all_events(start: str = "2025-06-01", end: str | None = None):
    raise NotImplementedError


def match_events(events, window_origin, lookback_days: int, title=None, keywords=(), categories=None, k: int = 10):
    raise NotImplementedError
