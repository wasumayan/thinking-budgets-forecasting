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
  (case-insensitive, whole-word with optional plural suffix, e.g. 'tariff' matches 'tariffs' but 'gold' does not
  match 'golden')
  in text, and category in categories when categories is given; newest first; truncate to k.

Implementation notes: a bullet that is immediately followed by a deeper bullet is a topic heading (e.g.
"*[[Gaza war]]") and is skipped, but its wikilinks are inherited by its child bullets (`links` = own links +
parent-chain links) so events under a topic heading still match the article title.
"""
from __future__ import annotations

import datetime as dt
import logging
import re

from ..config import load_series_config
from . import http

log = logging.getLogger(__name__)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
_EXTLINK = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?:\s+([^\]]*))?\]")
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_BOLD = re.compile(r"'{2,5}")
_TRAILING_CITE = re.compile(r"\s*\((?:[^()]{1,60})\)\s*$")


def _strip_templates(s: str) -> str:
    while "{{" in s:
        new = re.sub(r"\{\{[^{}]*\}\}", "", s)
        if new == s:
            break
        s = new
    return s


def _link_target(t: str) -> str:
    t = t.strip().replace(" ", "_")
    return (t[0].upper() + t[1:]) if t else t


def parse_bullet(raw: str) -> tuple[str, list[str]]:
    """wikitext bullet body -> (plain text, link targets)."""
    s = _REF.sub("", raw)
    s = _strip_templates(s)
    s = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", "", s, flags=re.IGNORECASE)
    links = [_link_target(m.group(1)) for m in _WIKILINK.finditer(s)]
    s = _WIKILINK.sub(lambda m: (m.group(2) if m.group(2) is not None else m.group(1)).strip(), s)
    s = _EXTLINK.sub(lambda m: (m.group(1) or "").strip(), s)
    s = _TAG.sub("", s)
    s = _BOLD.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    # trailing citation parentheses, possibly several: "(Reuters) (BBC News)"
    for _ in range(3):
        new = _TRAILING_CITE.sub("", s)
        if new == s:
            break
        s = new.strip()
    return s.strip(" ;:,"), links


def parse_wikitext(wikitext: str, date: dt.date, categories: list[str] | None = None) -> list[dict]:
    cats = set(categories or load_series_config()["defaults"]["all_event_categories"])
    lines = wikitext.split("\n")
    out, category = [], "Other"
    chain: list[tuple[int, list[str]]] = []  # (depth, links) of ancestor bullets
    avail = f"{date.isoformat()}T23:59:59Z"
    for i, line in enumerate(lines):
        line = line.rstrip()
        if line.startswith(";"):
            name = _BOLD.sub("", _strip_templates(line[1:])).strip()
            name = _WIKILINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), name)
            category = name if name in cats else "Other"
            chain = []
            continue
        if not line.startswith("*"):
            continue
        depth = len(line) - len(line.lstrip("*"))
        body = line[depth:].strip()
        text, links = parse_bullet(body)
        chain = [(d, l) for d, l in chain if d < depth]
        inherited = [l for _, ls in chain for l in ls]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        next_depth = len(next_line) - len(next_line.lstrip("*")) if next_line.startswith("*") else 0
        chain.append((depth, links))
        if not text or next_depth > depth:
            continue  # empty or topic heading line
        all_links = list(dict.fromkeys(links + inherited))
        out.append({"date": date.isoformat(), "category": category, "text": text, "links": all_links, "available_at": avail})
    return out


def daily_events(date) -> list[dict]:
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    cfg = load_series_config()["sources"]["current_events"]
    url = cfg["parse_api"].format(year=date.year, month_name=MONTHS[date.month - 1], day=date.day)
    try:
        data, _ = http.get(url, cache_key=f"current_events/{date.isoformat()}.json")
    except http.NotFound:
        log.warning("no current-events page for %s", date)
        return []
    if not isinstance(data, dict) or "parse" not in data:
        log.warning("current events %s: unexpected response %s", date, str(data)[:120])
        return []
    return parse_wikitext(data["parse"]["wikitext"], date)


def all_events(start: str = "2025-06-01", end: str | None = None) -> list[dict]:
    d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end) if end else dt.datetime.now(dt.timezone.utc).date()
    events = []
    while d <= end_d:
        events += daily_events(d)
        d += dt.timedelta(days=1)
    log.info("current events %s..%s: %d bullets", start, end_d, len(events))
    return events


def _keyword_regex(keywords) -> re.Pattern | None:
    parts = []
    for kw in keywords:
        kw = str(kw).strip()
        if not kw:
            continue
        parts.append(rf"\b{re.escape(kw)}(?:e?s)?\b")
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


def match_events(events, window_origin, lookback_days: int, title=None, keywords=(), categories=None, k: int = 10):
    origin = window_origin.date() if hasattr(window_origin, "date") and not isinstance(window_origin, dt.date) else window_origin
    if isinstance(origin, str):
        origin = dt.date.fromisoformat(origin[:10])
    lo = origin - dt.timedelta(days=lookback_days)
    rx = _keyword_regex(keywords)
    title_u = title.replace(" ", "_") if title else None
    cats = set(categories) if categories else None
    keep = []
    for e in events:
        d = dt.date.fromisoformat(str(e["date"])[:10])
        if not (lo < d <= origin):
            continue
        if cats is not None and e["category"] not in cats:
            continue
        hit = bool(title_u and title_u in e.get("links", [])) or bool(rx and rx.search(e["text"]))
        if hit:
            keep.append(e)
    keep.sort(key=lambda e: e["date"], reverse=True)
    return keep[:k]
