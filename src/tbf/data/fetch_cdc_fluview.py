"""TIER 2 (optional). CDC FluView weekly report 'Key Points' (SPEC §2.1 item 4).

key_points(year, week) -> dict(date=week_ending_saturday, source="CDC FluView", text=<joined bullets, <= 800 chars>,
                               available_at=<Friday of week+1, 17:00 ET>) or None if the page is missing.
URL pattern: series.yaml:sources.cdc_fluview_reports.url. Parse the first <ul> after a heading containing
'Key Points' (fallback: first paragraph). Off-season pages are one sentence — keep them.
"""
from __future__ import annotations

import datetime as dt
import html as _html
import logging
import re
from zoneinfo import ZoneInfo

from ..config import load_series_config
from . import http
from .fetch_delphi import epiweek_to_saturday

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(_TAG.sub("", s))).strip()


def parse_key_points(html: str, max_chars: int = 800) -> str:
    m = re.search(r"<h[1-6][^>]*>[^<]*Key Points[^<]*</h[1-6]>(.*)", html, re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else html
    ul = re.search(r"<ul[^>]*>(.*?)</ul>", body, re.DOTALL)
    items = []
    if ul:
        items = [_clean(li) for li in re.findall(r"<li[^>]*>(.*?)</li>", ul.group(1), re.DOTALL)]
        items = [x for x in items if x]
    if not items:
        p = re.search(r"<p[^>]*>(.*?)</p>", body, re.DOTALL)
        items = [_clean(p.group(1))] if p and _clean(p.group(1)) else []
    text = " ".join(items)
    return text[:max_chars].rsplit(" ", 1)[0] if len(text) > max_chars else text


def available_at_friday_next_week(week_ending: dt.date, hour: int = 17) -> str:
    friday = week_ending + dt.timedelta(days=6)  # Saturday + 6 = next Friday
    local = dt.datetime.combine(friday, dt.time(hour, 0), tzinfo=ET)
    return local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def key_points(year: int, week: int):
    cfg = load_series_config()["sources"]["cdc_fluview_reports"]
    url = cfg["url"].format(year=year, week=week)
    try:
        html, _ = http.get(url, cache_key=f"cdc_fluview/{year}-week-{week:02d}.html")
    except http.NotFound:
        return None
    text = parse_key_points(html)
    if not text:
        log.warning("CDC FluView %d-week-%02d: no key points parsed", year, week)
        return None
    week_ending = epiweek_to_saturday(year * 100 + week)
    return {"date": week_ending.isoformat(), "source": "CDC FluView", "text": text,
            "available_at": available_at_friday_next_week(week_ending)}
