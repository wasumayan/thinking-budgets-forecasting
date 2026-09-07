"""Wikimedia pageviews + Wikipedia summaries as-of 2025-07-31 (SPEC §2.1 web domain).

pageviews(title, start="20240101", end=<yesterday UTC>) -> pd.Series indexed by date (daily, complete; missing days
  filled with 0 and counted in n_missing_days). Endpoint pattern in series.yaml:sources.wikimedia_pageviews.
  Titles are URL-encoded with underscores; resolve redirects first via
  https://en.wikipedia.org/w/api.php?action=query&titles={t}&redirects=1&format=json (record canonical title).
summary_asof(title, asof="2025-07-31T23:59:59Z") -> (text, revid, fetched_asof: bool)
  1) revisions API (series.yaml) -> revid; 2) parse API oldid=revid, prop=text, section=0 -> HTML -> strip tags,
  keep the first 1–2 paragraphs (<= 1200 chars); 3) on failure use page/summary extract and set fetched_asof=False.

The returned Series carries .attrs = {"canonical_title", "n_missing_days"}.
"""
from __future__ import annotations

import datetime as dt
import html as _html
import logging
import re
import urllib.parse

import pandas as pd

from ..config import load_series_config
from . import http

log = logging.getLogger(__name__)
REDIRECT_API = "https://en.wikipedia.org/w/api.php?action=query&titles={title}&redirects=1&format=json&formatversion=2"


def _yesterday_utc() -> str:
    return (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).strftime("%Y%m%d")


def _safe(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", title)[:120]


def resolve_title(title: str) -> str:
    """Canonical (redirect-resolved) title with underscores. Falls back to the input on any error."""
    t = title.replace("_", " ")
    try:
        data, _ = http.get(REDIRECT_API.format(title=urllib.parse.quote(t)), cache_key=f"wikipedia_redirect/{_safe(title)}.json")
        pages = data.get("query", {}).get("pages", [])
        if pages and "title" in pages[0] and not pages[0].get("missing"):
            return pages[0]["title"].replace(" ", "_")
    except Exception as e:  # noqa: BLE001
        log.warning("redirect resolution failed for %s: %s", title, e)
    return title.replace(" ", "_")


def pageviews(title: str, start: str = "20240101", end: str | None = None):
    cfg = load_series_config()["sources"]["wikimedia_pageviews"]
    end = end or _yesterday_utc()
    canonical = resolve_title(title)
    url = cfg["url"].format(title=urllib.parse.quote(canonical, safe=""), start=start + "00", end=end + "00")
    data, _ = http.get(url, cache_key=f"wikimedia_pageviews/{_safe(canonical)}_{start}_{end}.json")
    items = data.get("items", []) if isinstance(data, dict) else []
    got = {pd.Timestamp(it["timestamp"][:8]).date(): float(it["views"]) for it in items}
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    values = [got.get(d.date(), 0.0) for d in idx]
    n_missing = sum(1 for d in idx if d.date() not in got)
    s = pd.Series(values, index=idx, name=canonical, dtype=float)
    s.attrs = {"canonical_title": canonical, "n_missing_days": n_missing, "n_items": len(items)}
    if n_missing:
        log.info("pageviews %s: %d missing days filled with 0", canonical, n_missing)
    return s


_TAG = re.compile(r"<[^>]+>")
_REF = re.compile(r"<sup[^>]*class=\"[^\"]*reference[^\"]*\"[^>]*>.*?</sup>", re.DOTALL)
_STYLE = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.DOTALL)
_P = re.compile(r"<p(?:\s[^>]*)?>(.*?)</p>", re.DOTALL)
_BRACKET_CITE = re.compile(r"\[(?:\d+|[a-z]|note \d+|citation needed)\]", re.IGNORECASE)


def _strip_html(fragment: str) -> str:
    text = _TAG.sub("", _REF.sub("", fragment))
    text = _html.unescape(text)
    text = _BRACKET_CITE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def lead_paragraphs(html: str, max_chars: int = 1200, max_paragraphs: int = 2) -> str:
    """First 1–2 non-empty <p> paragraphs of a parsed section-0 HTML, tags stripped, <= max_chars."""
    html = _STYLE.sub("", html)
    paras = []
    for m in _P.finditer(html):
        t = _strip_html(m.group(1))
        if len(t) < 40 or t.startswith("Coordinates:"):
            continue
        paras.append(t)
        if len(paras) >= max_paragraphs:
            break
    text = " ".join(paras)
    if len(text) > max_chars:
        cut = text[:max_chars]
        text = cut[: cut.rfind(". ") + 1] if ". " in cut[max_chars // 2:] else cut
    return text.strip()


def summary_asof(title: str, asof: str = "2025-07-31T23:59:59Z"):
    cfg = load_series_config()["sources"]["wikipedia_summary_asof"]
    canonical = resolve_title(title)
    q = urllib.parse.quote(canonical.replace("_", " "))
    try:
        rev_url = cfg["revisions_api"].format(title=q).replace("rvstart=2025-07-31T23:59:59Z", f"rvstart={asof}")
        data, _ = http.get(rev_url, cache_key=f"wikipedia_revisions/{_safe(canonical)}_{asof[:10]}.json")
        page = data["query"]["pages"][0]
        revid = int(page["revisions"][0]["revid"])
        parsed, _ = http.get(cfg["parse_api"].format(revid=revid), cache_key=f"wikipedia_parse/{_safe(canonical)}_{revid}.json")
        text = lead_paragraphs(parsed["parse"]["text"])
        if not text:
            raise ValueError("no lead paragraph found")
        return text, revid, True
    except Exception as e:  # noqa: BLE001
        log.warning("historical summary failed for %s (%s); falling back to current page/summary", title, e)
    data, _ = http.get(cfg["fallback_summary"].format(title=urllib.parse.quote(canonical, safe="")),
                       cache_key=f"wikipedia_summary/{_safe(canonical)}.json")
    extract = (data.get("extract") or "").strip()
    return extract[:1200], int(data.get("revision") or 0), False
