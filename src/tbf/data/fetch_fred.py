"""FRED no-key CSV + notes (SPEC §2.1 energy/macro).

fred_series(series_id) -> pd.Series indexed by observation_date, float, with '.' -> NaN (kept; the builder
  forward-fills inside windows and drops windows with NaN targets). URL: series.yaml:sources.fred_csv.url.
fred_notes(series_id) -> dict(title, units, frequency, seasonal_adjustment, notes)
  Scrape the series page (notes_url) — the 'Notes' section and the header fields; if FRED_API_KEY is set,
  use https://api.stlouisfed.org/fred/series?series_id=...&api_key=...&file_type=json instead. Cache both.
"""
from __future__ import annotations

import html as _html
import io
import logging
import os
import re

import pandas as pd

from ..config import load_series_config
from . import http

log = logging.getLogger(__name__)
FRED_API = "https://api.stlouisfed.org/fred/series?series_id={id}&api_key={key}&file_type=json"


def parse_fred_csv(text: str, series_id: str | None = None) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = series_id if series_id in df.columns else [c for c in df.columns if c != date_col][0]
    s = pd.Series(pd.to_numeric(df[val_col].replace(".", None), errors="coerce").to_numpy(dtype=float),
                  index=pd.to_datetime(df[date_col]), name=val_col)
    return s.sort_index()


def fred_series(series_id: str) -> pd.Series:
    cfg = load_series_config()["sources"]["fred_csv"]
    text, _ = http.get(cfg["url"].format(id=series_id), cache_key=f"fred_csv/{series_id}.csv")
    s = parse_fred_csv(text, series_id)
    log.info("FRED %s: %d obs %s..%s, %d missing", series_id, len(s), s.index.min().date(), s.index.max().date(), int(s.isna().sum()))
    return s


_TAG = re.compile(r"<[^>]+>")


def _text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL)
    html = re.sub(r"</(p|div|li|h\d|tr|br)>", "\n", html)
    return _html.unescape(_TAG.sub(" ", html))


def parse_fred_page(html: str, series_id: str) -> dict:
    """Best-effort scrape of a FRED series page: title, units, frequency, seasonal adjustment, notes."""
    out = {"title": "", "units": "", "frequency": "", "seasonal_adjustment": "", "notes": ""}
    m = re.search(r"<span[^>]*id=\"series-title-text-container\"[^>]*>(.*?)</span>", html, re.DOTALL)
    if m:
        out["title"] = _html.unescape(_TAG.sub("", m.group(1))).strip()
    else:
        m = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
        if m:
            t = _html.unescape(m.group(1)).strip()
            out["title"] = re.sub(r"\s*\(" + re.escape(series_id) + r"\).*$", "", t).split("|")[0].strip()
    text = _text(html)
    text = re.sub(r"[ \t]+", " ", text)
    m = re.search(r"Units:\s*([^\n]+)", text)
    if m:
        units = m.group(1).strip()
        out["units"] = units
        if re.search(r"not seasonally adjusted", units, re.I):
            out["seasonal_adjustment"] = "Not Seasonally Adjusted"
        elif re.search(r"seasonally adjusted", units, re.I):
            out["seasonal_adjustment"] = "Seasonally Adjusted"
    m = re.search(r"Frequency:\s*([^\n]+)", text)
    if m:
        out["frequency"] = m.group(1).strip()
    m = re.search(r"\bNotes\b\s*\n(.*?)(?:\n\s*Suggested Citation|\n\s*Release Tables|\n\s*Related Categories|\Z)", text, re.DOTALL)
    if m:
        notes = re.sub(r"\n\s*\n+", "\n", m.group(1)).strip()
        notes = re.sub(r"^(Source:|Release:)[^\n]*\n?", "", notes, flags=re.M).strip()
        out["notes"] = re.sub(r"\s+", " ", notes)[:1500]
    return out


def fred_notes(series_id: str) -> dict:
    cfg = load_series_config()["sources"]["fred_csv"]
    key = os.environ.get("FRED_API_KEY")
    if key:
        data, _ = http.get(FRED_API.format(id=series_id, key=key), cache_key=f"fred_api_series/{series_id}.json")
        s = (data.get("seriess") or [{}])[0]
        return {"title": s.get("title", ""), "units": s.get("units", ""), "frequency": s.get("frequency", ""),
                "seasonal_adjustment": s.get("seasonal_adjustment", ""), "notes": (s.get("notes") or "")[:1500]}
    html, _ = http.get(cfg["notes_url"].format(id=series_id), cache_key=f"fred_page/{series_id}.html")
    out = parse_fred_page(html, series_id)
    if not out["title"]:
        log.warning("FRED page for %s: no title parsed; builder falls back to series.yaml", series_id)
    return out
