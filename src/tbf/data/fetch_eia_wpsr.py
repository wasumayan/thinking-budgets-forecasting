"""TIER 2 (optional). EIA Weekly Petroleum Status Report highlights (SPEC §2.1 item 4).

highlights(release_date) -> dict(date=release_date, source="EIA WPSR", text=<first 2 paragraphs, <= 800 chars>,
                                 available_at=release_date 10:30 ET) or None.
Archive PDF pattern: series.yaml:sources.eia_wpsr.archive_url; use `pdftotext -layout` (poppler) or pypdf.
Release dates: Wednesdays (Thursday after Monday holidays); enumerate from 2025-06-01.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo

from ..config import load_series_config
from . import http

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def pdf_to_text(pdf: bytes) -> str:
    if shutil.which("pdftotext"):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(pdf)
            f.flush()
            return subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True, text=True, check=True).stdout
    try:
        import io

        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(pdf)).pages)
    except ImportError as e:
        raise RuntimeError("need `pdftotext` (poppler-utils) on PATH or `pip install pypdf` for WPSR highlights") from e


def first_paragraphs(text: str, n: int = 2, max_chars: int = 800) -> str:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p) > 60 and not p.lower().startswith(("weekly petroleum status report", "highlights"))]
    out = " ".join(paras[:n])
    return out[:max_chars].rsplit(" ", 1)[0] if len(out) > max_chars else out


def release_dates(start: str = "2025-06-01", end: str | None = None) -> list[dt.date]:
    """Wednesdays (Thursday when Monday of that week is a US federal holiday)."""
    import holidays

    us = holidays.US(years=range(2025, 2028))
    d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end) if end else dt.datetime.now(dt.timezone.utc).date()
    out = []
    while d <= end_d:
        if d.weekday() == 2:
            monday = d - dt.timedelta(days=2)
            out.append(d + dt.timedelta(days=1) if monday in us else d)
        d += dt.timedelta(days=1)
    return [x for x in out if x <= end_d]


def highlights(release_date):
    if isinstance(release_date, str):
        release_date = dt.date.fromisoformat(release_date)
    cfg = load_series_config()["sources"]["eia_wpsr"]
    url = cfg["archive_url"].format(year=release_date.year, month=release_date.month, day=release_date.day)
    try:
        pdf, _ = http.get(url, cache_key=f"eia_wpsr/{release_date.isoformat()}.pdf.json", binary=True)
    except http.NotFound:
        log.warning("no WPSR highlights PDF for %s", release_date)
        return None
    text = first_paragraphs(pdf_to_text(pdf))
    if not text:
        return None
    local = dt.datetime.combine(release_date, dt.time(10, 30), tzinfo=ET)
    return {"date": release_date.isoformat(), "source": "EIA WPSR", "text": text,
            "available_at": local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
