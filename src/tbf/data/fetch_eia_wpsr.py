"""TIER 2 (optional). EIA Weekly Petroleum Status Report highlights (SPEC §2.1 item 4).

highlights(release_date) -> dict(date=release_date, source="EIA WPSR", text=<first 2 paragraphs, <= 800 chars>,
                                 available_at=release_date 10:30 ET) or None.
Archive PDF pattern: series.yaml:sources.eia_wpsr.archive_url; use `pdftotext -layout` (poppler) or pypdf.
Release dates: Wednesdays (Thursday after Monday holidays); enumerate from 2025-06-01.
"""
from __future__ import annotations


def highlights(release_date):
    raise NotImplementedError
