"""TIER 2 (optional). CDC FluView weekly report 'Key Points' (SPEC §2.1 item 4).

key_points(year, week) -> dict(date=week_ending_saturday, source="CDC FluView", text=<joined bullets, <= 800 chars>,
                               available_at=<Friday of week+1, 17:00 ET>) or None if the page is missing.
URL pattern: series.yaml:sources.cdc_fluview_reports.url. Parse the first <ul> after a heading containing
'Key Points' (fallback: first paragraph). Off-season pages are one sentence — keep them.
"""
from __future__ import annotations


def key_points(year: int, week: int):
    raise NotImplementedError
