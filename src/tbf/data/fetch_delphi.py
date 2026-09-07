"""Delphi Epidata fluview (SPEC §2.1 health).

fluview(regions: list[str], epiweeks="202301-<current>") -> pd.DataFrame[region, epiweek, week_ending(date), ili, num_patients]
  One request for all regions (comma-separated). Field 'ili' (state-level available); week_ending = Saturday of the
  epiweek (MMWR week) — implement epiweek->date conversion and unit-test it against 202634 -> 2026-08-29.
  Final values (no issues= parameter). Drop regions with > 5 % missing in 2025-08..now and log them.

If the single request does not return result == 1 (e.g. Delphi's row cap), the fetch falls back to one request
per region. Dropped regions are listed in the DataFrame's .attrs["dropped"] = {region: reason}.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from ..config import load_series_config
from . import http

log = logging.getLogger(__name__)


def epiweek_to_saturday(epiweek: int) -> dt.date:
    """202634 -> date(2026, 8, 29). MMWR weeks end on Saturday; week 1 contains Jan 4."""
    year, week = divmod(int(epiweek), 100)
    jan4 = dt.date(year, 1, 4)
    week1_sunday = jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)
    return week1_sunday + dt.timedelta(weeks=week - 1, days=6)


def date_to_epiweek(d: dt.date) -> int:
    """Inverse of epiweek_to_saturday for any day (MMWR week containing d)."""
    for year in (d.year + 1, d.year, d.year - 1):
        jan4 = dt.date(year, 1, 4)
        week1_sunday = jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)
        if d >= week1_sunday:
            return year * 100 + (d - week1_sunday).days // 7 + 1
    raise ValueError(d)


def current_epiweek() -> int:
    return date_to_epiweek(dt.datetime.now(dt.timezone.utc).date())


def _fetch(regions: list[str], epiweeks: str) -> list[dict]:
    cfg = load_series_config()["sources"]["delphi_fluview"]
    last = epiweeks.split("-")[-1]
    url = cfg["url"].format(regions=",".join(regions), last_epiweek=last)
    url = url.replace("epiweeks=202301-", f"epiweeks={epiweeks.split('-')[0]}-")
    key = f"delphi_fluview/{'_'.join(regions)[:100]}_{epiweeks}.json"
    data, _ = http.get(url, cache_key=key)
    if not isinstance(data, dict) or data.get("result") != 1:
        raise RuntimeError(f"Delphi fluview returned result={data.get('result') if isinstance(data, dict) else data!r}: "
                           f"{data.get('message') if isinstance(data, dict) else ''}")
    return data.get("epidata", [])


def fluview(regions: list[str], epiweeks: str | None = None) -> pd.DataFrame:
    cfg = load_series_config()["sources"]["delphi_fluview"]
    field = cfg.get("value_field", "ili")
    epiweeks = epiweeks or f"202301-{current_epiweek()}"
    try:
        rows = _fetch(list(regions), epiweeks)
    except Exception as e:  # noqa: BLE001
        log.warning("single fluview request failed (%s); falling back to per-region requests", e)
        rows = []
        for r in regions:
            try:
                rows += _fetch([r], epiweeks)
            except Exception as e2:  # noqa: BLE001
                log.error("fluview %s failed: %s", r, e2)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Delphi fluview returned no rows")
    df = df[["region", "epiweek", field, "num_patients"]].rename(columns={field: "ili"})
    df["epiweek"] = df["epiweek"].astype(int)
    df["ili"] = pd.to_numeric(df["ili"], errors="coerce")
    df["week_ending"] = df["epiweek"].map(epiweek_to_saturday)
    df = df.sort_values(["region", "epiweek"]).reset_index(drop=True)

    # missingness check over 2025-08 .. now
    start_ew, end_ew = 202531, int(epiweeks.split("-")[-1])
    expected = [ew for ew in _epiweek_range(start_ew, end_ew)]
    dropped = {}
    for region in list(regions):
        have = set(df[(df.region == region) & df.ili.notna()].epiweek)
        missing = [ew for ew in expected if ew not in have]
        frac = len(missing) / max(len(expected), 1)
        if frac > 0.05:
            dropped[region] = f"{len(missing)}/{len(expected)} epiweeks missing ({frac:.1%})"
            log.warning("fluview: dropping region %s: %s", region, dropped[region])
    df = df[~df.region.isin(dropped)].reset_index(drop=True)
    df.attrs = {"dropped": dropped, "epiweeks": epiweeks}
    return df


def _epiweek_range(start: int, end: int):
    d = epiweek_to_saturday(start)
    while True:
        ew = date_to_epiweek(d)
        if ew > end:
            return
        yield ew
        d += dt.timedelta(days=7)
