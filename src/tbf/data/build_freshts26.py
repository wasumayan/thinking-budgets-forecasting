"""Build FreshTS-26 (SPEC §2.1) -> data/freshts26/{windows.parquet, series.parquet, shuffle_map.parquet, BUILD.json}

CLI: python -m tbf.data.build_freshts26 --series configs/series.yaml --out data/freshts26 [--tier2] [--end YYYY-MM-DD]

Algorithm:
 1. Load series.yaml. Materialize the series list: web (60 titles from clusters), health (regions), energy (ids),
    macro (daily_ids + weekly_ids). series.parquet columns: series_id (e.g. "web__Influenza", "health__hhs3",
    "energy__GASREGW", "macro__DEXUSEU"), domain, cluster, freq, seasonality, precision, title, units, keywords
    (list), event_categories (list), metadata_base (str), source, canonical_title/revid (web), dropped (bool), drop_reason.
 2. Fetch numbers (fetch_wikimedia.pageviews / fetch_fred.fred_series / fetch_delphi.fluview). Business-daily FRED:
    keep FRED's own observation dates (no reindexing to calendar days). Drop series with > max_missing_frac
    missing in [min_origin - T steps, end]; log.
 3. Fetch text: web summaries as-of 2025-07-31; FRED notes; all current events since 2025-06-01; tier-2 reports if
    --tier2. Metadata string per SPEC §2.1 item 1 (always ends with "Frequency: ... Prediction target period: ...",
    the last sentence is filled per window).
 4. Windows: for each series, origins o = every `stride` steps from the first index >= min_origin such that
    index o-95 exists and o+12 <= last observed; context = values[o-95..o] (forward-fill NaNs, count n_ffill),
    target = values[o+1..o+12] (any NaN -> drop window). window_id = f"{series_id}__{origin_date}".
 5. Text per window: calendar (holidays.US() in [t_{o+1}, t_{o+12}]); events = match_events(...) with lookback
    28 days, title (web only), keywords, categories, k=10; reports (tier 2) with available_at <= origin.
    text_available_at_max = max(available_at of attached items) (NaT if none). strict_2026 = origin >= strict_origin.
 6. shuffle_map.parquet: for each window, a window_id from a different domain chosen uniformly with
    numpy default_rng(0); the mapped window supplies events/reports in context_mode=shuffled.
 7. Assert for every window: text_available_at_max <= origin_ts (end of day) — raise if violated.
 8. BUILD.json: build_time, end date, per-domain n_series/n_windows, n_strict_2026, dropped series with reasons,
    fetch counts, n_windows_with_events, mean n_events, tier2 flag, code git sha.
Print a summary table at the end.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

from ..config import ROOT, load_yaml
from . import http
from .fetch_current_events import all_events, match_events
from .fetch_delphi import fluview
from .fetch_fred import fred_notes, fred_series
from .fetch_wikimedia import pageviews, summary_asof

log = logging.getLogger(__name__)
FREQ_DESC = {"D": "daily", "B": "business days", "W": "weekly"}
STATE_NAMES = {"ca": "California", "tx": "Texas", "ny": "New York", "il": "Illinois", "pa": "Pennsylvania", "oh": "Ohio",
               "ga": "Georgia", "nc": "North Carolina", "mi": "Michigan", "nj": "New Jersey", "va": "Virginia",
               "wa": "Washington", "az": "Arizona", "ma": "Massachusetts", "tn": "Tennessee", "in": "Indiana",
               "mo": "Missouri", "md": "Maryland", "wi": "Wisconsin", "co": "Colorado", "fl": "Florida", "mn": "Minnesota",
               "or": "Oregon", "sc": "South Carolina", "al": "Alabama", "la": "Louisiana", "ky": "Kentucky", "ok": "Oklahoma",
               "ct": "Connecticut", "ut": "Utah", "ia": "Iowa", "nv": "Nevada", "ar": "Arkansas", "ms": "Mississippi",
               "ks": "Kansas", "nm": "New Mexico", "ne": "Nebraska", "id": "Idaho", "wv": "West Virginia", "hi": "Hawaii",
               "nh": "New Hampshire", "me": "Maine", "ri": "Rhode Island", "mt": "Montana", "de": "Delaware", "sd": "South Dakota",
               "nd": "North Dakota", "ak": "Alaska", "dc": "District of Columbia", "vt": "Vermont", "wy": "Wyoming"}


# ------------------------------------------------------------------------------------------------------------
# 1. Series registry
# ------------------------------------------------------------------------------------------------------------

def _split_title_units(text: str) -> tuple[str, str]:
    if text.endswith(")") and " (" in text:
        i = text.rfind(" (")
        return text[:i].strip(), text[i + 2:-1].strip()
    return text.strip(), ""


def materialize_series(cfg: dict) -> list[dict]:
    d = cfg["defaults"]
    out = []
    for cluster, spec in cfg["web_clusters"].items():
        for title in spec["titles"]:
            out.append(dict(series_id=f"web__{title}", domain="web", cluster=cluster, freq="D", seasonality=d["seasonality"]["D"],
                            precision=0, title=title.replace("_", " "), units="daily user pageviews, English Wikipedia",
                            keywords=list(spec["keywords"]), event_categories=list(spec["event_categories"]),
                            source="wikimedia_pageviews", fetch_key=title, metadata_base=""))
    h = cfg["health"]
    for region in h["regions"]:
        name = h["region_names"].get(region) or STATE_NAMES.get(region, region.upper())
        out.append(dict(series_id=f"health__{region}", domain="health", cluster=None, freq=h["freq"], seasonality=d["seasonality"][h["freq"]],
                        precision=h["precision"], title=f"ILINet % ILI, {name}", units="percent of outpatient visits",
                        keywords=list(h["keywords"]), event_categories=list(h["event_categories"]), source=h["source"],
                        fetch_key=region, metadata_base=h["metadata_template"].format(region_name=name)))
    e = cfg["energy"]
    for sid, text in e["ids"].items():
        title, units = _split_title_units(text)
        out.append(dict(series_id=f"energy__{sid}", domain="energy", cluster=None, freq=e["freq"], seasonality=d["seasonality"][e["freq"]],
                        precision=e["precision"], title=title, units=units, keywords=list(e["keywords"]),
                        event_categories=list(e["event_categories"]), source=e["source"], fetch_key=sid, metadata_base=""))
    m = cfg["macro"]
    for block, freq in (("daily_ids", "B"), ("weekly_ids", "W")):
        for sid, spec in m[block].items():
            title, units = _split_title_units(spec["title"])
            group = m["keyword_group_by_id"][sid]
            out.append(dict(series_id=f"macro__{sid}", domain="macro", cluster=group, freq=freq, seasonality=d["seasonality"][freq],
                            precision=int(spec["precision"]), title=title, units=units, keywords=list(m["keywords"][group]),
                            event_categories=list(m["event_categories"]), source="fred_csv", fetch_key=sid, metadata_base=""))
    return out


# ------------------------------------------------------------------------------------------------------------
# 4–5. Windows (pure; unit-tested on synthetic series)
# ------------------------------------------------------------------------------------------------------------

def us_holidays_between(start: dt.date, end: dt.date) -> str:
    import holidays

    us = holidays.US(years=range(start.year, end.year + 1))
    items = [f"{day.isoformat()} {name}" for day, name in sorted(us.items()) if start <= day <= end]
    return ", ".join(items) if items else "None"


def make_windows(meta: dict, values: pd.Series, cfg: dict, events: list[dict], reports: list[dict] | None = None,
                 end: dt.date | None = None) -> list[dict]:
    """Rolling windows for one series. `values` is indexed by date (NaN allowed), sorted ascending."""
    d = cfg["defaults"]
    T, H = int(d["T"]), int(d["H"])
    stride = int(d["stride"][meta["freq"]])
    min_origin = pd.Timestamp(d["min_origin"]).date()
    strict_origin = pd.Timestamp(d["strict_origin"]).date()
    s = values.sort_index()
    if end is not None:
        s = s[s.index.date <= end]
    dates = [ts.date() for ts in s.index]
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    first = next((i for i, dd in enumerate(dates) if dd >= min_origin and i - (T - 1) >= 0), None)
    if first is None:
        return []
    freq_desc = "weekly, week ending Saturday" if meta["domain"] == "health" else FREQ_DESC[meta["freq"]]
    out = []
    for o in range(first, n, stride):
        if o + H > n - 1:
            break
        ctx = vals[o - (T - 1): o + 1].copy()
        tgt = vals[o + 1: o + 1 + H]
        if np.isnan(tgt).any() or np.isnan(ctx[0]):
            continue
        n_ffill = 0
        for i in range(1, T):
            if np.isnan(ctx[i]):
                ctx[i] = ctx[i - 1]
                n_ffill += 1
        origin = dates[o]
        ctx_ts, tgt_ts = dates[o - (T - 1): o + 1], dates[o + 1: o + 1 + H]
        h_start, h_end = tgt_ts[0].isoformat(), tgt_ts[-1].isoformat()
        ev = match_events(events, origin, int(d["events_lookback_days"]), title=meta.get("canonical_title") if meta["domain"] == "web" else None,
                          keywords=meta["keywords"], categories=None if meta["domain"] == "web" else meta["event_categories"],
                          k=int(d["events_max"]))
        origin_end = f"{origin.isoformat()}T23:59:59Z"
        rep = sorted([r for r in (reports or []) if r["available_at"] <= origin_end], key=lambda r: r["available_at"], reverse=True)[:2]
        avail = [x["available_at"] for x in ev] + [x["available_at"] for x in rep]
        metadata = (meta["metadata_base"].strip() + " " if meta["metadata_base"].strip() else "") + \
                   f"Frequency: {freq_desc}. Prediction target period: {h_start} to {h_end}."
        out.append({
            "window_id": f"{meta['series_id']}__{origin.isoformat()}", "series_id": meta["series_id"], "domain": meta["domain"],
            "cluster": meta.get("cluster"), "freq": meta["freq"], "seasonality": int(meta["seasonality"]),
            "origin_ts": pd.Timestamp(origin, tz="UTC"),
            "context_ts": [pd.Timestamp(x, tz="UTC") for x in ctx_ts], "context": [float(x) for x in ctx],
            "target_ts": [pd.Timestamp(x, tz="UTC") for x in tgt_ts], "target": [float(x) for x in tgt],
            "n_ffill": n_ffill, "metadata": metadata, "calendar": us_holidays_between(tgt_ts[0], tgt_ts[-1]),
            "events": ev, "reports": rep, "n_events": len(ev),
            "text_available_at_max": pd.Timestamp(max(avail).replace("Z", "+00:00")) if avail else pd.NaT,
            "strict_2026": origin >= strict_origin,
            "title": meta["title"], "units": meta["units"], "precision": int(meta["precision"]),
        })
    return out


def make_shuffle_map(windows: list[dict], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = sorted(w["window_id"] for w in windows)
    dom = {w["window_id"]: w["domain"] for w in windows}
    by_dom = {}
    for wid in ids:
        by_dom.setdefault(dom[wid], []).append(wid)
    rows = []
    for wid in ids:
        others = [x for d, xs in sorted(by_dom.items()) if d != dom[wid] for x in xs]
        if not others:
            raise ValueError("shuffle map needs at least two domains")
        rows.append({"window_id": wid, "source_window_id": others[int(rng.integers(0, len(others)))]})
    return pd.DataFrame(rows)


def assert_no_leakage(windows: list[dict]) -> None:
    for w in windows:
        limit = w["origin_ts"] + pd.Timedelta(hours=23, minutes=59, seconds=59)
        for item in list(w["events"]) + list(w["reports"]):
            a = pd.Timestamp(item["available_at"].replace("Z", "+00:00"))
            if a > limit:
                raise AssertionError(f"leakage: {w['window_id']} item available_at {a} > origin {limit}")
        if w["text_available_at_max"] is not pd.NaT and pd.notna(w["text_available_at_max"]) and w["text_available_at_max"] > limit:
            raise AssertionError(f"leakage: {w['window_id']} text_available_at_max {w['text_available_at_max']} > {limit}")


# ------------------------------------------------------------------------------------------------------------
# 2–3. Fetching (network; cached)
# ------------------------------------------------------------------------------------------------------------

def _missing_frac(s: pd.Series, freq: str, T: int, min_origin: str, end: dt.date, n_missing_days: int = 0) -> float:
    start = pd.Timestamp(min_origin) - (pd.Timedelta(days=T) if freq in ("D", "B") else pd.Timedelta(weeks=T))
    win = s[(s.index >= start) & (s.index.date <= end)]
    if len(win) == 0:
        return 1.0
    return float(win.isna().sum() + (n_missing_days if freq == "D" else 0)) / max(len(win), 1)


def fetch_numbers(series: list[dict], cfg: dict, end: dt.date) -> dict[str, pd.Series]:
    d = cfg["defaults"]
    numbers: dict[str, pd.Series] = {}
    health = [m for m in series if m["domain"] == "health"]
    if health:
        try:
            fv = fluview([m["fetch_key"] for m in health])
            for m in health:
                sub = fv[fv.region == m["fetch_key"]]
                if sub.empty:
                    m.update(dropped=True, drop_reason=fv.attrs.get("dropped", {}).get(m["fetch_key"], "no rows"))
                    continue
                numbers[m["series_id"]] = pd.Series(sub["ili"].to_numpy(float), index=pd.to_datetime(sub["week_ending"]))
        except Exception as e:  # noqa: BLE001
            log.error("fluview fetch failed: %s", e)
            for m in health:
                m.update(dropped=True, drop_reason=f"fetch failed: {e}")
    for m in series:
        if m["domain"] == "health" or m.get("dropped"):
            continue
        try:
            if m["source"] == "wikimedia_pageviews":
                s = pageviews(m["fetch_key"], start=cfg["sources"]["wikimedia_pageviews"]["start"], end=end.strftime("%Y%m%d"))
                m["canonical_title"] = s.attrs["canonical_title"]
                n_missing = s.attrs["n_missing_days"]
            else:
                s = fred_series(m["fetch_key"])
                n_missing = 0
            frac = _missing_frac(s, m["freq"], int(d["T"]), d["min_origin"], end, n_missing)
            if frac > float(d["max_missing_frac"]):
                m.update(dropped=True, drop_reason=f"{frac:.1%} missing in evaluation range")
                log.warning("dropping %s: %s", m["series_id"], m["drop_reason"])
                continue
            numbers[m["series_id"]] = s
        except Exception as e:  # noqa: BLE001
            log.error("fetch failed for %s: %s", m["series_id"], e)
            m.update(dropped=True, drop_reason=f"fetch failed: {e}")
    return numbers


def fetch_text(series: list[dict], cfg: dict) -> None:
    for m in series:
        if m.get("dropped"):
            continue
        try:
            if m["domain"] == "web":
                text, revid, asof = summary_asof(m.get("canonical_title") or m["fetch_key"])
                m.update(metadata_base=text, revid=revid, summary_asof=asof)
            elif m["source"] == "fred_csv":
                n = fred_notes(m["fetch_key"])
                title = n.get("title") or m["title"]
                units = n.get("units") or m["units"]
                m["units"] = units or m["units"]
                parts = [f"{title}.", f"Units: {units}." if units else "", f"Frequency: {n['frequency']}." if n.get("frequency") else "",
                         f"Seasonal adjustment: {n['seasonal_adjustment']}." if n.get("seasonal_adjustment") else "",
                         n.get("notes", "")]
                m["metadata_base"] = " ".join(p for p in parts if p).strip()
        except Exception as e:  # noqa: BLE001
            log.warning("text fetch failed for %s (%s); using series.yaml title", m["series_id"], e)
            m["metadata_base"] = m["metadata_base"] or f"{m['title']} ({m['units']})."
            m["summary_asof"] = False


def fetch_reports(cfg: dict, end: dt.date) -> dict[str, list[dict]]:
    from .fetch_cdc_fluview import key_points
    from .fetch_delphi import date_to_epiweek, epiweek_to_saturday
    from .fetch_eia_wpsr import highlights, release_dates

    health, energy = [], []
    ew = date_to_epiweek(dt.date.fromisoformat(cfg["sources"]["current_events"]["start"]))
    while epiweek_to_saturday(ew) <= end:
        r = key_points(ew // 100, ew % 100)
        if r:
            health.append(r)
        ew = date_to_epiweek(epiweek_to_saturday(ew) + dt.timedelta(days=7))
    for rd in release_dates(cfg["sources"]["current_events"]["start"], end.isoformat()):
        r = highlights(rd)
        if r:
            energy.append(r)
    log.info("tier-2 reports: %d CDC FluView, %d EIA WPSR", len(health), len(energy))
    return {"health": health, "energy": energy}


# ------------------------------------------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build(series_yaml: str, out_dir: str, tier2: bool = False, end: str | None = None) -> dict:
    cfg = load_yaml(series_yaml)
    end_d = dt.date.fromisoformat(end) if end else dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = dt.datetime.now(dt.timezone.utc)

    series = materialize_series(cfg)
    for m in series:
        m.setdefault("dropped", False)
        m.setdefault("drop_reason", "")
    log.info("registry: %d series", len(series))
    numbers = fetch_numbers(series, cfg, end_d)
    fetch_text(series, cfg)
    events = all_events(cfg["sources"]["current_events"]["start"], end_d.isoformat())
    reports = fetch_reports(cfg, end_d) if tier2 else {}

    windows: list[dict] = []
    for m in series:
        if m.get("dropped"):
            continue
        ws = make_windows(m, numbers[m["series_id"]], cfg, events, reports.get(m["domain"], []), end_d)
        if not ws:
            m.update(dropped=True, drop_reason="no valid windows")
            log.warning("dropping %s: no valid windows", m["series_id"])
        windows += ws
    if not windows:
        raise SystemExit("no windows built")
    windows.sort(key=lambda w: w["window_id"])
    assert_no_leakage(windows)

    wdf = pd.DataFrame(windows)
    wdf.to_parquet(out / "windows.parquet", index=False)
    scols = ["series_id", "domain", "cluster", "freq", "seasonality", "precision", "title", "units", "keywords", "event_categories",
             "metadata_base", "source", "canonical_title", "revid", "summary_asof", "dropped", "drop_reason"]
    sdf = pd.DataFrame([{k: m.get(k) for k in scols} for m in series])
    sdf.to_parquet(out / "series.parquet", index=False)
    smap = make_shuffle_map(windows)
    smap.to_parquet(out / "shuffle_map.parquet", index=False)

    per_domain = {}
    for dom in sorted(wdf.domain.unique()):
        sub = wdf[wdf.domain == dom]
        per_domain[dom] = {"n_series": int(sub.series_id.nunique()), "n_windows": int(len(sub)),
                           "n_windows_with_events": int((sub.n_events > 0).sum()), "mean_n_events": float(sub.n_events.mean())}
    info = {
        "build_time": t0.isoformat(timespec="seconds"), "end": end_d.isoformat(), "git_sha": _git_sha(), "tier2": tier2,
        "n_series": int(wdf.series_id.nunique()), "n_windows": int(len(wdf)), "n_strict_2026": int(wdf.strict_2026.sum()),
        "per_domain": per_domain, "n_windows_with_events": int((wdf.n_events > 0).sum()), "mean_n_events": float(wdf.n_events.mean()),
        "n_events_total": len(events), "dropped": {m["series_id"]: m["drop_reason"] for m in series if m.get("dropped")},
        "web_summary_not_asof": [m["series_id"] for m in series if m["domain"] == "web" and m.get("summary_asof") is False],
        "fetch_counts": http.fetch_stats(),
        "text_available_at_max": str(wdf.text_available_at_max.max()),
    }
    (out / "BUILD.json").write_text(json.dumps(info, indent=1, default=str))
    log.info("BUILD summary:\n%s", pd.DataFrame(per_domain).T.to_string())
    log.info("total: %d windows, %d series, %d strict_2026, %d dropped series -> %s",
             info["n_windows"], info["n_series"], info["n_strict_2026"], len(info["dropped"]), out)
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=str(ROOT / "configs" / "series.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "freshts26"))
    ap.add_argument("--tier2", action="store_true")
    ap.add_argument("--end", default=None, help="last observed date (YYYY-MM-DD); default yesterday UTC")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build(a.series, a.out, a.tier2, a.end)


if __name__ == "__main__":
    main()
