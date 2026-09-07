"""Offline tests for the FreshTS-26 builder (windowing, calendar, shuffle map, leakage assertion, parquet
round trip) on synthetic series, plus CiK rendering / RCRPS on synthetic rows."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tbf.config import load_series_config
from tbf.data import build_freshts26 as B
from tbf.data import windows as WIN
from tbf.data.cik import crps_pwm, rcrps_one
from tbf.prompts import render_messages


def _meta(**kw):
    m = dict(series_id="web__Influenza", domain="web", cluster="health", freq="D", seasonality=7, precision=0, title="Influenza",
             units="daily user pageviews, English Wikipedia", keywords=["flu", "influenza"], event_categories=["Health and environment"],
             source="wikimedia_pageviews", fetch_key="Influenza", metadata_base="Influenza is an infectious disease.",
             canonical_title="Influenza")
    m.update(kw)
    return m


def _events():
    return [{"date": "2025-08-20", "category": "Health and environment", "text": "Flu cases rise.", "links": [], "available_at": "2025-08-20T23:59:59Z"},
            {"date": "2025-08-30", "category": "Sports", "text": "A match about Influenza (link).", "links": ["Influenza"], "available_at": "2025-08-30T23:59:59Z"},
            {"date": "2025-09-10", "category": "Business and economy", "text": "flu vaccine maker profits", "links": [], "available_at": "2025-09-10T23:59:59Z"}]


def test_daily_windows_stride_ffill_and_events():
    cfg = load_series_config()
    idx = pd.date_range("2025-04-01", "2025-12-31", freq="D")
    vals = np.arange(len(idx), dtype=float)
    vals[100] = np.nan  # 2025-07-10: inside the first window's context
    s = pd.Series(vals, index=idx)
    ws = B.make_windows(_meta(), s, cfg, _events(), end=dt.date(2025, 12, 31))
    assert ws, "no windows"
    origins = [w["origin_ts"].date() for w in ws]
    assert origins[0] == dt.date(2025, 8, 1) and (origins[1] - origins[0]).days == 24
    for w in ws:
        assert len(w["context"]) == 96 and len(w["target"]) == 12
        assert w["context_ts"][-1].date() == w["origin_ts"].date() and (w["target_ts"][0] - w["context_ts"][-1]).days == 1
        assert w["metadata"].endswith(f"Prediction target period: {w['target_ts'][0].date()} to {w['target_ts'][-1].date()}.")
        assert "Frequency: daily." in w["metadata"]
    # window whose context contains the NaN has n_ffill == 1 and a forward-filled value
    nan_day = idx[100].date()
    hit = [w for w in ws if w["context_ts"][0].date() <= nan_day <= w["context_ts"][-1].date()]
    assert hit and all(w["n_ffill"] == 1 for w in hit)
    # events: web series = title link OR keyword, no category filter, 28-day lookback, newest first
    w0 = ws[0]  # origin 2025-08-01: only the 08-20 event is after the origin -> none before it
    assert w0["n_events"] == 0
    w1 = ws[1]  # origin 2025-08-25: 08-20 (keyword) in window; 08-30 not yet
    assert [e["date"] for e in w1["events"]] == ["2025-08-20"]
    w2 = ws[2]  # origin 2025-09-18: 09-10 (keyword) and 08-30 (title link, category Sports still kept for web)
    assert [e["date"] for e in w2["events"]] == ["2025-09-10", "2025-08-30"]
    assert w2["text_available_at_max"] == pd.Timestamp("2025-09-10T23:59:59Z")
    assert all(w["text_available_at_max"] <= w["origin_ts"] + pd.Timedelta(days=1) for w in ws if pd.notna(w["text_available_at_max"]))
    B.assert_no_leakage(ws)


def test_non_web_needs_category_and_keyword():
    cfg = load_series_config()
    idx = pd.date_range("2025-01-04", periods=120, freq="7D")  # Saturdays
    s = pd.Series(np.linspace(1, 3, len(idx)), index=idx)
    meta = _meta(series_id="health__nat", domain="health", freq="W", seasonality=1, precision=2, keywords=["flu"],
                 event_categories=["Health and environment"], canonical_title=None, metadata_base="ILI nat.")
    ws = B.make_windows(meta, s, cfg, _events(), end=dt.date(2027, 6, 1))
    assert ws and all("Frequency: weekly, week ending Saturday." in w["metadata"] for w in ws)
    assert (ws[1]["origin_ts"] - ws[0]["origin_ts"]).days == 28  # stride 4 weeks
    w = [w for w in ws if w["origin_ts"].date() >= dt.date(2025, 9, 13)][0]
    assert [e["date"] for e in w["events"]] == ["2025-09-10"] or w["n_events"] == 0  # 'flu vaccine' is Business -> excluded
    for w in ws:
        assert all(e["category"] in meta["event_categories"] for e in w["events"])


def test_nan_target_drops_window_and_business_days():
    cfg = load_series_config()
    idx = pd.bdate_range("2025-03-03", "2025-12-31")
    vals = np.random.default_rng(0).normal(size=len(idx)) + 100
    s = pd.Series(vals, index=idx)
    meta = _meta(series_id="macro__DEXUSEU", domain="macro", freq="B", seasonality=1, precision=4, canonical_title=None)
    ws_ok = B.make_windows(meta, s, cfg, [], end=dt.date(2025, 12, 31))
    assert ws_ok and all(t.weekday() < 5 for w in ws_ok for t in w["context_ts"] + w["target_ts"])
    first_origin = ws_ok[0]["origin_ts"].date()
    pos = list(idx.date).index(first_origin)
    s2 = s.copy()
    s2.iloc[pos + 3] = np.nan  # in the first window's target
    ws_drop = B.make_windows(meta, s2, cfg, [], end=dt.date(2025, 12, 31))
    assert len(ws_drop) == len(ws_ok) - 1 and ws_drop[0]["origin_ts"] == ws_ok[1]["origin_ts"]


def test_calendar_and_reports_available_at():
    assert B.us_holidays_between(dt.date(2025, 11, 20), dt.date(2025, 11, 30)) == "2025-11-27 Thanksgiving Day"
    assert B.us_holidays_between(dt.date(2025, 8, 5), dt.date(2025, 8, 20)) == "None"
    cfg = load_series_config()
    idx = pd.date_range("2023-10-07", periods=160, freq="7D")  # Saturdays; index 95 = 2025-08-02
    s = pd.Series(np.linspace(1, 3, len(idx)), index=idx)
    meta = _meta(series_id="health__nat", domain="health", freq="W", seasonality=1, precision=2, canonical_title=None)
    reports = [{"date": "2025-08-02", "source": "CDC FluView", "text": "low", "available_at": "2025-08-08T21:00:00Z"},
               {"date": "2025-08-09", "source": "CDC FluView", "text": "still low", "available_at": "2025-08-15T21:00:00Z"},
               {"date": "2025-08-30", "source": "CDC FluView", "text": "future", "available_at": "2025-09-05T21:00:00Z"}]
    ws = B.make_windows(meta, s, cfg, [], reports, end=dt.date(2027, 6, 1))
    assert ws[0]["origin_ts"].date() == dt.date(2025, 8, 2)
    w = [w for w in ws if w["origin_ts"].date() == dt.date(2025, 8, 30)]
    assert w, [x["origin_ts"] for x in ws][:3]
    assert [r["text"] for r in w[0]["reports"]] == ["still low", "low"]  # <= 2 most recent with available_at <= origin
    B.assert_no_leakage(ws)
    w[0]["reports"].append({"date": "2025-09-01", "source": "x", "text": "leak", "available_at": "2025-09-05T00:00:00Z"})
    with pytest.raises(AssertionError):
        B.assert_no_leakage(ws)


def test_shuffle_map_and_parquet_roundtrip(tmp_path, monkeypatch):
    cfg = load_series_config()
    idx = pd.date_range("2025-04-01", "2025-12-31", freq="D")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    ws = B.make_windows(_meta(), s, cfg, _events(), end=dt.date(2025, 12, 31))
    ws += B.make_windows(_meta(series_id="web__Measles", title="Measles", canonical_title="Measles"), s, cfg, [], end=dt.date(2025, 12, 31))
    idx_w = pd.date_range("2025-01-04", periods=120, freq="7D")
    ws += B.make_windows(_meta(series_id="health__nat", domain="health", freq="W", seasonality=1, precision=2, canonical_title=None),
                         pd.Series(np.linspace(1, 3, 120), index=idx_w), cfg, [], end=dt.date(2027, 6, 1))
    smap = B.make_shuffle_map(ws)
    dom = {w["window_id"]: w["domain"] for w in ws}
    assert len(smap) == len(ws) and all(dom[a] != dom[b] for a, b in zip(smap.window_id, smap.source_window_id))
    assert smap.equals(B.make_shuffle_map(ws))  # deterministic (seed 0)
    out = tmp_path / "freshts26"
    out.mkdir()
    pd.DataFrame(ws).to_parquet(out / "windows.parquet", index=False)
    smap.to_parquet(out / "shuffle_map.parquet", index=False)
    monkeypatch.setattr(WIN, "FRESHTS_DIR", out)
    loaded = WIN.load_windows("freshts26")
    assert len(loaded) == len(ws) and loaded[0]["window_id"] == sorted(w["window_id"] for w in ws)[0]
    w = [x for x in loaded if x["series_id"] == "web__Influenza"][0]
    assert len(w["context"]) == 96 and isinstance(w["context"][0], float) and w["context_ts"][0].count("-") == 2 and len(w["context_ts"][0]) == 10
    assert isinstance(w["events"], list) and w["origin_ts"].startswith("2025-")
    msgs = render_messages(w, "direct", "full")
    assert "## History (96 values" in msgs[1]["content"]
    df = pd.read_parquet(out / "windows.parquet")
    assert (df.origin_ts >= pd.Timestamp("2025-08-01", tz="UTC")).all()
    from tests.test_leakage import check_window
    for r in loaded:
        check_window(r)


def test_cik_rendering_and_rcrps():
    w = {"window_id": "cik__T__1", "series_id": "cik__T", "dataset": "cik", "domain": "cik", "freq": "hourly", "seasonality": 24,
         "precision": 4, "title": "T", "units": "y", "origin_ts": "2024-01-05 23:00",
         "context_ts": [f"2024-01-0{1 + i // 24} {i % 24:02d}:00" for i in range(120)], "context": [float(i) for i in range(120)],
         "target_ts": [f"2024-01-06 {i:02d}:00" for i in range(24)], "target": [1.0] * 24, "metadata": "Background:\nA sensor.\n\nScenario:\nMaintenance.",
         "calendar": "None", "events": [], "reports": [], "n_events": 0, "strict_2026": False,
         "metric_scaling": 0.5, "region_of_interest": [0, 1], "constraint_min": 0.0, "constraint_max": None,
         "constraint_variable_max_index": [], "constraint_variable_max_values": [], "weight": "1/3"}
    text = render_messages(w, "direct", "full")[1]["content"]
    assert "## Calendar" not in text and "## Recent events" not in text
    assert "## History (120 values" in text and "Forecast the next 24 values" in text and "v24]" in text and "containing 24 numbers" in text
    assert "Background:\nA sensor." in text and "2024-01-01 00:00: 0.0000" in text and "(hourly)" in text
    assert "Prediction target period: 2024-01-06 00:00 to 2024-01-06 23:00." in text
    # CRPS-PWM: degenerate ensemble equal to the truth -> 0; constant ensemble -> |x-y|
    assert crps_pwm(np.array([1.0, 2.0]), np.array([[1.0, 2.0]] * 5)) == pytest.approx([0.0, 0.0])
    assert crps_pwm(np.array([1.0]), np.array([[3.0]] * 4))[0] == pytest.approx(2.0)
    r = rcrps_one(w, np.ones((5, 24)))
    assert r["metric"] == pytest.approx(0.0) and r["violation_crps"] == 0.0
    r2 = rcrps_one(w, -np.ones((5, 24)))  # violates constraint_min=0 by 1 everywhere -> scaled viol 0.5 * 10 = 5 penalty
    assert r2["violation_crps"] == pytest.approx(5.0) and r2["metric"] == pytest.approx(0.5 * 2.0 + 5.0)
