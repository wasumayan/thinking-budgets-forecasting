"""Leakage audit: every text item attached to a window must have available_at <= origin (end of day UTC).
Runs on fixtures always; on data/freshts26/windows.parquet when it exists (make build-data)."""
import copy
import datetime as dt
import pathlib

import pytest

from tbf.data.fixtures import load_windows

REAL = pathlib.Path(__file__).resolve().parents[1] / "data" / "freshts26" / "windows.parquet"


def _origin_end_of_day(w):
    o = dt.datetime.fromisoformat(str(w["origin_ts"]).replace("Z", "+00:00"))
    if o.tzinfo is None:
        o = o.replace(tzinfo=dt.timezone.utc)
    return o.replace(hour=23, minute=59, second=59)


def check_window(w):
    limit = _origin_end_of_day(w)
    for item in list(w.get("events") or []) + list(w.get("reports") or []):
        a = dt.datetime.fromisoformat(str(item["available_at"]).replace("Z", "+00:00"))
        if a.tzinfo is None:
            a = a.replace(tzinfo=dt.timezone.utc)
        assert a <= limit, f"{w['window_id']}: item available_at {a} > origin {limit}"
    assert len(w["context"]) == 96 and len(w["target"]) == 12
    assert all(isinstance(v, (int, float)) for v in w["context"] + w["target"])


def test_fixtures_clean():
    for w in load_windows():
        check_window(w)


def test_corrupted_fixture_fails():
    w = copy.deepcopy(load_windows()[0])
    w["events"] = [{"date": "2099-01-01", "category": "Sports", "text": "future", "links": [], "available_at": "2099-01-01T23:59:59Z"}]
    with pytest.raises(AssertionError):
        check_window(w)


@pytest.mark.skipif(not REAL.exists(), reason="real dataset not built")
def test_real_dataset_clean():
    import pandas as pd
    df = pd.read_parquet(REAL)
    assert len(df) >= 1200 and df.series_id.nunique() >= 100 and df.domain.nunique() >= 3
    for w in df.to_dict("records"):
        check_window(w)
    assert (df.origin_ts >= pd.Timestamp("2025-08-01", tz="UTC")).all()
