"""Offline tests for the data layer: parsers and the cached/polite HTTP helper (network mocked)."""
import datetime as dt
import json
import types

import pytest

from tbf.data import fetch_current_events as CE
from tbf.data import fetch_delphi as D
from tbf.data import fetch_fred as F
from tbf.data import fetch_wikimedia as W
from tbf.data import http as H
from tbf.data.fetch_cdc_fluview import available_at_friday_next_week, parse_key_points
from tbf.data.fetch_eia_wpsr import first_paragraphs, release_dates


def test_epiweek_to_saturday():
    assert D.epiweek_to_saturday(202634) == dt.date(2026, 8, 29)
    assert D.epiweek_to_saturday(202501) == dt.date(2025, 1, 4)
    assert D.epiweek_to_saturday(202552) == dt.date(2025, 12, 27)
    for ew in (202501, 202531, 202552, 202601, 202634):
        assert D.date_to_epiweek(D.epiweek_to_saturday(ew)) == ew
    assert list(D._epiweek_range(202551, 202602)) == [202551, 202552, 202553, 202601, 202602]  # 2025 has 53 MMWR weeks


def test_fred_csv_parse():
    s = F.parse_fred_csv("observation_date,DEXUSEU\n2025-08-01,1.1584\n2025-08-04,.\n2025-08-05,1.1567\n", "DEXUSEU")
    assert len(s) == 3 and s.isna().sum() == 1 and s.iloc[0] == pytest.approx(1.1584)
    assert str(s.index[0].date()) == "2025-08-01"


def test_fred_page_parse():
    html = """<html><head><title>U.S. Dollars to Euro Spot Exchange Rate (DEXUSEU) | FRED | St. Louis Fed</title></head>
    <body><span id="series-title-text-container">U.S. Dollars to Euro Spot Exchange Rate</span>
    <p>Units: U.S. Dollars to One Euro, Not Seasonally Adjusted</p><p>Frequency: Daily</p>
    <h2>Notes</h2><p>Source: Board of Governors</p><p>Noon buying rates in New York City for cable transfers payable in foreign currencies.</p>
    <h2>Suggested Citation</h2><p>Board of Governors...</p></body></html>"""
    d = F.parse_fred_page(html, "DEXUSEU")
    assert d["title"] == "U.S. Dollars to Euro Spot Exchange Rate"
    assert d["units"].startswith("U.S. Dollars to One Euro") and d["seasonal_adjustment"] == "Not Seasonally Adjusted"
    assert d["frequency"] == "Daily" and "Noon buying rates" in d["notes"] and "Suggested" not in d["notes"]


WIKITEXT = """{{Current events header|2025|08|3}}
<!-- All news items below this line -->
;Armed conflicts and attacks
*[[Gaza war]]
**[[Israel–Hamas war]]
***Israeli strikes kill at least 20 people in [[Gaza City]], health officials say. [https://example.org/a (Reuters)]
;Business and economy
*The [[Federal Reserve]] holds its {{nowrap|benchmark}} rate steady, citing '''inflation''' risks.<ref>x</ref> [https://example.org/b (AP)]
*[[OPEC]]+ agrees to raise oil output by 548,000 barrels per day. [https://example.org/c (BBC News)]
;Health and environment
*The [[World Health Organization]] reports a new [[Ebola]] outbreak in the [[Democratic Republic of the Congo]]. [https://x (WHO)]
;Made-up category
*Something else happens. [https://x (AP)]
"""


def test_parse_wikitext_bullets():
    ev = CE.parse_wikitext(WIKITEXT, dt.date(2025, 8, 3))
    texts = [e["text"] for e in ev]
    assert texts[0] == "Israeli strikes kill at least 20 people in Gaza City, health officials say."
    assert ev[0]["category"] == "Armed conflicts and attacks"
    assert "Gaza_City" in ev[0]["links"] and "Gaza_war" in ev[0]["links"] and "Israel–Hamas_war" in ev[0]["links"]
    assert texts[1] == "The Federal Reserve holds its rate steady, citing inflation risks."
    assert ev[1]["links"] == ["Federal_Reserve"]
    assert texts[2].startswith("OPEC+ agrees") and ev[2]["links"] == ["OPEC"]
    assert ev[3]["category"] == "Health and environment" and "Ebola" in ev[3]["links"]
    assert ev[4]["category"] == "Other"
    assert all(e["available_at"] == "2025-08-03T23:59:59Z" for e in ev)
    assert len(ev) == 5  # topic-heading lines skipped


def test_match_events_rules():
    ev = CE.parse_wikitext(WIKITEXT, dt.date(2025, 8, 3))
    origin = dt.date(2025, 8, 10)
    # title match via (inherited) link
    m = CE.match_events(ev, origin, 28, title="Gaza_war")
    assert len(m) == 1 and "Gaza" in m[0]["text"]
    # short keyword needs word boundary: 'fed' must not match 'federal'; 'opec' (4 chars) substring-matches 'OPEC+'
    assert CE.match_events(ev, origin, 28, keywords=["fed"]) == []
    assert len(CE.match_events(ev, origin, 28, keywords=["opec"])) == 1
    assert len(CE.match_events(ev, origin, 28, keywords=["federal reserve"])) == 1
    # category filter
    assert CE.match_events(ev, origin, 28, keywords=["ebola"], categories=["Sports"]) == []
    assert len(CE.match_events(ev, origin, 28, keywords=["ebola"], categories=["Health and environment"])) == 1
    # lookback window: origin-lookback < date <= origin
    assert CE.match_events(ev, dt.date(2025, 8, 2), 28, keywords=["ebola"]) == []
    assert CE.match_events(ev, dt.date(2025, 8, 31), 28, keywords=["ebola"]) == []
    assert len(CE.match_events(ev, dt.date(2025, 8, 30), 28, keywords=["ebola"])) == 1
    assert len(CE.match_events(ev, origin, 28, keywords=["a"], k=2)) <= 2


def test_lead_paragraphs():
    html = ('<div><table class="infobox"><tr><td><p>short</p></td></tr></table><p class="mw-empty-elt"></p>'
            '<p><b>Influenza</b>, commonly known as the flu, is an infectious disease caused by influenza viruses.'
            '<sup class="reference">[1]</sup> Symptoms range from mild to severe.</p><p>Second paragraph with enough '
            'characters to be kept as the second lead paragraph of the article.</p><p>Third paragraph that must not '
            'appear in the two-paragraph summary at all.</p></div>')
    t = W.lead_paragraphs(html)
    assert t.startswith("Influenza, commonly known as the flu") and "[1]" not in t
    assert "Second paragraph" in t and "Third paragraph" not in t


def test_cdc_and_eia_parsers():
    html = "<h2>Key Points</h2><ul><li>Seasonal influenza activity is <b>low</b>.</li><li>Two pediatric deaths were reported.</li></ul>"
    assert parse_key_points(html) == "Seasonal influenza activity is low. Two pediatric deaths were reported."
    # week ending Sat 2026-08-29 -> Friday 2026-09-04 17:00 EDT = 21:00 UTC
    assert available_at_friday_next_week(dt.date(2026, 8, 29)) == "2026-09-04T21:00:00Z"
    txt = "Highlights\n\n" + "U.S. crude oil refinery inputs averaged 16.9 million barrels per day during the week. " * 2 + "\n\n" + "Second para " * 12
    assert first_paragraphs(txt).startswith("U.S. crude oil refinery inputs")
    rd = release_dates("2025-08-25", "2025-09-10")
    assert dt.date(2025, 8, 27) in rd and dt.date(2025, 9, 4) in rd and dt.date(2025, 9, 3) not in rd  # Labor Day week


class _Resp:
    def __init__(self, status, text="", headers=None, url="u"):
        self.status_code, self.text, self.headers, self.url = status, text, headers or {}, url
        self.content = text.encode()


def test_http_cache_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "RAW_DIR", tmp_path)
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    monkeypatch.setenv("TBF_CONTACT_EMAIL", "x@example.org")
    calls = []
    responses = [_Resp(503), _Resp(429, headers={"Retry-After": "1"}), _Resp(200, '{"a": 1}', {"Content-Type": "application/json"})]

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(headers["User-Agent"])
        return responses.pop(0)

    fake_requests = types.SimpleNamespace(get=fake_get, RequestException=Exception)
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)
    data, fetched = H.get("https://example.org/x", cache_key="t/x.json")
    assert data == {"a": 1} and len(calls) == 3 and "x@example.org" in calls[0]
    assert (tmp_path / "t" / "x.json").exists()
    entry = json.loads((tmp_path / "t" / "x.json").read_text())
    assert entry["fetched_at"] == fetched and entry["status"] == 200
    # second call is served from cache: no network
    data2, fetched2 = H.get("https://example.org/x", cache_key="t/x.json")
    assert data2 == {"a": 1} and fetched2 == fetched and len(calls) == 3
    # 404 is not retried
    responses.append(_Resp(404))
    with pytest.raises(H.NotFound):
        H.get("https://example.org/missing", cache_key="t/missing.json")
