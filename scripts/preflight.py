#!/usr/bin/env python
"""Preflight for a machine WITH internet (e.g. the Della login node): one request per data endpoint and one
tokenizer load, printing PASS/FAIL per item. Exit code 1 if anything fails.

Usage: python scripts/preflight.py [--no-cache] [--skip-tokenizer]
Requires TBF_CONTACT_EMAIL (used in the User-Agent). Responses are cached under data/raw/preflight/.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tbf.config import load_grid, load_series_config  # noqa: E402
from tbf.data import http  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    t0 = time.time()
    try:
        detail = fn()
        RESULTS.append((name, True, f"{detail} ({time.time() - t0:.1f}s)"))
        print(f"PASS  {name}: {detail}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:300]}")
        if os.environ.get("TBF_PREFLIGHT_DEBUG"):
            traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="ignore cached responses (TBF_HTTP_REFRESH=1)")
    ap.add_argument("--skip-tokenizer", action="store_true")
    a = ap.parse_args()
    if a.no_cache:
        os.environ["TBF_HTTP_REFRESH"] = "1"
    if not os.environ.get("TBF_CONTACT_EMAIL"):
        print("WARN  TBF_CONTACT_EMAIL is not set; Wikimedia asks for a contact in the User-Agent")
    cfg = load_series_config()
    src = cfg["sources"]

    def wikimedia():
        url = src["wikimedia_pageviews"]["url"].format(title="Influenza", start="2025080100", end="2025080700")
        data, at = http.get(url, cache_key="preflight/wikimedia_pageviews.json")
        items = data["items"]
        return f"{len(items)} days, first={items[0]['timestamp']} views={items[0]['views']}"

    def wikipedia_revisions():
        url = src["wikipedia_summary_asof"]["revisions_api"].format(title="Influenza")
        data, _ = http.get(url, cache_key="preflight/wikipedia_revisions.json")
        rev = data["query"]["pages"][0]["revisions"][0]
        return f"revid={rev['revid']} @ {rev['timestamp']}"

    def wikipedia_parse():
        url = src["current_events"]["parse_api"].format(year=2025, month_name="August", day=1)
        data, _ = http.get(url, cache_key="preflight/current_events_2025-08-01.json")
        from tbf.data.fetch_current_events import parse_wikitext
        import datetime as dt
        ev = parse_wikitext(data["parse"]["wikitext"], dt.date(2025, 8, 1))
        cats = sorted({e["category"] for e in ev})
        return f"{len(ev)} bullets, categories={cats[:4]}..."

    def wikipedia_summary_fallback():
        url = src["wikipedia_summary_asof"]["fallback_summary"].format(title="Influenza")
        data, _ = http.get(url, cache_key="preflight/wikipedia_summary.json")
        return f"extract={data['extract'][:60]!r}..."

    def fred():
        url = src["fred_csv"]["url"].format(id="DEXUSEU")
        text, _ = http.get(url, cache_key="preflight/fred_DEXUSEU.csv")
        from tbf.data.fetch_fred import parse_fred_csv
        s = parse_fred_csv(text, "DEXUSEU")
        return f"{len(s)} obs, last={s.index[-1].date()} value={s.dropna().iloc[-1]}"

    def fred_page():
        url = src["fred_csv"]["notes_url"].format(id="DEXUSEU")
        html, _ = http.get(url, cache_key="preflight/fred_DEXUSEU.html")
        from tbf.data.fetch_fred import parse_fred_page
        d = parse_fred_page(html, "DEXUSEU")
        if not d["title"]:
            raise ValueError("no title parsed from the FRED page (scraper needs updating)")
        return f"title={d['title']!r} units={d['units'][:40]!r} notes={len(d['notes'])} chars"

    def delphi():
        url = src["delphi_fluview"]["url"].format(regions="nat,ca", last_epiweek="202634").replace("epiweeks=202301-", "epiweeks=202620-")
        data, _ = http.get(url, cache_key="preflight/delphi_fluview.json")
        if data.get("result") != 1:
            raise ValueError(f"result={data.get('result')} message={data.get('message')}")
        rows = data["epidata"]
        return f"{len(rows)} rows, last={max(r['epiweek'] for r in rows)}, fields={sorted(rows[0])[:6]}..."

    def cdc():
        url = src["cdc_fluview_reports"]["url"].format(year=2026, week=34)
        html, _ = http.get(url, cache_key="preflight/cdc_2026-week-34.html")
        from tbf.data.fetch_cdc_fluview import parse_key_points
        kp = parse_key_points(html)
        return f"key points: {kp[:80]!r}"

    def tokenizer():
        from transformers import AutoTokenizer
        hf = load_grid()["models"]["qwen3-0.6b"]["hf"]
        tok = AutoTokenizer.from_pretrained(hf)
        ids = tok("</think><|im_end|>", add_special_tokens=False).input_ids
        if ids != [151668, 151645]:
            raise ValueError(f"special ids {ids} != [151668, 151645]")
        msgs = [{"role": "user", "content": "hi"}]
        off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if "<think>" not in off:
            raise ValueError("enable_thinking=False did not pre-fill an empty think block")
        return f"{hf}: vocab={len(tok)}, </think>=151668, <|im_end|>=151645, enable_thinking OK"

    check("Wikimedia pageviews API", wikimedia)
    check("Wikipedia revisions API (as-of summary)", wikipedia_revisions)
    check("Wikipedia parse API (Portal:Current events)", wikipedia_parse)
    check("Wikipedia REST page/summary (fallback)", wikipedia_summary_fallback)
    check("FRED fredgraph.csv", fred)
    check("FRED series page (notes scrape)", fred_page)
    check("Delphi Epidata fluview", delphi)
    check("CDC FluView weekly report (tier 2)", cdc)
    if not a.skip_tokenizer:
        check("Qwen3-0.6B tokenizer", tokenizer)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
