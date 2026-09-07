#!/usr/bin/env python
"""Generate data/fixtures/windows.jsonl (12 synthetic windows, 3 per domain) + shuffle_map.json.
Deterministic (seed 0). Schema = SPEC §7.1 + title/units/precision/freq. Synthetic numbers and text —
for tests and `make smoke` only; never for paper numbers."""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "fixtures"
rng = np.random.default_rng(0)

SPECS = [
    # (series_id, domain, cluster, freq, m, precision, title, units, origin, level, amp)
    ("web__Influenza", "web", "health", "D", 7, 0, "Influenza", "daily user pageviews, English Wikipedia", "2025-09-05", 9000, 2500),
    ("web__Bitcoin", "web", "economy", "D", 7, 0, "Bitcoin", "daily user pageviews, English Wikipedia", "2025-10-15", 30000, 8000),
    ("web__OPEC", "web", "energy", "D", 7, 0, "OPEC", "daily user pageviews, English Wikipedia", "2025-11-20", 2500, 700),
    ("health__nat", "health", None, "W", 1, 2, "ILINet % ILI, United States (national)", "percent of outpatient visits", "2025-11-29", 2.0, 1.2),
    ("health__hhs3", "health", None, "W", 1, 2, "ILINet % ILI, HHS Region 3 (Mid-Atlantic)", "percent of outpatient visits", "2026-01-03", 2.5, 1.5),
    ("health__ca", "health", None, "W", 1, 2, "ILINet % ILI, California", "percent of outpatient visits", "2026-02-07", 3.0, 1.0),
    ("energy__GASREGW", "energy", None, "W", 1, 2, "US Regular All Formulations Gas Price", "dollars per gallon", "2025-09-08", 3.15, 0.15),
    ("energy__WCESTUS1", "energy", None, "W", 1, 0, "Weekly U.S. Ending Stocks excluding SPR of Crude Oil", "thousand barrels", "2025-10-06", 420000, 8000),
    ("energy__WCRFPUS2", "energy", None, "W", 1, 0, "Weekly U.S. Field Production of Crude Oil", "thousand barrels per day", "2026-03-02", 13400, 150),
    ("macro__DEXUSEU", "macro", None, "B", 1, 4, "U.S. Dollars to Euro Spot Exchange Rate", "USD per EUR", "2025-09-12", 1.16, 0.02),
    ("macro__DGS10", "macro", None, "B", 1, 2, "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity", "percent", "2025-12-05", 4.2, 0.3),
    ("macro__DCOILWTICO", "macro", None, "B", 1, 2, "Crude Oil Prices: West Texas Intermediate (WTI)", "dollars per barrel", "2026-04-10", 68.0, 6.0),
]

EVENT_POOL = {
    "web": [("Health and environment", "The CDC reports rising {kw} activity across several regions.", ["Influenza"]),
            ("Business and economy", "{kw} prices swing after a regulatory announcement.", ["Bitcoin"]),
            ("International relations", "{kw} ministers meet to discuss production quotas.", ["OPEC"])],
    "health": [("Health and environment", "State health officials note an early start to the influenza season.", ["Influenza"])],
    "energy": [("Business and economy", "OPEC+ agrees to extend voluntary output cuts.", ["OPEC"]),
               ("Disasters and accidents", "A hurricane forces precautionary shutdowns of Gulf Coast refineries.", ["Hurricane"])],
    "macro": [("Business and economy", "The Federal Reserve holds interest rates steady and signals patience.", ["Federal_Reserve"]),
              ("International relations", "New tariffs on imported goods take effect.", ["Tariff"])],
}


def step(freq):
    return {"D": dt.timedelta(days=1), "W": dt.timedelta(weeks=1), "B": None}[freq]


def dates(origin: dt.date, n_before: int, n_after: int, freq: str):
    """Return (context_dates ending at origin, target_dates after origin)."""
    if freq in ("D", "W"):
        s = step(freq)
        before = [origin - s * k for k in range(n_before - 1, -1, -1)]
        after = [origin + s * k for k in range(1, n_after + 1)]
        return before, after
    assert origin.weekday() < 5, "business-day origin must be a weekday"

    def bdays(start: dt.date, n: int, direction: int) -> list[dt.date]:
        out, d = [], start
        while len(out) < n:
            d = d + dt.timedelta(days=direction)
            if d.weekday() < 5:
                out.append(d)
        return out

    before = sorted(bdays(origin, n_before - 1, -1)) + [origin]
    after = bdays(origin, n_after, +1)
    return before, after


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, domain, cluster, freq, m, prec, title, units, origin_s, level, amp in SPECS:
        origin = dt.date.fromisoformat(origin_s)
        before, after = dates(origin, 96, 12, freq)
        n = 108
        t = np.arange(n)
        season = np.sin(2 * np.pi * t / (7 if freq == "D" else 52 if freq == "W" else 20))
        noise = rng.normal(0, 0.15, n)
        vals = level + amp * (0.6 * season + noise + 0.002 * t)
        if prec == 0:
            vals = np.round(np.maximum(vals, 0))
        vals = [round(float(v), prec) for v in vals]
        kw = title.split(",")[0].split(" ")[0]
        events = []
        for i, (cat, text, links) in enumerate(EVENT_POOL[domain][: 2 + (sid == "web__Influenza")]):
            d = origin - dt.timedelta(days=3 + 5 * i)
            events.append({"date": d.isoformat(), "category": cat, "text": text.format(kw=kw), "links": links,
                           "available_at": f"{d.isoformat()}T23:59:59Z"})
        reports = []
        if domain == "health":
            d = origin - dt.timedelta(days=6)
            reports.append({"date": d.isoformat(), "source": "CDC FluView", "text": "Seasonal influenza activity is increasing nationally.",
                            "available_at": f"{d.isoformat()}T21:00:00Z"})
        avail = max([e["available_at"] for e in events] + [r["available_at"] for r in reports])
        h_start, h_end = after[0].isoformat(), after[-1].isoformat()
        freq_txt = {"D": "daily", "W": "weekly", "B": "business-daily"}[freq]
        metadata = f"{title} ({units}). Synthetic fixture series for tests. Frequency: {freq_txt}. Prediction target period: {h_start} to {h_end}."
        rows.append({
            "window_id": f"{sid}__{origin_s}", "series_id": sid, "domain": domain, "cluster": cluster, "freq": freq,
            "seasonality": m, "precision": prec, "title": title, "units": units,
            "origin_ts": f"{origin_s}T00:00:00Z",
            "context_ts": [d.isoformat() for d in before], "context": vals[:96],
            "target_ts": [d.isoformat() for d in after], "target": vals[96:], "n_ffill": 0,
            "metadata": metadata, "calendar": "None", "events": events, "reports": reports,
            "n_events": len(events), "text_available_at_max": avail, "strict_2026": origin >= dt.date(2026, 7, 1),
        })
    with open(OUT / "windows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ids = [r["window_id"] for r in rows]
    doms = {r["window_id"]: r["domain"] for r in rows}
    smap = {}
    for wid in ids:
        others = [o for o in ids if doms[o] != doms[wid]]
        smap[wid] = others[int(rng.integers(0, len(others)))]
    with open(OUT / "shuffle_map.json", "w") as f:
        json.dump(smap, f, indent=1)
    print(f"wrote {len(rows)} windows -> {OUT}")


if __name__ == "__main__":
    main()
