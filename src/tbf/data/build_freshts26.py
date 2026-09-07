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


def main(argv=None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
