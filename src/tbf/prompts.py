"""Prompt rendering (SPEC §4, configs/prompts/README.md).

render_messages(window: dict, setup: str, context_mode: str, prior_forecast: list[float] | None,
                shuffle_source: dict | None) -> list[dict]   # [{"role":"system",...},{"role":"user",...}]

- window: one row of windows.parquet as a dict (schema SPEC §7.1) + "title", "units", "precision", "freq".
- setup="reviser" requires prior_forecast (12 floats) rendered with the series precision.
- context_mode: full | none | shuffled. In shuffled mode `shuffle_source` is the mapped window whose
  events/reports replace this window's (metadata and calendar stay real).
- Section omission rules are in configs/prompts/README.md. Templates are read from configs/prompts/*.txt
  at import time; never inline them.
- Deterministic: same inputs -> byte-identical output (snapshot-tested).

format_value(v: float, precision: int) -> str   # "12345" for precision 0, "1.0834" for 4, no thousands separators
"""
from __future__ import annotations

import re

from .config import CONFIGS

PROMPT_DIR = CONFIGS / "prompts"
SYSTEM = (PROMPT_DIR / "system.txt").read_text().strip()
DIRECT = (PROMPT_DIR / "direct.txt").read_text()
REVISER = (PROMPT_DIR / "reviser.txt").read_text()

FREQ_DESC = {"D": "daily", "B": "business days", "W": "weekly"}
_TARGET_PERIOD = re.compile(r"Frequency: .*?\. Prediction target period: .*?\.\s*$", re.DOTALL)


def format_value(v: float, precision: int) -> str:
    return f"{v:.{precision}f}"


def _date(ts, sub_daily: bool = False) -> str:
    """ISO YYYY-MM-DD from a str / datetime / pandas Timestamp; with sub_daily=True (CiK hourly data) keep
    'YYYY-MM-DD HH:MM' for every stamp so midnight rows line up with the others."""
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M" if sub_daily else "%Y-%m-%d")
    s = str(ts)
    if sub_daily:
        return (s[:16].replace("T", " ") if len(s) >= 16 and s[10] in "T " else s[:10] + " 00:00")
    return s[:10]


def _is_sub_daily(window: dict) -> bool:
    for t in list(window.get("context_ts") or []) + list(window.get("target_ts") or []):
        s = str(t)
        if hasattr(t, "strftime"):
            if t.strftime("%H:%M") != "00:00":
                return True
        elif len(s) >= 16 and s[10] in "T " and s[11:16] != "00:00":
            return True
    return False


def _freq_desc(window: dict) -> str:
    freq = window.get("freq", "D")
    if freq == "W" and window.get("domain") == "health":
        return "weekly, week ending Saturday"
    return FREQ_DESC.get(freq, str(freq))


def _target_period_sentence(window: dict, h_start: str, h_end: str) -> str:
    return f"Frequency: {_freq_desc(window)}. Prediction target period: {h_start} to {h_end}."


def _metadata(window: dict, context_mode: str, h_start: str, h_end: str) -> str:
    tail = _target_period_sentence(window, h_start, h_end)
    if context_mode == "none":
        title, units = window.get("title", window["series_id"]), (window.get("units") or "").strip()
        return f"{title} ({units}). {tail}" if units else f"{title}. {tail}"
    meta = str(window.get("metadata") or "").strip()
    # The builder fills the frequency/target sentence per window; if it is present, keep the base text and
    # re-render the sentence from the window so the rendering is a pure function of the window fields.
    if _TARGET_PERIOD.search(meta):
        meta = _TARGET_PERIOD.sub("", meta).rstrip()
    return f"{meta} {tail}".strip() if meta else tail


def _events_lines(events) -> str:
    events = list(events or [])
    if not events:
        return "None"
    events = sorted(events, key=lambda e: _date(e["date"]), reverse=True)[:10]
    return "\n".join(f"- {_date(e['date'])} [{e['category']}] {str(e['text']).strip()}" for e in events)


def _reports_section(reports) -> str:
    reports = list(reports or [])
    if not reports:
        return ""
    reports = sorted(reports, key=lambda r: _date(r["date"]), reverse=True)[:2]
    lines = "\n".join(f"- {_date(r['date'])} [{r['source']}] {str(r['text']).strip()}" for r in reports)
    return f"## Reports (dated)\n{lines}\n\n"


def _history(window: dict, precision: int, sub_daily: bool = False) -> str:
    ts, vals = window["context_ts"], window["context"]
    return "\n".join(f"{_date(t, sub_daily)}: {format_value(float(v), precision)}" for t, v in zip(ts, vals))


def render_messages(window: dict, setup: str, context_mode: str, prior_forecast=None, shuffle_source=None) -> list[dict]:
    if setup not in ("direct", "reviser"):
        raise ValueError(f"unknown setup {setup!r}")
    if context_mode not in ("full", "none", "shuffled"):
        raise ValueError(f"unknown context_mode {context_mode!r}")
    if context_mode == "shuffled" and shuffle_source is None:
        raise ValueError("shuffled mode needs shuffle_source")
    if setup == "reviser" and prior_forecast is None:
        raise ValueError("reviser setup needs prior_forecast")

    precision = int(window.get("precision", 2))
    target_ts = list(window["target_ts"])
    sub_daily = _is_sub_daily(window)
    h_start, h_end = _date(target_ts[0], sub_daily), _date(target_ts[-1], sub_daily)
    fields = {
        "metadata": _metadata(window, context_mode, h_start, h_end),
        "h_start": h_start,
        "h_end": h_end,
        "freq_desc": _freq_desc(window),
        "history": _history(window, precision, sub_daily),
    }
    if context_mode == "none":
        fields["calendar"] = None
        fields["events"] = None
        fields["reports_section"] = ""
    else:
        src = shuffle_source if context_mode == "shuffled" else window
        fields["calendar"] = str(window.get("calendar") or "None").strip()
        fields["events"] = _events_lines(src.get("events"))
        fields["reports_section"] = _reports_section(src.get("reports"))

    is_cik = window.get("dataset") == "cik" or window.get("domain") == "cik"
    if is_cik and context_mode != "none":
        # CiK: the context text IS the metadata; there are no calendar / events / reports sections (data/cik.py).
        fields["metadata"] = f"{str(window.get('metadata') or '').strip()}\n\n{_target_period_sentence(window, h_start, h_end)}".strip()
        fields["calendar"] = None
        fields["events"] = None
        fields["reports_section"] = ""
    template = DIRECT if setup == "direct" else REVISER
    T, H = len(window["context"]), len(target_ts)
    if (T, H) != (96, 12):  # CiK task-specific lengths (SPEC "Deviations (implementation)")
        template = template.replace("96 values", f"{T} values").replace("next 12 values", f"next {H} values") \
                           .replace("containing 12 numbers", f"containing {H} numbers").replace("v12", f"v{H}")
    if setup == "reviser":
        pf = list(prior_forecast)
        if len(pf) != len(target_ts):
            raise ValueError(f"prior_forecast has {len(pf)} values, expected {len(target_ts)}")
        fields["prior_forecast"] = "\n".join(f"{_date(t, sub_daily)}: {format_value(float(v), precision)}" for t, v in zip(target_ts, pf))

    text = _fill(template, fields)
    if context_mode == "none" or is_cik:
        text = _drop_section(text, "## Calendar")
        text = _drop_section(text, "## Recent events (dated; newest first)")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text.rstrip("\n")}]


def _fill(template: str, fields: dict) -> str:
    out = template
    for k, v in fields.items():
        out = out.replace("{" + k + "}", "" if v is None else str(v))
    return out


def _drop_section(text: str, heading: str) -> str:
    """Remove a '## heading' block (heading line through the blank line before the next '## ')."""
    lines = text.split("\n")
    out, skipping = [], False
    for line in lines:
        if line.startswith("## "):
            skipping = line.strip() == heading
        if not skipping:
            out.append(line)
    return "\n".join(out)
