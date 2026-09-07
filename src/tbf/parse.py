"""Answer parsing (SPEC §4). Take the LAST {...} JSON object in the answer text; require a
"forecast" list of finite numbers. Accept 11–13 values (truncate / pad with last value, flag).
Returns ParsedForecast(values: list[float] | None, valid: bool, n_values_mismatch: bool, reason: str).
"""
from __future__ import annotations

import dataclasses
import json
import math
import re

H = 12
_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclasses.dataclass
class ParsedForecast:
    values: list[float] | None
    valid: bool
    n_values_mismatch: bool
    reason: str = ""


def _coerce_list(obj) -> list[float] | None:
    if isinstance(obj, dict):
        obj = obj.get("forecast")
    if not isinstance(obj, list):
        return None
    out = []
    for v in obj:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            f = float(v)
        elif isinstance(v, str):
            m = _NUM.search(v.replace(",", ""))
            if not m:
                return None
            f = float(m.group())
        else:
            return None
        if not math.isfinite(f):
            return None
        out.append(f)
    return out


def _fix_length(vals: list[float], h: int = H) -> tuple[list[float] | None, bool]:
    if len(vals) == h:
        return vals, False
    if h - 1 <= len(vals) <= h + 1:
        if len(vals) > h:
            return vals[:h], True
        return vals + [vals[-1]] * (h - len(vals)), True
    return None, True


def parse_forecast(answer: str, h: int = H) -> ParsedForecast:
    """`h` = expected horizon length (12 for FreshTS-26; task-specific for CiK, see data/cik.py)."""
    if not answer or not answer.strip():
        return ParsedForecast(None, False, False, "empty")
    text = answer.strip()
    # strip ```json fences
    text = re.sub(r"```(?:json)?", "", text)
    candidates = _JSON_OBJ.findall(text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            # tolerate trailing commas
            try:
                obj = json.loads(re.sub(r",\s*([\]}])", r"\1", cand))
            except json.JSONDecodeError:
                continue
        vals = _coerce_list(obj)
        if vals is None or len(vals) == 0:
            continue
        fixed, mismatch = _fix_length(vals, h)
        if fixed is None:
            return ParsedForecast(None, False, True, f"bad_length_{len(vals)}")
        return ParsedForecast(fixed, True, mismatch, "")
    # fallback: a bare bracketed list anywhere
    m = re.findall(r"\[[^\[\]]*\]", text)
    for cand in reversed(m):
        try:
            vals = _coerce_list(json.loads(cand))
        except json.JSONDecodeError:
            vals = None
        if vals:
            fixed, mismatch = _fix_length(vals, h)
            if fixed is not None:
                return ParsedForecast(fixed, True, mismatch, "bare_list")
    return ParsedForecast(None, False, False, "no_json")
