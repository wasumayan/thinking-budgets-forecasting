"""Uniform window loading for every dataset (fixtures | freshts26 | cik) -> list[dict] in the SPEC §7.1 shape
plus title / units / precision / freq, sorted by window_id (stable). Also the shuffle map per dataset."""
from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
from typing import Any

from ..config import ROOT
from . import fixtures as _fx

log = logging.getLogger(__name__)
FRESHTS_DIR = ROOT / "data" / "freshts26"
DATASETS = ("fixtures", "freshts26", "cik")


def _to_py(v: Any):
    """Convert numpy / pandas scalars and arrays to plain python for JSON-safe dicts."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _normalize(row: dict) -> dict:
    w = {k: _to_py(v) for k, v in row.items()}
    for key in ("context_ts", "target_ts"):
        w[key] = [str(_to_py(t))[:10] for t in (w.get(key) or [])]
    w["context"] = [float(x) for x in w["context"]]
    w["target"] = [float(x) for x in w["target"]]
    for key in ("events", "reports"):
        items = w.get(key)
        if items is None:
            w[key] = []
        else:
            w[key] = [{k: _to_py(v) for k, v in dict(it).items()} for it in items]
    if isinstance(w.get("origin_ts"), (dt.date, dt.datetime)):
        w["origin_ts"] = w["origin_ts"].isoformat()
    w.setdefault("n_events", len(w["events"]))
    w.setdefault("strict_2026", False)
    w.setdefault("precision", 2)
    return w


def load_windows(dataset: str) -> list[dict]:
    if dataset == "fixtures":
        rows = _fx.load_windows()
    elif dataset == "freshts26":
        path = FRESHTS_DIR / "windows.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `make build-data` first")
        import pandas as pd
        df = pd.read_parquet(path)
        rows = df.to_dict("records")
        series_path = FRESHTS_DIR / "series.parquet"
        if series_path.exists():
            sdf = pd.read_parquet(series_path).set_index("series_id")
            for r in rows:
                if r["series_id"] in sdf.index:
                    s = sdf.loc[r["series_id"]]
                    for k in ("title", "units", "precision"):
                        if k not in r or r[k] is None:
                            r[k] = s[k]
    elif dataset == "cik":
        from .cik import load_windows as _cik
        rows = _cik()
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    out = [_normalize(dict(r)) for r in rows]
    out.sort(key=lambda w: w["window_id"])
    return out


def load_shuffle_map(dataset: str) -> dict[str, str]:
    if dataset == "fixtures":
        return _fx.load_shuffle_map()
    if dataset == "freshts26":
        import pandas as pd
        df = pd.read_parquet(FRESHTS_DIR / "shuffle_map.parquet")
        return dict(zip(df["window_id"], df["source_window_id"]))
    raise ValueError(f"no shuffle map for dataset {dataset!r}")


def load_windows_by_id(dataset: str) -> dict[str, dict]:
    return {w["window_id"]: w for w in load_windows(dataset)}


# ------------------------------------------------------------------------------------------------
# Results JSONL helpers (SPEC §7.2): append-only, resumable.
# ------------------------------------------------------------------------------------------------

def read_results(path: str | pathlib.Path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping malformed line in %s", path)
    return rows


def done_window_ids(path: str | pathlib.Path) -> set[str]:
    return {r["window_id"] for r in read_results(path) if "window_id" in r}


def append_results(path: str | pathlib.Path, rows: list[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
