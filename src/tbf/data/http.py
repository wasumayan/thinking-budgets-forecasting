"""Cached, polite HTTP (see data/__init__.py). Implement with requests.

get(url, params=None, headers=None, cache_key=None, retries=5, backoff=1.5, binary=False)
    -> (text_or_json_or_bytes, fetched_at_iso)

- Every response is cached under data/raw/<cache_key> (cache_key like "wikimedia_pageviews/Influenza.json"; when
  None a sha1 of url+params under data/raw/misc/) as JSON {url, params, fetched_at, status, content_type, text}
  and re-used without a network call. Set env TBF_HTTP_REFRESH=1 to ignore the cache.
- User-Agent from series.yaml (sources.wikimedia_pageviews.user_agent) with <EMAIL> replaced by env
  TBF_CONTACT_EMAIL (SPEC rule 5). Retries with exponential backoff (max `retries`) on 429 and 5xx, honouring
  Retry-After; >= 0.2 s between calls to the same host. 404 raises NotFound (not retried).
- JSON responses (content-type json or parseable text for .json keys) are returned parsed; else text (or bytes
  when binary=True, cached base64-encoded).
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
import urllib.parse
from typing import Any

from ..config import ROOT, load_series_config

log = logging.getLogger(__name__)
RAW_DIR = ROOT / "data" / "raw"
MIN_INTERVAL_S = 0.2
_last_call: dict[str, float] = {}
_lock = threading.Lock()
_STATS = {"network": 0, "cached": 0}


class HTTPError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status, self.url = status, url


class NotFound(HTTPError):
    pass


def user_agent() -> str:
    ua = load_series_config()["sources"]["wikimedia_pageviews"]["user_agent"]
    email = os.environ.get("TBF_CONTACT_EMAIL")
    if not email:
        log.warning("TBF_CONTACT_EMAIL is not set; User-Agent will carry a placeholder contact")
        email = "unset@example.org"
    return ua.replace("<EMAIL>", email)


def _cache_path(url: str, params: dict | None, cache_key: str | None) -> pathlib.Path:
    if cache_key:
        return RAW_DIR / cache_key
    h = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()[:20]
    return RAW_DIR / "misc" / f"{h}.json"


def _throttle(host: str) -> None:
    with _lock:
        last = _last_call.get(host, 0.0)
        wait = MIN_INTERVAL_S - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def _decode(entry: dict, binary: bool) -> Any:
    if binary or entry.get("encoding") == "base64":
        return base64.b64decode(entry["text"])
    text = entry["text"]
    ctype = (entry.get("content_type") or "").lower()
    if "json" in ctype or entry.get("is_json"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def fetch_stats() -> dict[str, int]:
    return dict(_STATS)


def get(url: str, params=None, headers=None, cache_key: str | None = None, retries: int = 5, backoff: float = 1.5,
        binary: bool = False, timeout: float = 60.0):
    path = _cache_path(url, params, cache_key)
    if path.exists() and not os.environ.get("TBF_HTTP_REFRESH"):
        entry = json.loads(path.read_text())
        _STATS["cached"] += 1
        return _decode(entry, binary), entry["fetched_at"]

    import requests  # local import keeps the module importable without network deps in tests

    hdrs = {"User-Agent": user_agent(), "Accept": "application/octet-stream" if binary else "application/json, text/*;q=0.9, */*;q=0.5"}
    if headers:
        hdrs.update(headers)
    host = urllib.parse.urlsplit(url).netloc
    attempt, last_err = 0, None
    while attempt <= retries:
        _throttle(host)
        try:
            r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        except requests.RequestException as e:  # connection reset etc.
            last_err = e
            r = None
        if r is not None:
            if r.status_code == 404:
                raise NotFound(404, r.url, r.text)
            if r.status_code < 400:
                fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                ctype = r.headers.get("Content-Type", "")
                entry = {"url": r.url, "params": params, "fetched_at": fetched_at, "status": r.status_code,
                         "content_type": ctype}
                if binary:
                    entry.update(encoding="base64", text=base64.b64encode(r.content).decode())
                else:
                    entry["text"] = r.text
                    entry["is_json"] = "json" in ctype.lower() or (cache_key or "").endswith(".json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(entry))
                _STATS["network"] += 1
                return _decode(entry, binary), fetched_at
            if r.status_code not in (429,) and r.status_code < 500:
                raise HTTPError(r.status_code, r.url, r.text)
            last_err = HTTPError(r.status_code, r.url, r.text)
            retry_after = r.headers.get("Retry-After")
        else:
            retry_after = None
        attempt += 1
        if attempt > retries:
            break
        sleep = backoff ** attempt
        if retry_after:
            try:
                sleep = max(sleep, float(retry_after))
            except ValueError:
                pass
        log.warning("retry %d/%d for %s in %.1fs (%s)", attempt, retries, url, sleep, last_err)
        time.sleep(sleep)
    raise RuntimeError(f"giving up on {url} after {retries} retries: {last_err}")
