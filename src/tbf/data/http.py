"""Cached, polite HTTP (see data/__init__.py). Implement with requests."""
from __future__ import annotations


def get(url: str, params=None, headers=None, cache_key: str | None = None, retries: int = 5, backoff: float = 1.5):
    raise NotImplementedError
