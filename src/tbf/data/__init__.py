"""Data fetchers and the FreshTS-26 builder. All fetchers cache raw responses under data/raw/<source>/
with a fetch timestamp (SPEC §2.1; CODEX_PROMPT rule 6) and are pure functions of their arguments otherwise.

Shared helper contract (implement in .http):
  get(url, params=None, headers=None, cache_key=None, retries=5, backoff=1.5) -> (text_or_json, fetched_at_iso)
  Sends User-Agent from series.yaml with TBF_CONTACT_EMAIL substituted; retries on 429/5xx; sleeps >= 0.2 s
  between calls to the same host.
"""
