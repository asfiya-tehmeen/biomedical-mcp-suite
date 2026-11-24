"""HTTP client with caching, per-host rate limiting, retries and provenance.

Modes (env BIOMED_MODE or constructor arg):
  live    - use cache if fresh (ttl), otherwise hit the network and store the response.
  replay  - NEVER touch the network; serve only from the store (used for evaluation).
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import logging

import httpx

logging.getLogger("httpx").setLevel(logging.WARNING)

from .provenance import Envelope, Provenance, cache_key, sha256_hex
from .store import SnapshotStore, StoredResponse

DEFAULT_DB = os.environ.get("BIOMED_DB", "data/snapshots.duckdb")
DEFAULT_TTL = float(os.environ.get("BIOMED_CACHE_TTL", 24 * 3600))

# Minimum seconds between requests per host (public API etiquette; keys raise the limits).
MIN_INTERVAL = {
    "eutils.ncbi.nlm.nih.gov": 0.34 if not os.environ.get("NCBI_API_KEY") else 0.11,
    "api.fda.gov": 0.3 if not os.environ.get("OPENFDA_API_KEY") else 0.1,
    "clinicaltrials.gov": 0.2,
    "dailymed.nlm.nih.gov": 0.2,
    "rxnav.nlm.nih.gov": 0.1,
}


class ReplayMiss(RuntimeError):
    """Raised in replay mode when a request is not in the frozen snapshot."""


class UpstreamError(RuntimeError):
    pass


class BiomedClient:
    def __init__(
        self,
        store: SnapshotStore | None = None,
        mode: str | None = None,
        ttl: float = DEFAULT_TTL,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
    ):
        self.store = store or SnapshotStore(DEFAULT_DB)
        self.mode = mode or os.environ.get("BIOMED_MODE", "live")
        if self.mode not in ("live", "replay"):
            raise ValueError("mode must be 'live' or 'replay'")
        self.ttl = ttl
        self.max_retries = max_retries
        self._http = httpx.AsyncClient(
            transport=transport, timeout=30.0, follow_redirects=True,
            headers={"User-Agent": "biomed-mcp/0.1 (research tool)"},
        )
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _throttle(self, host: str) -> None:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = MIN_INTERVAL.get(host, 0.2) - (time.monotonic() - self._last.get(host, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[host] = time.monotonic()

    async def fetch(self, source: str, url: str, params: dict[str, Any] | None = None) -> tuple[StoredResponse, bool]:
        """Return (stored_response, from_cache)."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        key = cache_key(url, params)
        cached = self.store.get(key)
        if self.mode == "replay":
            if cached is None:
                raise ReplayMiss(f"not in frozen snapshot: {url} {params}")
            return cached, True
        if cached is not None and (cached.frozen or self._fresh(cached)):
            return cached, True

        host = urlsplit(url).netloc
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self._throttle(host)
            try:
                resp = await self._http.get(url, params=params)
            except httpx.HTTPError as e:
                last_exc = e
            else:
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = UpstreamError(f"{resp.status_code} from {host}")
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        await asyncio.sleep(min(float(retry_after), 10))
                else:
                    stored = StoredResponse(
                        key=key, source=source, source_url=str(resp.request.url), query=params,
                        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        status=resp.status_code, sha256=sha256_hex(resp.content),
                        body=resp.text, frozen=False,
                    )
                    # 404 is a legitimate "no results" for several APIs (openFDA); cache it too.
                    if resp.status_code < 400 or resp.status_code == 404:
                        self.store.put(stored)
                    return stored, False
            await asyncio.sleep(min(2 ** attempt * 0.5, 8))
        raise UpstreamError(f"{url} failed after retries: {last_exc}")

    def _fresh(self, r: StoredResponse) -> bool:
        try:
            age = time.time() - datetime.fromisoformat(r.retrieved_at).timestamp()
        except ValueError:
            return False
        return age < self.ttl

    async def get_json(self, source: str, url: str, params: dict[str, Any] | None = None,
                       allow_404: bool = False) -> tuple[Any, Provenance]:
        import json
        r, cached = await self.fetch(source, url, params)
        prov = self._prov(r, cached)
        if r.status == 404 and allow_404:
            return None, prov
        if r.status >= 400:
            raise UpstreamError(f"{r.status} from {r.source_url}: {r.body[:200]}")
        return json.loads(r.body), prov

    async def get_text(self, source: str, url: str, params: dict[str, Any] | None = None) -> tuple[str, Provenance]:
        r, cached = await self.fetch(source, url, params)
        if r.status >= 400:
            raise UpstreamError(f"{r.status} from {r.source_url}")
        return r.body, self._prov(r, cached)

    @staticmethod
    def _prov(r: StoredResponse, cached: bool) -> Provenance:
        return Provenance(source=r.source, source_url=r.source_url, query=r.query,
                          retrieved_at=r.retrieved_at, sha256=r.sha256, from_cache=cached, frozen=r.frozen)

    @staticmethod
    def envelope(prov: Provenance, result: Any) -> dict[str, Any]:
        return Envelope(provenance=prov, result=result).model_dump()

    async def aclose(self) -> None:
        await self._http.aclose()
