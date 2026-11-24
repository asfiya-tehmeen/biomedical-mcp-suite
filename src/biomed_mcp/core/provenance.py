"""Provenance envelope attached to every tool result."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


class Provenance(BaseModel):
    source: str          # e.g. "clinicaltrials.gov"
    source_url: str      # fully-resolved request URL
    query: dict[str, Any]
    retrieved_at: str    # ISO-8601 UTC timestamp of the original network fetch
    sha256: str          # hash of the raw response body
    from_cache: bool = False
    frozen: bool = False  # served from a frozen evaluation snapshot


class Envelope(BaseModel):
    """What every MCP tool returns: the (trimmed) result plus where it came from."""

    provenance: Provenance
    result: Any


def sha256_hex(body: str | bytes) -> str:
    if isinstance(body, str):
        body = body.encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def cache_key(url: str, params: dict[str, Any] | None) -> str:
    canon = json.dumps({"u": url, "p": params or {}}, sort_keys=True, default=str)
    return sha256_hex(canon)
