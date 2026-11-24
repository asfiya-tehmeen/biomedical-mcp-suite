"""Shared helpers for building provenance-first FastMCP servers."""
from __future__ import annotations

from typing import Any

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from ..core.client import BiomedClient, ReplayMiss, UpstreamError

_client: BiomedClient | None = None


_server_name = "shared"


def _default_store():
    """One cache file per server process (DuckDB allows a single read-write process per file), so five MCP
    servers launched side by side by a client don't collide. BIOMED_DB (explicit) overrides, e.g. benchmarks."""
    import os
    from pathlib import Path

    from ..core.store import SnapshotStore
    if os.environ.get("BIOMED_DB"):
        return SnapshotStore(os.environ["BIOMED_DB"])
    root = Path(os.environ.get("BIOMED_CACHE_DIR") or Path(__file__).resolve().parents[3] / "data" / "cache")
    return SnapshotStore(root / f"{_server_name}.duckdb")


def get_client() -> BiomedClient:
    """Process-wide client (one cache + rate limiter shared by all tools). Tests may override via set_client."""
    global _client
    if _client is None:
        _client = BiomedClient(_default_store())
    return _client


def set_client(c: BiomedClient | None) -> None:
    global _client
    _client = c


def make_server(name: str, instructions: str) -> FastMCP:
    global _server_name
    _server_name = name.replace("biomed-", "")
    return FastMCP(name, instructions=instructions)


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def text_of(el: Any) -> str:
    """Collapse whitespace of all text under an XML element."""
    return " ".join("".join(el.itertext()).split())


def tool(server: FastMCP):
    """Register an async tool whose upstream failures come back as {"error": "<reason>"} instead of the
    framework's opaque "Error executing tool X", so agents (and humans) can see what went wrong."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except (ReplayMiss, UpstreamError) as e:
                return {"error": str(e)[:300]}
        return server.tool()(wrapper)

    return deco
