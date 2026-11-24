"""NLM RxNav / RxNorm MCP server."""
from __future__ import annotations

from typing import Any

from ._base import get_client, make_server, tool

BASE = "https://rxnav.nlm.nih.gov/REST"
SRC = "rxnav.nlm.nih.gov"

mcp = make_server("biomed-rxnorm", "Normalise drug names to RxNorm concepts (RxCUI) and find ingredients.")


@tool(mcp)
async def find_rxcui(name: str) -> dict[str, Any]:
    """Exact-match lookup of a drug name (brand or generic) to RxCUI identifiers.

    Args:
        name: Drug name, e.g. "Lipitor" or "atorvastatin".
    """
    c = get_client()
    data, prov = await c.get_json(SRC, f"{BASE}/rxcui.json", {"name": name, "search": 2})
    ids = (data.get("idGroup") or {}).get("rxnormId", [])
    return c.envelope(prov, {"name": name, "rxcuis": ids})


@tool(mcp)
async def approximate_match(term: str, max_entries: int = 5) -> dict[str, Any]:
    """Fuzzy match a possibly misspelled drug name to candidate RxNorm concepts.

    Args:
        term: Free-text drug name.
        max_entries: Max candidates (1-20).
    """
    c = get_client()
    data, prov = await c.get_json(SRC, f"{BASE}/approximateTerm.json",
                                  {"term": term, "maxEntries": max(1, min(20, max_entries))})
    cands = (data.get("approximateGroup") or {}).get("candidate", []) or []
    seen, out = set(), []
    for cd in cands:
        if cd["rxcui"] not in seen:
            seen.add(cd["rxcui"])
            out.append({"rxcui": cd["rxcui"], "score": cd.get("score"), "rank": cd.get("rank")})
    return c.envelope(prov, {"term": term, "candidates": out})


@tool(mcp)
async def get_ingredients(rxcui: str) -> dict[str, Any]:
    """Return the active ingredient concept(s) (TTY=IN) related to an RxCUI (e.g. a brand or clinical drug).

    Args:
        rxcui: RxNorm concept id, e.g. "153165".
    """
    c = get_client()
    data, prov = await c.get_json(SRC, f"{BASE}/rxcui/{rxcui}/related.json", {"tty": "IN"})
    ings = []
    for grp in (data.get("relatedGroup") or {}).get("conceptGroup", []) or []:
        for cp in grp.get("conceptProperties", []) or []:
            ings.append({"rxcui": cp["rxcui"], "name": cp["name"]})
    return c.envelope(prov, {"rxcui": rxcui, "ingredients": ings})


@tool(mcp)
async def get_rxcui_properties(rxcui: str) -> dict[str, Any]:
    """Return name, term type (TTY) and synonym for an RxCUI.

    Args:
        rxcui: RxNorm concept id.
    """
    c = get_client()
    data, prov = await c.get_json(SRC, f"{BASE}/rxcui/{rxcui}/properties.json")
    return c.envelope(prov, (data or {}).get("properties") or {})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
