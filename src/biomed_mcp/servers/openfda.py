"""openFDA MCP server (drug adverse events, device recalls, 510(k), device events)."""
from __future__ import annotations

import os
from typing import Any

from ._base import clamp, get_client, make_server, tool

BASE = "https://api.fda.gov"
SRC = "openfda"

mcp = make_server(
    "biomed-openfda",
    "Query openFDA. Counts come from the API itself; always cite the provenance sha256. "
    "FAERS adverse-event reports show association, not causation.",
)


def _p(search: str | None, count: str | None = None, limit: int | None = None) -> dict[str, Any]:
    p: dict[str, Any] = {"search": search, "count": count, "limit": limit}
    if os.environ.get("OPENFDA_API_KEY"):
        p["api_key"] = os.environ["OPENFDA_API_KEY"]
    return p


async def _query(endpoint: str, search: str | None, count: str | None, limit: int,
                 totals_only: bool = False) -> dict[str, Any]:
    c = get_client()
    data, prov = await c.get_json(SRC, f"{BASE}/{endpoint}.json", _p(search, count, limit), allow_404=True)
    if data is None:  # openFDA returns 404 NOT_FOUND when nothing matches
        return c.envelope(prov, {"total": 0} if totals_only else {"total": 0, "results": []})
    total = data.get("meta", {}).get("results", {}).get("total")
    if totals_only:
        return c.envelope(prov, {"total": total})
    return c.envelope(prov, {"total": total, "results": data.get("results", [])})


@tool(mcp)
async def drug_event_count(drug: str, reaction: str | None = None,
                           date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
    """Count FAERS adverse-event reports mentioning a drug (optionally a reaction and receive-date range).

    Args:
        drug: Generic or brand name, matched against patient.drug.medicinalproduct / openfda.generic_name.
        reaction: MedDRA preferred term, e.g. "nausea".
        date_from: Receive date lower bound YYYYMMDD.
        date_to: Receive date upper bound YYYYMMDD.
    """
    q = [f'(patient.drug.openfda.generic_name:"{drug}" OR patient.drug.openfda.brand_name:"{drug}")']
    if reaction:
        q.append(f'patient.reaction.reactionmeddrapt:"{reaction}"')
    if date_from or date_to:
        q.append(f"receivedate:[{date_from or '19000101'} TO {date_to or '30000101'}]")
    return await _query("drug/event", " AND ".join(q), None, 1, totals_only=True)


@tool(mcp)
async def drug_event_top_reactions(drug: str, top: int = 10) -> dict[str, Any]:
    """Most frequently reported reactions (MedDRA terms) for a drug in FAERS.

    Args:
        drug: Generic or brand name.
        top: Number of terms (1-50).
    """
    s = f'patient.drug.openfda.generic_name:"{drug}" OR patient.drug.openfda.brand_name:"{drug}"'
    out = await _query("drug/event", s, "patient.reaction.reactionmeddrapt.exact", 50)  # canonical; sliced
    out["result"]["results"] = out["result"].get("results", [])[:clamp(top, 1, 50)]
    return out


@tool(mcp)
async def device_recall_count(product_code: str | None = None, date_from: str | None = None,
                              date_to: str | None = None) -> dict[str, Any]:
    """Count medical-device recall events (device/recall) by FDA product code and initiation-date range.
    This endpoint has NO recall class; use device_enforcement_count for Class I/II/III.

    Args:
        product_code: Three-letter FDA product code, e.g. "DXN".
        date_from: event_date_initiated lower bound YYYY-MM-DD.
        date_to: event_date_initiated upper bound YYYY-MM-DD.
    """
    q = []
    if product_code:
        q.append(f"product_code:{product_code}")
    if date_from or date_to:
        q.append(f"event_date_initiated:[{date_from or '1900-01-01'} TO {date_to or '3000-01-01'}]")
    return await _query("device/recall", " AND ".join(q) or None, None, 1, totals_only=True)


@tool(mcp)
async def device_enforcement_count(classification: str | None = None, date_from: str | None = None,
                                   date_to: str | None = None, product_text: str | None = None) -> dict[str, Any]:
    """Count device enforcement reports (device/enforcement), which carry the recall class.
    There is no product_code field here; narrow by words in the product description instead.

    Args:
        classification: "Class I", "Class II" or "Class III".
        date_from: report_date lower bound YYYYMMDD.
        date_to: report_date upper bound YYYYMMDD.
        product_text: Word(s) that must appear in product_description, e.g. "pacemaker".
    """
    q = []
    if classification:
        q.append(f'classification:"{classification}"')
    if date_from or date_to:
        q.append(f"report_date:[{date_from or '19000101'} TO {date_to or '30000101'}]")
    if product_text:
        q.append(f'product_description:"{product_text}"')
    return await _query("device/enforcement", " AND ".join(q) or None, None, 1, totals_only=True)


@tool(mcp)
async def device_510k_search(applicant: str | None = None, product_code: str | None = None,
                             limit: int = 5) -> dict[str, Any]:
    """Search FDA 510(k) clearances by applicant name and/or product code.

    Args:
        applicant: Company name.
        product_code: FDA product code.
        limit: Max records (1-25).
    """
    q = []
    if applicant:
        q.append(f'applicant:"{applicant}"')
    if product_code:
        q.append(f"product_code:{product_code}")
    out = await _query("device/510k", " AND ".join(q) or None, None, 25)  # canonical; sliced
    out["result"]["results"] = out["result"].get("results", [])[:clamp(limit, 1, 25)]
    return out


@tool(mcp)
async def device_event_count(product_code: str, event_type: str | None = None) -> dict[str, Any]:
    """Count MAUDE device adverse-event reports for a product code.

    Args:
        product_code: FDA product code.
        event_type: "Death", "Injury" or "Malfunction".
    """
    q = [f"device.device_report_product_code:{product_code}"]
    if event_type:
        q.append(f'event_type:"{event_type}"')
    return await _query("device/event", " AND ".join(q), None, 1, totals_only=True)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
