"""ClinicalTrials.gov API v2 MCP server."""
from __future__ import annotations

from typing import Any

from ._base import clamp, get_client, make_server, tool

BASE = "https://clinicaltrials.gov/api/v2"
SRC = "clinicaltrials.gov"

mcp = make_server(
    "biomed-trials",
    "Search ClinicalTrials.gov. Every result includes a `provenance` block (source_url, retrieved_at, sha256); "
    "cite the sha256 of the response that supports each claim.",
)


def _summ(study: dict[str, Any]) -> dict[str, Any]:
    p = study.get("protocolSection", {})
    ident, status, design = p.get("identificationModule", {}), p.get("statusModule", {}), p.get("designModule", {})
    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle"),
        "status": status.get("overallStatus"),
        "phases": design.get("phases", []),
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
        "start_date": status.get("startDateStruct", {}).get("date"),
    }


def _params(condition, intervention, status, phase, country, page_size, count_total, page_token=None):
    adv = f"AREA[Phase]{phase}" if phase else None
    return {
        "query.cond": condition, "query.intr": intervention, "query.locn": country,
        "filter.overallStatus": status, "filter.advanced": adv,
        "pageSize": page_size, "countTotal": "true" if count_total else None, "pageToken": page_token,
    }


@tool(mcp)
async def search_trials(
    condition: str | None = None,
    intervention: str | None = None,
    status: str | None = None,
    phase: str | None = None,
    country: str | None = None,
    page_size: int = 10,
    page_token: str | None = None,
) -> dict[str, Any]:
    """Search clinical trials on ClinicalTrials.gov and return a page of summaries plus the total match count.

    Args:
        condition: Disease/condition, e.g. "type 2 diabetes".
        intervention: Drug/device/procedure name, e.g. "semaglutide".
        status: Overall status enum, e.g. RECRUITING, COMPLETED, NOT_YET_RECRUITING, TERMINATED.
        phase: One of EARLY_PHASE1, PHASE1, PHASE2, PHASE3, PHASE4, NA.
        country: Country name for trial locations, e.g. "Saudi Arabia".
        page_size: Results per page (1-50).
        page_token: nextPageToken from a previous call.
    """
    c = get_client()
    n = clamp(page_size, 1, 50)
    # Without a page token fetch a canonical 50 and slice (cache key independent of page_size).
    data, prov = await c.get_json(SRC, f"{BASE}/studies",
                                  _params(condition, intervention, status, phase, country,
                                          n if page_token else 50, True, page_token))
    return c.envelope(prov, {
        "total_count": data.get("totalCount"),
        "studies": [_summ(s) for s in data.get("studies", [])[:n]],
        "next_page_token": data.get("nextPageToken"),
    })


@tool(mcp)
async def count_trials(
    condition: str | None = None,
    intervention: str | None = None,
    status: str | None = None,
    phase: str | None = None,
    country: str | None = None,
) -> dict[str, Any]:
    """Return only the number of trials matching the filters (cheaper than search_trials).

    Args: same filter semantics as search_trials.
    """
    c = get_client()
    data, prov = await c.get_json(SRC, f"{BASE}/studies",
                                  _params(condition, intervention, status, phase, country, 1, True))
    return c.envelope(prov, {"total_count": data.get("totalCount")})


@tool(mcp)
async def get_trial(nct_id: str) -> dict[str, Any]:
    """Fetch one trial by NCT id (e.g. NCT04280705): summary, eligibility criteria and primary outcomes.

    Args:
        nct_id: ClinicalTrials.gov identifier, 'NCT' followed by 8 digits.
    """
    c = get_client()
    data, prov = await c.get_json(SRC, f"{BASE}/studies/{nct_id.strip().upper()}")
    p = data.get("protocolSection", {})
    out = _summ(data)
    out["brief_summary"] = p.get("descriptionModule", {}).get("briefSummary")
    out["eligibility"] = p.get("eligibilityModule", {}).get("eligibilityCriteria")
    out["primary_outcomes"] = [o.get("measure") for o in p.get("outcomesModule", {}).get("primaryOutcomes", [])]
    out["enrollment"] = p.get("designModule", {}).get("enrollmentInfo", {}).get("count")
    return c.envelope(prov, out)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
