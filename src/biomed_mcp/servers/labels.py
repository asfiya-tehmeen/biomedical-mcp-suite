"""DailyMed (SPL drug label) MCP server."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from ._base import clamp, get_client, make_server, text_of, tool

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
SRC = "dailymed"

mcp = make_server("biomed-labels", "Find current FDA drug labels on DailyMed and read their sections.")


@tool(mcp)
async def search_labels(drug_name: str, page_size: int = 10) -> dict[str, Any]:
    """Search DailyMed SPL labels by drug name (brand or generic).

    Args:
        drug_name: e.g. "atorvastatin" or "Lipitor".
        page_size: Max labels to return (1-50).
    """
    c = get_client()
    data, prov = await c.get_json(SRC, f"{BASE}/spls.json",
                                  {"drug_name": drug_name, "pagesize": 50})  # canonical upstream size; sliced below
    rows = [{"setid": d["setid"], "title": d["title"], "version": d.get("spl_version"),
             "published": d.get("published_date")} for d in data.get("data", [])[:clamp(page_size, 1, 50)]]
    return c.envelope(prov, {"total": data.get("metadata", {}).get("total_elements"), "labels": rows})


@tool(mcp)
async def get_label_sections(setid: str, section_contains: str | None = None,
                             max_chars: int = 3000) -> dict[str, Any]:
    """Read a label's sections. Without `section_contains`, returns only the list of section titles.

    Args:
        setid: DailyMed set id (UUID) from search_labels.
        section_contains: Case-insensitive substring of a section title, e.g. "warnings" or "adverse reactions".
        max_chars: Truncate each returned section's text to this many characters.
    """
    c = get_client()
    body, prov = await c.get_text(SRC, f"{BASE}/spls/{setid}.xml")
    root = ET.fromstring(body)
    ns = {"h": "urn:hl7-org:v3"}
    secs = []
    for s in root.iterfind(".//h:section", ns):
        title = s.findtext("h:title", default="", namespaces=ns).strip()
        if title:
            secs.append((title, s))
    if section_contains is None:
        return c.envelope(prov, {"setid": setid, "section_titles": [t for t, _ in secs]})
    needle = section_contains.lower()
    out = [{"title": t, "text": text_of(s)[:max_chars]} for t, s in secs if needle in t.lower()]
    return c.envelope(prov, {"setid": setid, "sections": out[:5]})


@tool(mcp)
async def find_label_sections_mentioning(setid: str, term: str) -> dict[str, Any]:
    """List the titles of label sections whose text mentions `term` (case-insensitive).

    Args:
        setid: DailyMed set id.
        term: Word or phrase to look for, e.g. "hepatotoxicity".
    """
    c = get_client()
    body, prov = await c.get_text(SRC, f"{BASE}/spls/{setid}.xml")
    root = ET.fromstring(body)
    ns = {"h": "urn:hl7-org:v3"}
    hits = []
    for s in root.iterfind(".//h:section", ns):
        title = s.findtext("h:title", default="", namespaces=ns).strip()
        # Only the section's own text (exclude nested sections) to avoid double-reporting parents.
        own = " ".join(
            " ".join("".join(p.itertext()).split()) for p in s.iterfind("h:text", ns)
        )
        if title and term.lower() in own.lower():
            hits.append(title)
    return c.envelope(prov, {"setid": setid, "term": term, "sections": hits})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
