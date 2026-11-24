"""PubMed E-utilities MCP server."""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from ._base import clamp, get_client, make_server, text_of, tool

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SRC = "pubmed"

mcp = make_server("biomed-pubmed", "Search PubMed and fetch abstracts. Cite PMIDs and the provenance sha256.")


def _common() -> dict[str, Any]:
    p: dict[str, Any] = {"tool": "biomed-mcp"}
    if os.environ.get("NCBI_API_KEY"):
        p["api_key"] = os.environ["NCBI_API_KEY"]
    return p


def _canon_date(s: str | None, end: bool) -> str | None:
    """Equivalent date spellings share one cache key: YYYY-MM-DD -> YYYY/MM/DD, and Jan 1 / Dec 31 bounds -> YYYY."""
    if not s:
        return s
    m = re.fullmatch(r"(\d{4})(?:[/-](\d{2})[/-](\d{2}))?", s.strip())
    if not m:
        return s
    y, mo, d = m.groups()
    if mo is None:
        return y
    if (mo, d) == (("12", "31") if end else ("01", "01")):
        return y
    return f"{y}/{mo}/{d}"


@tool(mcp)
async def search_pubmed(term: str, retmax: int = 10, min_date: str | None = None,
                        max_date: str | None = None, sort: str = "relevance") -> dict[str, Any]:
    """Search PubMed; returns total hit count and PMIDs.

    Args:
        term: PubMed query, supports field tags e.g. "metformin[Title] AND cancer".
        retmax: Number of PMIDs to return (1-100).
        min_date: Publication date lower bound, YYYY/MM/DD or YYYY.
        max_date: Publication date upper bound.
        sort: "relevance" or "pub_date".
    """
    c = get_client()
    min_date, max_date = _canon_date(min_date, False), _canon_date(max_date, True)
    # Upstream request is canonical (retmax=100, default sort omitted) so equivalent calls share one
    # cache entry / snapshot row; we slice to the caller's retmax below.
    params = {**_common(), "db": "pubmed", "term": term, "retmode": "json", "retmax": 100,
              "sort": None if sort == "relevance" else sort}
    if min_date or max_date:
        params.update({"datetype": "pdat", "mindate": min_date or "1800", "maxdate": max_date or "3000"})
    data, prov = await c.get_json(SRC, f"{BASE}/esearch.fcgi", params)
    r = data["esearchresult"]
    return c.envelope(prov, {"count": int(r.get("count", 0)), "pmids": r.get("idlist", [])[:clamp(retmax, 1, 100)]})


@tool(mcp)
async def fetch_abstracts(pmids: list[str]) -> dict[str, Any]:
    """Fetch title, journal, year and abstract for up to 20 PMIDs.

    Args:
        pmids: List of PubMed ids.
    """
    c = get_client()
    ids = ",".join(p.strip() for p in pmids[:20])
    body, prov = await c.get_text(SRC, f"{BASE}/efetch.fcgi",
                                  {**_common(), "db": "pubmed", "id": ids, "retmode": "xml"})
    root = ET.fromstring(body)
    arts = []
    for a in root.iter("PubmedArticle"):
        title = a.find("MedlineCitation/Article/ArticleTitle")
        year = a.findtext(".//PubDate/Year") or (a.findtext(".//PubDate/MedlineDate") or "")[:4]
        arts.append({
            "pmid": a.findtext("MedlineCitation/PMID"),
            "title": text_of(title) if title is not None else None,
            "journal": a.findtext(".//Journal/Title"),
            "year": year,
            "abstract": " ".join(text_of(t) for t in a.findall(".//Abstract/AbstractText")) or None,
        })
    return c.envelope(prov, {"articles": arts})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
