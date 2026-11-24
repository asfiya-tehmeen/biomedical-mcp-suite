# Prior art (Phase 0) — snippet-level only, verify before citing

Existing biomedical MCP servers found by one web search (not yet inspected in depth):

- PubCrawl — https://github.com/nickjlamb/pubcrawl — PubMed, FDA labels (openFDA), UK medicines, ClinicalTrials.gov; MIT.
- medical-mcps — https://pypi.org/project/medical-mcps/ — many bio/medical APIs incl. openFDA, PubTator3, ClinicalTrials.gov; Streamable HTTP.
- BioMCP (GenomOncology) — https://mcpservers.org/servers/genomoncology/biomcp — ClinicalTrials.gov, PubMed, MyVariant.info, openFDA, NCI CTS; stdio.

**Implication (per ideation §1/§9):** the APIs are already wrapped by others, so the contribution here is
(1) the provenance envelope (source_url, query, retrieved_at, sha256) + snapshot/replay store, and
(2) the benchmark with programmatically computed answers. No found server was checked for per-result hashing
or frozen replay; confirm that before claiming novelty.

## TODO
- [ ] Read the three repos' tool lists and licences; credit anything reused.
- [ ] Read usage policies: NCBI, openFDA, ClinicalTrials.gov, DailyMed, RxNav.
