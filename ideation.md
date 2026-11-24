# H. Biomedical MCP Suite + Agent Tool-Use Benchmark — Ideation

**Status:** Ideation | **Data:** REAL (live public biomedical APIs, frozen snapshots for evaluation) | **Target roles:** AI engineer (pharma R&D/informatics, health-tech), forward-deployed AI engineer, AI platform engineer

**One-line pitch:** A set of provenance-first MCP servers over public biomedical APIs, plus a benchmark that measures how reliably agents choose tools, fill arguments, cite sources and resist injected content.

---

## 1. Why this matters

- MCP appears in real pharma AI postings: Pfizer's Informatics and AI Engineer role asks for MCP/RAG systems; a Genentech/Roche foundation-models role asks for LLM agents and Model Context Protocol work.
- Agents in pharma fail in predictable ways: wrong tool, wrong argument, stale data, invented citations. Few public benchmarks measure these on biomedical tools.
- This project is also the **shared tool layer** for projects F, G, I and J, so it pays off several times.
- Before building: check GitHub for existing PubMed/ClinicalTrials.gov MCP servers. If good ones exist, the contribution shifts towards the provenance layer and the benchmark (which is the more valuable part anyway).

## 2. How the idea was improved from v1

| v1 idea | Improved version | Why |
|---|---|---|
| Wrap 4 APIs as MCP tools | Provenance-first servers: every result carries source URL, query, retrieval time and a content hash; caching; rate-limit handling | "Regulatory-grade" traceability is what pharma cares about |
| ~60 hand-written questions | Benchmark where answers are **computed programmatically** from frozen API snapshots | Real data, exact ground truth, regenerable |
| Measure accuracy | Measure tool selection, argument correctness, answer accuracy, citation validity, cost and latency | Separates where agents fail |
| — | Experiments on tool-description quality, number of tools (tool overload), and prompt injection hidden in API content | These are open, practical questions for anyone shipping MCP |

## 3. Research questions

1. **Tool selection:** How often does an agent pick the right tool, and how does that change as the number of available tools grows (5 vs 15 vs 30)?
2. **Description quality:** Do better tool descriptions and schemas measurably improve selection and argument accuracy?
3. **Grounding:** How often do answers cite a result that actually supports them?
4. **Injection:** If an API response contains instruction-like text (simulated inside cached responses), how often does the agent follow it?
5. **Cost:** What is the accuracy–cost trade-off across small and large models?

## 4. Data plan (all real)

| Tool group | Source | Notes / to verify |
|---|---|---|
| Clinical trials | ClinicalTrials.gov API v2 | Free; check rate guidance. |
| Drug labels | DailyMed web services (SPL) | Free, no account. |
| Adverse events, recalls, devices | openFDA (drug event, drug label, device event, recall, 510(k)) | Free; API key optional but recommended; check paging limits. |
| Literature | PubMed E-utilities (NCBI) | Free; API key recommended; follow NCBI usage policy. |
| Drug names and normalisation | NLM RxNav / RxNorm API | Free. |
| Optional | MeSH (NLM) | Free. |

Ground truth: freeze snapshots of API responses (store in DuckDB/JSON with timestamps), then generate questions whose answers are computed from those snapshots by code. Example templates:
- "How many recruiting phase 3 trials for [condition] list [country]?"
- "Which section of the current [drug] label mentions [term]?"
- "How many [device product code] recalls were classified as class I between [dates]?"
- "What is the RxNorm ingredient for [brand name]?"
- Multi-hop: "For the drug with RxCUI X, find the number of FAERS reports mentioning [reaction] in [year]."

No synthetic data except the injected strings for the injection experiment (clearly labelled).

## 5. System design

```
                ┌──────────────────────── MCP servers (Python SDK) ────────────────────────┐
                │ trials-server │ labels-server │ openfda-server │ pubmed-server │ rxnorm  │
                └───────┬────────────────┬───────────────┬────────────────┬────────────────┘
                        │ shared layer: typed schemas (Pydantic), cache, rate limiter,
                        │ provenance envelope {source_url, query, retrieved_at, sha256}
                        ▼
                ┌─────────────────────────────┐
                │ Snapshot store (DuckDB)      │◄── freeze for evaluation
                └─────────────────────────────┘
                        │
     ┌──────────────────┴──────────────────┐
     ▼                                     ▼
 Question generator                 Agent runners (different models, tool counts,
 (templates → answers by code)      description variants, injection on/off)
     │                                     │
     └──────────────► Scorer ◄─────────────┘
         tool choice · arguments · answer · citation validity · cost · latency
                        │
                        ▼
                 Leaderboard + failure taxonomy
```

## 6. Evaluation plan

| Metric | How it is measured |
|---|---|
| Tool-selection accuracy | Called tool vs expected tool(s) per question |
| Argument accuracy | Parameters vs expected (exact or normalised match) |
| Answer accuracy | Compare with computed answer (numeric tolerance / set match) |
| Citation validity | Cited provenance hash exists and contains the supporting value |
| Injection success rate | Share of runs where agent follows injected instruction |
| Cost/latency | Tokens and seconds per question |

Experiments:
1. Tool count: 5 vs 15 vs 30 tools exposed.
2. Description variants: minimal vs detailed vs detailed + examples.
3. Models: two or three free models (one small, one large).
4. Injection: none vs injected text in 10% of cached responses, with and without a sanitising layer.

Target size: 150–300 generated questions across 5 tool groups, of which about 20% are multi-hop.

## 7. Free stack and budget

- Python MCP SDK, httpx, Pydantic, DuckDB, pytest, a simple CLI runner, optional Next.js or Streamlit leaderboard.
- Free LLM tiers; questions are short, so 300 questions × a few models is feasible over a few days with caching.

## 8. Step-by-step plan

### Phase 0 — Check and decide (½ day)
- [ ] Search GitHub for existing biomedical MCP servers; write `notes/prior-art.md`.
- [ ] Read usage policies for NCBI, openFDA, ClinicalTrials.gov, DailyMed, RxNav.

### Phase 1 — Servers (2–3 days)
- [ ] Shared provenance envelope, cache and rate limiter.
- [ ] Five servers with typed schemas and tests (recorded responses for tests).
- [ ] Connect to an MCP client and test interactively.

### Phase 2 — Benchmark (2 days)
- [ ] Freeze snapshots.
- [ ] Question templates + programmatic answers; multi-hop set.
- [ ] Scorer for all metrics.

### Phase 3 — Experiments (2 days)
- [ ] Run models × tool counts × description variants × injection.
- [ ] Build the failure taxonomy (wrong tool, wrong argument, stale data, unsupported citation, followed injection).

### Phase 4 — Ship (1 day)
- [ ] README with install steps for Claude Desktop/other MCP clients; `REPORT.md`; short demo video.
- [ ] Publish servers as a package (optional).

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Existing servers already cover the APIs | Focus on provenance + benchmark; reuse or extend existing servers with credit |
| APIs change or rate-limit | Snapshots for evaluation; retries and caching in servers |
| Generated questions are too easy | Add multi-hop, ambiguous names, date filters, and "unanswerable" questions |
| Benchmark overfits to your own prompts | Keep a held-out question set you never tune on |

## 10. Do not claim

- That results generalise to all biomedical agents; they apply to these tools and question types.
- Medical accuracy of any answer beyond what the source APIs say.

## 11. Stretch ideas

- "Unanswerable" detection: questions where the right answer is "no data".
- Agent memory/caching experiments (does caching change accuracy?).
- Gulf extension: an SFDA drug-list lookup tool (only if scraping is allowed by SFDA's terms).

## 12. Open questions to decide together

1. Which five tool groups first?
2. Which MCP client(s) to support for the demo?
3. Which models to benchmark on the free tiers?

## 13. Resume/DM hook (fill in only after measuring)

> Built 5 provenance-first MCP servers over public biomedical APIs and a [N]-question benchmark with programmatically computed answers; showed tool-selection accuracy fell from [x]% to [y]% as tools grew from 5 to 30, and a sanitising layer cut injection success from [x]% to [y]%.
