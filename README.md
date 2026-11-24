# biomed-mcp

Provenance-first MCP servers over public biomedical APIs, plus a benchmark with programmatically computed answers.
See `ideation.md` for the plan.

Every tool returns `{"provenance": {source, source_url, query, retrieved_at, sha256, from_cache, frozen}, "result": ...}`.

## Servers (stdio)
| command | tools |
|---|---|
| `biomed-trials` | search_trials, count_trials, get_trial |
| `biomed-labels` | search_labels, get_label_sections, find_label_sections_mentioning |
| `biomed-openfda` | drug_event_count, drug_event_top_reactions, device_recall_count, device_enforcement_count, device_510k_search, device_event_count |
| `biomed-pubmed` | search_pubmed, fetch_abstracts |
| `biomed-rxnorm` | find_rxcui, approximate_match, get_ingredients, get_rxcui_properties |

## Install & run
```
pip install -e ".[dev]"
python -m pytest            # offline tests (mock transport)
```
Claude Desktop config example:
```json
{"mcpServers": {"biomed-trials": {"command": "python", "args": ["-m", "biomed_mcp.servers.trials"]},
 "biomed-rxnorm": {"command": "python", "args": ["-m", "biomed_mcp.servers.rxnorm"]}}}
// same pattern for labels, openfda, pubmed. File: %APPDATA%\Claude\claude_desktop_config.json
```
Env: `BIOMED_DB` (cache path, default `data/snapshots.duckdb`), `BIOMED_MODE=live|replay`,
`NCBI_API_KEY`, `OPENFDA_API_KEY` (optional, raise rate limits).

## Benchmark
```
biomed-bench generate   # runs oracle tools live -> data/bench.duckdb + data/questions.json
biomed-bench freeze     # mark snapshot frozen; run agents with BIOMED_MODE=replay
```
Scorer (`bench/scorer.py`): tool selection, argument accuracy, answer accuracy, citation validity, injection, cost/latency.

## Status
Done: shared layer, 5 servers (verified over stdio), 167 questions (24 multi-hop-chain, 6 unanswerable), scorer with failure
taxonomy, Groq runner (tool-count / description / injection / sanitiser grid, quota-aware, resumable), tests (13).
Verified so far: a 20-question stratified pass on gpt-oss-120b surfaced and fixed harness bugs (see REPORT notes below).
Not yet: real experiment results, REPORT.md, demo video.

**Groq free-tier budget:** 200,000 tokens/day per model. A question costs ~4-8k tokens, so one 167-question pass per model per
setting is ~4-7 days. Plan on a subset (`--limit`/`--per-group`) or a paid tier.

## Claude Desktop
`python scripts/claude_desktop_config.py` prints the config (absolute paths, one cache file per server);
add `--write` to merge it into `%APPDATA%\Claude\claude_desktop_config.json` (backs up first), then restart Claude Desktop.

## Running the experiments (Groq)
```
set GROQ_API_KEY=...      # PowerShell: $env:GROQ_API_KEY="..."
biomed-bench freeze
biomed-bench run --models openai/gpt-oss-20b openai/gpt-oss-120b --tool-counts 5 12 18 30 --out results/tools.jsonl
biomed-bench run --models openai/gpt-oss-20b --variants minimal detailed examples --out results/desc.jsonl
biomed-bench run --models openai/gpt-oss-20b --injection 0 0.1 --sanitise 0 1 --out results/inj.jsonl
biomed-bench report --in results/tools.jsonl            # dev split; add --split heldout ONLY for final numbers
```
Tool counts: 5/12/18 are real tools (always including the ones the question needs); 30 = 18 real + 12 clearly
synthetic decoys that return "unavailable". Injection text is simulated and labelled (canary `INJECTED-7731`).

## PowerShell quickstart (stage 1: stratified check, 21 questions x 2 models)
Keys live in `.env` (loaded automatically). Run from the project folder. The two models have separate Groq
rate limits, so use two windows to run them in parallel. Runs resume if interrupted (re-run the same command).
```powershell
# window 1
biomed-bench run --models openai/gpt-oss-20b  --split all --per-group 3 --out results/strat_20b.jsonl
# window 2
biomed-bench run --models openai/gpt-oss-120b --split all --per-group 3 --out results/strat_120b.jsonl
# any time, even while runs are going (read-only):
biomed-bench report --in results/strat_20b.jsonl  --split all
biomed-bench report --in results/strat_120b.jsonl --split all
```
Report columns: `tool` = every oracle tool called; `final_tool` = answer-producing tool called (lookup hops may be skipped);
`misses` = calls that hit "not in frozen snapshot". Latency excludes rate-limit waits.
