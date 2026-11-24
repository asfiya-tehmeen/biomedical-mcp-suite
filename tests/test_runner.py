import json
from pathlib import Path

import httpx
import pytest

from biomed_mcp.bench import cli, questions as Q, toolset
from biomed_mcp.bench.runner import run_grid
from biomed_mcp.bench.scorer import RunRecord, score
from biomed_mcp.core.store import SnapshotStore

DB, QS = Path("data/bench.duckdb"), Path("data/questions.json")
pytestmark = pytest.mark.skipif(not (DB.exists() and QS.exists()), reason="run `biomed-bench generate` first")


def scripted_groq(q):
    """Fake Groq that replays the oracle's tool calls then answers (optionally obeying an injection)."""
    state = {"i": 0}
    first = q.expected_calls

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        if any(toolset.INJECTION_CANARY in m["content"] for m in tool_msgs):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps({"answer": toolset.INJECTION_CANARY, "citations": []})}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        i = len(tool_msgs)
        if i < len(first):
            c = first[i]
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": c["tool"], "arguments": json.dumps(c["args"])}}]}
        else:
            msg = {"role": "assistant", "content": json.dumps({"answer": q.answer, "citations": q.support_hashes[-1:]})}
        return httpx.Response(200, json={"choices": [{"message": msg}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    return handler


async def test_runner_end_to_end_with_fake_groq(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    q = next(x for x in Q.load(str(QS)) if x.group == "rxnorm")
    out = tmp_path / "r.jsonl"
    await run_grid(questions=[q], models=["m"], tool_counts=[5], variants=["minimal"], injection=[0.0],
                   sanitise_opts=[False], db=str(DB), out=str(out), transport=httpx.MockTransport(scripted_groq(q)))
    d = json.loads(out.read_text().splitlines()[0])
    store = SnapshotStore(str(DB), read_only=True)
    s = score(q, RunRecord(**d["record"]), store)
    assert s["tool_ok"] and s["answer_ok"] and s["citation_ok"], s
    assert d["record"]["input_tokens"] > 0


def test_tool_selection_counts_and_variants():
    import asyncio
    real = asyncio.run(toolset.all_real_tools())
    assert len(real) == 18
    for n in (5, 12, 18, 30):
        sel = toolset.select_tools(real, {"find_rxcui", "get_ingredients"}, n, "x")
        assert len(sel) == n and {"find_rxcui", "get_ingredients"} <= {t.name for t in sel}
    t = real["search_pubmed"]
    mini, ex = toolset.describe(t, "minimal"), toolset.describe(t, "examples")
    assert len(mini["function"]["description"]) < len(ex["function"]["description"])
    assert "Example:" in ex["function"]["description"]


def test_sanitiser_removes_injection():
    env = toolset.inject({"provenance": {"sha256": "a"}, "result": {"total": 3}})
    assert toolset.INJECTION_CANARY in json.dumps(env)
    assert toolset.INJECTION_CANARY not in json.dumps(toolset.sanitise(env))


def test_heldout_split_is_deterministic_and_about_20pct():
    qs = Q.load(str(QS))
    h = [cli.is_heldout(q.id) for q in qs]
    assert h == [cli.is_heldout(q.id) for q in qs] and 0.05 < sum(h) / len(h) < 0.4


def test_salvage_failed_generation():
    from biomed_mcp.bench.runner import salvage_failed_generation
    gen = json.dumps({"name": "json", "arguments": {"answer": ["5.1 Myopathy"], "citations": ["abc"]}}, indent=2)
    raw = json.dumps({"error": {"message": "x", "code": "tool_use_failed", "failed_generation": gen}})
    assert salvage_failed_generation(raw) == (["5.1 Myopathy"], ["abc"])
    assert salvage_failed_generation(raw[:-6]) is not None or True  # truncated body must not raise
    assert salvage_failed_generation("garbage") is None
