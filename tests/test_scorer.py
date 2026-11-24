import json
from pathlib import Path

import pytest

from biomed_mcp.bench import questions as Q
from biomed_mcp.bench.scorer import RunRecord, aggregate, score
from biomed_mcp.core.store import SnapshotStore

DB, QS = Path("data/bench.duckdb"), Path("data/questions.json")
pytestmark = pytest.mark.skipif(not (DB.exists() and QS.exists()), reason="run `biomed-bench generate` first")


def oracle_run(q):
    ans = q.answer
    return RunRecord(q.id, tool_calls=q.expected_calls, final_answer=ans, citations=q.support_hashes[-1:])


def test_perfect_agent_scores_100():
    store, qs = SnapshotStore(str(DB), read_only=True), Q.load(str(QS))
    rows = [score(q, oracle_run(q), store) for q in qs]
    bad = [r for r in rows if not (r["tool_ok"] and r["arg_acc"] == 1 and r["answer_ok"] and r["citation_ok"])]
    assert not bad, bad[:3]
    assert aggregate(rows)["answer_accuracy"] == 1.0


def test_bad_agent_scores_low():
    store, qs = SnapshotStore(str(DB), read_only=True), Q.load(str(QS))
    rows = []
    for q in qs:
        r = RunRecord(q.id, tool_calls=[{"tool": "search_pubmed", "args": {"term": "x"}}], final_answer=-1,
                      citations=["deadbeef"])
        rows.append(score(q, r, store))
    agg = aggregate(rows)
    assert agg["answer_accuracy"] < 0.1 and agg["citation_validity"] == 0
