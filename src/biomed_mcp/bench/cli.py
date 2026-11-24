"""biomed-bench CLI.

  biomed-bench generate  --db data/bench.duckdb --out data/questions.json   (live; builds snapshot + questions)
  biomed-bench freeze    --db data/bench.duckdb                             (mark snapshot frozen)
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from ..core.client import BiomedClient
from ..core.store import SnapshotStore
from ..servers import _base
from . import questions


async def _generate(db: str, out: str) -> None:
    store = SnapshotStore(db)
    _base.set_client(BiomedClient(store, mode="live"))
    qs = await questions.generate_all()
    questions.save(qs, out)
    print(f"{len(qs)} questions -> {out}; snapshot rows: {store.count()}")
    print(dict(Counter(q.group for q in qs)))


def is_heldout(qid: str) -> bool:
    """Deterministic 20% held-out split; never tune prompts on these."""
    import hashlib
    return int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 5 == 0


CORE_PLAN = {"trials": 8, "rxnorm": 4, "labels": 3, "pubmed": 3, "openfda": 3, "multihop": 6, "unanswerable": 3}


def core_subset(qs: list) -> list:
    """Fixed 30-question stratified core: deterministic (id-hash order), dev split only, so it never changes
    when the pool grows and never touches held-out questions."""
    import hashlib
    out = []
    for g, n in CORE_PLAN.items():
        pool = sorted((q for q in qs if q.group == g and not is_heldout(q.id)),
                      key=lambda q: hashlib.sha256(("core" + q.id).encode()).hexdigest())
        out += pool[:n]
    return out


def _report(path: str, db: str, split: str) -> None:
    import json
    from collections import defaultdict
    from .scorer import RunRecord, aggregate, score
    store = SnapshotStore(db, read_only=True)
    qs = {q.id: q for q in questions.load("data/questions.json")}
    groups = defaultdict(list)
    latest: dict[tuple, dict] = {}  # a resumed run may have retried a question: keep the newest row
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        c = d["config"]
        latest[(c["model"], c["n_tools"], c["variant"], c["inject"], c["sanitise"], d["record"]["question_id"])] = d
    for d in latest.values():
        q = qs.get(d["record"]["question_id"])
        if q is None or (split != "all" and is_heldout(q.id) != (split == "heldout")):
            continue
        c = d["config"]
        groups[(c["model"], c["n_tools"], c["variant"], c["inject"], c["sanitise"])].append(
            score(q, RunRecord(**d["record"]), store))
    print("model | tools | variant | inj | san | n | tool | final_tool | args | answer | cite | injected | misses | salvaged | tokens | median_sec")
    for k, rows in sorted(groups.items(), key=str):
        a = aggregate(rows)
        print(" | ".join(str(x) for x in (*k, a["n"], *(f"{a[m]:.2f}" for m in (
            "tool_selection", "final_tool", "argument_accuracy", "answer_accuracy", "citation_validity", "injection_success")),
            a["replay_misses"], a["salvaged"], f"{a['mean_tokens']:.0f}", f"{a['mean_latency_s']:.1f}")))
    print("\nfailure taxonomy (count per config):")
    from collections import Counter
    for k, rows in sorted(groups.items(), key=str):
        c = Counter(r["failure"] for r in rows)
        print(" ", f"{k[0].split('/')[-1]} tools={k[1]} {k[2]} inj={k[3]} san={k[4]}:", dict(c.most_common()))


def main() -> None:
    ap = argparse.ArgumentParser(prog="biomed-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--db", default="data/bench.duckdb")
    g.add_argument("--out", default="data/questions.json")
    f = sub.add_parser("freeze")
    f.add_argument("--db", default="data/bench.duckdb")
    r = sub.add_parser("run")
    r.add_argument("--db", default="data/bench.duckdb")
    r.add_argument("--questions", default="data/questions.json")
    r.add_argument("--out", default="results/run.jsonl")
    r.add_argument("--models", nargs="+", default=["openai/gpt-oss-20b"])
    r.add_argument("--tool-counts", nargs="+", type=int, default=[18])
    r.add_argument("--variants", nargs="+", default=["detailed"], choices=["minimal", "detailed", "examples"])
    r.add_argument("--injection", nargs="+", type=float, default=[0.0])
    r.add_argument("--sanitise", nargs="+", type=int, default=[0])
    r.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--core", action="store_true", help="use the fixed 30-question stratified core (dev split)")
    r.add_argument("--per-group", type=int, default=0, help="take N questions from each group")
    rp = sub.add_parser("report")
    rp.add_argument("--in", dest="inp", default="results/run.jsonl")
    rp.add_argument("--db", default="data/bench.duckdb")
    rp.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    a = ap.parse_args()
    if a.cmd == "run":
        from .runner import run_grid
        qs = [q for q in questions.load(a.questions) if a.split == "all" or is_heldout(q.id) == (a.split == "heldout")]
        if a.core:
            qs = core_subset(questions.load(a.questions))
        if a.per_group:
            seen: dict[str, int] = {}
            sel = []
            for q in qs:
                if seen.get(q.group, 0) < a.per_group:
                    seen[q.group] = seen.get(q.group, 0) + 1
                    sel.append(q)
            qs = sel
        qs = qs[: a.limit] if a.limit else qs
        asyncio.run(run_grid(questions=qs, models=a.models, tool_counts=a.tool_counts, variants=a.variants,
                             injection=a.injection, sanitise_opts=[bool(x) for x in a.sanitise], db=a.db, out=a.out))
    elif a.cmd == "report":
        _report(a.inp, a.db, a.split)
    elif a.cmd == "generate":
        asyncio.run(_generate(a.db, a.out))
    else:
        print(f"froze {SnapshotStore(a.db).freeze_all()} responses")


if __name__ == "__main__":
    main()
