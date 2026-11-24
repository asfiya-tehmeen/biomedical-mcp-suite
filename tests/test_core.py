import json

import httpx
import pytest

from biomed_mcp.core.client import BiomedClient, ReplayMiss
from biomed_mcp.core.provenance import sha256_hex
from biomed_mcp.core.store import SnapshotStore
from biomed_mcp.servers import _base, openfda, rxnorm, trials


def make_client(handler, mode="live", store=None):
    calls = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return handler(request)

    c = BiomedClient(store or SnapshotStore(":memory:"), mode=mode, transport=httpx.MockTransport(wrapped),
                     max_retries=1)
    c.calls = calls
    return c


def ok(payload):
    return lambda req: httpx.Response(200, json=payload)


async def test_provenance_and_cache_hit():
    c = make_client(ok({"a": 1}))
    data, p1 = await c.get_json("t", "https://rxnav.nlm.nih.gov/x", {"q": "v"})
    data2, p2 = await c.get_json("t", "https://rxnav.nlm.nih.gov/x", {"q": "v"})
    assert data == data2 == {"a": 1}
    assert len(c.calls) == 1                      # second call served from cache
    assert not p1.from_cache and p2.from_cache
    assert p1.sha256 == p2.sha256 == sha256_hex(c.store.get_by_hash(p1.sha256).body)
    assert p1.retrieved_at == p2.retrieved_at     # retrieval time is the ORIGINAL fetch time
    assert "q=v" in p1.source_url


async def test_replay_mode_never_hits_network():
    store = SnapshotStore(":memory:")
    live = make_client(ok({"a": 1}), store=store)
    await live.get_json("t", "https://rxnav.nlm.nih.gov/x", {"q": 1})
    store.freeze_all()
    replay = make_client(lambda r: pytest.fail("network used"), mode="replay", store=store)
    data, prov = await replay.get_json("t", "https://rxnav.nlm.nih.gov/x", {"q": 1})
    assert data == {"a": 1} and prov.frozen and prov.from_cache
    with pytest.raises(ReplayMiss):
        await replay.get_json("t", "https://rxnav.nlm.nih.gov/x", {"q": 2})


async def test_retry_on_429_then_success():
    state = {"n": 0}

    def h(req):
        state["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "0"}) if state["n"] == 1 else httpx.Response(200, json={"ok": 1})

    c = make_client(h)
    data, _ = await c.get_json("t", "https://rxnav.nlm.nih.gov/y")
    assert data == {"ok": 1} and state["n"] == 2


async def test_openfda_404_means_zero_results():
    c = make_client(lambda r: httpx.Response(404, json={"error": {"code": "NOT_FOUND"}}))
    _base.set_client(c)
    try:
        out = await openfda.drug_event_count("nosuchdrug")
        assert out["result"] == {"total": 0}
        assert out["provenance"]["sha256"]
    finally:
        _base.set_client(None)


async def test_trials_count_builds_expected_query():
    c = make_client(ok({"totalCount": 42, "studies": []}))
    _base.set_client(c)
    try:
        out = await trials.count_trials(condition="asthma", status="RECRUITING", phase="PHASE3", country="Saudi Arabia")
        assert out["result"]["total_count"] == 42
        url = c.calls[0]
        assert "query.cond=asthma" in url and "AREA%5BPhase%5DPHASE3" in url and "countTotal=true" in url
    finally:
        _base.set_client(None)


async def test_rxnorm_ingredients_parsing():
    payload = {"relatedGroup": {"conceptGroup": [{"tty": "IN", "conceptProperties": [{"rxcui": "83367", "name": "atorvastatin"}]}]}}
    c = make_client(ok(payload))
    _base.set_client(c)
    try:
        out = await rxnorm.get_ingredients("153165")
        assert out["result"]["ingredients"] == [{"rxcui": "83367", "name": "atorvastatin"}]
        json.dumps(out)  # envelope must be JSON-serialisable
    finally:
        _base.set_client(None)
