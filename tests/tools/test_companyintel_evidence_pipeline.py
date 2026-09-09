import json
from pathlib import Path
import pytest


@pytest.fixture
def companyintel_home(tmp_path, monkeypatch):
    home = tmp_path / "companyintel"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "companyintel")
    return home



def test_inventory_promotion_creates_code_owned_finding_and_ambiguous_candidate(companyintel_home):
    from tools.companyintel_graph_tool import _promote_inventory_identity, _connect, companyintel_graph

    run_id = "run_promotion_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    observation = json.loads(companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "url",
        "value": "https://example.com/about", "source_url": "https://example.com/about",
        "excerpt": "Example Company", "retrieval_method": "bounded_http_extraction",
    }))
    assert observation["source_tier"] == "B"
    _promote_inventory_identity(run_id, init["seed_node_id"], {"metadata": {"title": "Example Company"}})
    graph = json.loads(companyintel_graph({"action": "projection", "run_id": run_id, "view": "graph"}))["data"]
    assert any(node["node_type"] == "finding" for node in graph["nodes"])
    assert graph["findings"][0]["finding_type"] == "first_party_web_presence"
    assert graph["identity_candidates"][0]["status"] == "AMBIGUOUS"
    assert graph["identity_candidates"][0]["evidence_ids"]
    conn = _connect(run_id)
    assert conn.execute("SELECT COUNT(*) FROM findings WHERE run_id=?", (run_id,)).fetchone()[0] == 1
    conn.close()
