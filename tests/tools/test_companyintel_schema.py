import copy
import json
import sqlite3
from pathlib import Path

import pytest

from companyintel.schema import (
    SCHEMA_VERSION,
    adapt_legacy_graph,
    canonical_chain_fixture,
    validate_graph,
)


@pytest.fixture
def companyintel_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "companyintel"))
    monkeypatch.setenv("HERMES_PROFILE", "companyintel")
    return Path(tmp_path / "companyintel")


def test_canonical_chain_fixture_validates():
    graph = canonical_chain_fixture()
    receipt = validate_graph(graph)
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["nodes"] == 5
    assert receipt["edges"] == 4


def test_checked_in_chain_fixture_validates():
    path = Path(__file__).parents[1] / "fixtures" / "companyintel" / "url-phone-marketplace-legal-registry.v2.json"
    receipt = validate_graph(json.loads(path.read_text(encoding="utf-8")))
    assert receipt["edges"] == 4


def test_evidence_rows_are_immutable(companyintel_home):
    from tools.companyintel_graph_tool import _connect, companyintel_graph

    run_id = "run_evidence_immutable_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    observation = json.loads(companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "immutable@example.com", "source_url": "https://example.com/contact",
        "excerpt": "immutable@example.com", "relation_from_node_id": init["seed_node_id"],
    }))
    conn = _connect(run_id)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE evidence SET excerpt='mutated' WHERE evidence_id=?", (observation["evidence_id"],))
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM evidence WHERE evidence_id=?", (observation["evidence_id"],))
    conn.close()


def test_legacy_shape_is_imported_but_not_accepted_as_canonical():
    legacy = {
        "schema_version": "corporate-intelligence-graph/v1",
        "target": "https://example.test",
        "nodes": [{"id": "n1", "type": "url", "value": "https://example.test"}],
        "edges": [],
    }
    with pytest.raises(ValueError, match="noncanonical"):
        validate_graph(legacy)
    migrated = adapt_legacy_graph(legacy)
    assert migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["nodes"][0]["node_id"] == "n1"


def test_legacy_edge_keys_are_normalized_and_canonical_shape_rejects_them():
    graph = canonical_chain_fixture()
    broken = copy.deepcopy(graph)
    broken["edges"][0]["from"] = broken["edges"][0].pop("from_node")
    with pytest.raises(ValueError, match="from_node/to_node"):
        validate_graph(broken)


def test_relation_rejects_invalid_endpoint_types():
    graph = canonical_chain_fixture()
    broken = copy.deepcopy(graph)
    broken["nodes"][1]["node_type"] = "legal_entity_candidate"
    with pytest.raises(ValueError, match="endpoint types"):
        validate_graph(broken)


def test_adapter_does_not_mutate_legacy_input():
    legacy = {"target": "https://example.test", "nodes": [], "edges": []}
    snapshot = copy.deepcopy(legacy)
    adapt_legacy_graph(legacy)
    assert legacy == snapshot


def test_graph_tool_rejects_noncanonical_relation(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_relation_guard_001"
    init = json.loads(companyintel_graph({
        "action": "init_run", "run_id": run_id, "target_url": "https://example.com",
    }))
    result = json.loads(companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "invalid-relation@example.com", "source_url": "https://example.com/contact",
        "excerpt": "invalid relation fixture", "relation_from_node_id": init["seed_node_id"],
        "relation": "arbitrary_noncanonical_relation",
    }))
    assert result["ok"] is False
    assert "unsupported graph relation" in result["error"]
