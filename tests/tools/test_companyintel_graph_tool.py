import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def companyintel_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "companyintel"))
    monkeypatch.setenv("HERMES_PROFILE", "companyintel")
    return Path(tmp_path / "companyintel")


def test_companyintel_graph_tool_is_profile_gated(companyintel_home):
    from tools.companyintel_graph_tool import check_companyintel_requirements

    assert check_companyintel_requirements() is True


def test_init_run_persists_canonical_graph_and_checkpoint(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    result = json.loads(companyintel_graph({
        "action": "init_run",
        "run_id": "run_test_001",
        "target_url": "https://example.com",
    }))

    assert result["ok"] is True
    graph_path = companyintel_home / "companyintel" / "runs" / "run_test_001" / "graph.json"
    checkpoint_path = companyintel_home / "companyintel" / "runs" / "run_test_001" / "checkpoints" / "latest.json"
    assert graph_path.exists()
    assert checkpoint_path.exists()
    graph = json.loads(graph_path.read_text())
    assert graph["schema_version"] == "corporate-intelligence-graph/v2"
    assert graph["target"]["domain"] == "example.com"
    assert graph["research_state"]["status"] == "RUNNING"


def test_observation_creates_durable_node_evidence_edge_and_typed_frontier(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    init = json.loads(companyintel_graph({"action": "init_run", "run_id": "run_test_002", "target_url": "https://example.com"}))
    result = json.loads(companyintel_graph({
        "action": "record_observation",
        "run_id": "run_test_002",
        "node_type": "phone",
        "value": "+380 50 123 4567",
        "source_url": "https://example.com/contact",
        "excerpt": "Call +380 50 123 4567",
        "relation_from_node_id": init["seed_node_id"],
        "relation": "publishes",
    }))

    assert result["ok"] is True
    graph_path = companyintel_home / "companyintel" / "runs" / "run_test_002" / "graph.json"
    graph = json.loads(graph_path.read_text())
    assert len(graph["nodes"]) >= 2
    assert any(node["node_type"] == "phone" for node in graph["nodes"])
    assert len(graph["evidence"]) == 1
    assert len(graph["edges"]) == 1
    assert len(graph["frontier"]) >= 1
    assert graph["search_log"] == []
    assert graph["research_state"]["round"] == 0


def test_retry_wait_does_not_block_other_open_tasks(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import companyintel_graph, _connect
    run_id = "run_task_local_retry_001"
    companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"})
    conn = _connect(run_id)
    rows = conn.execute("SELECT task_id FROM frontier_tasks ORDER BY task_id").fetchall()
    conn.execute("UPDATE frontier_tasks SET state='RETRY_WAIT',available_at=? WHERE task_id=?", (9999999999, rows[0]["task_id"]))
    conn.execute("UPDATE run_state SET status='RETRY_WAIT',retry_at=?,retry_reason=? WHERE run_id=?", (9999999999, "timeout", run_id))
    conn.commit(); conn.close()
    claimed = json.loads(companyintel_graph({"action": "claim_frontier", "run_id": run_id, "worker_id": "local-retry-worker", "pivot_type": "site_inventory"}))
    assert claimed["task"] is not None
    assert claimed["task"]["state"] == "CLAIMED"


def test_run_frontier_records_saturation_when_all_tasks_completed(companyintel_home, monkeypatch):
    run_id = "run_saturation_001"
    monkeypatch.setattr("tools.companyintel_graph_tool._execute_inventory_frontier", lambda args: {"outcome": "NO_TASK"})
    monkeypatch.setattr("tools.companyintel_graph_tool._execute_frontier", lambda args: {"outcome": "NO_TASK"})
    from tools.companyintel_graph_tool import companyintel_graph, _connect
    companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"})
    conn = _connect(run_id)
    conn.execute("UPDATE frontier_tasks SET state='COMPLETED'")
    conn.commit(); conn.close()
    result = json.loads(companyintel_graph({"action": "run_frontier", "run_id": run_id, "worker_id": "sat-worker", "max_tasks": 2}))
    assert result["ok"] is True
    summary = json.loads(companyintel_graph({"action": "summary", "run_id": run_id}))
    assert summary["frontier_open"] == 0
    assert summary["saturation_reason"].startswith("FRONTIER_EXHAUSTED:")
    conn = _connect(run_id)
    assert conn.execute("SELECT COUNT(*) FROM frontier_tasks WHERE state='SKIPPED_UNSUPPORTED'").fetchone()[0] == 0
    conn.close()


def test_run_frontier_invokes_durable_reasoning_cycle_and_persists_decisions(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_reasoning_cycle_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    evidence = json.loads(companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "research@example.com", "source_url": "https://example.com/contact",
        "excerpt": "Email research@example.com", "relation_from_node_id": init["seed_node_id"],
    }))
    calls = []

    class FakeController:
        def generate_hypotheses_with_llm(self, *, payload, evidence_ids):
            calls.append("generate_hypotheses")
            return [{"decision_type": "GENERATE_HYPOTHESIS", "hypothesis_id": "h1", "run_id": run_id, "kind": "identity", "claim": "candidate", "subject_node_ids": [init["seed_node_id"]], "supporting_evidence_ids": [evidence["evidence_id"]], "confidence": 0.5, "status": "HYPOTHESIS"}]
        def compare_candidates_with_llm(self, *, payload):
            calls.append("compare_candidates")
            return {"decision_type": "COMPARE_CANDIDATES", "decision_id": "c1", "run_id": run_id, "candidates": [], "dominance": "NONE", "recommended_decision": "REQUIRE_MORE_EVIDENCE", "verified_candidate_ids": [], "rationale": "compare"}
        def analyze_contradictions_with_llm(self, *, payload):
            calls.append("analyze_contradictions")
            return {"decision_type": "ANALYZE_CONTRADICTIONS", "run_id": run_id, "contradictions": [], "unresolved": [], "rationale": "none"}
        def select_equal_priority_pivot_with_llm(self, *, payload):
            calls.append("select_equal_priority_pivot")
            return {"decision_type": "SELECT_EQUAL_PRIORITY_PIVOT", "decision_id": "p1", "run_id": run_id, "node_id": payload["node_id"], "selected_pivot": payload["eligible_pivots"][0]["pivot_type"], "eligible_pivots": payload["eligible_pivots"], "rationale": "highest value"}
        def interpret_ambiguity_with_llm(self, *, payload):
            calls.append("interpret_ambiguity")
            return {"decision_type": "INTERPRET_AMBIGUITY", "decision_id": "a1", "run_id": run_id, "ambiguity_type": "candidate", "subject_ids": payload["subject_ids"], "evidence_ids": payload["evidence_ids"], "decision": "REQUIRE_MORE_EVIDENCE", "conflicts": [], "rationale": "more evidence"}
        def decide_weak_branch_with_llm(self, *, payload):
            calls.append("decide_weak_branch")
            return {"decision_type": "WEAK_CLUE_BRANCH", "decision_id": "w1", "run_id": run_id, "clue_id": "clue", "branch": False, "decision": "DO_NOT_BRANCH", "allowed_pivots": [], "max_new_tasks": 0, "max_budget_cost": 0, "current_depth": 0, "max_depth": 2, "rationale": "too weak"}

    monkeypatch.setattr("tools.companyintel_reasoning_controller.CompanyIntelReasoningController", FakeController)
    result = json.loads(companyintel_graph({"action": "reasoning_cycle", "run_id": run_id, "force_all": True}))
    assert result["ok"] is True
    assert calls == ["generate_hypotheses", "compare_candidates", "analyze_contradictions", "select_equal_priority_pivot", "interpret_ambiguity", "decide_weak_branch"]
    conn = sqlite3.connect(companyintel_home / "companyintel" / "runs" / run_id / "graph.sqlite3")
    rows = conn.execute("SELECT decision_type FROM reasoning_decisions WHERE run_id=? ORDER BY created_at, decision_type", (run_id,)).fetchall()
    assert {row[0] for row in rows} >= {"GENERATE_HYPOTHESIS", "SELECT_EQUAL_PRIORITY_PIVOT", "INTERPRET_AMBIGUITY", "WEAK_CLUE_BRANCH"}


def test_existing_v1_database_records_v2_migration_and_integrity(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_schema_migration_001"
    db_path = companyintel_home / "companyintel" / "runs" / run_id / "graph.sqlite3"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE run_state (run_id TEXT PRIMARY KEY, target_url TEXT NOT NULL, status TEXT NOT NULL, round INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, resume_count INTEGER NOT NULL DEFAULT 0)")
    conn.commit()
    conn.close()
    result = json.loads(companyintel_graph({"action": "summary", "run_id": run_id}))
    assert result["ok"] is False or result["status"] == "RUNNING" or result["status"] == ""
    conn = sqlite3.connect(db_path)
    migration = conn.execute("SELECT from_version,to_version,integrity_json FROM schema_migrations WHERE migration_id=?", ("corporate-intelligence-graph-v1-to-v2",)).fetchone()
    assert migration[0] == "corporate-intelligence-graph/v1"
    assert migration[1] == "corporate-intelligence-graph/v2"
    assert json.loads(migration[2])["missing_tables"] == []


def test_reasoning_cycle_never_closes_frontier_or_sets_terminal_status(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_reasoning_authority_001"
    json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    result = json.loads(companyintel_graph({"action": "reasoning_cycle", "run_id": run_id}))
    assert result["ok"] is True
    summary = json.loads(companyintel_graph({"action": "summary", "run_id": run_id}))
    assert summary["status"] == "RUNNING"
    assert summary["frontier_open"] >= 1


def test_checkpoint_is_atomic_and_summary_is_compact(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    companyintel_graph({"action": "init_run", "run_id": "run_test_003", "target_url": "https://example.com"})
    result = json.loads(companyintel_graph({"action": "summary", "run_id": "run_test_003"}))

    assert result["ok"] is True
    assert result["run_id"] == "run_test_003"
    assert result["nodes"] == 1
    assert result["edges"] == 0
    assert result["evidence"] == 0
    assert result["frontier_open"] == 1
    assert result["search_log"] == 0
    assert result["round"] == 0
    assert result["status"] == "RUNNING"
    assert result["retry_count"] == 0
    assert result["resume_count"] == 0
    assert result["worker_checkpoints"] == []


def test_run_service_event_log_is_ordered_idempotent_and_replayable(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    companyintel_graph({"action": "init_run", "run_id": "run_service_001", "target_url": "https://example.com"})
    first = json.loads(companyintel_graph({
        "action": "append_event", "run_id": "run_service_001", "event_type": "inventory.completed",
        "payload": {"items": 3}, "idempotency_key": "inventory:1",
    }))
    duplicate = json.loads(companyintel_graph({
        "action": "append_event", "run_id": "run_service_001", "event_type": "inventory.completed",
        "payload": {"items": 999}, "idempotency_key": "inventory:1",
    }))
    second = json.loads(companyintel_graph({
        "action": "append_event", "run_id": "run_service_001", "event_type": "frontier.scheduled",
        "payload": {"tasks": 4}, "idempotency_key": "frontier:1",
    }))

    assert first["event_id"] == 2
    assert duplicate["event_id"] == first["event_id"]
    assert duplicate["idempotent"] is True
    assert duplicate["payload"] == {"items": 3}
    assert second["event_id"] == 3

    replay = json.loads(companyintel_graph({
        "action": "replay_events", "run_id": "run_service_001", "after_event_id": 2,
    }))
    assert [event["event_id"] for event in replay["events"]] == [3]
    assert replay["events"][0]["event"] == "frontier.scheduled"


def test_checkpoint_after_five_retrievals_survives_reopen_with_durable_graph(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_retrieval_checkpoint_001"
    init = json.loads(companyintel_graph({
        "action": "init_run", "run_id": run_id, "target_url": "https://example.com",
    }))
    observed = json.loads(companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "phone",
        "value": "+380****4567", "source_url": "https://example.com/contact",
        "excerpt": "bounded phone evidence", "relation_from_node_id": init["seed_node_id"],
        "relation": "publishes",
    }))
    assert observed["ok"] is True

    for index in range(5):
        result = json.loads(companyintel_graph({
            "action": "record_search", "run_id": run_id, "search_id": f"search-{index}",
            "pivot_node_id": init["seed_node_id"], "pivot_type": "exact_search",
            "query": f"example query {index}", "source": "fixture",
            "status": "COMPLETED_ZERO_RESULTS", "new_node_ids": [],
        }))
        assert result["ok"] is True

    # A fresh connection is the process-boundary/crash analogue: the durable
    # SQLite state, not an in-memory projection, must contain the checkpoint.
    db = companyintel_home / "companyintel" / "runs" / run_id / "graph.sqlite3"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM nodes WHERE node_id=?", (init["seed_node_id"],)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0] == 5
        checkpoint = conn.execute(
            "SELECT reason FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        assert checkpoint == ("retrieval_batch_5",)

    projection = json.loads(companyintel_graph({"action": "projection", "run_id": run_id, "view": "graph"}))
    assert projection["ok"] is True
    assert len(projection["data"]["nodes"]) >= 2
    assert len(projection["data"]["edges"]) >= 1
    assert len(projection["data"]["evidence"]) >= 1
    assert len(projection["data"]["search_log"]) == 5


def test_run_service_transition_is_durable_and_fail_closed(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    companyintel_graph({"action": "init_run", "run_id": "run_service_002", "target_url": "https://example.com"})
    retry = json.loads(companyintel_graph({
        "action": "transition_run", "run_id": "run_service_002", "status": "RETRY_WAIT",
        "reason": "provider timeout", "idempotency_key": "transition:retry",
    }))
    resumable = json.loads(companyintel_graph({
        "action": "transition_run", "run_id": "run_service_002", "status": "RESUMABLE",
        "reason": "retry window elapsed", "idempotency_key": "transition:resumable",
    }))
    summary = json.loads(companyintel_graph({"action": "summary", "run_id": "run_service_002"}))
    replay = json.loads(companyintel_graph({"action": "replay_events", "run_id": "run_service_002"}))

    assert retry["from"] == "RUNNING"
    assert resumable["from"] == "RETRY_WAIT"
    assert summary["status"] == "RESUMABLE"
    assert summary["event_count"] == 3
    assert [event["event"] for event in replay["events"]] == ["run.started", "run.status", "run.status"]
    assert [event["payload"].get("to", event["payload"].get("status")) for event in replay["events"]] == ["RUNNING", "RETRY_WAIT", "RESUMABLE"]

    invalid = json.loads(companyintel_graph({
        "action": "transition_run", "run_id": "run_service_002", "status": "COMPLETED",
    }))
    assert invalid["ok"] is False
    assert "invalid run transition" in invalid["error"]


def test_projection_views_are_durable_and_report_is_gated(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    companyintel_graph({"action": "init_run", "run_id": "run_projection_001", "target_url": "https://example.com"})
    for view in ("graph", "frontier", "evidence", "report"):
        result = json.loads(companyintel_graph({"action": "projection", "run_id": "run_projection_001", "view": view}))
        assert result["ok"] is True
        assert result["view"] == view
        assert result["run_id"] == "run_projection_001"
    report = json.loads(companyintel_graph({"action": "projection", "run_id": "run_projection_001", "view": "report"}))
    assert report["data"]["ready"] is False
    assert report["data"]["reason"] == "pre_report_gate_not_satisfied"


def test_resume_frontier_requires_resumable_state(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    companyintel_graph({"action": "init_run", "run_id": "run_resume_001", "target_url": "https://example.com"})
    not_resumable = json.loads(companyintel_graph({"action": "resume_frontier", "run_id": "run_resume_001"}))
    assert not_resumable["ok"] is False
    assert not_resumable["reason"] == "run_not_resumable"

    companyintel_graph({"action": "transition_run", "run_id": "run_resume_001", "status": "RETRY_WAIT", "reason": "timeout"})
    companyintel_graph({"action": "transition_run", "run_id": "run_resume_001", "status": "RESUMABLE", "reason": "retry ready"})
    resumed = json.loads(companyintel_graph({"action": "resume_frontier", "run_id": "run_resume_001", "additional_tasks": 1}))
    assert resumed["ok"] is True
    assert resumed["resumed"] is True
    assert resumed["status"] in {"RUNNING", "RETRY_WAIT", "PARTIAL", "COMPLETED"}


def test_report_artifact_persists_after_gate_and_get_is_read_only(companyintel_home):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph

    run_id = "run_report_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    observation = json.loads(companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "info@example.com", "source_url": "https://example.com/contact",
        "excerpt": "info@example.com", "relation_from_node_id": init["seed_node_id"], "relation": "publishes",
    }))
    assert observation["ok"] is True
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET status='COMPLETED',saturation_reason='coverage_complete',updated_at=? WHERE run_id=?", ("2026-08-11T00:00:00Z", run_id))
    conn.execute("UPDATE frontier SET state='EXHAUSTED' WHERE node_id=?", (init["seed_node_id"],))
    conn.commit()
    _materialize(conn, run_id)
    conn.close()

    generated = json.loads(companyintel_graph({"action": "generate_report", "run_id": run_id, "report_text": "validated report", "model": "test-model", "provider": "test-provider", "review_text": "same-model review"}))
    assert generated["ok"] is True
    artifact = generated["report"]
    assert artifact["graph_revision"]
    assert artifact["evidence_ids"]
    assert artifact["model"] == "test-model"
    assert artifact["provider"] == "test-provider"
    assert artifact["review_label"] == "same-model-review"
    assert artifact["review_independence"] == "not_independent_same_model"
    assert artifact["gate_result"]["result"] == "PASS"

    read_back = json.loads(companyintel_graph({"action": "projection", "run_id": run_id, "view": "report"}))
    assert read_back["data"]["ready"] is True
    assert read_back["data"]["artifact"]["report_id"] == artifact["report_id"]
    assert read_back["data"]["artifact"]["text"] == "validated report"
    conn = _connect(run_id)
    assert conn.execute("SELECT COUNT(*) FROM reasoning_decisions WHERE run_id=?", (run_id,)).fetchone()[0] == 3
    decision_types = {row[0] for row in conn.execute("SELECT decision_type FROM reasoning_decisions WHERE run_id=?", (run_id,)).fetchall()}
    assert {"SYNTHESIZE_REPORT", "COMPARE_CANDIDATES", "ANALYZE_CONTRADICTIONS"} <= decision_types
    assert conn.execute("SELECT COUNT(*) FROM report_reviews WHERE report_id=?", (artifact["report_id"],)).fetchone()[0] == 1
    review_row = conn.execute("SELECT independent_review,review_label FROM report_reviews WHERE report_id=?", (artifact["report_id"],)).fetchone()
    assert review_row[0] == 0
    assert review_row[1] == "same-model-review"
    conn.close()


def test_report_artifact_uses_fixed_openrouter_reasoning_model(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph

    run_id = "run_report_llm_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    companyintel_graph({"action": "record_observation", "run_id": run_id, "node_type": "email", "value": "info@example.com", "source_url": "https://example.com/contact", "excerpt": "info@example.com", "relation_from_node_id": init["seed_node_id"], "relation": "publishes"})
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET status='COMPLETED',saturation_reason='coverage_complete' WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier SET state='EXHAUSTED' WHERE node_id=?", (init["seed_node_id"],))
    conn.commit()
    _materialize(conn, run_id)
    conn.close()

    calls = []
    def fake_request(operation, payload):
        calls.append(operation)
        if operation == "synthesize_report":
            return {"report_text": "LLM validated report", "claims": [{"text": "Observed email", "evidence_ids": [json.loads(companyintel_graph({"action": "projection", "run_id": run_id, "view": "evidence"}))["data"]["evidence"][0]["evidence_id"]]}]}
        return {"verdict": "PASS_WITH_LIMITATIONS", "review_text": "Same model review", "findings": []}

    monkeypatch.setattr("tools.companyintel_reasoning_controller.CompanyIntelReasoningController._request", lambda self, operation, payload: fake_request(operation, payload))
    generated = json.loads(companyintel_graph({"action": "generate_report", "run_id": run_id, "reasoning_enabled": True}))
    assert generated["ok"] is True
    assert calls == ["synthesize_report", "review_report"]
    assert generated["report"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert generated["report"]["provider"] == "openrouter"
    assert generated["report"]["independent_review"] is False


def test_non_companyintel_profile_does_not_expose_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    from tools.companyintel_graph_tool import check_companyintel_requirements

    assert check_companyintel_requirements() is False
