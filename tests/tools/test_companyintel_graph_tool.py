import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def companyintel_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "companyintel"))
    monkeypatch.setenv("HERMES_PROFILE", "companyintel")
    return Path(tmp_path / "companyintel")


def _prepare_report_gate(companyintel_graph, _connect, _materialize, run_id, init):
    """Build a code-owned, bounded fixture for strict report-gate tests."""
    companyintel_graph({
        "action": "record_search", "run_id": run_id, "search_id": f"fixture-search-{run_id}",
        "pivot_node_id": init["seed_node_id"], "pivot_type": "exact_search",
        "query": "fixture", "source": "fixture", "provider": "fixture",
        "status": "COMPLETED_ZERO_RESULTS", "new_node_ids": [],
    })
    companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "finding",
        "value": "bounded first-party fixture", "source_url": "https://example.com/about",
        "excerpt": "Bounded first-party fixture evidence",
        "retrieval_method": "bounded_http_extraction",
    })
    conn = _connect(run_id)
    conn.execute("UPDATE frontier_tasks SET state='COMPLETED', last_error=NULL WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier SET state='EXHAUSTED' WHERE node_id IN (SELECT node_id FROM nodes)")
    conn.commit()
    _materialize(conn, run_id)
    conn.close()


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
    planned_pivots = set(graph["frontier"][0]["planned_pivots"])
    frontier_tasks = graph["frontier_tasks"]
    assert frontier_tasks
    assert {task["pivot_type"] for task in frontier_tasks} == planned_pivots
    assert all(isinstance(task["priority"], int) and task["priority"] >= 0 for task in frontier_tasks)
    assert all(
        task["state"]
        in {
            "OPEN",
            "BLOCKED_CAPABILITY",
            "SKIPPED_UNSUPPORTED",
            "SKIPPED_BREADTH_CAP",
            "SKIPPED_IDENTITY_FOCUS",
        }
        for task in frontier_tasks
    )


def test_blocked_capability_is_not_reported_as_completed_zero_results(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_blocked_capability_coverage"
    init = json.loads(companyintel_graph({
        "action": "init_run", "run_id": run_id, "target_url": "https://example.com",
    }))
    observed = json.loads(companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "legal_entity",
        "value": "Example LLC", "source_url": "https://example.com/about",
        "excerpt": "Example LLC legal entity", "retrieval_method": "bounded_http_extraction",
    }))
    assert observed["ok"] is True
    result = json.loads(companyintel_graph({
        "action": "legal_identity_exhaustion", "run_id": run_id,
    }))
    assert result["coverage"]["official_registry"] == "UNAVAILABLE"


def test_unknown_run_operation_does_not_create_graph_database(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_unknown_must_not_create_db"
    result = json.loads(companyintel_graph({
        "action": "append_event",
        "run_id": run_id,
        "event_type": "run.status",
        "payload": {"status": "RUNNING"},
    }))

    assert result["ok"] is False
    assert "not initialized" in result["error"]
    assert not (companyintel_home / "companyintel" / "runs" / run_id / "graph.sqlite3").exists()


def test_concurrent_bootstrap_connections_create_one_migration_row(companyintel_home):
    from tools.companyintel_graph_tool import _connect

    def bootstrap(_index):
        conn = _connect("run_bootstrap_race_001", create=True)
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE migration_id=?",
                ("corporate-intelligence-graph-v1-to-v2",),
            ).fetchone()[0]
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(bootstrap, range(12)))
    assert results == [1] * 12


def test_concurrent_projection_and_event_access_does_not_lock_database(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_concurrent_sqlite_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    assert init["ok"] is True

    actions = [
        {"action": "summary", "run_id": run_id},
        {"action": "projection", "run_id": run_id, "view": "graph"},
        {"action": "projection", "run_id": run_id, "view": "frontier"},
        {"action": "projection", "run_id": run_id, "view": "evidence"},
        {"action": "append_event", "run_id": run_id, "event_type": "run.status", "payload": {"status": "running"}, "idempotency_key": "concurrent-status"},
    ]
    with ThreadPoolExecutor(max_workers=len(actions)) as pool:
        results = list(pool.map(companyintel_graph, actions))
    decoded = [json.loads(result) for result in results]
    assert all(result.get("ok") is True for result in decoded), decoded


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


def test_reasoning_provider_failures_are_durable_rejections_not_run_failures(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_reasoning_provider_failure_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    companyintel_graph({
        "action": "record_observation",
        "run_id": run_id,
        "node_type": "email",
        "value": "research@example.com",
        "source_url": "https://example.com/contact",
        "excerpt": "Email research@example.com",
        "relation_from_node_id": init["seed_node_id"],
    })

    class FailingController:
        def _fail(self, **_kwargs):
            raise RuntimeError("provider connection failed")

        generate_hypotheses_with_llm = _fail
        compare_candidates_with_llm = _fail
        analyze_contradictions_with_llm = _fail
        select_equal_priority_pivot_with_llm = _fail
        interpret_ambiguity_with_llm = _fail
        decide_weak_branch_with_llm = _fail

    monkeypatch.setattr("tools.companyintel_reasoning_controller.CompanyIntelReasoningController", FailingController)
    result = json.loads(companyintel_graph({"action": "reasoning_cycle", "run_id": run_id, "force_all": True}))

    assert result["ok"] is True
    assert result["decision_count"] == 6
    conn = sqlite3.connect(companyintel_home / "companyintel" / "runs" / run_id / "graph.sqlite3")
    rows = conn.execute(
        "SELECT decision_json FROM reasoning_decisions WHERE run_id=? AND decision_type='REASONING_REJECTED'",
        (run_id,),
    ).fetchall()
    conn.close()
    assert {json.loads(row[0])["operation"] for row in rows} == {
        "generate_hypotheses",
        "compare_candidates",
        "analyze_contradictions",
        "select_equal_priority_pivot",
        "interpret_ambiguity",
        "decide_weak_branch",
    }
    summary = json.loads(companyintel_graph({"action": "summary", "run_id": run_id}))
    assert summary["status"] == "RUNNING"


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

    orphan_init = json.loads(companyintel_graph({"action": "init_run", "run_id": "run_service_orphan", "target_url": "https://example.com"}))
    orphan_resumable = json.loads(companyintel_graph({"action": "transition_run", "run_id": "run_service_orphan", "status": "RESUMABLE", "reason": "gateway restarted"}))
    assert orphan_init["ok"] is True
    assert orphan_resumable["ok"] is True
    assert orphan_resumable["from"] == "RUNNING"

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


def test_pre_report_gate_rejects_auth_failed_low_quality_search(companyintel_home):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph

    run_id = "run_report_degraded_search_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "info@example.com", "source_url": "https://example.com/contact",
        "excerpt": "inventory-only evidence", "relation_from_node_id": init["seed_node_id"],
        "relation": "publishes", "source_tier": "C",
    })
    for index in range(4):
        companyintel_graph({
            "action": "record_search", "run_id": run_id, "search_id": f"auth-failed-{index}",
            "pivot_node_id": init["seed_node_id"], "pivot_type": "exact_search",
            "query": f"query-{index}", "source": "tavily", "provider": "tavily",
            "status": "AUTH_FAILED", "new_node_ids": [],
        })
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET status='COMPLETED',saturation_reason='frontier_saturated' WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier SET state='EXHAUSTED' WHERE node_id=?", (init["seed_node_id"],))
    conn.commit(); _materialize(conn, run_id); conn.close()

    result = json.loads(companyintel_graph({"action": "generate_report", "run_id": run_id, "terminal_transition_pending": True}))
    assert result["ok"] is False
    assert result["reason"] == "pre_report_gate_not_satisfied"
    assert result["upstream_reason"] == "SEARCH_PROVIDER_UNAVAILABLE"
    assert result["gate"]["search_provider_auth_failed"] is False
    assert result["gate"]["evidence_quality"] is False
    assert result["gate"]["critical_coverage"] is False


def test_pre_report_gate_rejects_search_success_with_only_tier_c(companyintel_home):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph

    run_id = "run_report_c_only_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "url",
        "value": "https://news.example.net/acme", "source_url": "https://news.example.net/acme",
        "excerpt": "unverified discovery snippet", "source_tier": "C",
        "retrieval_method": "public_search",
    })
    companyintel_graph({
        "action": "record_search", "run_id": run_id, "search_id": "c-only-search",
        "pivot_node_id": init["seed_node_id"], "pivot_type": "exact_search",
        "query": "acme", "source": "tavily", "provider": "tavily",
        "status": "COMPLETED_WITH_RESULTS", "new_node_ids": [],
    })
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET status='COMPLETED',saturation_reason='coverage_complete' WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier_tasks SET state='COMPLETED' WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier SET state='EXHAUSTED'")
    conn.commit(); _materialize(conn, run_id); conn.close()

    result = json.loads(companyintel_graph({"action": "generate_report", "run_id": run_id, "terminal_transition_pending": True}))
    assert result["ok"] is False
    assert result["code"] == "COMPANYINTEL_INSUFFICIENT_EVIDENCE"
    assert "evidence_quality" in result["failed_checks"]
    assert "critical_coverage" in result["failed_checks"]
    assert result["gate"]["search_coverage"] is True


def test_pre_report_gate_rejects_unresolved_candidate_contradiction(companyintel_home):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph

    run_id = "run_report_contradiction_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    evidence = json.loads(companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "info@example.com", "source_url": "https://example.com/contact",
        "excerpt": "info@example.com", "relation_from_node_id": init["seed_node_id"], "relation": "publishes",
        "retrieval_method": "bounded_http_extraction",
    }))
    _prepare_report_gate(companyintel_graph, _connect, _materialize, run_id, init)
    candidate = json.loads(companyintel_graph({
        "action": "record_identity_candidate", "run_id": run_id,
        "candidate_id": "candidate:contradiction",
        "legal_name": "Example One", "legal_id": "UA-11111111", "country": "UA",
        "match_types": ["exact_official_legal_id"],
        "conflicting_dimensions": ["country"],
        "evidence_ids": [evidence["evidence_id"]],
    }))
    assert candidate["ok"] is True
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET status='COMPLETED',saturation_reason='coverage_complete' WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier SET state='EXHAUSTED'")
    conn.commit(); _materialize(conn, run_id); conn.close()

    result = json.loads(companyintel_graph({"action": "generate_report", "run_id": run_id, "terminal_transition_pending": True}))
    assert result["ok"] is False
    assert "no_unresolved_blocking_contradictions" in result["failed_checks"]
    assert result["gate"]["blocking_contradiction_count"] == 1


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
    _prepare_report_gate(companyintel_graph, _connect, _materialize, run_id, init)
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
    assert artifact["review_verdict"] == "PASS"
    assert artifact["review_findings"] == []
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


def test_report_artifact_uses_configured_hermes_reasoning_model(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph

    run_id = "run_report_llm_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    companyintel_graph({"action": "record_observation", "run_id": run_id, "node_type": "email", "value": "info@example.com", "source_url": "https://example.com/contact", "excerpt": "info@example.com", "relation_from_node_id": init["seed_node_id"], "relation": "publishes"})
    _prepare_report_gate(companyintel_graph, _connect, _materialize, run_id, init)
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
    assert generated["report"]["model"] == "gpt-5.6-luna"
    assert generated["report"]["provider"] == "openai-codex"
    assert generated["report"]["independent_review"] is False
    assert generated["report"]["review_verdict"] == "PASS_WITH_LIMITATIONS"
    assert generated["report"]["review_findings"] == []


def test_non_companyintel_profile_does_not_expose_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    from tools.companyintel_graph_tool import check_companyintel_requirements

    assert check_companyintel_requirements() is False


def test_atomic_publication_routes_through_canonical_contract(companyintel_home, monkeypatch):
    from tools import companyintel_graph_tool as module

    artifact = companyintel_home / "companyintel" / "runs" / "run_acl_001" / "graph.json"
    calls = []

    def fake_publish(root, relative_path, content, **kwargs):
        calls.append((root, relative_path, content, kwargs))
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    monkeypatch.setattr(module, "publish_text", fake_publish)

    module._atomic_json(artifact, {"ok": True})

    assert len(calls) == 1
    assert calls[0][0] == companyintel_home / "companyintel"
    assert calls[0][1].as_posix() == "runs/run_acl_001/graph.json"
    assert calls[0][3]["reader_principal"] == "gpt-agent"
    assert calls[0][3]["apply_acl"] is True
    assert artifact.exists()


def test_atomic_publication_failure_has_structured_event(companyintel_home, monkeypatch):
    from tools import companyintel_graph_tool as module

    artifact = companyintel_home / "companyintel" / "runs" / "run_acl_002" / "graph.json"

    def fail_publish(*args, **kwargs):
        raise module.ArtifactFailure("PERMISSION_DENIED", "denied")

    monkeypatch.setattr(module, "publish_text", fail_publish)

    with pytest.raises(module.CompanyIntelArtifactPublicationError) as error:
        module._atomic_json(artifact, {"ok": True})

    assert error.value.event == {
        "code": "ARTIFACT_ACCESS_POSTCONDITION_FAILED",
        "run_id": "run_acl_002",
        "artifact_class": "run_graph",
        "relative_path": "runs/run_acl_002/graph.json",
        "reason_code": "PERMISSION_DENIED",
    }


def test_lifecycle_projection_exposes_authoritative_snapshot_and_breakdown(companyintel_home):
    from tools.companyintel_graph_tool import companyintel_graph

    run_id = "run_lifecycle_snapshot_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    graph = json.loads(companyintel_graph({"action": "projection", "run_id": run_id, "view": "graph"}))["data"]
    summary = json.loads(companyintel_graph({"action": "summary", "run_id": run_id}))

    assert init["execution_generation"] == 0
    assert graph["graph_revision"]
    assert graph["projection_source"] == "durable"
    assert graph["execution_generation"] == 0
    assert graph["resume_generation"] == 0
    assert graph["research_state"]["runtime_status"] == "RUNNING"
    assert graph["research_state"]["terminal_reason"] is None
    assert graph["counts"]["graph_nodes"] == 1
    assert graph["node_breakdown"] == {"domain": 1}
    assert summary["graph_revision"] == graph["graph_revision"]
    assert summary["runtime_status"] == "RUNNING"
    assert summary["counts"]["graph_nodes"] == 1


def test_budget_stop_and_resume_are_generation_aware(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import companyintel_graph, _connect

    run_id = "run_lifecycle_resume_001"
    companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com", "max_tasks": 1})
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET tasks_executed=max_tasks WHERE run_id=?", (run_id,))
    conn.commit()
    conn.close()

    monkeypatch.setattr("tools.companyintel_graph_tool._execute_inventory_frontier", lambda args: {"outcome": "NO_TASK"})
    monkeypatch.setattr("tools.companyintel_graph_tool._execute_frontier", lambda args: {"outcome": "NO_TASK"})
    reasoning_calls = []
    monkeypatch.setattr("tools.companyintel_graph_tool._reasoning_cycle", lambda args: reasoning_calls.append(args))
    budget = json.loads(companyintel_graph({
        "action": "run_frontier",
        "run_id": run_id,
        "worker_id": "budget-worker",
        "max_tasks": 1,
        "reasoning_enabled": True,
    }))
    assert budget["ok"] is True
    assert budget["durable_status"] == "PARTIAL"
    assert budget["reasoning"] == {"ok": True, "skipped": "run_not_running", "durable_status": "PARTIAL"}
    assert reasoning_calls == []
    stopped = json.loads(companyintel_graph({"action": "summary", "run_id": run_id}))
    assert stopped["status"] == "PARTIAL"
    assert stopped["saturation_reason"] == "BUDGET_EXHAUSTED:max_tasks"
    assert stopped["last_stop_reason"] == "BUDGET_EXHAUSTED:frontier_budget"
    assert stopped["terminal_reason"] is None
    assert stopped["execution_generation"] == 0

    monkeypatch.setattr("tools.companyintel_graph_tool._run_frontier", lambda args: {"ok": True, "run_id": run_id, "tasks_executed": 0})
    resumed = json.loads(companyintel_graph({"action": "resume_frontier", "run_id": run_id, "additional_tasks": 1}))
    assert resumed["resumed"] is True
    assert resumed["execution_generation"] == 1
    current = json.loads(companyintel_graph({"action": "summary", "run_id": run_id}))
    assert current["status"] == "RUNNING"
    assert current["saturation_reason"] is None
    assert current["runtime_status"] == "RUNNING"
    assert current["terminal_reason"] is None
    assert current["last_stop_reason"] == "BUDGET_EXHAUSTED:frontier_budget"

    events = json.loads(companyintel_graph({"action": "replay_events", "run_id": run_id}))["events"]
    resumed_events = [event for event in events if event["event"] == "run.resumed"]
    assert len(resumed_events) == 1
    assert resumed_events[0]["payload"]["from_generation"] == 0
    assert resumed_events[0]["payload"]["execution_generation"] == 1
    assert all(event["payload"].get("execution_generation", 0) == 0 for event in events if event["event"] != "run.resumed")


def test_provider_captcha_opens_durable_circuit_and_skips_pending_searches(companyintel_home, monkeypatch):
    from tools import companyintel_graph_tool as graph_module
    from tools import companyintel_public_search as search_module

    def fake_publish(root, relative_path, content, **_kwargs):
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    monkeypatch.setattr(graph_module, "publish_text", fake_publish)

    class Provider:
        name = "brave-free"

        def supports_search(self):
            return True

        def search(self, _query, _limit):
            return {"success": False, "error": "captcha challenge"}

    monkeypatch.setattr(search_module, "_active_provider", lambda: ("brave-free", Provider()))
    monkeypatch.setattr(search_module, "_allow_legacy_html_fallback", lambda: False)

    run_id = "run_provider_circuit_001"
    init = json.loads(graph_module.companyintel_graph({
        "action": "init_run", "run_id": run_id, "target_url": "https://example.com",
    }))
    result = json.loads(graph_module.companyintel_graph({
        "action": "execute_frontier", "run_id": run_id,
        "worker_id": "provider-worker", "pivot_type": "exact_search",
    }))

    assert result["outcome"] == "BLOCKED_BY_CAPTCHA"
    conn = graph_module._connect(run_id)
    tasks = conn.execute(
        "SELECT pivot_type,state FROM frontier_tasks WHERE run_id=? ORDER BY pivot_type",
        (run_id,),
    ).fetchall()
    circuit = conn.execute(
        "SELECT provider,state,failure_count FROM search_provider_circuits WHERE run_id=?",
        (run_id,),
    ).fetchone()
    events = conn.execute(
        "SELECT event_type FROM run_events WHERE run_id=? AND event_type='search.provider_circuit'",
        (run_id,),
    ).fetchall()
    conn.close()

    assert init["ok"] is True
    assert circuit["provider"] == "brave-free"
    assert circuit["state"] == "OPEN"
    assert circuit["failure_count"] == 1
    assert len(events) == 1
    assert all(state in {"SKIPPED_PROVIDER_BLOCKED", "COMPLETED", "OPEN"} for _pivot, state in tasks)


def test_resume_reconciles_old_url_search_tasks_without_touching_completed(companyintel_home, monkeypatch):
    from tools import companyintel_graph_tool as module

    monkeypatch.setattr(module, "publish_text", lambda root, relative_path, content, **_kwargs: (
        (root / relative_path).parent.mkdir(parents=True, exist_ok=True),
        (root / relative_path).write_text(content, encoding="utf-8"),
    )[-1])

    run_id = "run_url_reconciliation_001"
    init = json.loads(module.companyintel_graph({
        "action": "init_run", "run_id": run_id, "target_url": "https://example.com",
    }))
    conn = module._connect(run_id)
    now = module.time.time()
    url_node = "old-url-node"
    conn.execute(
        "INSERT INTO nodes(node_id,node_type,value,normalized_value,status,confidence,evidence_ids,discovered_from,first_seen_round,frontier_state) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (url_node, "url", "https://example.com/old", "https://example.com/old", "OBSERVED", 0.5, "[]", "[]", 0, "OPEN"),
    )
    conn.execute(
        "INSERT INTO frontier_tasks(task_id,run_id,node_id,pivot_type,priority,state,attempt,max_attempts,available_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("old-open-task", run_id, url_node, "exact_search", 1, "OPEN", 0, 3, now, now, now),
    )
    conn.execute(
        "INSERT INTO frontier_tasks(task_id,run_id,node_id,pivot_type,priority,state,attempt,max_attempts,available_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("old-completed-task", run_id, url_node, "documents", 1, "COMPLETED", 1, 3, now, now, now),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(module, "_execute_inventory_frontier", lambda _args: {"outcome": "NO_TASK"})
    monkeypatch.setattr(module, "_execute_frontier", lambda _args: {"outcome": "NO_TASK"})
    result = json.loads(module.companyintel_graph({
        "action": "run_frontier", "run_id": run_id, "worker_id": "migration-worker",
    }))
    assert result["ok"] is True

    conn = module._connect(run_id)
    states = {
        row["task_id"]: row["state"]
        for row in conn.execute(
            "SELECT task_id,state FROM frontier_tasks WHERE run_id=? AND task_id LIKE 'old-%'",
            (run_id,),
        ).fetchall()
    }
    reason = conn.execute(
        "SELECT last_error FROM frontier_tasks WHERE task_id='old-open-task'",
    ).fetchone()["last_error"]
    migration = conn.execute(
        "SELECT COUNT(*) FROM run_events WHERE run_id=? AND event_type='frontier.migrated'",
        (run_id,),
    ).fetchone()[0]
    conn.close()

    assert states == {
        "old-open-task": "SKIPPED_REDUNDANT",
        "old-completed-task": "COMPLETED",
    }
    assert reason == "redundant_url_search_removed_by_planner_v2"
    assert migration == 1


def test_completed_provider_query_is_deduplicated(companyintel_home, monkeypatch):
    from tools import companyintel_graph_tool as module

    monkeypatch.setattr(module, "publish_text", lambda root, relative_path, content, **_kwargs: (
        (root / relative_path).parent.mkdir(parents=True, exist_ok=True),
        (root / relative_path).write_text(content, encoding="utf-8"),
    )[-1])
    run_id = "run_query_dedup_001"
    init = json.loads(module.companyintel_graph({
        "action": "init_run", "run_id": run_id, "target_url": "https://example.com",
    }))
    recorded = json.loads(module.companyintel_graph({
        "action": "record_search", "run_id": run_id,
        "pivot_node_id": init["seed_node_id"], "pivot_type": "exact_search",
        "query": '"example.com"', "source": "none", "provider": "none",
        "mode": "exact", "status": "COMPLETED_ZERO_RESULTS",
    }))
    result = json.loads(module.companyintel_graph({
        "action": "execute_frontier", "run_id": run_id,
        "worker_id": "dedup-worker", "pivot_type": "exact_search",
    }))

    assert recorded["ok"] is True
    assert result["outcome"] == "SKIPPED_DEDUPLICATED"
    conn = module._connect(run_id)
    task_state = conn.execute(
        "SELECT state FROM frontier_tasks WHERE run_id=? AND pivot_type='exact_search'",
        (run_id,),
    ).fetchone()[0]
    conn.close()
    assert task_state == "SKIPPED_DEDUPLICATED"


def test_provider_failure_does_not_prevent_inventory_task(companyintel_home, monkeypatch):
    from tools import companyintel_graph_tool as graph_module
    from tools import companyintel_public_search as search_module

    monkeypatch.setattr(graph_module, "_inventory", lambda _args: {"findings": 0})
    monkeypatch.setattr(search_module, "_active_provider", lambda: ("", None))
    monkeypatch.setattr(search_module, "_allow_legacy_html_fallback", lambda: False)
    run_id = "run_failure_continuation_001"
    json.loads(graph_module.companyintel_graph({
        "action": "init_run", "run_id": run_id, "target_url": "https://example.com",
    }))

    result = json.loads(graph_module.companyintel_graph({
        "action": "run_frontier", "run_id": run_id,
        "worker_id": "continuation-worker", "max_tasks": 2,
    }))

    assert result["ok"] is True
    assert result["outcomes"][0]["pivot_type"] == "site_inventory"
    assert result["outcomes"][0]["outcome"] == "COMPLETED_ZERO_RESULTS"
    assert result["outcomes"][1]["outcome"] == "UNAVAILABLE"


def test_report_review_revise_then_pass_persists_verdict(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph

    run_id = "run_report_revise_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "info@example.com", "source_url": "https://example.com/contact",
        "excerpt": "info@example.com", "relation_from_node_id": init["seed_node_id"],
        "relation": "publishes",
    })
    _prepare_report_gate(companyintel_graph, _connect, _materialize, run_id, init)
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET status='COMPLETED',saturation_reason='coverage_complete' WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier SET state='EXHAUSTED' WHERE node_id=?", (init["seed_node_id"],))
    conn.commit()
    _materialize(conn, run_id)
    conn.close()

    calls = []
    def fake_request(self, operation, payload):
        calls.append(operation)
        evidence_id = json.loads(companyintel_graph({"action": "projection", "run_id": run_id, "view": "evidence"}))[
            "data"]["evidence"][0]["evidence_id"]
        if operation == "synthesize_report":
            return {"report_text": "draft", "claims": [{"text": "draft claim", "evidence_ids": [evidence_id]}]}
        if operation == "review_report":
            review_count = calls.count("review_report")
            if review_count == 1:
                return {"verdict": "REVISE", "review_text": "needs correction", "findings": [{"code": "C1", "severity": "WARNING", "message": "clarify"}]}
            return {"verdict": "PASS", "review_text": "approved", "findings": []}
        if operation == "revise_report":
            return {"report_text": "revised", "claims": [{"text": "revised claim", "evidence_ids": [evidence_id]}]}
        raise AssertionError(operation)

    monkeypatch.setattr("tools.companyintel_reasoning_controller.CompanyIntelReasoningController._request", fake_request)
    generated = json.loads(companyintel_graph({"action": "generate_report", "run_id": run_id, "reasoning_enabled": True, "terminal_transition_pending": True}))
    assert generated["ok"] is True
    assert generated["report"]["text"] == "revised"
    assert generated["report"]["review_verdict"] == "PASS"
    assert calls == ["synthesize_report", "review_report", "revise_report", "review_report"]
    conn = _connect(run_id)
    review = json.loads(conn.execute("SELECT review_json FROM report_reviews WHERE run_id=?", (run_id,)).fetchone()[0])
    draft = json.loads(conn.execute("SELECT draft_json FROM report_drafts WHERE run_id=?", (run_id,)).fetchone()[0])
    conn.close()
    assert review["verdict"] == "PASS"
    assert draft["review_verdict"] == "PASS"


def test_report_review_failure_preserves_draft_for_resume(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph
    from tools.openrouter_client import OpenRouterClientError

    run_id = "run_report_resume_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "info@example.com", "source_url": "https://example.com/contact",
        "excerpt": "info@example.com", "relation_from_node_id": init["seed_node_id"],
        "relation": "publishes",
    })
    _prepare_report_gate(companyintel_graph, _connect, _materialize, run_id, init)
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET saturation_reason='coverage_complete' WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier SET state='EXHAUSTED' WHERE node_id=?", (init["seed_node_id"],))
    conn.commit()
    _materialize(conn, run_id)
    conn.close()

    calls = []
    reviews = 0
    def fake_request(self, operation, payload):
        nonlocal reviews
        calls.append(operation)
        if operation == "synthesize_report":
            evidence_id = json.loads(companyintel_graph({"action": "projection", "run_id": run_id, "view": "evidence"}))["data"]["evidence"][0]["evidence_id"]
            return {"report_text": "durable draft", "claims": [{"text": "Observed email", "evidence_ids": [evidence_id]}]}
        reviews += 1
        if reviews == 1:
            raise OpenRouterClientError("provider unavailable", code="REASONING_PROVIDER_CONNECTION", retryable=True, operation="review_report")
        return {"verdict": "PASS", "review_text": "reviewed", "findings": []}

    monkeypatch.setattr("tools.companyintel_reasoning_controller.CompanyIntelReasoningController._request", fake_request)
    first = json.loads(companyintel_graph({"action": "generate_report", "run_id": run_id, "reasoning_enabled": True, "terminal_transition_pending": True}))
    assert first["durable_status"] == "RETRY_WAIT"
    second = json.loads(companyintel_graph({"action": "generate_report", "run_id": run_id, "reasoning_enabled": True, "terminal_transition_pending": True}))
    assert second["ok"] is True
    assert calls == ["synthesize_report", "review_report", "review_report"]
    conn = _connect(run_id)
    assert conn.execute("SELECT stage FROM report_drafts WHERE run_id=?", (run_id,)).fetchone()[0] == "REPORT_READY"
    assert conn.execute("SELECT COUNT(*) FROM report_artifacts WHERE run_id=?", (run_id,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM report_reviews WHERE run_id=?", (run_id,)).fetchone()[0] == 1
    conn.close()


def test_nonretryable_report_failure_is_resumable(companyintel_home, monkeypatch):
    from tools.companyintel_graph_tool import _connect, _materialize, companyintel_graph
    from tools.openrouter_client import OpenRouterClientError

    run_id = "run_report_nonretryable_001"
    init = json.loads(companyintel_graph({"action": "init_run", "run_id": run_id, "target_url": "https://example.com"}))
    companyintel_graph({
        "action": "record_observation", "run_id": run_id, "node_type": "email",
        "value": "info@example.com", "source_url": "https://example.com/contact",
        "excerpt": "info@example.com", "relation_from_node_id": init["seed_node_id"],
        "relation": "publishes",
    })
    _prepare_report_gate(companyintel_graph, _connect, _materialize, run_id, init)
    conn = _connect(run_id)
    conn.execute("UPDATE run_state SET saturation_reason='coverage_complete' WHERE run_id=?", (run_id,))
    conn.execute("UPDATE frontier SET state='EXHAUSTED' WHERE node_id=?", (init["seed_node_id"],))
    conn.commit()
    _materialize(conn, run_id)
    conn.close()

    def fail_request(self, operation, payload):
        raise OpenRouterClientError(
            "Reasoning client response contract failed",
            code="REASONING_CLIENT_CONTRACT_ERROR",
            retryable=False,
            operation=operation,
            cause_class="TypeError",
            failure_stage="provider_request",
        )

    monkeypatch.setattr("tools.companyintel_reasoning_controller.CompanyIntelReasoningController._request", fail_request)
    result = json.loads(companyintel_graph({
        "action": "generate_report", "run_id": run_id,
        "reasoning_enabled": True, "terminal_transition_pending": True,
    }))

    assert result["ok"] is False
    assert result["retryable"] is False
    assert result["durable_status"] == "RESUMABLE"
    assert result["retry_at"] is None
    conn = _connect(run_id)
    state = conn.execute("SELECT status,retry_at FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    stage = conn.execute("SELECT stage FROM report_drafts WHERE run_id=?", (run_id,)).fetchone()[0]
    conn.close()
    assert tuple(state) == ("RESUMABLE", None)
    assert stage == "REPORT_RESUMABLE"
