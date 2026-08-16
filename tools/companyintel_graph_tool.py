"""Deterministic, profile-gated graph runtime for companyintel.

This is intentionally a narrow Hermes tool boundary: the tool is exposed only
when the active profile is ``companyintel``.  SQLite owns the authoritative
state; JSON/JSONL files are projections for validators and operators.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from hermes_constants import get_hermes_home
from tools.registry import registry
from companyintel.schema import RELATIONS, SCHEMA_VERSION, _endpoint_allowed, deterministic_id, validate_graph

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_NODE_TYPES = {
    "domain", "url", "phone", "email", "address", "place", "brand", "legal_entity",
    "legal_entity_candidate", "official_registry", "person", "document", "image", "favicon", "marketplace_profile",
    "social_profile", "analytics_id", "merchant_id", "iban", "legal_id",
    "product", "organization_candidate",
}


def _home() -> Path:
    raw = os.environ.get("HERMES_HOME")
    return Path(raw).expanduser() if raw else Path(get_hermes_home())


def check_companyintel_requirements() -> bool:
    profile = os.environ.get("HERMES_PROFILE", "")
    if profile:
        return profile == "companyintel"
    return _home().name == "companyintel"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_id(prefix: str, *parts: str) -> str:
    return deterministic_id(prefix, *parts)


def _normalize_url(value: str) -> str:
    value = value.strip()
    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("target_url must be a public HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("target_url must not contain credentials")
    parsed = parsed._replace(fragment="")
    return parsed.geturl().rstrip("/")


def _domain(value: str) -> str:
    return (urlparse(value).hostname or "").lower().rstrip(".")


def _normalize_value(node_type: str, value: str) -> str:
    value = " ".join(str(value).strip().split())
    if node_type in {"domain", "email"}:
        value = value.lower()
    if node_type == "domain":
        value = value.rstrip(".")
    if node_type == "email" and "@" not in value:
        raise ValueError("email node requires an email address")
    if node_type == "phone":
        value = re.sub(r"[^0-9+]", "", value)
    if not value:
        raise ValueError("node value must not be empty")
    return value


def _run_dir(run_id: str) -> Path:
    return _home() / "companyintel" / "runs" / run_id


def _connect(run_id: str) -> sqlite3.Connection:
    run_dir = _run_dir(run_id)
    db_path = run_dir / "graph.sqlite3"
    preexisting_db = db_path.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_state (
            run_id TEXT PRIMARY KEY,
            target_url TEXT NOT NULL,
            status TEXT NOT NULL,
            round INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            max_tasks INTEGER NOT NULL DEFAULT 50,
            max_searches INTEGER NOT NULL DEFAULT 200,
            max_nodes INTEGER NOT NULL DEFAULT 250,
            max_edges INTEGER NOT NULL DEFAULT 500,
            deadline_at REAL,
            tasks_executed INTEGER NOT NULL DEFAULT 0,
            source_calls INTEGER NOT NULL DEFAULT 0,
            saturation_reason TEXT,
            legal_identity_status TEXT NOT NULL DEFAULT 'UNRESOLVED',
            legal_identity_reason TEXT,
            legal_identity_checked_at TEXT,
            legal_identity_coverage TEXT NOT NULL DEFAULT '{}',
            retry_at REAL,
            retry_reason TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            resume_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            value TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            discovered_from TEXT NOT NULL DEFAULT '[]',
            first_seen_round INTEGER NOT NULL,
            last_expanded_round INTEGER,
            frontier_state TEXT NOT NULL DEFAULT 'OPEN'
        );
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            source_title TEXT NOT NULL DEFAULT '',
            source_tier TEXT NOT NULL DEFAULT 'C',
            retrieved_at TEXT NOT NULL,
            excerpt TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            retrieval_method TEXT NOT NULL,
            access_status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS edges (
            edge_id TEXT PRIMARY KEY,
            from_node TEXT NOT NULL,
            to_node TEXT NOT NULL,
            relation TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            first_seen_round INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS frontier (
            node_id TEXT PRIMARY KEY,
            priority INTEGER NOT NULL,
            reason TEXT NOT NULL,
            planned_pivots TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'OPEN'
        );
        CREATE TABLE IF NOT EXISTS frontier_tasks (
            task_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            pivot_type TEXT NOT NULL,
            priority INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'OPEN',
            attempt INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            available_at REAL NOT NULL,
            worker_id TEXT,
            lease_token TEXT,
            lease_expires_at REAL,
            last_error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(run_id, node_id, pivot_type)
        );
        CREATE INDEX IF NOT EXISTS frontier_tasks_claim_idx
            ON frontier_tasks(state, available_at, priority DESC, created_at);
        CREATE TABLE IF NOT EXISTS search_log (
            search_id TEXT PRIMARY KEY,
            round INTEGER NOT NULL,
            pivot_node_id TEXT NOT NULL,
            pivot_type TEXT NOT NULL,
            query TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            new_node_ids TEXT NOT NULL DEFAULT '[]',
            limitation TEXT
        );
        CREATE TABLE IF NOT EXISTS identity_candidates (
            candidate_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            legal_name TEXT NOT NULL DEFAULT '',
            legal_id TEXT NOT NULL DEFAULT '',
            country TEXT NOT NULL DEFAULT '',
            score INTEGER NOT NULL,
            status TEXT NOT NULL,
            matching_dimensions TEXT NOT NULL DEFAULT '[]',
            conflicting_dimensions TEXT NOT NULL DEFAULT '[]',
            missing_dimensions TEXT NOT NULL DEFAULT '[]',
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(run_id, candidate_id)
        );
        CREATE INDEX IF NOT EXISTS identity_candidates_run_idx
            ON identity_candidates(run_id, score DESC, updated_at DESC);
        CREATE TABLE IF NOT EXISTS worker_checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            phase TEXT NOT NULL,
            status TEXT NOT NULL,
            cursor_json TEXT NOT NULL DEFAULT '{}',
            graph_digest TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(run_id, task_id, attempt)
        );
        CREATE INDEX IF NOT EXISTS worker_checkpoints_latest_idx
            ON worker_checkpoints(run_id, task_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            reason TEXT NOT NULL,
            graph_digest TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS run_events (
            run_id TEXT NOT NULL,
            event_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            idempotency_key TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            PRIMARY KEY(run_id, event_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS run_events_idempotency_idx
            ON run_events(run_id, idempotency_key)
            WHERE idempotency_key <> '';
        CREATE TABLE IF NOT EXISTS report_artifacts (
            report_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            graph_revision TEXT NOT NULL,
            report_json TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '',
            review_label TEXT NOT NULL DEFAULT 'not_reviewed',
            gate_json TEXT NOT NULL DEFAULT '{}',
            generated_at TEXT NOT NULL,
            UNIQUE(run_id, graph_revision)
        );
        CREATE INDEX IF NOT EXISTS report_artifacts_run_idx
            ON report_artifacts(run_id, generated_at DESC);
        CREATE TABLE IF NOT EXISTS reasoning_decisions (
            decision_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            decision_type TEXT NOT NULL,
            decision_json TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '',
            graph_revision TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(run_id, decision_id)
        );
        CREATE INDEX IF NOT EXISTS reasoning_decisions_run_idx
            ON reasoning_decisions(run_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS report_reviews (
            review_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            graph_revision TEXT NOT NULL,
            review_json TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '',
            review_label TEXT NOT NULL,
            independent_review INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS report_reviews_run_idx
            ON report_reviews(run_id, created_at DESC);
        CREATE TRIGGER IF NOT EXISTS evidence_immutable_update
            BEFORE UPDATE ON evidence
            BEGIN SELECT RAISE(ABORT, 'evidence rows are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS evidence_immutable_delete
            BEFORE DELETE ON evidence
            BEGIN SELECT RAISE(ABORT, 'evidence rows are immutable'); END;
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            from_version TEXT NOT NULL,
            to_version TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            integrity_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(run_state)").fetchall()}
    for column, definition in {
        "max_tasks": "INTEGER NOT NULL DEFAULT 50",
        "max_searches": "INTEGER NOT NULL DEFAULT 200",
        "max_nodes": "INTEGER NOT NULL DEFAULT 250",
        "max_edges": "INTEGER NOT NULL DEFAULT 500",
        "deadline_at": "REAL",
        "tasks_executed": "INTEGER NOT NULL DEFAULT 0",
        "source_calls": "INTEGER NOT NULL DEFAULT 0",
        "saturation_reason": "TEXT",
        "legal_identity_status": "TEXT NOT NULL DEFAULT 'UNRESOLVED'",
        "legal_identity_reason": "TEXT",
        "legal_identity_checked_at": "TEXT",
        "legal_identity_coverage": "TEXT NOT NULL DEFAULT '{}'",
        "retry_at": "REAL",
        "retry_reason": "TEXT",
        "retry_count": "INTEGER NOT NULL DEFAULT 0",
        "resume_count": "INTEGER NOT NULL DEFAULT 0",
    }.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE run_state ADD COLUMN {column} {definition}")
    migration = conn.execute("SELECT migration_id FROM schema_migrations WHERE migration_id='corporate-intelligence-graph-v1-to-v2'").fetchone()
    if migration is None:
        from_version = "corporate-intelligence-graph/v1" if preexisting_db else "bootstrap"
        required_tables = {"run_state", "nodes", "edges", "evidence", "frontier", "frontier_tasks", "run_events", "report_artifacts", "reasoning_decisions", "report_reviews"}
        present_tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        missing_tables = sorted(required_tables - present_tables)
        if missing_tables:
            raise RuntimeError(f"schema v2 migration incomplete; missing tables: {', '.join(missing_tables)}")
        conn.execute(
            "INSERT INTO schema_migrations(migration_id,from_version,to_version,applied_at,integrity_json) VALUES(?,?,?,?,?)",
            ("corporate-intelligence-graph-v1-to-v2", from_version, SCHEMA_VERSION, _now(), _json({"required_tables": sorted(required_tables), "missing_tables": missing_tables})),
        )
    conn.commit()
    return conn


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _materialize(conn: sqlite3.Connection, run_id: str) -> None:
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    nodes = [dict(row) for row in conn.execute("SELECT * FROM nodes ORDER BY node_id")]
    edges = [dict(row) for row in conn.execute("SELECT * FROM edges ORDER BY edge_id")]
    evidence = [dict(row) for row in conn.execute("SELECT * FROM evidence ORDER BY evidence_id")]
    frontier = [dict(row) for row in conn.execute("SELECT * FROM frontier ORDER BY priority DESC,node_id")]
    frontier_tasks = [dict(row) for row in conn.execute("SELECT * FROM frontier_tasks ORDER BY priority DESC,created_at ASC,task_id")]
    search_log = [dict(row) for row in conn.execute("SELECT * FROM search_log ORDER BY search_id")]
    identity_candidates = [dict(row) for row in conn.execute("SELECT * FROM identity_candidates WHERE run_id=? ORDER BY score DESC,updated_at DESC", (run_id,))]
    worker_checkpoints = [dict(row) for row in conn.execute("SELECT * FROM worker_checkpoints WHERE run_id=? ORDER BY updated_at DESC,checkpoint_id DESC", (run_id,))]
    for row in frontier_tasks:
        row.pop("lease_token", None)
    for row in nodes + edges + evidence + frontier + search_log + identity_candidates:
        for key in ("evidence_ids", "discovered_from", "planned_pivots", "new_node_ids", "matching_dimensions", "conflicting_dimensions", "missing_dimensions"):
            if key in row and isinstance(row[key], str):
                row[key] = json.loads(row[key])
    for row in worker_checkpoints:
        row.pop("graph_digest", None)
        if isinstance(row.get("cursor_json"), str):
            row["cursor"] = json.loads(row.pop("cursor_json"))
    open_frontier = sum(row["state"] == "OPEN" for row in frontier)
    graph = {
        "schema_version": SCHEMA_VERSION,
        "target": {"url": run["target_url"], "domain": _domain(run["target_url"])},
        "research_state": {
            "status": run["status"], "round": run["round"], "frontier_open": open_frontier,
            "stale_rounds": 0,
            "budgets": {"max_rounds": 8, "max_tasks": run["max_tasks"], "max_searches": run["max_searches"], "max_nodes": run["max_nodes"], "max_edges": run["max_edges"], "deadline_at": run["deadline_at"]},
            "usage": {"tasks_executed": run["tasks_executed"], "source_calls": run["source_calls"]},
            "saturation_reason": run["saturation_reason"],
            "retry_at": run["retry_at"], "retry_reason": run["retry_reason"], "retry_count": run["retry_count"], "resume_count": run["resume_count"],
        },
        "identity_resolution": {"status": "AMBIGUOUS", "candidate_ids": [row["candidate_id"] for row in identity_candidates]},
        "identity_candidates": identity_candidates,
        "worker_checkpoints": worker_checkpoints,
        "nodes": nodes, "edges": edges, "evidence": evidence,
        "frontier": frontier, "frontier_tasks": frontier_tasks, "search_log": search_log,
        "pivot_coverage": {},
        "legal_identity_exhaustion": {"status": run["legal_identity_status"], "reason": run["legal_identity_reason"], "checked_at": run["legal_identity_checked_at"], "coverage": json.loads(run["legal_identity_coverage"] or "{}")},
    }
    inventory_path = _run_dir(run_id) / "inventory.json"
    if inventory_path.exists():
        try:
            graph["inventory"] = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            graph["inventory"] = {"error": "inventory_projection_unavailable"}
    validate_graph(graph)
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_dir / "graph.json", graph)
    for name, rows in (("nodes.jsonl", nodes), ("edges.jsonl", edges), ("evidence.jsonl", evidence), ("frontier_tasks.jsonl", frontier_tasks), ("search_log.jsonl", search_log), ("identity_candidates.jsonl", identity_candidates), ("worker_checkpoints.jsonl", worker_checkpoints)):
        _atomic_text(run_dir / name, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _checkpoint(conn: sqlite3.Connection, run_id: str, reason: str) -> None:
    graph_path = _run_dir(run_id) / "graph.json"
    graph_json = graph_path.read_bytes() if graph_path.exists() else b""
    digest = hashlib.sha256(graph_json).hexdigest()
    created_at = _now()
    conn.execute("INSERT INTO checkpoints(created_at,reason,graph_digest) VALUES(?,?,?)", (created_at, reason, digest))
    conn.commit()
    _atomic_json(_run_dir(run_id) / "checkpoints" / "latest.json", {
        "run_id": run_id,
        "created_at": created_at,
        "reason": reason,
        "graph_sha256": digest,
    })


def _validate_worker_cursor(cursor) -> tuple[dict, str]:
    if cursor is None:
        cursor = {}
    if not isinstance(cursor, dict):
        raise ValueError("worker checkpoint cursor must be an object")
    encoded = _json(cursor)
    if len(encoded.encode("utf-8")) > 8192:
        raise ValueError("worker checkpoint cursor exceeds 8192 bytes")
    return cursor, encoded


def _upsert_worker_checkpoint(conn: sqlite3.Connection, run_id: str, worker_id: str, task: sqlite3.Row | dict, *, phase: str, status: str, cursor=None) -> dict:
    cursor, cursor_json = _validate_worker_cursor(cursor)
    phase = " ".join(str(phase).split())[:128] or "worker"
    status = " ".join(str(status).split())[:32]
    now = time.time()
    checkpoint_id = _stable_id("worker-checkpoint", run_id, task["task_id"], task["attempt"])
    graph_path = _run_dir(run_id) / "graph.json"
    digest = hashlib.sha256(graph_path.read_bytes()).hexdigest() if graph_path.exists() else ""
    conn.execute(
        """INSERT OR REPLACE INTO worker_checkpoints(checkpoint_id,run_id,worker_id,task_id,attempt,phase,status,cursor_json,graph_digest,created_at,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (checkpoint_id, run_id, worker_id[:128], task["task_id"], int(task["attempt"]), phase, status, cursor_json, digest, now, now),
    )
    return {"checkpoint_id": checkpoint_id, "run_id": run_id, "worker_id": worker_id[:128], "task_id": task["task_id"], "attempt": int(task["attempt"]), "phase": phase, "status": status, "cursor": cursor, "created_at": now, "updated_at": now}


def _latest_worker_checkpoint(conn: sqlite3.Connection, run_id: str, task_id: str = "") -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM worker_checkpoints WHERE run_id=? AND (?='' OR task_id=?) ORDER BY updated_at DESC,checkpoint_id DESC LIMIT 1",
        (run_id, task_id, task_id),
    ).fetchone()


def _enqueue_frontier_tasks(conn: sqlite3.Connection, run_id: str, node_id: str, priority: int, planned_pivots: list[str]) -> int:
    from tools.companyintel_pivots import get_pivot
    now = time.time()
    inserted = 0
    for pivot_type in sorted({str(value).strip() for value in planned_pivots if str(value).strip()}):
        task_id = _stable_id("frontier-task", run_id, node_id, pivot_type)
        spec = get_pivot(pivot_type)
        initial_state = "OPEN" if spec is not None and spec.implemented else "SKIPPED_UNSUPPORTED"
        initial_error = None if initial_state == "OPEN" else "typed_worker_not_implemented"
        cursor = conn.execute(
            """INSERT OR IGNORE INTO frontier_tasks(
                task_id,run_id,node_id,pivot_type,priority,state,attempt,max_attempts,
                available_at,last_error,created_at,updated_at
            ) VALUES(?,?,?,?,?, ?,0,3,?,?,?,?)""",
            (task_id, run_id, node_id, pivot_type, int(priority), initial_state, now, initial_error, now, now),
        )
        inserted += cursor.rowcount
    return inserted


def _task_dict(row: sqlite3.Row | None, *, include_lease: bool = True) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    if not include_lease:
        result.pop("lease_token", None)
    return result


def _recover_expired_leases(conn: sqlite3.Connection, now: float) -> int:
    cursor = conn.execute(
        """UPDATE frontier_tasks
           SET state=CASE WHEN attempt < max_attempts THEN 'RETRY_WAIT' ELSE 'FAILED' END,
               available_at=CASE WHEN attempt < max_attempts THEN ? ELSE available_at END,
               worker_id=NULL, lease_token=NULL, lease_expires_at=NULL,
               last_error=CASE WHEN attempt < max_attempts THEN 'lease_expired' ELSE COALESCE(last_error, 'lease_expired') END,
               updated_at=?
         WHERE state='CLAIMED' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?""",
        (now, now, now),
    )
    return cursor.rowcount


def _refresh_run_retry_state(conn: sqlite3.Connection, run_id: str, now: float, *, promote_ready: bool = True) -> sqlite3.Row | None:
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None or run["status"] in {"COMPLETED", "FAILED", "CANCELLED", "PARTIAL"}:
        return run
    retry = conn.execute("SELECT MIN(available_at) AS retry_at, COUNT(*) AS count, MAX(last_error) AS reason FROM frontier_tasks WHERE run_id=? AND state='RETRY_WAIT'", (run_id,)).fetchone()
    if retry["count"]:
        retry_at = float(retry["retry_at"])
        runnable = conn.execute("SELECT COUNT(*) FROM frontier_tasks WHERE run_id=? AND state='OPEN'", (run_id,)).fetchone()[0]
        if run["status"] in {"RETRY_WAIT", "RESUMABLE"}:
            status = "RESUMABLE" if promote_ready and retry_at <= now else "RETRY_WAIT"
        elif runnable:
            status = "RUNNING"
        else:
            status = "RUNNING" if run["status"] == "RUNNING" else "RETRY_WAIT"
        conn.execute("UPDATE run_state SET status=?,retry_at=?,retry_reason=?,updated_at=? WHERE run_id=?", (status, retry_at, retry["reason"], _now(), run_id))
    elif run["status"] in {"RETRY_WAIT", "RESUMABLE"}:
        # Preserve an explicit durable lifecycle state.  In particular,
        # RESUMABLE may represent a process-boundary checkpoint with no
        # frontier task currently in RETRY_WAIT; silently changing it to
        # RUNNING makes the public resume action fail closed after it has
        # already lost the resumable marker.
        conn.execute("UPDATE run_state SET retry_at=NULL,retry_reason=NULL,updated_at=? WHERE run_id=?", (_now(), run_id))
    return conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()


def _schedule_frontier(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM run_state WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("run is not initialized")
        node_id = str(args.get("node_id", "")).strip()
        rows = conn.execute(
            "SELECT node_id,priority,planned_pivots FROM frontier WHERE state='OPEN' AND (?='' OR node_id=?) ORDER BY priority DESC,node_id ASC",
            (node_id, node_id),
        ).fetchall()
        scheduled = 0
        for row in rows:
            try:
                pivots = json.loads(row["planned_pivots"])
            except json.JSONDecodeError:
                pivots = []
            if isinstance(pivots, list):
                scheduled += _enqueue_frontier_tasks(conn, run_id, row["node_id"], row["priority"], pivots)
        conn.commit()
        return {"ok": True, "run_id": run_id, "scheduled": scheduled, "frontier_rows": len(rows)}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _claim_frontier(args: dict) -> dict:
    run_id = args.get("run_id", "")
    worker_id = str(args.get("worker_id", "")).strip()[:128]
    if not worker_id:
        raise ValueError("worker_id is required")
    lease_seconds = min(3600.0, max(0.0, float(args.get("lease_seconds", 300))))
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM run_state WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("run is not initialized")
        recovered = _recover_expired_leases(conn, now)
        run = _refresh_run_retry_state(conn, run_id, now, promote_ready=True)
        if run["status"] in {"COMPLETED", "FAILED", "CANCELLED", "PARTIAL"}:
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered, "run_status": run["status"], "stop_reason": run["saturation_reason"] or run["status"]}
        if run["status"] == "RETRY_WAIT" and not conn.execute("SELECT 1 FROM frontier_tasks WHERE run_id=? AND state='OPEN' LIMIT 1", (run_id,)).fetchone():
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered, "run_status": "RETRY_WAIT", "retry_at": run["retry_at"], "stop_reason": "RETRY_WAIT"}
        if run["status"] == "RESUMABLE":
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered, "run_status": "RESUMABLE", "retry_at": run["retry_at"], "stop_reason": "RESUMABLE"}
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        budget_reason = None
        if run["deadline_at"] is not None and now >= run["deadline_at"]:
            budget_reason = "BUDGET_EXHAUSTED:deadline"
        elif run["tasks_executed"] >= run["max_tasks"]:
            budget_reason = "BUDGET_EXHAUSTED:max_tasks"
        elif run["source_calls"] >= run["max_searches"]:
            budget_reason = "BUDGET_EXHAUSTED:max_searches"
        elif node_count >= run["max_nodes"]:
            budget_reason = "BUDGET_EXHAUSTED:max_nodes"
        elif edge_count >= run["max_edges"]:
            budget_reason = "BUDGET_EXHAUSTED:max_edges"
        if budget_reason:
            conn.execute("UPDATE run_state SET status='PARTIAL',saturation_reason=?,updated_at=? WHERE run_id=?", (budget_reason, _now(), run_id))
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered, "stop_reason": budget_reason}
        row = conn.execute(
            """SELECT * FROM frontier_tasks
               WHERE (state='OPEN' OR (state='RETRY_WAIT' AND available_at <= ?))
                 AND (?='' OR pivot_type=?)
                 AND (?='' OR task_id=?)
               ORDER BY priority DESC, created_at ASC, task_id ASC LIMIT 1""",
            (now, str(args.get("pivot_type", "")).strip(), str(args.get("pivot_type", "")).strip(), str(args.get("task_id", "")).strip(), str(args.get("task_id", "")).strip()),
        ).fetchone()
        if row is None:
            conn.commit()
            return {"ok": True, "task": None, "recovered": recovered}
        token = secrets.token_hex(16)
        expires = now + lease_seconds
        requested_max_attempts = min(10, max(1, int(args.get("max_attempts", 3))))
        if row["attempt"] == 0:
            conn.execute("UPDATE frontier_tasks SET max_attempts=? WHERE task_id=?", (requested_max_attempts, row["task_id"]))
        conn.execute(
            """UPDATE frontier_tasks
               SET state='CLAIMED', attempt=attempt+1, worker_id=?, lease_token=?,
                   lease_expires_at=?, updated_at=?
             WHERE task_id=?""",
            (worker_id, token, expires, now, row["task_id"]),
        )
        conn.execute(
            "UPDATE run_state SET tasks_executed=tasks_executed+1,source_calls=source_calls+1,updated_at=? WHERE run_id=?",
            (_now(), run_id),
        )
        claimed = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=?", (row["task_id"],)).fetchone()
        checkpoint = _upsert_worker_checkpoint(conn, run_id, worker_id, claimed, phase="claimed", status="CLAIMED")
        conn.commit()
        return {"ok": True, "task": _task_dict(claimed), "checkpoint": checkpoint, "recovered": 0}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _lease_mutation(args: dict, action: str) -> dict:
    run_id = args.get("run_id", "")
    task_id = str(args.get("task_id", "")).strip()
    worker_id = str(args.get("worker_id", "")).strip()
    lease_token = str(args.get("lease_token", "")).strip()
    if not task_id or not worker_id or not lease_token:
        raise ValueError("task_id, worker_id and lease_token are required")
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=? AND run_id=?", (task_id, run_id)).fetchone()
        if row is None:
            raise ValueError("frontier task not found")
        if row["state"] != "CLAIMED" or row["worker_id"] != worker_id or row["lease_token"] != lease_token:
            raise ValueError("frontier lease is not owned by worker")
        if row["lease_expires_at"] is None or row["lease_expires_at"] <= now:
            raise ValueError("frontier lease expired")
        if action == "complete":
            conn.execute("UPDATE frontier_tasks SET state='COMPLETED',worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,updated_at=? WHERE task_id=?", (now, task_id))
            state = "COMPLETED"
        elif action == "renew":
            lease_seconds = min(3600.0, max(1.0, float(args.get("lease_seconds", 300))))
            conn.execute("UPDATE frontier_tasks SET lease_expires_at=?,updated_at=? WHERE task_id=?", (now + lease_seconds, now, task_id))
            state = "CLAIMED"
        else:
            error = " ".join(str(args.get("error", "frontier task failed")).split())[:500]
            retry_after = min(3600.0, max(0.0, float(args.get("retry_after_seconds", 30))))
            if row["attempt"] < row["max_attempts"]:
                state = "RETRY_WAIT"
                conn.execute("UPDATE frontier_tasks SET state=?,available_at=?,worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,last_error=?,updated_at=? WHERE task_id=?", (state, now + retry_after, error, now, task_id))
            else:
                state = "FAILED"
                conn.execute("UPDATE frontier_tasks SET state=?,worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,last_error=?,updated_at=? WHERE task_id=?", (state, error, now, task_id))
            if state == "RETRY_WAIT":
                conn.execute("UPDATE run_state SET status='RETRY_WAIT',retry_at=?,retry_reason=?,retry_count=retry_count+1,updated_at=? WHERE run_id=? AND status NOT IN ('COMPLETED','FAILED','CANCELLED','PARTIAL')", (now + retry_after, error, _now(), run_id))
        updated = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=?", (task_id,)).fetchone()
        prior = _latest_worker_checkpoint(conn, run_id, task_id)
        cursor = json.loads(prior["cursor_json"] or "{}") if prior else {}
        checkpoint = _upsert_worker_checkpoint(conn, run_id, worker_id, updated, phase=prior["phase"] if prior else action, status=state, cursor=cursor)
        conn.commit()
        return {"ok": True, "task_id": task_id, "state": state, "attempt": updated["attempt"], "task": _task_dict(updated), "checkpoint": checkpoint}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _checkpoint_worker(args: dict) -> dict:
    run_id = args.get("run_id", "")
    task_id = str(args.get("task_id", "")).strip()
    worker_id = str(args.get("worker_id", "")).strip()
    lease_token = str(args.get("lease_token", "")).strip()
    if not task_id or not worker_id or not lease_token:
        raise ValueError("task_id, worker_id and lease_token are required")
    cursor, _ = _validate_worker_cursor(args.get("cursor", {}))
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        task = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=? AND run_id=?", (task_id, run_id)).fetchone()
        if task is None or task["state"] != "CLAIMED" or task["worker_id"] != worker_id or task["lease_token"] != lease_token:
            raise ValueError("frontier lease is not owned by worker")
        if task["lease_expires_at"] is None or task["lease_expires_at"] <= now:
            raise ValueError("frontier lease expired")
        checkpoint = _upsert_worker_checkpoint(conn, run_id, worker_id, task, phase=args.get("phase", "worker"), status="CHECKPOINTED", cursor=cursor)
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (_now(), run_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    conn = _connect(run_id)
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, "worker_checkpoint")
    conn.close()
    return {"ok": True, "run_id": run_id, "task_id": task_id, "checkpoint": checkpoint, "cursor": cursor}


def _resume_worker(args: dict) -> dict:
    run_id = args.get("run_id", "")
    worker_id = str(args.get("worker_id", "")).strip()[:128]
    requested_task_id = str(args.get("task_id", "")).strip()
    if not worker_id:
        raise ValueError("worker_id is required")
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM run_state WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("run is not initialized")
        _recover_expired_leases(conn, now)
        checkpoint = _latest_worker_checkpoint(conn, run_id, requested_task_id)
        if checkpoint is None:
            conn.commit()
            return {"ok": False, "run_id": run_id, "resumed_from_checkpoint": False, "reason": "checkpoint_not_found"}
        task = conn.execute("SELECT * FROM frontier_tasks WHERE task_id=? AND run_id=?", (checkpoint["task_id"], run_id)).fetchone()
        if task is None:
            raise ValueError("checkpoint task not found")
        if task["state"] == "CLAIMED" and task["lease_expires_at"] and task["lease_expires_at"] > now:
            conn.commit()
            return {"ok": False, "run_id": run_id, "resumed_from_checkpoint": False, "reason": "worker_lease_active"}
        cursor = json.loads(checkpoint["cursor_json"] or "{}")
        pivot_type = task["pivot_type"]
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    claimed = _claim_frontier({**args, "task_id": task["task_id"], "pivot_type": pivot_type})
    if claimed.get("task") is None:
        return {"ok": False, "run_id": run_id, "resumed_from_checkpoint": False, "reason": claimed.get("stop_reason", "task_not_ready"), "checkpoint": {"checkpoint_id": checkpoint["checkpoint_id"], "task_id": checkpoint["task_id"], "phase": checkpoint["phase"], "status": checkpoint["status"], "cursor": cursor}}
    return {"ok": True, "run_id": run_id, "worker_id": worker_id, "resumed_from_checkpoint": True, "task": claimed["task"], "checkpoint": {"checkpoint_id": checkpoint["checkpoint_id"], "task_id": checkpoint["task_id"], "phase": checkpoint["phase"], "status": checkpoint["status"], "cursor": cursor}, "new_checkpoint": claimed.get("checkpoint")}


def _frontier_status(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        recovered = _recover_expired_leases(conn, now)
        run = _refresh_run_retry_state(conn, run_id, now, promote_ready=True)
        rows = conn.execute("SELECT state,COUNT(*) AS count FROM frontier_tasks WHERE run_id=? GROUP BY state", (run_id,)).fetchall()
        tasks = conn.execute("SELECT * FROM frontier_tasks WHERE run_id=? ORDER BY priority DESC,created_at ASC,task_id ASC", (run_id,)).fetchall()
        conn.commit()
        counts = {row["state"]: row["count"] for row in rows}
        return {"ok": True, "run_id": run_id, "recovered": recovered, "run_status": run["status"], "retry_at": run["retry_at"], "retry_reason": run["retry_reason"], "counts": counts, "tasks": [_task_dict(row, include_lease=False) for row in tasks]}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_run(args: dict) -> dict:
    run_id = args.get("run_id", "")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id is required and must be safe")
    target_url = _normalize_url(args.get("target_url", ""))
    max_tasks = min(500, max(1, int(args.get("max_tasks", 50))))
    max_searches = min(2000, max(1, int(args.get("max_searches", 200))))
    max_nodes = min(5000, max(1, int(args.get("max_nodes", 250))))
    max_edges = min(10000, max(1, int(args.get("max_edges", 500))))
    deadline_seconds = min(86400.0, max(60.0, float(args.get("deadline_seconds", 3600))))
    deadline_at = time.time() + deadline_seconds
    conn = _connect(run_id)
    now = _now()
    with conn:
        conn.execute("INSERT OR REPLACE INTO run_state(run_id,target_url,status,round,created_at,updated_at,max_tasks,max_searches,max_nodes,max_edges,deadline_at,tasks_executed,source_calls,saturation_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, target_url, "RUNNING", 0, now, now, max_tasks, max_searches, max_nodes, max_edges, deadline_at, 0, 0, None))
        domain = _domain(target_url)
        node_id = _stable_id("domain", domain)
        conn.execute("INSERT OR IGNORE INTO nodes(node_id,node_type,value,normalized_value,status,confidence,first_seen_round,frontier_state) VALUES(?,?,?,?,?,?,?,?)", (node_id, "domain", domain, domain, "OBSERVED", 1.0, 0, "OPEN"))
        from tools.companyintel_pivots import expand_pivots
        planned_specs = expand_pivots("domain", domain)
        planned_pivots = [spec.pivot_type for spec in planned_specs]
        priority = max(spec.priority for spec in planned_specs)
        conn.execute("INSERT OR REPLACE INTO frontier(node_id,priority,reason,planned_pivots,state) VALUES(?,?,?,?,?)", (node_id, priority, "seed domain inventory", _json(planned_pivots), "OPEN"))
        _enqueue_frontier_tasks(conn, run_id, node_id, priority, planned_pivots)
        start_event = _append_event_conn(conn, run_id, "run.started", {"target_url": target_url, "status": "RUNNING"}, "run.started")
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, "run_initialized")
    conn.close()
    return {"ok": True, "run_id": run_id, "status": "RUNNING", "seed_node_id": node_id, "event": start_event}


def _inventory(args: dict) -> dict:
    """Run the bounded code-owned inventory and persist its findings."""
    from tools.companyintel_inventory import InventoryLimits, extract_inventory

    run_id = args.get("run_id", "")
    target_url = _normalize_url(args.get("target_url", ""))
    conn = _connect(run_id)
    existing = conn.execute("SELECT target_url FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if existing:
        if existing["target_url"] != target_url:
            raise ValueError("run_id is already bound to a different target_url")
        initialized = {"seed_node_id": _stable_id("domain", _domain(target_url))}
    else:
        initialized = _init_run(args)
    limits = InventoryLimits(
        max_urls=int(args.get("max_urls", 32)),
        max_bytes_per_url=int(args.get("max_bytes_per_url", 256 * 1024)),
        max_total_bytes=int(args.get("max_total_bytes", 2 * 1024 * 1024)),
    )
    inventory = extract_inventory(args.get("target_url", ""), limits=limits)
    inventory_path = _run_dir(run_id) / "inventory.json"
    _atomic_json(inventory_path, inventory)
    persisted = 0
    for finding in inventory.get("findings", []):
        result = _record_observation({
            "run_id": run_id,
            "node_type": finding["node_type"],
            "value": finding["value"],
            "source_url": finding["source_url"],
            "excerpt": finding["excerpt"],
            "relation_from_node_id": initialized["seed_node_id"],
            "relation": "inventory_discovered",
            "retrieval_method": "bounded_http_inventory",
            "access_status": "success",
        })
        if result.get("ok"):
            persisted += 1
    return {
        "ok": True,
        "run_id": run_id,
        "inventory_path": str(inventory_path),
        "findings": persisted,
        "fetched_urls": inventory["stats"]["fetched_urls"],
        "errors": len(inventory["errors"]),
        "inventory": {key: inventory[key] for key in ("metadata", "urls", "documents", "images", "scripts", "identifiers", "external_domains", "discovered_sources", "stats")},
    }


def _record_observation(args: dict) -> dict:
    run_id = args.get("run_id", "")
    node_type = args.get("node_type", "")
    if node_type not in _ALLOWED_NODE_TYPES:
        raise ValueError("unsupported node_type")
    value = _normalize_value(node_type, args.get("value", ""))
    source_url = _normalize_url(args.get("source_url", ""))
    excerpt = " ".join(str(args.get("excerpt", "")).split())[:2000]
    if not excerpt:
        raise ValueError("excerpt is required")
    conn = _connect(run_id)
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError("run is not initialized")
    node_id = _stable_id(node_type, value)
    source_node_id = args.get("relation_from_node_id", "")
    source_type = ""
    if source_node_id:
        source_row = conn.execute("SELECT node_type FROM nodes WHERE node_id=?", (source_node_id,)).fetchone()
        if source_row is None:
            conn.close()
            raise ValueError("relation_from_node_id does not exist")
        source_type = source_row["node_type"]
    relation = str(args.get("relation", "observed_in")).strip()
    if source_node_id and relation not in RELATIONS:
        conn.close()
        raise ValueError("unsupported graph relation")
    if source_node_id and not _endpoint_allowed(relation, source_type, node_type):
        conn.close()
        raise ValueError("edge endpoint types are incompatible with relation")
    evidence_id = _stable_id("evidence", source_url, excerpt)
    edge_id = _stable_id("edge", source_node_id, node_id, relation, evidence_id)
    now = _now()
    with conn:
        conn.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?,?,?,?,?,?)", (evidence_id, source_url, args.get("source_title", ""), args.get("source_tier", "C"), now, excerpt, hashlib.sha256(excerpt.encode()).hexdigest(), args.get("retrieval_method", "browser"), args.get("access_status", "success")))
        node_insert = conn.execute("INSERT OR IGNORE INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)", (node_id, node_type, value, value, "OBSERVED", float(args.get("confidence", 0.7)), _json([evidence_id]), _json([args.get("relation_from_node_id")] if args.get("relation_from_node_id") else []), run["round"], None, "OPEN"))
        if source_node_id:
            conn.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?,?)", (edge_id, source_node_id, node_id, relation, float(args.get("confidence", 0.7)), _json([evidence_id]), run["round"]))
        if node_insert.rowcount:
            from tools.companyintel_pivots import expand_pivots
            planned_specs = expand_pivots(node_type, value)
            pivots = [spec.pivot_type for spec in planned_specs]
            priority = max(spec.priority for spec in planned_specs)
            conn.execute("INSERT OR REPLACE INTO frontier(node_id,priority,reason,planned_pivots,state) VALUES(?,?,?,?,?)", (node_id, priority, f"typed pivots for {node_type}", _json(pivots), "OPEN"))
            _enqueue_frontier_tasks(conn, run_id, node_id, priority, pivots)
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (now, run_id))
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, "observation_recorded")
    conn.close()
    return {"ok": True, "run_id": run_id, "node_id": node_id, "evidence_id": evidence_id, "edge_id": edge_id}


def _record_search(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError("run is not initialized")
    search_id = args.get("search_id") or _stable_id("search", args.get("query", ""), args.get("source", ""), _now())
    with conn:
        conn.execute("INSERT OR REPLACE INTO search_log VALUES(?,?,?,?,?,?,?,?,?)", (search_id, run["round"], args.get("pivot_node_id", ""), args.get("pivot_type", ""), str(args.get("query", ""))[:1000], args.get("source", ""), args.get("status", "completed"), _json(args.get("new_node_ids", [])), args.get("limitation")))
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (_now(), run_id))
    _materialize(conn, run_id)
    count = conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]
    if count % 5 == 0:
        _checkpoint(conn, run_id, f"retrieval_batch_{count}")
    conn.close()
    return {"ok": True, "run_id": run_id, "search_id": search_id, "retrievals": count, "checkpoint_created": count % 5 == 0}


def _execute_frontier(args: dict) -> dict:
    """Execute the first registered worker slice: bounded public exact search."""
    from tools.companyintel_pivots import get_pivot
    from tools.companyintel_public_search import SearchLimits, execute_public_search

    run_id = args.get("run_id", "")
    pivot_type = str(args.get("pivot_type", "exact_search")).strip() or "exact_search"
    spec = get_pivot(pivot_type)
    worker_configs = {
        "public_search": {"mode": "exact", "node_type": "url", "relation": "search_result", "source": "duckduckgo_html"},
        "maps_search": {"mode": "maps", "node_type": "place", "relation": "maps_result", "source": "duckduckgo_maps_search"},
        "marketplace_search": {"mode": "marketplace", "node_type": "marketplace_profile", "relation": "marketplace_result", "source": "duckduckgo_marketplace_search"},
        "document_search": {"mode": "document", "node_type": "document", "relation": "document_result", "source": "duckduckgo_document_search"},
    }
    if spec is None:
        raise ValueError("unknown pivot type")
    if not spec.implemented:
        raise ValueError(f"pivot type {pivot_type} is not implemented; refusing generic fallback")
    worker_config = worker_configs.get(spec.worker if spec else "")
    if worker_config is None:
        raise ValueError("pivot type is not backed by an implemented typed worker")
    worker_id = str(args.get("worker_id", "")).strip()
    claimed = _claim_frontier({**args, "pivot_type": pivot_type})
    task = claimed.get("task")
    if task is None:
        return {"ok": True, "outcome": "NO_TASK", "task": None, "persisted_results": 0}
    conn = _connect(run_id)
    node = conn.execute("SELECT node_type,value FROM nodes WHERE node_id=?", (task["node_id"],)).fetchone()
    conn.close()
    if node is None:
        _lease_mutation({**args, "task_id": task["task_id"], "worker_id": worker_id, "lease_token": task["lease_token"], "error": "pivot node not found"}, "fail")
        return {"ok": False, "outcome": "FAILED", "task_id": task["task_id"], "persisted_results": 0}
    result = execute_public_search(
        node["node_type"],
        node["value"],
        limits=SearchLimits(
            timeout_seconds=min(15.0, max(1.0, float(args.get("search_timeout_seconds", 8)))),
            max_bytes=min(512 * 1024, max(4096, int(args.get("search_max_bytes", 256 * 1024)))),
            max_results=min(20, max(1, int(args.get("search_max_results", 10)))),
        ),
        mode=worker_config["mode"],
    )
    if result["outcome"] in {"RETRYABLE_ERROR", "UNAVAILABLE"}:
        failure = _lease_mutation({
            **args,
            "task_id": task["task_id"],
            "worker_id": worker_id,
            "lease_token": task["lease_token"],
            "error": result.get("error") or result["outcome"],
            "retry_after_seconds": args.get("retry_after_seconds", 30),
        }, "fail")
        return {"ok": True, "outcome": result["outcome"], "task_id": task["task_id"], "state": failure["state"], "persisted_results": 0, "error": result.get("error")}
    new_node_ids = []
    persisted = 0
    for item in result["results"]:
        observation = _record_observation({
            "run_id": run_id,
            "node_type": worker_config["node_type"],
            "value": item["url"],
            "source_url": item["url"],
            "source_title": item["title"],
            "excerpt": item["snippet"] or item["title"] or item["url"],
            "relation_from_node_id": task["node_id"],
            "relation": worker_config["relation"],
            "retrieval_method": "public_search",
            "access_status": "success",
        })
        if observation.get("ok"):
            persisted += 1
            new_node_ids.append(observation["node_id"])
    _record_search({
        "run_id": run_id,
        "pivot_node_id": task["node_id"],
        "pivot_type": pivot_type,
        "query": result["query"],
        "source": worker_config["source"],
        "status": result["outcome"],
        "new_node_ids": new_node_ids,
        "limitation": None,
    })
    completed = _lease_mutation({
        **args,
        "task_id": task["task_id"],
        "worker_id": worker_id,
        "lease_token": task["lease_token"],
    }, "complete")
    return {"ok": True, "outcome": result["outcome"], "pivot_type": pivot_type, "task_id": task["task_id"], "state": completed["state"], "persisted_results": persisted, "new_node_ids": new_node_ids}


def _execute_inventory_frontier(args: dict) -> dict:
    run_id = args.get("run_id", "")
    worker_id = str(args.get("worker_id", "")).strip()
    claimed = _claim_frontier({**args, "pivot_type": "site_inventory"})
    task = claimed.get("task")
    if task is None:
        return {"ok": True, "outcome": "NO_TASK", "pivot_type": "site_inventory", "task": None, "persisted_results": 0}
    conn = _connect(run_id)
    run = conn.execute("SELECT target_url FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if run is None:
        raise ValueError("run is not initialized")
    try:
        result = _inventory({"run_id": run_id, "target_url": run["target_url"]})
        completed = _lease_mutation({
            **args,
            "task_id": task["task_id"],
            "worker_id": worker_id,
            "lease_token": task["lease_token"],
        }, "complete")
        return {
            "ok": True,
            "outcome": "COMPLETED_WITH_RESULTS" if result.get("findings", 0) else "COMPLETED_ZERO_RESULTS",
            "pivot_type": "site_inventory",
            "task_id": task["task_id"],
            "state": completed["state"],
            "persisted_results": result.get("findings", 0),
        }
    except Exception as exc:
        failure = _lease_mutation({
            **args,
            "task_id": task["task_id"],
            "worker_id": worker_id,
            "lease_token": task["lease_token"],
            "error": str(exc),
        }, "fail")
        return {"ok": True, "outcome": "RETRYABLE_ERROR", "pivot_type": "site_inventory", "task_id": task["task_id"], "state": failure["state"], "persisted_results": 0}


def _persist_reasoning_decision(conn: sqlite3.Connection, run_id: str, decision: dict) -> dict:
    """Persist a validated controller decision without granting graph authority."""
    decision_type = str(decision.get("decision_type", "")).strip()
    if not decision_type:
        raise ValueError("reasoning decision_type is required")
    decision_id = str(decision.get("decision_id") or _stable_id("decision", run_id, decision_type, decision)).strip()
    now = _now()
    conn.execute(
        "INSERT OR REPLACE INTO reasoning_decisions(decision_id,run_id,decision_type,decision_json,model,provider,graph_revision,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (decision_id, run_id, decision_type, _json(decision), "nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter", "", now),
    )
    return {"decision_id": decision_id, "decision_type": decision_type}


def _persist_reasoning_rejection(conn: sqlite3.Connection, run_id: str, operation: str, error: Exception) -> dict:
    """Persist malformed LLM proposals without granting them graph authority."""
    decision = {
        "decision_type": "REASONING_REJECTED",
        "operation": operation,
        "error_code": "INVALID_LLM_PROPOSAL",
        "error": str(error)[:1000],
        "status": "REJECTED",
    }
    return _persist_reasoning_decision(conn, run_id, decision)


def _reasoning_cycle(args: dict) -> dict:
    """Run one bounded, code-owned reasoning cycle over durable graph state.

    The controller may propose typed decisions only. This function persists the
    audit trail, while frontier/task/run mutations remain owned by graph code.
    """
    run_id = str(args.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("run_id is required")
    from tools.companyintel_reasoning_controller import CompanyIntelReasoningController

    conn = _connect(run_id)
    try:
        run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise ValueError("run is not initialized")
        evidence_rows = [dict(row) for row in conn.execute("SELECT evidence_id FROM evidence ORDER BY evidence_id").fetchall()]
        evidence_ids = {row["evidence_id"] for row in evidence_rows}
        if not evidence_ids:
            return {"ok": True, "run_id": run_id, "decisions": [], "skipped": "no_evidence"}
        nodes = [dict(row) for row in conn.execute("SELECT node_id,node_type,value FROM nodes ORDER BY node_id").fetchall()]
        candidates = []
        for row in conn.execute("SELECT * FROM identity_candidates WHERE run_id=? ORDER BY score DESC,candidate_id", (run_id,)).fetchall():
            item = dict(row)
            for key in ("matching_dimensions", "conflicting_dimensions", "missing_dimensions", "evidence_ids"):
                item[key] = json.loads(item[key] or "[]")
            candidates.append(item)
        contradictions = [
            {"candidate_id": item["candidate_id"], "conflicting_dimensions": item.get("conflicting_dimensions", []), "evidence_ids": item.get("evidence_ids", []), "status": "UNRESOLVED"}
            for item in candidates if item.get("conflicting_dimensions")
        ]
        tasks = [dict(row) for row in conn.execute("SELECT task_id,node_id,pivot_type,priority,state FROM frontier_tasks WHERE run_id=? AND state='OPEN' ORDER BY priority DESC,task_id", (run_id,)).fetchall()]
        eligible = [{"pivot_type": row["pivot_type"], "priority": int(row["priority"])} for row in tasks[:8]]
        node_id = nodes[0]["node_id"] if nodes else ""
        subject_ids = [item["candidate_id"] for item in candidates] or [node_id]
        payload = {
            "run_id": run_id, "round": run["round"], "nodes": nodes[:50],
            "evidence_ids": sorted(evidence_ids), "candidates": candidates[:50],
            "contradictions": contradictions, "eligible_pivots": eligible,
        }
        controller = CompanyIntelReasoningController()
        decisions = []
        try:
            hypotheses = controller.generate_hypotheses_with_llm(payload=payload, evidence_ids=evidence_ids)
            for decision in hypotheses:
                decisions.append(_persist_reasoning_decision(conn, run_id, decision))
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            decisions.append(_persist_reasoning_rejection(conn, run_id, "generate_hypotheses", exc))
        if candidates or args.get("force_all"):
            decision = controller.compare_candidates_with_llm(payload=payload)
            decisions.append(_persist_reasoning_decision(conn, run_id, decision))
        if contradictions or args.get("force_all"):
            decision = controller.analyze_contradictions_with_llm(payload={**payload, "contradictions": contradictions})
            decisions.append(_persist_reasoning_decision(conn, run_id, decision))
        if eligible and (args.get("force_all") or len({item["priority"] for item in eligible}) == 1):
            decision = controller.select_equal_priority_pivot_with_llm(payload={**payload, "node_id": node_id})
            decisions.append(_persist_reasoning_decision(conn, run_id, decision))
        if contradictions or args.get("force_all"):
            ambiguity_payload = {**payload, "ambiguity_type": "candidate_conflict", "subject_ids": subject_ids, "conflicts": [item["candidate_id"] for item in contradictions]}
            decision = controller.interpret_ambiguity_with_llm(payload={**ambiguity_payload, "evidence_ids": sorted(evidence_ids)})
            decisions.append(_persist_reasoning_decision(conn, run_id, decision))
        if args.get("force_all"):
            decision = controller.decide_weak_branch_with_llm(payload={
                **payload, "clue_id": node_id, "clue_strength": "weak", "allowed_pivots": eligible,
                "max_new_tasks": 2, "max_budget_cost": 2, "current_depth": 0, "max_depth": 2,
            })
            decisions.append(_persist_reasoning_decision(conn, run_id, decision))
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (_now(), run_id))
        conn.commit()
        _materialize(conn, run_id)
        _checkpoint(conn, run_id, "reasoning_cycle")
        return {"ok": True, "run_id": run_id, "decisions": decisions, "decision_count": len(decisions), "frontier_mutation": "code_owned_only"}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _run_frontier(args: dict) -> dict:
    """Run a bounded automatic dispatch loop over supported typed pivots."""
    run_id = args.get("run_id", "")
    worker_id = str(args.get("worker_id", "")).strip()
    if not worker_id:
        raise ValueError("worker_id is required")
    max_tasks = min(50, max(1, int(args.get("max_tasks", 10))))
    outcomes = []
    supported = ("site_inventory", "exact_search", "maps", "marketplaces", "documents")
    while len(outcomes) < max_tasks:
        dispatched = None
        for pivot_type in supported:
            if pivot_type == "site_inventory":
                candidate = _execute_inventory_frontier(args)
            else:
                candidate = _execute_frontier({**args, "pivot_type": pivot_type})
            if candidate.get("outcome") != "NO_TASK":
                dispatched = candidate
                break
        if dispatched is None:
            break
        outcomes.append(dispatched)
    conn = _connect(run_id)
    from tools.companyintel_pivots import get_pivot
    for row in conn.execute("SELECT task_id,pivot_type FROM frontier_tasks WHERE run_id=? AND state='OPEN'", (run_id,)).fetchall():
        spec = get_pivot(row["pivot_type"])
        if spec is None or not spec.implemented:
            conn.execute("UPDATE frontier_tasks SET state='SKIPPED_UNSUPPORTED',last_error='typed_worker_not_implemented',updated_at=? WHERE task_id=?", (_now(), row["task_id"]))
    conn.commit()
    run_row = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    unresolved = conn.execute("SELECT COUNT(*) FROM frontier_tasks WHERE run_id=? AND state IN ('OPEN','CLAIMED','RETRY_WAIT')", (run_id,)).fetchone()[0]
    if unresolved == 0:
        now = _now()
        conn.execute("UPDATE frontier SET state='CLOSED' WHERE node_id IN (SELECT node_id FROM frontier_tasks WHERE run_id=?) AND state='OPEN'", (run_id,))
        conn.execute("UPDATE nodes SET frontier_state='CLOSED' WHERE node_id IN (SELECT node_id FROM frontier_tasks WHERE run_id=?) AND frontier_state='OPEN'", (run_id,))
        conn.execute("UPDATE run_state SET saturation_reason=COALESCE(saturation_reason,'FRONTIER_EXHAUSTED:all_scheduled_tasks_completed'),updated_at=? WHERE run_id=?", (now, run_id))
    elif run_row and run_row["status"] == "RUNNING" and run_row["tasks_executed"] >= run_row["max_tasks"]:
        conn.execute("UPDATE run_state SET status='PARTIAL',saturation_reason='BUDGET_EXHAUSTED:max_tasks',updated_at=? WHERE run_id=?", (_now(), run_id))
        conn.commit()
        _materialize(conn, run_id)
        _checkpoint(conn, run_id, "budget_exhausted_max_tasks")
    else:
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (_now(), run_id))
        conn.commit()
        _materialize(conn, run_id)
        _checkpoint(conn, run_id, "frontier_saturation_check")
    conn.commit()
    if unresolved == 0 or not (run_row and run_row["status"] == "RUNNING" and run_row["tasks_executed"] >= run_row["max_tasks"]):
        _materialize(conn, run_id)
        _checkpoint(conn, run_id, "frontier_saturation_check")

    legal_identity = _legal_identity_exhaustion({"run_id": run_id})
    reasoning = None
    if args.get("reasoning_enabled"):
        reasoning = _reasoning_cycle({**args, "run_id": run_id})
    return {
        "ok": True,
        "run_id": run_id,
        "worker_id": worker_id,
        "tasks_executed": len(outcomes),
        "legal_identity_exhaustion": legal_identity,
        "outcomes": outcomes,
        "remaining_supported_tasks": sum(1 for item in outcomes if item.get("state") in {"RETRY_WAIT", "CLAIMED"}),
        "reasoning": reasoning,
    }


_IDENTITY_MATCH_SCORES = {
    "exact_official_legal_id": 100,
    "domain_authoritative_record": 95,
    "legal_name_phone_address": 90,
    "exact_phone_email_marketplace_legal_card": 90,
    "name_director_country": 80,
    "brand_phone_email": 75,
    "name_only": 40,
    "logo_favicon_only": 20,
}


def _candidate_status(score: int, conflicts: list[str]) -> str:
    if conflicts:
        return "REJECTED" if "exact_official_legal_id" in conflicts else "AMBIGUOUS"
    if score >= 100:
        return "VERIFIED"
    if score >= 90:
        return "HIGH_CONFIDENCE"
    if score >= 75:
        return "PROBABLE"
    if score >= 40:
        return "AMBIGUOUS"
    return "REJECTED"


def _record_identity_candidate(args: dict) -> dict:
    run_id = args.get("run_id", "")
    candidate_id = str(args.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError("candidate_id is required")
    match_types = [str(item).strip() for item in args.get("match_types", []) if str(item).strip()]
    unknown = sorted(set(match_types) - set(_IDENTITY_MATCH_SCORES))
    if unknown:
        raise ValueError(f"unknown match_types: {', '.join(unknown)}")
    matching = sorted(set(match_types), key=lambda item: (-_IDENTITY_MATCH_SCORES[item], item))
    conflicts = sorted(set(str(item).strip() for item in args.get("conflicting_dimensions", []) if str(item).strip()))
    missing = sorted(set(str(item).strip() for item in args.get("missing_dimensions", []) if str(item).strip()))
    evidence_ids = sorted(set(str(item).strip() for item in args.get("evidence_ids", []) if str(item).strip()))
    score = max((_IDENTITY_MATCH_SCORES[item] for item in matching), default=0)
    status = _candidate_status(score, conflicts)
    conn = _connect(run_id)
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError("run is not initialized")
    if evidence_ids:
        placeholders = ",".join("?" * len(evidence_ids))
        existing_evidence = {row[0] for row in conn.execute(f"SELECT evidence_id FROM evidence WHERE evidence_id IN ({placeholders})", evidence_ids).fetchall()}
        unknown_evidence = sorted(set(evidence_ids) - existing_evidence)
        if unknown_evidence:
            conn.close()
            raise ValueError(f"unknown evidence_ids: {', '.join(unknown_evidence)}")
    now = _now()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO identity_candidates(candidate_id,run_id,legal_name,legal_id,country,score,status,matching_dimensions,conflicting_dimensions,missing_dimensions,evidence_ids,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (candidate_id, run_id, str(args.get("legal_name", "")).strip(), str(args.get("legal_id", "")).strip(), str(args.get("country", "")).strip(), score, status, _json(matching), _json(conflicts), _json(missing), _json(evidence_ids), now, now),
        )
        conn.execute("UPDATE run_state SET updated_at=? WHERE run_id=?", (now, run_id))
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, "identity_candidate_recorded")
    conn.close()
    return {"ok": True, "run_id": run_id, "candidate_id": candidate_id, "score": score, "status": status, "matching_dimensions": matching, "conflicting_dimensions": conflicts, "missing_dimensions": missing, "evidence_ids": evidence_ids}


def _legal_identity_coverage(conn: sqlite3.Connection, run_id: str) -> dict:
    required = (
        "structured_data", "phone", "email", "address", "domain_variants",
        "document_metadata", "javascript_identifiers", "marketplace", "maps",
        "archive", "social", "image_or_favicon", "official_registry",
        "procurement", "court_or_ip",
    )
    pivot_map = {
        "structured_data": ("site_inventory",), "phone": ("site_inventory",),
        "email": ("site_inventory",), "address": ("site_inventory",),
        "domain_variants": ("exact_search",), "document_metadata": ("documents",),
        "javascript_identifiers": ("site_inventory",), "marketplace": ("marketplaces",),
        "maps": ("maps",), "archive": ("archive",), "social": ("social",),
        "image_or_favicon": ("site_inventory",), "official_registry": ("official_registry",),
        "procurement": ("procurement",), "court_or_ip": ("court_or_ip",),
    }
    unsupported = {"archive", "social", "official_registry", "procurement", "court_or_ip"}
    coverage = {}
    for coverage_class in required:
        pivots = pivot_map[coverage_class]
        rows = conn.execute(
            "SELECT pivot_type,state,last_error FROM frontier_tasks WHERE run_id=? AND pivot_type IN ({}) ORDER BY updated_at DESC".format(",".join("?" * len(pivots))),
            (run_id, *pivots),
        ).fetchall()
        if not rows:
            if coverage_class == "maps":
                inventory_rows = conn.execute("SELECT state FROM frontier_tasks WHERE run_id=? AND pivot_type='site_inventory'", (run_id,)).fetchall()
                relevant_nodes = conn.execute("SELECT COUNT(*) FROM nodes WHERE node_type IN ('address','phone','place')").fetchone()[0]
                if inventory_rows and all(row["state"] == "COMPLETED" for row in inventory_rows) and relevant_nodes == 0:
                    coverage[coverage_class] = "NOT_APPLICABLE"
                    continue
            coverage[coverage_class] = "UNAVAILABLE" if coverage_class in unsupported else "MISSING"
            continue
        states = [row["state"] for row in rows]
        if any(state in {"OPEN", "CLAIMED", "RETRY_WAIT"} for state in states):
            coverage[coverage_class] = "PENDING"
            continue
        failures = [str(row["last_error"] or "").lower() for row in rows if row["state"] == "FAILED"]
        if failures:
            text = " ".join(failures)
            if "captcha" in text:
                coverage[coverage_class] = "BLOCKED_BY_CAPTCHA"
            elif "rate" in text:
                coverage[coverage_class] = "RATE_LIMITED"
            elif "budget" in text:
                coverage[coverage_class] = "BUDGET_EXHAUSTED"
            else:
                coverage[coverage_class] = "RETRYABLE_ERROR"
            continue
        logs = conn.execute(
            "SELECT new_node_ids,limitation,status FROM search_log WHERE pivot_type IN ({}) ORDER BY rowid DESC".format(",".join("?" * len(pivots))),
            pivots,
        ).fetchall()
        limitations = " ".join(str(row["limitation"] or "").lower() for row in logs)
        if "captcha" in limitations:
            coverage[coverage_class] = "BLOCKED_BY_CAPTCHA"
        elif "rate" in limitations:
            coverage[coverage_class] = "RATE_LIMITED"
        elif "budget" in limitations:
            coverage[coverage_class] = "BUDGET_EXHAUSTED"
        else:
            has_results = False
            for log in logs:
                try:
                    has_results = has_results or bool(json.loads(log["new_node_ids"] or "[]"))
                except json.JSONDecodeError:
                    pass
            coverage[coverage_class] = "COMPLETED_WITH_RESULTS" if has_results else "COMPLETED_ZERO_RESULTS"
    return coverage


def _legal_identity_exhaustion(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError("run is not initialized")
    coverage = _legal_identity_coverage(conn, run_id)
    candidates = conn.execute("SELECT candidate_id,status,score FROM identity_candidates WHERE run_id=? ORDER BY score DESC,updated_at DESC", (run_id,)).fetchall()
    viable_candidates = [dict(row) for row in candidates if row["status"] != "REJECTED"]
    pending_states = {"PENDING", "MISSING"}
    blocked_states = {"UNAVAILABLE", "RATE_LIMITED", "BLOCKED_BY_CAPTCHA", "RETRYABLE_ERROR", "BUDGET_EXHAUSTED"}
    pending = [name for name, state in coverage.items() if state in pending_states]
    blocked = [name for name, state in coverage.items() if state in blocked_states]
    if viable_candidates:
        status = "PARTIAL"
        reason = "identity_candidate_present"
    elif pending:
        status = "PENDING"
        reason = "coverage_pending"
    elif blocked:
        status = "PARTIAL"
        reason = "coverage_blocked"
    else:
        status = "EXHAUSTED"
        reason = "coverage_complete"
    allowed = status == "EXHAUSTED" and run["status"] not in {"PARTIAL", "FAILED", "CANCELLED"}
    identity_status = "NOT_FOUND_ELIGIBLE" if allowed else "UNRESOLVED"
    with conn:
        conn.execute(
            "UPDATE run_state SET legal_identity_status=?,legal_identity_reason=?,legal_identity_checked_at=?,legal_identity_coverage=?,updated_at=? WHERE run_id=?",
            (status, reason, _now(), _json(coverage), _now(), run_id),
        )
    _materialize(conn, run_id)
    _checkpoint(conn, run_id, f"legal_identity_exhaustion_{status.lower()}")
    conn.close()
    return {"ok": True, "run_id": run_id, "status": status, "identity_status": identity_status, "allowed_not_found": allowed, "reason": reason, "coverage": coverage, "pending_classes": pending, "blocking_classes": blocked, "candidates": [dict(row) for row in candidates], "saturation_reason": run["saturation_reason"]}


def _exhaustion_gate(args: dict) -> dict:
    result = _legal_identity_exhaustion(args)
    result["reason"] = "coverage_complete" if result["allowed_not_found"] else "coverage_pending_or_missing"
    return result


def _resume_frontier(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("run is not initialized")
        if row["status"] == "RETRY_WAIT":
            row = _refresh_run_retry_state(conn, run_id, now, promote_ready=True)
        if row["status"] == "RETRY_WAIT" and float(row["retry_at"] or 0) > now:
            conn.commit()
            return {"ok": False, "run_id": run_id, "resumed": False, "status": "RETRY_WAIT", "retry_at": row["retry_at"], "reason": "retry_not_ready"}
        if row["status"] not in {"RESUMABLE", "PARTIAL", "RETRY_WAIT"}:
            conn.commit()
            return {"ok": False, "run_id": run_id, "resumed": False, "status": row["status"], "reason": "run_not_resumable"}
        additional_tasks = min(500, max(1, int(args.get("additional_tasks", 25))))
        additional_searches = min(2000, max(1, int(args.get("additional_searches", additional_tasks))))
        deadline_seconds = min(86400.0, max(60.0, float(args.get("deadline_seconds", 3600))))
        conn.execute("UPDATE run_state SET status='RUNNING',max_tasks=max_tasks+?,max_searches=max_searches+?,deadline_at=?,saturation_reason=NULL,retry_at=NULL,retry_reason=NULL,resume_count=resume_count+1,updated_at=? WHERE run_id=?", (additional_tasks, additional_searches, now + deadline_seconds, _now(), run_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    result = _run_frontier({**args, "worker_id": args.get("worker_id") or f"resume-{run_id}"})
    result["status"] = _summary({"run_id": run_id})["status"]
    result["resumed"] = True
    return result


_RUN_TRANSITIONS = {
    "RUNNING": {"RUNNING", "RETRY_WAIT", "PARTIAL", "COMPLETED", "FAILED", "CANCELLED"},
    "RETRY_WAIT": {"RETRY_WAIT", "RESUMABLE", "FAILED", "CANCELLED"},
    "RESUMABLE": {"RESUMABLE", "RUNNING", "CANCELLED"},
    "PARTIAL": {"PARTIAL", "RESUMABLE", "CANCELLED"},
    "COMPLETED": {"COMPLETED"},
    "FAILED": {"FAILED"},
    "CANCELLED": {"CANCELLED"},
}
_RUN_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "PARTIAL"}


def _event_payload(args: dict) -> dict:
    payload = args.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("event payload must be an object")
    encoded = _json(payload)
    if len(encoded.encode("utf-8")) > 16384:
        raise ValueError("event payload exceeds 16384 bytes")
    return payload


def _append_event_conn(conn: sqlite3.Connection, run_id: str, event_type: str, payload: dict, idempotency_key: str = "") -> dict:
    event_type = " ".join(str(event_type).split())[:128]
    if not event_type:
        raise ValueError("event_type is required")
    idempotency_key = " ".join(str(idempotency_key).split())[:256]
    if idempotency_key:
        existing = conn.execute(
            "SELECT event_id,event_type,payload_json,created_at FROM run_events WHERE run_id=? AND idempotency_key=?",
            (run_id, idempotency_key),
        ).fetchone()
        if existing:
            return {"event_id": existing["event_id"], "run_id": run_id, "event": existing["event_type"], "payload": json.loads(existing["payload_json"]), "created_at": existing["created_at"], "idempotent": True}
    row = conn.execute("SELECT COALESCE(MAX(event_id), 0) + 1 AS next_id FROM run_events WHERE run_id=?", (run_id,)).fetchone()
    event_id = int(row["next_id"])
    created_at = _now()
    conn.execute(
        "INSERT INTO run_events(run_id,event_id,event_type,payload_json,idempotency_key,created_at) VALUES(?,?,?,?,?,?)",
        (run_id, event_id, event_type, _json(payload), idempotency_key, created_at),
    )
    return {"event_id": event_id, "run_id": run_id, "event": event_type, "payload": payload, "created_at": created_at, "idempotent": False}


def _append_event(args: dict) -> dict:
    run_id = args["run_id"]
    conn = _connect(run_id)
    try:
        if conn.execute("SELECT 1 FROM run_state WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("run is not initialized")
        event = _append_event_conn(conn, run_id, args.get("event_type", ""), _event_payload(args), args.get("idempotency_key", ""))
        conn.commit()
        return {"ok": True, **event}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _replay_events(args: dict) -> dict:
    run_id = args["run_id"]
    after_event_id = max(0, int(args.get("after_event_id", 0)))
    limit = min(max(1, int(args.get("limit", 500))), 5000)
    conn = _connect(run_id)
    try:
        if conn.execute("SELECT 1 FROM run_state WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("run is not initialized")
        rows = conn.execute(
            "SELECT event_id,event_type,payload_json,created_at FROM run_events WHERE run_id=? AND event_id>? ORDER BY event_id ASC LIMIT ?",
            (run_id, after_event_id, limit),
        ).fetchall()
        return {"ok": True, "run_id": run_id, "events": [{"event_id": row["event_id"], "event": row["event_type"], "payload": json.loads(row["payload_json"]), "created_at": row["created_at"]} for row in rows]}
    finally:
        conn.close()


def _transition_run(args: dict) -> dict:
    run_id = args["run_id"]
    target = " ".join(str(args.get("status", "")).upper().split())
    if target not in _RUN_TRANSITIONS:
        raise ValueError(f"unsupported run status: {target}")
    conn = _connect(run_id)
    try:
        row = conn.execute("SELECT status FROM run_state WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("run is not initialized")
        current = row["status"]
        if target not in _RUN_TRANSITIONS.get(current, set()):
            raise ValueError(f"invalid run transition: {current} -> {target}")
        reason = str(args.get("reason", "")).strip()[:512]
        if target == "COMPLETED":
            open_tasks = conn.execute("SELECT COUNT(*) FROM frontier_tasks WHERE run_id=? AND state IN ('OPEN','CLAIMED','RETRY_WAIT')", (run_id,)).fetchone()[0]
            saturation = conn.execute("SELECT saturation_reason FROM run_state WHERE run_id=?", (run_id,)).fetchone()[0]
            artifacts = conn.execute("SELECT COUNT(*) FROM report_artifacts WHERE run_id=?", (run_id,)).fetchone()[0]
            reviews = conn.execute("SELECT COUNT(*) FROM report_reviews WHERE run_id=?", (run_id,)).fetchone()[0]
            if open_tasks or not saturation or str(saturation).startswith("BUDGET_EXHAUSTED:") or not artifacts or not reviews:
                raise ValueError("completion requires frontier exhaustion, non-budget saturation_reason, report_artifact and report_review")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE run_state SET status=?,retry_reason=CASE WHEN ? IN ('RUNNING','COMPLETED','FAILED','CANCELLED','PARTIAL') THEN NULL ELSE retry_reason END,updated_at=? WHERE run_id=?", (target, target, _now(), run_id))
        event = _append_event_conn(conn, run_id, "run.status", {"from": current, "to": target, "reason": reason}, args.get("idempotency_key", f"status:{current}:{target}:{reason}"))
        if target == "COMPLETED":
            _append_event_conn(conn, run_id, "run.completed", {"status": "COMPLETED", "reason": reason}, f"completed:{run_id}")
        conn.commit()
        return {"ok": True, "run_id": run_id, "from": current, "status": target, "event": event}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _generate_report(args: dict) -> dict:
    run_id = args["run_id"]
    conn = _connect(run_id)
    try:
        run = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise ValueError("run is not initialized")
        graph_path = _run_dir(run_id) / "graph.json"
        if not graph_path.exists():
            raise ValueError("run projection is not available")
        graph_bytes = graph_path.read_bytes()
        graph = json.loads(graph_bytes)
        graph_revision = hashlib.sha256(graph_bytes).hexdigest()
        frontier_open = int(graph["research_state"].get("frontier_open", 0))
        saturation_reason = graph["research_state"].get("saturation_reason")
        gate = {
            "terminal": run["status"] == "COMPLETED" or bool(args.get("terminal_transition_pending")),
            "has_nodes": len(graph.get("nodes", [])) > 0,
            "has_evidence": len(graph.get("evidence", [])) > 0,
            "frontier_exhausted": frontier_open == 0 or bool(saturation_reason),
            "saturation_reason": saturation_reason,
        }
        gate["result"] = "PASS" if all(gate[key] for key in ("terminal", "has_nodes", "has_evidence", "frontier_exhausted")) else "FAIL"
        if gate["result"] != "PASS":
            return {"ok": False, "run_id": run_id, "reason": "pre_report_gate_not_satisfied", "gate": gate}
        from tools.companyintel_reasoning_controller import CompanyIntelReasoningController
        controller = CompanyIntelReasoningController()
        evidence_ids = [row.get("evidence_id") for row in graph.get("evidence", [])]
        candidates = graph.get("identity_candidates", [])
        contradictions = [
            {"candidate_id": row.get("candidate_id"), "conflicting_dimensions": row.get("conflicting_dimensions", []), "status": "UNRESOLVED"}
            for row in candidates if row.get("conflicting_dimensions")
        ]
        report_text = str(args.get("report_text", "")).strip()[:200000]
        review_text = str(args.get("review_text", "")).strip()[:50000]
        report_model = str(args.get("model", ""))
        report_provider = str(args.get("provider", ""))
        reasoning_claims = []
        if args.get("reasoning_enabled"):
            graph_digest = {
                "run_id": run_id,
                "graph_revision": graph_revision,
                "target": graph.get("target", {}),
                "research_state": graph.get("research_state", {}),
                "nodes": graph.get("nodes", []),
                "edges": graph.get("edges", []),
                "evidence": graph.get("evidence", []),
                "evidence_ids": evidence_ids,
                "identity_candidates": candidates,
                "identity_resolution": graph.get("identity_resolution", {}),
                "contradictions": contradictions,
                "source_limitations": [row.get("limitation") for row in graph.get("search_log", []) if row.get("limitation")],
            }
            synthesis = controller.synthesize_report_with_llm(graph_digest=graph_digest)
            report_text = synthesis["report_text"]
            reasoning_claims = synthesis["claims"]
            review = controller.review_report_with_llm(report={
                **graph_digest,
                "report_text": report_text,
                "claims": reasoning_claims,
            })
            review_text = review["review_text"]
            report_model = review["model"]
            report_provider = review["provider"]
        artifact = controller.build_report_artifact(
            run_id=run_id,
            graph_revision=graph_revision,
            evidence_ids=evidence_ids,
            candidates=candidates,
            contradictions=contradictions,
            unresolved_ambiguity=graph.get("identity_resolution", {}),
            source_limitations=[row.get("limitation") for row in graph.get("search_log", []) if row.get("limitation")],
            report_text=report_text,
            model=report_model,
            provider=report_provider,
            review_text=review_text,
            gate=gate,
        )
        artifact["claims"] = reasoning_claims
        generated_at = artifact["generated_at"]
        conn.execute("INSERT OR REPLACE INTO report_artifacts(report_id,run_id,graph_revision,report_json,model,provider,review_label,gate_json,generated_at) VALUES(?,?,?,?,?,?,?,?,?)", (artifact["report_id"], run_id, graph_revision, _json(artifact), artifact["model"], artifact["provider"], artifact["review_label"], _json(gate), generated_at))
        conn.execute("INSERT OR REPLACE INTO reasoning_decisions(decision_id,run_id,decision_type,decision_json,model,provider,graph_revision,created_at) VALUES(?,?,?,?,?,?,?,?)", (_stable_id("decision", run_id, graph_revision, "SYNTHESIZE_REPORT"), run_id, "SYNTHESIZE_REPORT", _json({"artifact_id": artifact["report_id"], "gate": gate}), artifact["model"], artifact["provider"], graph_revision, generated_at))
        comparison = controller.compare_candidates(candidates)
        contradiction_result = controller.analyze_contradictions(run_id=run_id, contradictions=contradictions)
        conn.execute("INSERT OR REPLACE INTO reasoning_decisions(decision_id,run_id,decision_type,decision_json,model,provider,graph_revision,created_at) VALUES(?,?,?,?,?,?,?,?)", (_stable_id("decision", run_id, graph_revision, "COMPARE_CANDIDATES"), run_id, "COMPARE_CANDIDATES", _json(comparison), artifact["model"], artifact["provider"], graph_revision, generated_at))
        conn.execute("INSERT OR REPLACE INTO reasoning_decisions(decision_id,run_id,decision_type,decision_json,model,provider,graph_revision,created_at) VALUES(?,?,?,?,?,?,?,?)", (_stable_id("decision", run_id, graph_revision, "ANALYZE_CONTRADICTIONS"), run_id, "ANALYZE_CONTRADICTIONS", _json(contradiction_result), artifact["model"], artifact["provider"], graph_revision, generated_at))
        if review_text:
            review_id = _stable_id("review", artifact["report_id"], artifact["model"], artifact["provider"])
            review = {"review_id": review_id, "report_id": artifact["report_id"], "verdict": "RECORDED", "text": review_text, "review_label": artifact["review_label"], "independent_review": False, "model": artifact["model"], "provider": artifact["provider"], "graph_revision": graph_revision}
            conn.execute("INSERT OR REPLACE INTO report_reviews(review_id,report_id,run_id,graph_revision,review_json,model,provider,review_label,independent_review,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (review_id, artifact["report_id"], run_id, graph_revision, _json(review), artifact["model"], artifact["provider"], artifact["review_label"], 0, generated_at))
        conn.commit()
        return {"ok": True, "run_id": run_id, "report": artifact, "persisted": True}
    except Exception as exc:
        conn.rollback()
        try:
            failure_text = " ".join(str(exc).split())[:1000]
            conn.execute(
                "INSERT OR REPLACE INTO reasoning_decisions(decision_id,run_id,decision_type,decision_json,model,provider,graph_revision,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (_stable_id("decision", run_id, "REASONING_FAILURE", failure_text), run_id, "REASONING_FAILURE", _json({"error": failure_text, "reasoning_enabled": bool(args.get("reasoning_enabled")), "operation": "synthesize_report_or_review"}), "nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter", "", _now()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
        raise
    finally:
        conn.close()


def _projection(args: dict) -> dict:
    run_id = args["run_id"]
    view = str(args.get("view", "graph")).strip().lower()
    if view not in {"graph", "frontier", "evidence", "report"}:
        raise ValueError("unsupported projection view")
    conn = _connect(run_id)
    try:
        if view == "report":
            row = conn.execute("SELECT report_json FROM report_artifacts WHERE run_id=? ORDER BY generated_at DESC LIMIT 1", (run_id,)).fetchone()
            if row:
                return {"ok": True, "run_id": run_id, "view": view, "data": {"ready": True, "artifact": json.loads(row["report_json"])}}
    finally:
        conn.close()
    path = _run_dir(run_id) / "graph.json"
    if not path.exists():
        raise ValueError("run projection is not available")
    graph = json.loads(path.read_text(encoding="utf-8"))
    if view == "graph":
        return {"ok": True, "run_id": run_id, "view": view, "data": graph}
    if view == "frontier":
        return {"ok": True, "run_id": run_id, "view": view, "data": {"research_state": graph["research_state"], "frontier": graph["frontier"], "frontier_tasks": graph["frontier_tasks"]}}
    if view == "evidence":
        return {"ok": True, "run_id": run_id, "view": view, "data": {"evidence": graph["evidence"], "search_log": graph["search_log"]}}
    status = graph["research_state"]["status"]
    evidence_count = len(graph["evidence"])
    ready = status == "COMPLETED" and evidence_count > 0 and bool(graph["research_state"].get("saturation_reason"))
    return {"ok": True, "run_id": run_id, "view": view, "data": {"ready": ready, "status": status, "reason": None if ready else "pre_report_gate_not_satisfied", "summary": {"nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "evidence": evidence_count, "frontier_open": graph["research_state"]["frontier_open"]}, "identity_resolution": graph["identity_resolution"], "identity_candidates": graph["identity_candidates"]}}


def _summary(args: dict) -> dict:
    run_id = args.get("run_id", "")
    conn = _connect(run_id)
    row = conn.execute("SELECT * FROM run_state WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError("run is not initialized")
    identity_candidates = []
    for candidate in conn.execute("SELECT candidate_id,legal_name,legal_id,country,score,status,matching_dimensions,conflicting_dimensions,missing_dimensions,evidence_ids FROM identity_candidates WHERE run_id=? ORDER BY score DESC,updated_at DESC", (run_id,)).fetchall():
        item = dict(candidate)
        for key in ("matching_dimensions", "conflicting_dimensions", "missing_dimensions", "evidence_ids"):
            item[key] = json.loads(item[key] or "[]")
        identity_candidates.append(item)
    worker_checkpoints = []
    for checkpoint in conn.execute("SELECT checkpoint_id,run_id,worker_id,task_id,attempt,phase,status,cursor_json,created_at,updated_at FROM worker_checkpoints WHERE run_id=? ORDER BY updated_at DESC,checkpoint_id DESC", (run_id,)).fetchall():
        item = dict(checkpoint)
        item["cursor"] = json.loads(item.pop("cursor_json") or "{}")
        worker_checkpoints.append(item)
    event_count = conn.execute("SELECT COUNT(*) FROM run_events WHERE run_id=?", (run_id,)).fetchone()[0]
    latest_event = conn.execute("SELECT event_id,event_type,created_at FROM run_events WHERE run_id=? ORDER BY event_id DESC LIMIT 1", (run_id,)).fetchone()
    result = {"ok": True, "run_id": run_id, "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0], "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0], "evidence": conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], "frontier_open": conn.execute("SELECT COUNT(*) FROM frontier WHERE state='OPEN'").fetchone()[0], "search_log": conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0], "round": row["round"], "status": row["status"], "saturation_reason": row["saturation_reason"], "retry_at": row["retry_at"], "retry_reason": row["retry_reason"], "retry_count": row["retry_count"], "resume_count": row["resume_count"], "event_count": event_count, "latest_event": dict(latest_event) if latest_event else None, "legal_identity_exhaustion": {"status": row["legal_identity_status"], "reason": row["legal_identity_reason"], "checked_at": row["legal_identity_checked_at"], "coverage": json.loads(row["legal_identity_coverage"] or "{}")}, "identity_candidates": identity_candidates, "worker_checkpoints": worker_checkpoints, "budgets": {"max_tasks": row["max_tasks"], "max_searches": row["max_searches"], "max_nodes": row["max_nodes"], "max_edges": row["max_edges"], "deadline_at": row["deadline_at"]}, "usage": {"tasks_executed": row["tasks_executed"], "source_calls": row["source_calls"]}}
    conn.close()
    return result


def companyintel_graph(args: dict, **_kwargs) -> str:
    if not check_companyintel_requirements():
        return json.dumps({"ok": False, "error": "companyintel graph tool is profile-gated"})
    try:
        action = args.get("action", "")
        if action == "init_run": result = _init_run(args)
        elif action == "inventory": result = _inventory(args)
        elif action == "schedule_frontier": result = _schedule_frontier(args)
        elif action == "claim_frontier": result = _claim_frontier(args)
        elif action == "renew_frontier": result = _lease_mutation(args, "renew")
        elif action == "complete_frontier": result = _lease_mutation(args, "complete")
        elif action == "fail_frontier": result = _lease_mutation(args, "fail")
        elif action == "checkpoint_worker": result = _checkpoint_worker(args)
        elif action == "resume_worker": result = _resume_worker(args)
        elif action == "frontier_status": result = _frontier_status(args)
        elif action == "record_observation": result = _record_observation(args)
        elif action == "record_search": result = _record_search(args)
        elif action == "execute_frontier": result = _execute_frontier(args)
        elif action == "run_frontier": result = _run_frontier(args)
        elif action == "reasoning_cycle": result = _reasoning_cycle(args)
        elif action == "resume_frontier": result = _resume_frontier(args)
        elif action == "exhaustion_gate": result = _exhaustion_gate(args)
        elif action == "legal_identity_exhaustion": result = _legal_identity_exhaustion(args)
        elif action == "record_identity_candidate": result = _record_identity_candidate(args)
        elif action == "append_event": result = _append_event(args)
        elif action == "replay_events": result = _replay_events(args)
        elif action == "transition_run": result = _transition_run(args)
        elif action == "projection": result = _projection(args)
        elif action == "generate_report": result = _generate_report(args)
        elif action == "summary": result = _summary(args)
        else: raise ValueError("unsupported action")
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


_COMPANYINTEL_GRAPH_SCHEMA = {
    "name": "companyintel_graph",
    "description": "Deterministically persist companyintel graph nodes, evidence, edges and typed frontier. Required for graph state; profile-gated.",
    "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["init_run", "inventory", "schedule_frontier", "claim_frontier", "renew_frontier", "complete_frontier", "fail_frontier", "checkpoint_worker", "resume_worker", "frontier_status", "execute_frontier", "run_frontier", "reasoning_cycle", "resume_frontier", "exhaustion_gate", "legal_identity_exhaustion", "record_identity_candidate", "record_observation", "record_search", "append_event", "replay_events", "transition_run", "projection", "generate_report", "summary"]}, "run_id": {"type": "string"}, "target_url": {"type": "string"}, "node_type": {"type": "string"}, "value": {"type": "string"}, "source_url": {"type": "string"}, "source_title": {"type": "string"}, "excerpt": {"type": "string"}, "relation_from_node_id": {"type": "string"}, "relation": {"type": "string"}, "worker_id": {"type": "string"}, "task_id": {"type": "string"}, "lease_token": {"type": "string"}, "phase": {"type": "string"}, "cursor": {"type": "object", "additionalProperties": True}, "lease_seconds": {"type": "number"}, "max_attempts": {"type": "integer"}, "retry_after_seconds": {"type": "number"}, "pivot_type": {"type": "string"}, "error": {"type": "string"}, "search_timeout_seconds": {"type": "number"}, "search_max_bytes": {"type": "integer"}, "search_max_results": {"type": "integer"}, "max_urls": {"type": "integer"}, "max_bytes_per_url": {"type": "integer"}, "max_total_bytes": {"type": "integer"}, "event_type": {"type": "string"}, "payload": {"type": "object", "additionalProperties": True}, "idempotency_key": {"type": "string"}, "after_event_id": {"type": "integer"}, "limit": {"type": "integer"}, "status": {"type": "string"}, "reason": {"type": "string"}, "view": {"type": "string", "enum": ["graph", "frontier", "evidence", "report"]}, "report_text": {"type": "string"}, "review_text": {"type": "string"}, "model": {"type": "string"}, "provider": {"type": "string"}}, "required": ["action", "run_id"]},
}

registry.register(name="companyintel_graph", toolset="companyintel", schema=_COMPANYINTEL_GRAPH_SCHEMA, handler=companyintel_graph, check_fn=check_companyintel_requirements, description=_COMPANYINTEL_GRAPH_SCHEMA["description"], emoji="🕸️", max_result_size_chars=12000)

# Narrow typed tools: the LLM mutates the durable store only through these
# operations, never through generic file writes of graph.json.
def _typed_tool(action: str, args: dict, **_kwargs) -> str:
    return companyintel_graph({**args, "action": action})

_TYPED_SCHEMA = lambda name, description, properties: {
    "name": name,
    "description": description,
    "parameters": {"type": "object", "properties": properties, "required": ["run_id"]},
}
_COMMON = {"run_id": {"type": "string"}}

_INIT_SCHEMA = _TYPED_SCHEMA("companyintel_init_run", "Initialize a durable CompanyIntel run with explicit target URL.", {**_COMMON, "target_url": {"type": "string"}})
registry.register(name="companyintel_init_run", toolset="companyintel", schema=_INIT_SCHEMA, handler=lambda args, **kw: _typed_tool("init_run", args, **kw), check_fn=check_companyintel_requirements, description=_INIT_SCHEMA["description"], max_result_size_chars=12000)
registry.register(name="companyintel_record_observation", toolset="companyintel", schema=_TYPED_SCHEMA("companyintel_record_observation", "Persist a typed observation, evidence and exact graph relation.", {**_COMMON, "node_type": {"type": "string"}, "value": {"type": "string"}, "source_url": {"type": "string"}, "excerpt": {"type": "string"}, "relation_from_node_id": {"type": "string"}, "relation": {"type": "string"}}), handler=lambda args, **kw: _typed_tool("record_observation", args, **kw), check_fn=check_companyintel_requirements, description="Persist typed evidence-backed observation.", max_result_size_chars=12000)
registry.register(name="companyintel_enqueue_pivots", toolset="companyintel", schema=_TYPED_SCHEMA("companyintel_enqueue_pivots", "Enqueue code-registry pivots in the durable frontier.", {**_COMMON, "node_id": {"type": "string"}, "pivot_types": {"type": "array", "items": {"type": "string"}}}), handler=lambda args, **kw: _typed_tool("schedule_frontier", args, **kw), check_fn=check_companyintel_requirements, description="Enqueue typed pivots.", max_result_size_chars=12000)
registry.register(name="companyintel_claim_frontier", toolset="companyintel", schema=_TYPED_SCHEMA("companyintel_claim_frontier", "Claim one durable frontier task with a worker lease.", {**_COMMON, "worker_id": {"type": "string"}, "pivot_type": {"type": "string"}}), handler=lambda args, **kw: _typed_tool("claim_frontier", args, **kw), check_fn=check_companyintel_requirements, description="Claim frontier task.", max_result_size_chars=12000)
registry.register(name="companyintel_complete_frontier", toolset="companyintel", schema=_TYPED_SCHEMA("companyintel_complete_frontier", "Complete a claimed frontier task with lease ownership.", {**_COMMON, "worker_id": {"type": "string"}, "task_id": {"type": "string"}, "lease_token": {"type": "string"}}), handler=lambda args, **kw: _typed_tool("complete_frontier", args, **kw), check_fn=check_companyintel_requirements, description="Complete frontier task.", max_result_size_chars=12000)
registry.register(name="companyintel_checkpoint", toolset="companyintel", schema=_TYPED_SCHEMA("companyintel_checkpoint", "Persist worker cursor and graph checkpoint.", {**_COMMON, "worker_id": {"type": "string"}, "task_id": {"type": "string"}, "lease_token": {"type": "string"}, "cursor": {"type": "object"}}), handler=lambda args, **kw: _typed_tool("checkpoint_worker", args, **kw), check_fn=check_companyintel_requirements, description="Persist checkpoint.", max_result_size_chars=12000)
registry.register(name="companyintel_graph_summary", toolset="companyintel", schema=_TYPED_SCHEMA("companyintel_graph_summary", "Read durable graph summary and lifecycle truth.", _COMMON), handler=lambda args, **kw: _typed_tool("summary", args, **kw), check_fn=check_companyintel_requirements, description="Read graph summary.", max_result_size_chars=12000)
registry.register(name="companyintel_candidate_ledger", toolset="companyintel", schema=_TYPED_SCHEMA("companyintel_candidate_ledger", "Read durable identity candidate ledger.", _COMMON), handler=lambda args, **kw: _typed_tool("projection", {**args, "view": "graph"}, **kw), check_fn=check_companyintel_requirements, description="Read candidate ledger.", max_result_size_chars=12000)
registry.register(name="companyintel_validate", toolset="companyintel", schema=_TYPED_SCHEMA("companyintel_validate", "Run the code-owned validation/exhaustion gate.", _COMMON), handler=lambda args, **kw: _typed_tool("exhaustion_gate", args, **kw), check_fn=check_companyintel_requirements, description="Validate CompanyIntel run.", max_result_size_chars=12000)
