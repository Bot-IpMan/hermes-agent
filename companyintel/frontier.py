"""Durable frontier operations; no in-memory queue is authoritative."""
from __future__ import annotations
from typing import Any
from .store import CompanyIntelStore

def enqueue(store: CompanyIntelStore, run_id: str, node_id: str, pivots: list[str]) -> dict[str, Any]:
    return store.call("schedule_frontier", run_id, node_id=node_id, pivot_types=pivots)

def claim(store: CompanyIntelStore, run_id: str, worker_id: str, **kwargs: Any) -> dict[str, Any]:
    return store.call("claim_frontier", run_id, worker_id=worker_id, **kwargs)

def complete(store: CompanyIntelStore, run_id: str, task_id: str, lease_token: str, worker_id: str, **kwargs: Any) -> dict[str, Any]:
    return store.call("complete_frontier", run_id, task_id=task_id, lease_token=lease_token, worker_id=worker_id, **kwargs)
