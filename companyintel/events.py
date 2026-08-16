"""Durable lifecycle event helpers."""
from __future__ import annotations
from typing import Any
from .store import CompanyIntelStore

def append(store: CompanyIntelStore, run_id: str, event_type: str, payload: dict[str, Any], idempotency_key: str = "") -> dict[str, Any]:
    return store.call("append_event", run_id, event_type=event_type, payload=payload, idempotency_key=idempotency_key)

def replay(store: CompanyIntelStore, run_id: str, after: int = 0, limit: int = 200) -> dict[str, Any]:
    return store.replay_events(run_id, after=after, limit=limit)
