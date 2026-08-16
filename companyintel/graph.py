"""Typed graph projections and validation helpers."""
from __future__ import annotations
from typing import Any
from .store import CompanyIntelStore

def graph(store: CompanyIntelStore, run_id: str) -> dict[str, Any]:
    return store.projection(run_id, "graph")

def validate_graph(store: CompanyIntelStore, run_id: str) -> dict[str, Any]:
    result = store.call("exhaustion_gate", run_id)
    return {"run_id": run_id, "validated": bool(result.get("allowed_not_found", False)), **result}
