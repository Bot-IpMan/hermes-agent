"""Stable typed DTOs for CompanyIntel API/tool boundaries."""
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class RunProjection:
    run_id: str
    status: str
    target_url: str = ""
    round: int | None = None
    saturation_reason: str | None = None
    counts: dict[str, int] = field(default_factory=dict)

@dataclass(frozen=True)
class GraphProjection:
    run: dict[str, Any]
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    frontier: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)

@dataclass(frozen=True)
class FrontierTask:
    task_id: str
    state: str
    pivot_type: str
    priority: int = 0

@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_url: str
    excerpt: str
    retrieved_at: str
    access_status: str = "success"
