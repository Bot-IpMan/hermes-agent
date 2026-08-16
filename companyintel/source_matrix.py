"""Wave 5 source-expansion matrix.

Only sources with a typed, bounded worker are executable. Others remain explicit
fail-closed entries so UI/report coverage cannot overclaim implementation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceClass:
    name: str
    worker: str
    implemented: bool
    outcome: str
    reason: str


SOURCE_MATRIX: tuple[SourceClass, ...] = (
    SourceClass("marketplace", "marketplace_search", False, "SKIPPED_UNSUPPORTED", "provider-specific marketplace adapter not configured"),
    SourceClass("maps", "maps_search", False, "SKIPPED_UNSUPPORTED", "provider-specific maps adapter not configured"),
    SourceClass("social", "not_implemented", False, "SKIPPED_UNSUPPORTED", "no bounded social adapter"),
    SourceClass("wayback", "not_implemented", False, "SKIPPED_UNSUPPORTED", "archive adapter not configured"),
    SourceClass("rdap_dns_ct", "not_implemented", False, "SKIPPED_UNSUPPORTED", "network identity adapter not configured"),
    SourceClass("official_registry", "not_implemented", False, "SKIPPED_UNSUPPORTED", "authoritative query contract unavailable"),
    SourceClass("prozorro_spending", "not_implemented", False, "SKIPPED_UNSUPPORTED", "bounded local/import contract unavailable"),
    SourceClass("court_ip", "not_implemented", False, "SKIPPED_UNSUPPORTED", "source requires a reviewed provider contract"),
    SourceClass("reverse_image", "not_implemented", False, "SKIPPED_UNSUPPORTED", "no approved provider/tool available"),
)


def source_matrix() -> tuple[SourceClass, ...]:
    return SOURCE_MATRIX


def source_class(name: str) -> SourceClass | None:
    return next((item for item in SOURCE_MATRIX if item.name == name), None)
