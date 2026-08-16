"""Durable CompanyIntel facade over the profile-local graph runtime."""
from .models import RunProjection, GraphProjection, FrontierTask, EvidenceRecord
from .planners import plan_pivots
from .schema import SCHEMA_VERSION, adapt_legacy_graph, canonical_chain_fixture, validate_graph
from .source_matrix import SOURCE_MATRIX


def __getattr__(name: str):
    if name == "CompanyIntelStore":
        from .store import CompanyIntelStore
        return CompanyIntelStore
    raise AttributeError(name)


__all__ = [
    "CompanyIntelStore", "RunProjection", "GraphProjection", "FrontierTask", "EvidenceRecord",
    "SCHEMA_VERSION", "adapt_legacy_graph", "canonical_chain_fixture", "validate_graph",
    "plan_pivots", "SOURCE_MATRIX",
]
