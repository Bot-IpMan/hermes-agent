"""Code-owned typed pivot registry for companyintel.

The registry owns automatic fact-to-pivot expansion. Workers remain separate
and are selected by the ``worker`` field; this module performs no network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PivotSpec:
    pivot_type: str
    worker: str
    priority: int
    budget_cost: int
    max_attempts: int = 3
    source_class: str = "public_search"
    implemented: bool = True


_REGISTRY: dict[str, tuple[PivotSpec, ...]] = {
    "domain": (
        PivotSpec("site_inventory", "inventory", 100, 1),
        PivotSpec("exact_search", "public_search", 80, 1),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    # Search results are discovery nodes; each bounded URL extraction is a
    # separate verification task with redirect/SSRF/MIME limits.
    "url": (PivotSpec("document_extraction", "document_extraction", 86, 2, source_class="document"),),
    "phone": (
        PivotSpec("exact_search", "public_search", 95, 1),
        PivotSpec("maps", "maps_search", 90, 2),
        PivotSpec("marketplaces", "marketplace_search", 85, 2),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "email": (
        PivotSpec("exact_search", "public_search", 92, 1),
        PivotSpec("marketplaces", "marketplace_search", 80, 2),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "address": (
        PivotSpec("exact_search", "public_search", 88, 1),
        PivotSpec("maps", "maps_search", 90, 2),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "brand": (
        PivotSpec("exact_search", "public_search", 78, 1),
        PivotSpec("marketplaces", "marketplace_search", 72, 2),
    ),
    "organization_candidate": (
        PivotSpec("exact_search", "public_search", 78, 1),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "legal_entity": (
        PivotSpec("exact_search", "public_search", 90, 1),
        PivotSpec("documents", "document_search", 75, 1),
        PivotSpec("official_registry", "not_implemented", 95, 3, source_class="registry", implemented=False),
    ),
    "legal_entity_candidate": (
        PivotSpec("exact_search", "public_search", 90, 1),
        PivotSpec("documents", "document_search", 75, 1),
        PivotSpec("official_registry", "not_implemented", 95, 3, source_class="registry", implemented=False),
    ),
    "person": (
        PivotSpec("exact_search", "public_search", 82, 1),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "place": (
        PivotSpec("exact_search", "public_search", 80, 1),
        PivotSpec("maps", "maps_search", 90, 2),
    ),
    "marketplace_profile": (
        PivotSpec("exact_search", "public_search", 82, 1),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "document": (
        PivotSpec("document_extraction", "document_extraction", 80, 2, source_class="document"),
    ),
    "image": (
        PivotSpec("image_favicon", "not_implemented", 60, 2, source_class="image", implemented=False),
    ),
    "favicon": (
        PivotSpec("image_favicon", "not_implemented", 60, 2, source_class="image", implemented=False),
    ),
    "social_profile": (
        PivotSpec("social", "not_implemented", 55, 2, source_class="social", implemented=False),
    ),
    "analytics_id": (
        PivotSpec("analytics_identifier", "not_implemented", 65, 2, source_class="javascript", implemented=False),
    ),
    "merchant_id": (
        PivotSpec("merchant_identifier", "not_implemented", 65, 2, source_class="javascript", implemented=False),
    ),
    "iban": (
        PivotSpec("iban_lookup", "not_implemented", 65, 2, source_class="financial", implemented=False),
    ),
    "legal_id": (
        PivotSpec("exact_search", "public_search", 90, 1),
        PivotSpec("official_registry", "not_implemented", 95, 3, source_class="registry", implemented=False),
    ),
    "product": (
        PivotSpec("exact_search", "public_search", 70, 1),
    ),
    "html": (PivotSpec("html", "not_implemented", 70, 2, source_class="html", implemented=False),),
    "javascript": (PivotSpec("javascript", "not_implemented", 68, 2, source_class="javascript", implemented=False),),
    "image": (PivotSpec("image_favicon", "not_implemented", 60, 2, source_class="image", implemented=False),),
    "official_registry": (PivotSpec("official_registry", "not_implemented", 90, 3, source_class="registry", implemented=False),),
    "archive": (PivotSpec("archive", "not_implemented", 55, 2, source_class="archive", implemented=False),),
    "social": (PivotSpec("social", "not_implemented", 55, 2, source_class="social", implemented=False),),
    "procurement": (PivotSpec("procurement", "not_implemented", 50, 3, source_class="procurement", implemented=False),),
    "court_or_ip": (PivotSpec("court_or_ip", "not_implemented", 50, 3, source_class="court_or_ip", implemented=False),),
}


def expand_pivots(node_type: str, _value: str = "") -> tuple[PivotSpec, ...]:
    """Return deterministic pivot specs; unknown types fail closed."""
    return _REGISTRY.get(node_type, (PivotSpec("unsupported", "not_implemented", 0, 0, source_class="unsupported", implemented=False),))


def get_pivot(pivot_type: str) -> PivotSpec | None:
    for specs in _REGISTRY.values():
        for spec in specs:
            if spec.pivot_type == pivot_type:
                return spec
    return None
