"""Deterministic, bounded pivot planners.

Planning is pure: it creates typed frontier intents and never performs I/O.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from .pivots import registry_for


def _clean(value: str) -> str:
    return " ".join(str(value).strip().split())[:512]


def plan_pivots(node_type: str, value: str) -> list[dict[str, object]]:
    """Expand one high-value node into stable, deduplicated frontier intents."""
    value = _clean(value)
    if not value:
        return []
    variants = [value]
    if node_type in {"url", "domain"}:
        host = (urlparse(value if "://" in value else "https://" + value).hostname or value).lower().rstrip(".")
        variants = list(dict.fromkeys([host, host.removeprefix("www."), value]))
    if node_type == "phone":
        digits = re.sub(r"\D", "", value)
        variants = list(dict.fromkeys([value, digits, "+" + digits] if digits else [value]))
    if node_type in {"brand", "legal_entity", "legal_entity_candidate", "person", "address", "email"}:
        variants = [value, value.lower()]
    result: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for spec in registry_for(node_type):
        for variant in variants[:3]:
            key = (spec.pivot_type, variant)
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "pivot_type": spec.pivot_type,
                "worker": spec.worker,
                "input_value": variant,
                "priority": spec.priority,
                "budget_cost": spec.budget_cost,
                "implemented": spec.implemented,
                "source_class": spec.source_class,
                "reason": f"{node_type}:{variant}",
            })
    return result
