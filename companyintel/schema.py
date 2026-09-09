"""Canonical CompanyIntel graph/v2 contract and legacy adapters.

The SQLite runtime remains authoritative; this module is the single contract for
JSON/JSONL projections, fixtures and validators.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "corporate-intelligence-graph/v2"
LEGACY_SCHEMA_VERSION = "corporate-intelligence-graph/v1"

NODE_TYPES = frozenset({
    "domain", "url", "phone", "email", "address", "place", "brand",
    "legal_entity", "legal_entity_candidate", "official_registry", "person", "document", "image",
    "favicon", "marketplace_profile", "social_profile", "analytics_id",
    "merchant_id", "iban", "legal_id", "product", "organization_candidate", "finding",
})
RELATIONS = frozenset({
    "publishes", "contains", "mentions", "observed_on", "located_at", "operated_by",
    "owned_by", "same_as", "registered_as", "listed_on", "maps_to", "derived_from",
    "mentions_legal_id", "visually_matches", "inventory_discovered", "observed_in",
    "search_result", "maps_result", "marketplace_result", "document_result", "supports",
})


RELATION_ENDPOINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "publishes": (("url", "domain"), ("url", "phone"), ("url", "email"), ("url", "document"), ("domain", "phone"), ("domain", "email"), ("domain", "document")),
    "listed_on": (("phone", "marketplace_profile"), ("email", "marketplace_profile"), ("brand", "marketplace_profile")),
    "maps_to": (("marketplace_profile", "legal_entity_candidate"), ("marketplace_profile", "legal_entity")),
    "registered_as": (("legal_entity_candidate", "official_registry"), ("legal_entity", "official_registry")),
    "search_result": (("*", "url"),),
    "maps_result": (("*", "place"),),
    "marketplace_result": (("*", "marketplace_profile"),),
    "document_result": (("*", "document"),),
    "supports": (("finding", "*"),),
}


def _endpoint_allowed(relation: str, source_type: str, target_type: str) -> bool:
    rules = RELATION_ENDPOINTS.get(relation)
    return not rules or any((source == "*" or source == source_type) and target == target_type for source, target in rules)


def deterministic_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part).strip().lower() for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def normalize_node_value(node_type: str, value: object) -> str:
    result = " ".join(str(value).strip().split())
    if node_type in {"domain", "email"}:
        result = result.lower()
    if node_type == "domain":
        result = result.rstrip(".")
    if node_type == "phone":
        result = re.sub(r"[^0-9+]", "", result)
    if not result:
        raise ValueError("node value must not be empty")
    if node_type == "email" and "@" not in result:
        raise ValueError("email node requires an email address")
    return result


def validate_graph(document: dict[str, Any], *, allow_legacy: bool = False) -> dict[str, Any]:
    """Validate one graph projection and return a compact validation receipt."""
    if not isinstance(document, dict):
        raise ValueError("graph must be an object")
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        if allow_legacy and version in {None, LEGACY_SCHEMA_VERSION, "v1"}:
            document = adapt_legacy_graph(document)
        else:
            raise ValueError(f"noncanonical graph schema: {version!r}")
    required = ("target", "research_state", "nodes", "edges", "evidence", "frontier", "search_log")
    missing = [key for key in required if key not in document]
    if missing:
        raise ValueError(f"canonical graph missing fields: {', '.join(missing)}")
    node_ids = set()
    node_types_by_id: dict[str, str] = {}
    evidence_ids = set()
    for node in document["nodes"]:
        if set(("node_id", "node_type", "value", "normalized_value")) - node.keys():
            raise ValueError("canonical node shape is incomplete")
        if node["node_type"] not in NODE_TYPES:
            raise ValueError(f"unknown canonical node type: {node['node_type']}")
        if node["node_id"] in node_ids:
            raise ValueError("duplicate node_id")
        node_ids.add(node["node_id"])
        node_types_by_id[node["node_id"]] = node["node_type"]
    for evidence in document["evidence"]:
        if set(("evidence_id", "source_url", "retrieved_at", "excerpt", "content_sha256")) - evidence.keys():
            raise ValueError("canonical evidence shape is incomplete")
        if evidence["evidence_id"] in evidence_ids:
            raise ValueError("duplicate evidence_id")
        evidence_ids.add(evidence["evidence_id"])
    for edge in document["edges"]:
        if "from_node" not in edge or "to_node" not in edge:
            raise ValueError("edges must use from_node/to_node")
        if "from" in edge or "to" in edge:
            raise ValueError("legacy edge keys are forbidden")
        if edge.get("relation") not in RELATIONS:
            raise ValueError(f"unknown edge relation: {edge.get('relation')}")
        if edge["from_node"] not in node_ids or edge["to_node"] not in node_ids:
            raise ValueError("edge endpoint is not a graph node")
        if not _endpoint_allowed(edge["relation"], node_types_by_id[edge["from_node"]], node_types_by_id[edge["to_node"]]):
            raise ValueError("edge endpoint types are incompatible with relation")
        if any(ref not in evidence_ids for ref in edge.get("evidence_ids", [])):
            raise ValueError("edge references missing evidence")
    return {
        "validated": True,
        "schema_version": SCHEMA_VERSION,
        "nodes": len(document["nodes"]),
        "edges": len(document["edges"]),
        "evidence": len(document["evidence"]),
    }


def adapt_legacy_graph(legacy: dict[str, Any]) -> dict[str, Any]:
    """Convert known v1 id/type/from/to shapes without mutating the input."""
    source = deepcopy(legacy)
    nodes = []
    for item in source.get("nodes", []):
        node_type = item.get("node_type", item.get("type"))
        value = item.get("value", item.get("label", item.get("name", "")))
        if not node_type or not value:
            raise ValueError("legacy node lacks type/value")
        normalized = normalize_node_value(node_type, value)
        nodes.append({
            "node_id": item.get("node_id", item.get("id", deterministic_id("node", node_type, normalized))),
            "node_type": node_type,
            "value": value,
            "normalized_value": normalized,
            "status": item.get("status", "OBSERVED"),
            "confidence": item.get("confidence", 0.0),
            "evidence_ids": item.get("evidence_ids", item.get("evidence", [])),
            "discovered_from": item.get("discovered_from", []),
            "first_seen_round": item.get("first_seen_round", 0),
            "last_expanded_round": item.get("last_expanded_round"),
            "frontier_state": item.get("frontier_state", "OPEN"),
        })
    edges = []
    for item in source.get("edges", []):
        edge = dict(item)
        edge["edge_id"] = edge.get("edge_id", edge.get("id", deterministic_id("edge", edge.get("from_node", edge.get("from")), edge.get("to_node", edge.get("to")), edge.get("relation", ""))))
        edge["from_node"] = edge.pop("from_node", edge.pop("from", None))
        edge["to_node"] = edge.pop("to_node", edge.pop("to", None))
        edges.append(edge)
    target = source.get("target", {})
    if isinstance(target, str):
        target = {"url": target}
    target = dict(target)
    target.setdefault("domain", (urlparse(target.get("url", "")).hostname or "").lower())
    result = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "research_state": source.get("research_state", {"status": "RUNNING", "round": 0}),
        "identity_resolution": source.get("identity_resolution", {"status": "UNRESOLVED", "candidate_ids": []}),
        "identity_candidates": source.get("identity_candidates", []),
        "worker_checkpoints": source.get("worker_checkpoints", []),
        "nodes": nodes,
        "edges": edges,
        "evidence": source.get("evidence", []),
        "frontier": source.get("frontier", []),
        "frontier_tasks": source.get("frontier_tasks", []),
        "search_log": source.get("search_log", []),
        "pivot_coverage": source.get("pivot_coverage", {}),
    }
    validate_graph(result)
    return result


def canonical_chain_fixture() -> dict[str, Any]:
    """A deterministic URL → phone → marketplace → legal entity → registry fixture."""
    values = [
        ("url", "https://example.test"),
        ("phone", "+380501234567"),
        ("marketplace_profile", "https://market.example.test/acme"),
        ("legal_entity_candidate", "Acme LLC"),
        ("official_registry", "UA-EDRPOU-12345678"),
    ]
    nodes = []
    for node_type, value in values:
        normalized = normalize_node_value(node_type, value)
        nodes.append({"node_id": deterministic_id("node", node_type, normalized), "node_type": node_type, "value": value, "normalized_value": normalized, "status": "OBSERVED", "confidence": 0.9, "evidence_ids": ["e-chain"], "discovered_from": [], "first_seen_round": 0, "last_expanded_round": 0, "frontier_state": "EXHAUSTED"})
    evidence = [{"evidence_id": "e-chain", "source_url": "https://example.test/contact", "retrieved_at": "2026-01-01T00:00:00Z", "excerpt": "bounded fixture evidence", "content_sha256": hashlib.sha256(b"bounded fixture evidence").hexdigest(), "source_title": "Fixture", "source_tier": "A", "retrieval_method": "fixture", "access_status": "success"}]
    relations = ["publishes", "listed_on", "maps_to", "registered_as"]
    edges = [{"edge_id": deterministic_id("edge", nodes[i]["node_id"], nodes[i + 1]["node_id"], relations[i]), "from_node": nodes[i]["node_id"], "to_node": nodes[i + 1]["node_id"], "relation": relations[i], "confidence": 0.9, "evidence_ids": ["e-chain"], "first_seen_round": 0} for i in range(4)]
    graph = {"schema_version": SCHEMA_VERSION, "target": {"url": "https://example.test", "domain": "example.test"}, "research_state": {"status": "COMPLETED", "round": 1, "frontier_open": 0}, "nodes": nodes, "edges": edges, "evidence": evidence, "frontier": [], "frontier_tasks": [], "search_log": [{"search_id": "s-chain", "round": 0, "pivot_node_id": nodes[0]["node_id"], "pivot_type": "fixture", "query": "fixture", "source": "fixture", "status": "COMPLETED", "new_node_ids": [node["node_id"] for node in nodes[1:]]}], "identity_resolution": {"status": "CONFIRMED", "candidate_ids": [nodes[3]["node_id"]]}, "identity_candidates": [], "worker_checkpoints": [], "pivot_coverage": {"fixture": "COMPLETED_WITH_RESULTS"}}
    validate_graph(graph)
    return graph
