"""Deterministic source authority classification for CompanyIntel.

Search providers are discovery mechanisms only.  This module is deliberately
small and code-owned: model output cannot promote an observation to Tier A/B.
Unknown or ambiguous origins fail closed to Tier C.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourceClassification:
    tier: str
    source_class: str
    authority_reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_tier": self.tier,
            "source_class": self.source_class,
            "authority_reason": self.authority_reason,
        }


_REGISTRY_HOSTS = frozenset({
    # Explicit public-record endpoints only. Commercial aggregators and
    # arbitrary government-looking domains remain discovery Tier C until an
    # adapter proves provenance.
    "usr.minjust.gov.ua", "data.gov.ua", "prozorro.gov.ua",
    "public.nazk.gov.ua", "court.gov.ua", "reyestr.court.gov.ua",
})
_REGISTRY_SUFFIXES = (".gov.ua", ".gov.uk", ".gouv.fr", ".go.jp")
_FIRST_PARTY_PATHS = (
    "/about", "/contact", "/contacts", "/company", "/imprint", "/legal",
    "/terms", "/privacy", "/documents", "/investor", "/investors",
    "/careers", "/blog", "/news",
)
_MARKETPLACE_HOST_HINTS = ("market", "shop", "store", "catalog", "directory")


def _host(url: str) -> str:
    return (urlparse(str(url)).hostname or "").lower().rstrip(".")


def _same_or_subdomain(host: str, target: str) -> bool:
    return bool(host and target and (host == target or host.endswith("." + target)))


def classify_source(
    url: str,
    *,
    target_url: str = "",
    retrieval_method: str = "",
    content_type: str = "",
    search_snippet: bool = False,
) -> SourceClassification:
    """Return a stable A/B/C classification with an auditable reason."""
    parsed = urlparse(str(url))
    host = _host(url)
    target_host = _host(target_url)
    method = str(retrieval_method).lower()
    if search_snippet or method in {"public_search", "tavily", "search_snippet"}:
        return SourceClassification("C", "search_snippet", "search_provider_discovery_only")
    if not host or parsed.scheme not in {"http", "https"}:
        return SourceClassification("C", "unknown", "invalid_or_unknown_origin")
    if host in _REGISTRY_HOSTS or host.endswith(_REGISTRY_SUFFIXES):
        return SourceClassification("A", "registry", "allowlisted_government_or_public_record_host")
    path = (parsed.path or "/").lower()
    if target_host and _same_or_subdomain(host, target_host):
        if any(path == marker or path.startswith(marker + "/") for marker in _FIRST_PARTY_PATHS):
            return SourceClassification("B", "first_party", "target_domain_authoritative_page")
        if "html" in str(content_type).lower() or not content_type:
            return SourceClassification("B", "first_party", "target_domain_bounded_extraction")
        if any(path.endswith(ext) for ext in (".pdf", ".doc", ".docx", ".xls", ".xlsx")):
            return SourceClassification("B", "document", "target_domain_published_document")
    if any(hint in host for hint in _MARKETPLACE_HOST_HINTS):
        return SourceClassification("C", "marketplace", "unverified_marketplace_or_directory")
    if "pdf" in str(content_type).lower() or path.endswith(".pdf"):
        return SourceClassification("C", "document", "document_origin_not_verified")
    return SourceClassification("C", "aggregator", "origin_not_allowlisted")


def classify_search_result(url: str, *, target_url: str = "") -> SourceClassification:
    """Search results are always discovery Tier C, regardless of hostname."""
    return classify_source(url, target_url=target_url, retrieval_method="public_search", search_snippet=True)


__all__ = ["SourceClassification", "classify_source", "classify_search_result"]
