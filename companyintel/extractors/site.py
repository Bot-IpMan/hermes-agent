"""Safe, bounded site extractor primitives used by typed workers."""
from __future__ import annotations
from urllib.parse import urlparse

def same_public_origin(seed: str, candidate: str) -> bool:
    a, b = urlparse(seed), urlparse(candidate)
    return a.scheme in {"http", "https"} and b.scheme in {"http", "https"} and a.hostname == b.hostname

def bounded_text(value: str, limit: int = 2000) -> str:
    return " ".join(value.split())[:limit]
