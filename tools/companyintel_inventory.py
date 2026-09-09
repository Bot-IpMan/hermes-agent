"""Bounded deterministic website inventory extraction for the companyintel profile.

The extractor is deliberately independent from an LLM or browser runtime. It fetches
small, same-site HTTP resources, normalizes the useful identifiers, and returns a
compact JSON-compatible inventory plus evidence-ready findings.
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse, urlunparse


@dataclass(frozen=True)
class InventoryLimits:
    max_urls: int = 32
    max_bytes_per_url: int = 256 * 1024
    max_total_bytes: int = 2 * 1024 * 1024
    timeout_seconds: float = 5.0
    deadline_seconds: float = 120.0
    max_findings: int = 100

    def __post_init__(self) -> None:
        if not 1 <= self.max_urls <= 128:
            raise ValueError("max_urls must be between 1 and 128")
        if not 1024 <= self.max_bytes_per_url <= 1024 * 1024:
            raise ValueError("max_bytes_per_url is outside the safe bound")
        if not self.max_bytes_per_url <= self.max_total_bytes <= 8 * 1024 * 1024:
            raise ValueError("max_total_bytes is outside the safe bound")
        if not 5.0 <= self.deadline_seconds <= 600.0:
            raise ValueError("deadline_seconds must be between 5 and 600")
        if not 1 <= self.max_findings <= 500:
            raise ValueError("max_findings must be between 1 and 500")


@dataclass(frozen=True)
class _Fetched:
    content_type: str
    body: bytes


class InventoryFetchError(RuntimeError):
    """Bounded, non-secret classification for a failed inventory fetch."""

    def __init__(self, error_class: str, *, retryable: bool, detail: str = "") -> None:
        super().__init__(error_class)
        self.error_class = error_class
        self.retryable = retryable
        self.detail = " ".join(str(detail).split())[:240]


def _fetch_error(exc: BaseException) -> InventoryFetchError:
    """Map transport failures to stable classes without persisting raw errors."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            return InventoryFetchError("HTTP_429", retryable=True)
        if exc.code in {408, 425, 500, 502, 503, 504}:
            return InventoryFetchError(f"HTTP_{exc.code}", retryable=True)
        return InventoryFetchError(f"HTTP_{exc.code}", retryable=False)
    if isinstance(exc, urllib.error.URLError):
        reason = str(getattr(exc, "reason", exc)).lower()
        if "timed out" in reason or "timeout" in reason:
            return InventoryFetchError("CONNECT_OR_READ_TIMEOUT", retryable=True)
        if "reset" in reason or "refused" in reason or "temporarily" in reason:
            return InventoryFetchError("CONNECTION_RESET_OR_REFUSED", retryable=True)
        if "ssl" in reason or "certificate" in reason:
            return InventoryFetchError("TLS_FAILURE", retryable=False)
        if "name or service" in reason or "nodename" in reason or "dns" in reason:
            return InventoryFetchError("DNS_FAILURE", retryable=False)
        return InventoryFetchError("URL_ERROR", retryable=True)
    if isinstance(exc, TimeoutError):
        return InventoryFetchError("CONNECT_OR_READ_TIMEOUT", retryable=True)
    if isinstance(exc, ConnectionError):
        return InventoryFetchError("CONNECTION_RESET_OR_REFUSED", retryable=True)
    return InventoryFetchError(type(exc).__name__.upper()[:64], retryable=False)


class _HTMLInventoryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.metadata: dict[str, str] = {}
        self.links: list[str] = []
        self.assets: list[tuple[str, str]] = []
        self.jsonld: list[str] = []
        self.identifiers: set[str] = set()
        self._in_title = False
        self._jsonld = False
        self._script_parts: list[str] = []
        self._script_type = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = (values.get("name") or values.get("property") or "").lower().strip()
            if key and values.get("content"):
                self.metadata.setdefault(key, _clean_text(values["content"]))
        if tag == "link" and values.get("href"):
            self.links.append(values["href"])
            rel = values.get("rel", "").lower()
            kind = values.get("type", "").lower()
            if "canonical" in rel:
                self.metadata.setdefault("canonical", values["href"])
            if "icon" in rel or "shortcut icon" in rel:
                self.assets.append(("favicon", values["href"]))
            if "alternate" in rel and ("rss" in kind or "atom" in kind):
                self.assets.append(("feed", values["href"]))
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if tag in {"img", "script", "iframe", "source"}:
            attr = "src" if tag != "source" else "src"
            if values.get(attr):
                self.assets.append(("image" if tag == "img" else "script", values[attr]))
        if tag == "script":
            self._script_type = values.get("type", "").lower() or "text/javascript"
            self._jsonld = self._script_type == "application/ld+json"
            self._script_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag == "script":
            script = "".join(self._script_parts).strip()
            if self._jsonld and script:
                self.jsonld.append(script[:20000])
            self.identifiers.update(_extract_identifiers(script))
            self._jsonld = False
            self._script_parts = []
            self._script_type = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._script_type:
            self._script_parts.append(data)
            self.identifiers.update(_extract_identifiers(data))


def _clean_text(value: str) -> str:
    return " ".join(value.split())[:1000]


def _extract_identifiers(value: str) -> set[str]:
    found: set[str] = set()
    for pattern in (
        r"\bG-[A-Z0-9]{4,20}\b",
        r"\bUA-\d{4,12}-\d{1,4}\b",
        r"\bGTM-[A-Z0-9]{4,20}\b",
        r"\bAW-\d{4,20}\b",
        r"\bfbq\s*\(\s*['\"]init['\"]\s*,\s*['\"](\d{5,30})",
    ):
        for match in re.findall(pattern, value, flags=re.I):
            found.add(match if isinstance(match, str) else match[0])
    return found


def _public_url(value: str) -> str:
    raw = str(value).strip()
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("target_url must be a public HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("target_url must not contain credentials")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError("private target URL is not allowed")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror:
        addresses = []
    for address in addresses:
        ip = address[4][0]
        if _private_ip(ip):
            raise ValueError("private target URL is not allowed")
    path = parsed.path or "/"
    return urlunparse((parsed.scheme.lower(), host + ((":" + str(parsed.port)) if parsed.port else ""), path, "", parsed.query, ""))


def _private_ip(value: str) -> bool:
    import ipaddress
    ip = ipaddress.ip_address(value)
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved


def _site_host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def _same_site(url: str, target_host: str) -> bool:
    host = _site_host(url)
    return host == target_host or host.endswith("." + target_host)


def _canonical_link(raw: str, base: str) -> str | None:
    try:
        value = _public_url(urljoin(base, raw))
    except (TypeError, ValueError):
        return None
    parsed = urlparse(value)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", parsed.query, ""))


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, target_host: str) -> None:
        super().__init__()
        self.target_host = target_host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = _public_url(newurl)
        if not _same_site(target, self.target_host):
            raise ValueError("redirect leaves target site")
        return super().redirect_request(req, fp, code, msg, headers, target)


def _fetch_url(url: str, limits: InventoryLimits) -> tuple[str, bytes]:
    target = _public_url(url)
    request = urllib.request.Request(target, headers={"User-Agent": "Hermes-companyintel-inventory/1.0", "Accept": "text/html,application/xml,text/plain,*/*;q=0.1"})
    opener = urllib.request.build_opener(_SafeRedirect(_site_host(target)))
    attempts = 3
    deadline_at = time.monotonic() + min(600.0, max(5.0, float(limits.deadline_seconds)))
    last_error: InventoryFetchError | None = None
    for attempt in range(attempts):
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise InventoryFetchError("INVENTORY_DEADLINE_EXCEEDED", retryable=True)
        try:
            with opener.open(request, timeout=min(float(limits.timeout_seconds), remaining)) as response:
                content_type = response.headers.get_content_type() if response.headers else ""
                body = response.read(limits.max_bytes_per_url + 1)
                if len(body) > limits.max_bytes_per_url:
                    body = body[:limits.max_bytes_per_url]
                return content_type, body
        except Exception as exc:
            last_error = _fetch_error(exc)
            if not last_error.retryable or attempt + 1 >= attempts:
                raise last_error from exc
    raise last_error or InventoryFetchError("URL_ERROR", retryable=True)


def _text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _xml_links(body: bytes) -> list[str]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    return sorted({element.text.strip() for element in root.iter() if element.tag.rsplit("}", 1)[-1].lower() in {"loc", "link"} and element.text and element.text.strip()})


def _excerpt(value: str) -> str:
    return _clean_text(value)[:500]


def _add_unique(items: list[str], value: str, limit: int = 100) -> None:
    if value and value not in items and len(items) < limit:
        items.append(value)


def _is_document(url: str) -> bool:
    return bool(re.search(r"\.(pdf|xls[xm]?|doc[xm]?|zip|xml)(?:$|[?#])", url, re.I))


def extract_inventory(
    target_url: str,
    *,
    limits: InventoryLimits | None = None,
    fetcher: Callable[[str, InventoryLimits], tuple[str, bytes]] | None = None,
) -> dict:
    limits = limits or InventoryLimits()
    target = _public_url(target_url).rstrip("/")
    target_host = _site_host(target)
    fetcher = fetcher or _fetch_url
    queue: list[str] = [target + "/", target + "/robots.txt", target + "/sitemap.xml", target + "/llms.txt"]
    fetched: set[str] = set()
    source_types: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    total_bytes = 0
    metadata = {"title": "", "description": "", "canonical": ""}
    urls: list[str] = []
    documents: list[str] = []
    images: list[str] = []
    scripts: list[str] = []
    identifiers: list[str] = []
    external_domains: list[str] = []
    discovered_sources: list[str] = []
    jsonld: list[dict] = []
    observations: list[dict[str, str]] = []
    deadline_at = time.monotonic() + min(600.0, max(5.0, float(limits.deadline_seconds)))

    def add_observation(node_type: str, value: str, source_url: str, excerpt: str) -> None:
        if len(observations) < limits.max_findings:
            observations.append({"node_type": node_type, "value": value, "source_url": source_url, "excerpt": _excerpt(excerpt)})

    while queue and len(fetched) < limits.max_urls and total_bytes < limits.max_total_bytes:
        if time.monotonic() >= deadline_at:
            errors.append({"url": target, "error": "INVENTORY_DEADLINE_EXCEEDED", "retryable": "true"})
            break
        url = queue.pop(0)
        if url in fetched or not _same_site(url, target_host):
            continue
        fetched.add(url)
        try:
            content_type, body = fetcher(url, limits)
        except InventoryFetchError as exc:
            row = {"url": url, "error": exc.error_class, "retryable": str(exc.retryable).lower()}
            if exc.detail:
                row["detail"] = exc.detail
            errors.append(row)
            continue
        except Exception as exc:
            classified = _fetch_error(exc)
            row = {"url": url, "error": classified.error_class, "retryable": str(classified.retryable).lower()}
            if classified.detail:
                row["detail"] = classified.detail
            errors.append(row)
            continue
        source_types[url] = content_type
        total_bytes += len(body)
        if not body:
            continue
        text = _text(body)
        lower_url = url.lower()
        if url.endswith("/robots.txt"):
            for line in text.splitlines():
                if line.lower().startswith("sitemap:"):
                    candidate = _canonical_link(line.split(":", 1)[1].strip(), url)
                    if candidate and _same_site(candidate, target_host):
                        _add_unique(discovered_sources, candidate)
                        if candidate not in queue and candidate not in fetched:
                            queue.append(candidate)
            add_observation("url", url, url, "robots.txt inventory source")
        elif url.endswith("/llms.txt"):
            add_observation("url", url, url, "llms.txt inventory source")
            for link in re.findall(r"https?://[^\s)>]+", text):
                candidate = _canonical_link(link, url)
                if candidate and _same_site(candidate, target_host):
                    _add_unique(urls, candidate)
                    if _is_document(candidate):
                        _add_unique(documents, candidate)
                        add_observation("document", candidate, url, "document URL discovered in llms.txt")
                    add_observation("url", candidate, url, "URL discovered in llms.txt")
        elif "xml" in content_type or lower_url.endswith((".xml", ".rss", ".atom")):
            for link in _xml_links(body):
                candidate = _canonical_link(link, url)
                if not candidate:
                    continue
                if _same_site(candidate, target_host):
                    _add_unique(urls, candidate)
                    if _is_document(candidate):
                        _add_unique(documents, candidate)
                        add_observation("document", candidate, url, "document URL discovered in XML feed or sitemap")
                    add_observation("url", candidate, url, "URL discovered in XML feed or sitemap")
            add_observation("url", url, url, "XML/RSS/Atom inventory source")
        elif "html" in content_type or lower_url.endswith(("/", ".html", ".htm")):
            parser = _HTMLInventoryParser()
            parser.feed(text[:limits.max_bytes_per_url])
            title = _clean_text(" ".join(parser.title_parts))
            if title and not metadata["title"]:
                metadata["title"] = title
                add_observation("brand", title, url, "HTML title")
            for key, destination in (("description", "description"), ("og:description", "description"), ("canonical", "canonical")):
                if parser.metadata.get(key) and not metadata[destination]:
                    metadata[destination] = parser.metadata[key]
            for raw in parser.links:
                candidate = _canonical_link(raw, url)
                if not candidate:
                    continue
                if _same_site(candidate, target_host):
                    _add_unique(urls, candidate)
                    if re.search(r"\.(pdf|xls[xm]?|doc[xm]?|zip|xml)(?:$|[?#])", candidate, re.I):
                        _add_unique(documents, candidate)
                        add_observation("document", candidate, url, "document link discovered in HTML")
                else:
                    _add_unique(external_domains, _site_host(candidate))
            for kind, raw in parser.assets:
                candidate = _canonical_link(raw, url)
                if not candidate:
                    continue
                if _same_site(candidate, target_host):
                    if kind == "image":
                        _add_unique(images, candidate)
                        add_observation("image", candidate, url, "image asset discovered in HTML")
                    elif kind == "favicon":
                        _add_unique(images, candidate)
                        add_observation("favicon", candidate, url, "favicon discovered in HTML")
                    elif kind == "script":
                        _add_unique(scripts, candidate)
                    elif kind == "feed" and candidate not in queue and candidate not in fetched:
                        queue.append(candidate)
                        _add_unique(discovered_sources, candidate)
            for identifier in sorted(parser.identifiers):
                _add_unique(identifiers, identifier)
                add_observation("analytics_id", identifier, url, "JavaScript analytics identifier")
            for raw in parser.jsonld:
                try:
                    value = json.loads(raw)
                    if isinstance(value, dict):
                        jsonld.append(value)
                except json.JSONDecodeError:
                    errors.append({"url": url, "error": "invalid_jsonld"})
    for value in urls:
        if _is_document(value):
            _add_unique(documents, value)
    for value in documents:
        if not any(item["value"] == value for item in observations):
            add_observation("document", value, target + "/", "document discovered in bounded site inventory")
    for value in urls[: limits.max_findings]:
        if not any(item["value"] == value for item in observations):
            add_observation("url", value, target + "/", "same-site URL discovered in bounded inventory")

    return {
        "schema_version": "companyintel-inventory/v1",
        "target": {"url": target, "domain": target_host},
        "metadata": metadata,
        "urls": sorted(urls)[: limits.max_findings],
        "documents": sorted(documents),
        "images": sorted(images),
        "scripts": sorted(scripts),
        "identifiers": sorted(identifiers),
        "external_domains": sorted(external_domains),
        "discovered_sources": sorted(discovered_sources),
        "jsonld": jsonld[:20],
        "findings": observations,
        "errors": errors[:50],
        "stats": {
            "fetched_urls": len(fetched),
            "total_bytes": total_bytes,
            "max_urls": limits.max_urls,
            "max_bytes_per_url": limits.max_bytes_per_url,
            "content_sha256": hashlib.sha256("\n".join(sorted(fetched)).encode()).hexdigest(),
        },
    }


def inventory_limits_dict(limits: InventoryLimits) -> dict:
    return asdict(limits)
