"""Bounded, SSRF-safe URL extraction for CompanyIntel verification."""
from __future__ import annotations

import hashlib
import html
import ipaddress
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from tools.companyintel_source_classifier import classify_source


@dataclass(frozen=True)
class ExtractionLimits:
    timeout_seconds: float = 8.0
    max_bytes: int = 512 * 1024
    max_redirects: int = 3
    max_excerpt: int = 2000


class ExtractionError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: list[str] = []
        self.text: list[str] = []
        self.jsonld: list[str] = []
        self._in_title = False
        self._in_script = False
        self._script_type = ""
        self._script: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k.lower(): v or "" for k, v in attrs}
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag == "script":
            self._in_script = True
            self._script_type = values.get("type", "").lower()
            self._script = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag == "script":
            if self._script_type == "application/ld+json":
                value = "".join(self._script).strip()
                if value:
                    self.jsonld.append(value[:20000])
            self._in_script = False
            self._script_type = ""
            self._script = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        if self._in_script:
            self._script.append(data)
        elif data.strip():
            self.text.append(data)


def _private_host(host: str) -> bool:
    if not host or host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local")):
        return True
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            return True
    return False


def _safe_url(value: str) -> str:
    parsed = urlparse(str(value).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ExtractionError("UNSAFE_URL")
    if _private_host(parsed.hostname.lower().rstrip(".")):
        raise ExtractionError("SSRF_BLOCKED")
    return parsed._replace(fragment="").geturl()


class _BoundedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, target_host: str, max_redirects: int):
        super().__init__()
        self.target_host = target_host
        self.max_redirects = max_redirects
        self.count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.count += 1
        if self.count > self.max_redirects:
            raise ExtractionError("REDIRECT_LIMIT")
        candidate = _safe_url(newurl)
        host = (urlparse(candidate).hostname or "").lower().rstrip(".")
        if not (host == self.target_host or host.endswith("." + self.target_host)):
            raise ExtractionError("REDIRECT_OUTSIDE_TARGET")
        return super().redirect_request(req, fp, code, msg, headers, candidate)


def extract_url(url: str, *, target_url: str = "", limits: ExtractionLimits | None = None) -> dict:
    """Fetch one URL without JavaScript and return bounded, non-secret facts."""
    limits = limits or ExtractionLimits()
    target = _safe_url(url)
    target_host = (urlparse(target_url or target).hostname or "").lower().rstrip(".")
    host = (urlparse(target).hostname or "").lower().rstrip(".")
    opener = urllib.request.build_opener(_BoundedRedirect(target_host or host, limits.max_redirects))
    request = urllib.request.Request(target, headers={"User-Agent": "Hermes-companyintel-extractor/1.0", "Accept": "text/html,text/plain,application/pdf;q=0.8,*/*;q=0.1"})
    try:
        with opener.open(request, timeout=limits.timeout_seconds) as response:
            final_url = _safe_url(response.geturl())
            content_type = response.headers.get_content_type() if response.headers else ""
            body = response.read(limits.max_bytes + 1)
            truncated = len(body) > limits.max_bytes
            body = body[:limits.max_bytes]
    except ExtractionError:
        raise
    except urllib.error.HTTPError as exc:
        raise ExtractionError(f"HTTP_{exc.code}", retryable=exc.code in {408, 425, 429, 500, 502, 503, 504}) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ExtractionError("FETCH_FAILED", retryable=True) from exc
    if not body:
        raise ExtractionError("EMPTY_RESPONSE")
    text = body.decode("utf-8", errors="replace")
    parser = _TextParser()
    if "html" in content_type or "xml" in content_type or not content_type:
        parser.feed(text)
        title = " ".join(" ".join(parser.title).split())[:300]
        excerpt = " ".join(" ".join(parser.text).split())[:limits.max_excerpt]
        # Challenges are a bounded, explicit outcome rather than empty evidence.
        lower = text.lower()
        challenge = any(marker in lower for marker in ("captcha", "verify you are human", "cf-chl", "challenge-platform"))
        if challenge:
            excerpt = "access challenge detected; page content was not treated as a zero-result source"
    else:
        title = ""
        excerpt = " ".join(text.split())[:limits.max_excerpt]
    classification = classify_source(final_url, target_url=target_url, retrieval_method="bounded_http_extraction", content_type=content_type)
    return {
        "ok": True,
        "source_url": final_url,
        "source_title": title,
        "content_type": content_type,
        "excerpt": html.unescape(excerpt),
        "content_sha256": hashlib.sha256(body).hexdigest(),
        "truncated": truncated,
        "jsonld": parser.jsonld[:10],
        **classification.as_dict(),
        "retrieval_method": "bounded_http_extraction",
        "access_status": "challenge" if "challenge detected" in excerpt else "success",
    }


__all__ = ["ExtractionError", "ExtractionLimits", "extract_url"]
