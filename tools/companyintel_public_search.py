"""Bounded CompanyIntel public-search adapter.

CompanyIntel uses Hermes' configured WebSearchProvider registry as its primary
search path. The legacy DuckDuckGo HTML endpoint is an explicit, opt-in
fallback only; anti-bot pages are never interpreted as empty results.
"""
from __future__ import annotations

import ipaddress
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse


_ALLOWED_PROVIDER_NAMES = frozenset({
    "brave-free", "tavily", "exa", "searxng", "parallel", "firecrawl",
})
_CHALLENGE_MARKERS = (
    "anomaly.js",
    "challenge-form",
    "anomaly-modal",
    "bots use duckduckgo",
    "captcha",
)
_GENERIC_ERRORS = {
    "RETRYABLE_ERROR": "Search provider temporarily unavailable; execution continued with available sources.",
    "RATE_LIMITED": "Search provider rate limit reached; execution continued with available sources.",
    "BLOCKED_BY_CAPTCHA": "Search provider blocked automated search; execution continued with available sources.",
    "AUTH_FAILED": "Search provider authentication is unavailable; execution continued with available sources.",
    "INVALID_RESPONSE": "Search provider returned an invalid response; execution continued with available sources.",
    "UNAVAILABLE": "No supported search provider is available; execution continued with available sources.",
}


@dataclass(frozen=True)
class SearchLimits:
    timeout_seconds: float = 8.0
    max_bytes: int = 256 * 1024
    max_results: int = 10
    max_attempts: int = 3
    # One wall-clock budget for the complete legacy HTTP task, including retries.
    deadline_seconds: float = 30.0
    retry_backoff_seconds: float = 0.5
    retry_jitter_seconds: float = 0.25


class SearchProviderError(RuntimeError):
    """An error that must be surfaced with a stable CompanyIntel outcome."""

    def __init__(self, message: str, outcome: str) -> None:
        super().__init__(message)
        self.outcome = outcome


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            href = values.get("href", "")
            parsed = urllib.parse.urlparse(href)
            if parsed.query:
                redirected = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
                href = urllib.parse.unquote(redirected) or href
            self._current = {"url": href, "title": "", "snippet": ""}
            self._in_title = True
        elif tag in {"a", "div"} and "result__snippet" in classes and self._current is not None:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            if self._current and self._current["url"]:
                self.results.append(self._current)
                self._current = None
        if tag in {"a", "div"}:
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        if self._in_title:
            self._current["title"] += data
        elif self._in_snippet:
            self._current["snippet"] += data


def _safe_result_url(value: str) -> str | None:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    try:
        if ipaddress.ip_address(parsed.hostname).is_private:
            return None
    except ValueError:
        pass
    return parsed._replace(fragment="").geturl()


def _is_challenge_page(body: bytes) -> bool:
    sample = body[:512 * 1024].decode("utf-8", errors="replace").lower()
    return any(marker in sample for marker in _CHALLENGE_MARKERS)


def _safe_provider_detail(value: object) -> str:
    """Return a bounded diagnostic with credential-like values redacted."""
    detail = " ".join(str(value).split())
    for marker in ("authorization", "api_key", "apikey", "token", "secret", "password", "cookie"):
        detail = re.sub(
            rf"({re.escape(marker)})(?:\s*[:=]\s*)\S+",
            r"\1=<redacted>",
            detail,
            flags=re.IGNORECASE,
        )
    return detail[:300] or "provider failure"


def _provider_failure_outcome(error: object) -> str:
    """Classify provider errors without exposing transport details to callers."""
    text = str(error).lower()
    if any(marker in text for marker in _CHALLENGE_MARKERS):
        return "BLOCKED_BY_CAPTCHA"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "RATE_LIMITED"
    if "401" in text or "unauthor" in text or "api key" in text or "authentication" in text:
        return "AUTH_FAILED"
    if "403" in text and any(marker in text for marker in ("captcha", "challenge", "bot")):
        return "BLOCKED_BY_CAPTCHA"
    if any(marker in text for marker in (
        "timeout", "timed out", "connection reset", "connection refused",
        "temporarily", "unavailable", "502", "503", "504", "econnreset",
        "reset by peer",
    )):
        return "RETRYABLE_ERROR"
    return "UNAVAILABLE"


def _load_search_config() -> dict:
    """Read CompanyIntel search policy from config.yaml, never from .env."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:
        return {}
    section = config.get("companyintel", {}) if isinstance(config, dict) else {}
    search = section.get("search", {}) if isinstance(section, dict) else {}
    return search if isinstance(search, dict) else {}


def _coerce_config_number(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def configured_search_limits() -> SearchLimits:
    """Load bounded search budgets from the profile's YAML policy.

    The graph may still pass stricter per-task overrides, but no configuration
    value can expand a task beyond the hard safety ceilings here.
    """
    config = _load_search_config()
    return SearchLimits(
        timeout_seconds=_coerce_config_number(config.get("timeout_seconds"), 8.0, 1.0, 15.0),
        max_bytes=int(_coerce_config_number(config.get("max_bytes"), 256 * 1024, 4096, 512 * 1024)),
        max_results=int(_coerce_config_number(config.get("max_results"), 10, 1, 20)),
        max_attempts=int(_coerce_config_number(config.get("max_attempts"), 3, 1, 3)),
        deadline_seconds=_coerce_config_number(config.get("deadline_seconds"), 30.0, 1.0, 120.0),
        retry_backoff_seconds=_coerce_config_number(config.get("retry_backoff_seconds"), 0.5, 0.0, 8.0),
        retry_jitter_seconds=_coerce_config_number(config.get("retry_jitter_seconds"), 0.25, 0.0, 1.0),
    )


def _allow_legacy_html_fallback() -> bool:
    """The HTML fallback is opt-in and disabled for normal production runs."""
    search = _load_search_config()
    if "allow_legacy_duckduckgo_html" in search:
        return bool(search["allow_legacy_duckduckgo_html"])
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        web = config.get("web", {}) if isinstance(config, dict) else {}
        return bool(web.get("allow_legacy_duckduckgo_html", False)) if isinstance(web, dict) else False
    except Exception:
        return False


def _ensure_web_provider_registry() -> None:
    try:
        from tools.web_tools import _ensure_web_plugins_loaded

        _ensure_web_plugins_loaded()
    except Exception:
        # Provider resolution below fails closed when discovery is unavailable.
        return


def _active_provider() -> tuple[str, object | None]:
    """Resolve an allowlisted Hermes provider through the shared registry."""
    _ensure_web_provider_registry()
    try:
        from agent.web_search_registry import get_active_search_provider

        provider = get_active_search_provider()
    except Exception:
        return "", None
    if provider is None:
        return "", None
    name = str(getattr(provider, "name", "")).strip().lower()
    if name not in _ALLOWED_PROVIDER_NAMES:
        return "", None
    try:
        if not provider.supports_search():
            return "", None
    except Exception:
        return "", None
    return name, provider


def active_provider_name() -> str:
    """Return provider identity used for query deduplication and provenance."""
    name, _ = _active_provider()
    return name or "none"


_READINESS_CACHE: dict[tuple[str, int], tuple[float, dict[str, object]]] = {}
_READINESS_CACHE_TTL_SECONDS = 30.0
_READINESS_QUERY = "Hermes CompanyIntel readiness probe"


def _safe_readiness_detail(value: object) -> str:
    detail = " ".join(str(value or "").split())[:256]
    # Provider adapters must not return credentials, but keep the readiness
    # envelope safe even if a third-party adapter violates that contract.
    detail = re.sub(r"(?i)(api[_ -]?key|authorization|bearer|token|secret|password)\s*[:=]\s*\S+", r"\1=<redacted>", detail)
    return detail or "provider probe failed"


def _probe_provider(provider: object) -> dict[str, object]:
    """Perform one bounded provider request without exposing its credentials."""
    try:
        response = provider.search(_READINESS_QUERY, 1)
    except (TimeoutError, ConnectionError, OSError) as exc:
        return {"authenticated": False, "reachable": False, "reason": _safe_readiness_detail(exc)}
    except Exception as exc:  # noqa: BLE001 - plugin boundary
        return {"authenticated": False, "reachable": False, "reason": _safe_readiness_detail(exc)}
    if not isinstance(response, dict):
        return {"authenticated": False, "reachable": True, "reason": "provider probe returned an invalid response"}
    if bool(response.get("success")):
        return {"authenticated": True, "reachable": True, "reason": "ready"}
    detail = _safe_readiness_detail(response.get("error"))
    auth_failure = bool(re.search(r"(?i)\b(?:401|403|unauthori[sz]ed|forbidden)\b|api[_ -]?key|authentication", detail))
    transport_failure = bool(re.search(r"(?i)timeout|timed out|connection reset|connection refused|name resolution|network is unreachable|could not reach", detail))
    return {
        "authenticated": False if auth_failure else None,
        "reachable": not transport_failure,
        "reason": "provider authentication rejected" if auth_failure else detail,
    }


def provider_readiness(*, probe: bool = True) -> dict[str, object]:
    """Return configured/authenticated/reachable readiness without secrets.

    ``is_available()`` only proves that credentials/dependencies are present;
    ``ready`` is true only after a bounded live provider request succeeds.
    Probe results are cached briefly to avoid turning UI polling into a search
    storm. Tests and operators can request ``probe=False`` for metadata only.
    """
    name, provider = _active_provider()
    if provider is None:
        return {
            "provider": "none",
            "registered": False,
            "supports_search": False,
            "configured": False,
            "credentials_available": False,
            "authenticated": False,
            "reachable": False,
            "ready": False,
            "reason": "no allowlisted Hermes search provider is configured",
        }
    try:
        configured = bool(provider.is_available())
    except Exception:
        configured = False
    base = {
        "provider": name,
        "registered": True,
        "supports_search": True,
        "configured": configured,
        "credentials_available": configured,
        "authenticated": False,
        "reachable": False,
        "ready": False,
    }
    if not configured:
        base["reason"] = "provider credentials or dependencies unavailable"
        return base
    if not probe:
        base["reason"] = "configured; live provider probe not requested"
        return base
    cache_key = (name, id(provider))
    now = time.monotonic()
    cached = _READINESS_CACHE.get(cache_key)
    if cached and now - cached[0] < _READINESS_CACHE_TTL_SECONDS:
        return dict(cached[1])
    probe_result = _probe_provider(provider)
    base.update(probe_result)
    # A probe that reaches the endpoint but gets 401/403 is reachable but not
    # authenticated; only a successful response certifies readiness.
    base["ready"] = bool(base["configured"] and base["authenticated"] and base["reachable"])
    if base["ready"]:
        base["reason"] = "ready"
    elif not base.get("reason"):
        base["reason"] = "provider probe did not authenticate"
    _READINESS_CACHE[cache_key] = (now, dict(base))
    return base


def normalized_query(query: str) -> str:
    """Canonical query key component shared by the graph deduplication gate."""
    return " ".join(str(query).split()).casefold()[:1000]


def _retry_delay(limits: SearchLimits, attempt: int, retry_after: float | None = None) -> None:
    delay = min(8.0, max(0.0, float(limits.retry_backoff_seconds)) * (2 ** attempt))
    if retry_after is not None:
        delay = min(8.0, max(delay, retry_after))
    delay += min(1.0, max(0.0, float(limits.retry_jitter_seconds))) * random.random()
    time.sleep(delay)


def _interruptible_delay(limits: SearchLimits, attempt: int, retry_after: float | None = None) -> None:
    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            raise SearchProviderError("search interrupted", "UNAVAILABLE")
    except ImportError:
        pass
    _retry_delay(limits, attempt, retry_after)


def _fetch_search(query: str, limits: SearchLimits = SearchLimits()) -> tuple[str, bytes]:
    """Opt-in legacy HTML fetcher retained for compatibility and unit tests.

    ``deadline_seconds`` is a total task budget, not an additional budget per
    retry. This keeps a degraded fallback from holding a frontier lease
    indefinitely.
    """
    endpoint = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(endpoint, headers={
        "User-Agent": "Hermes-companyintel-public-search/1.0",
        "Accept": "text/html",
    })
    attempts = min(3, max(1, int(limits.max_attempts)))
    deadline_at = time.monotonic() + min(120.0, max(1.0, float(limits.deadline_seconds)))
    for attempt in range(attempts):
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise SearchProviderError("search task deadline exceeded", "RETRYABLE_ERROR")
        try:
            with urllib.request.urlopen(request, timeout=min(float(limits.timeout_seconds), remaining)) as response:
                status = getattr(response, "status", None)
                if status is None:
                    getcode = getattr(response, "getcode", None)
                    status = getcode() if callable(getcode) else 200
                status = int(status)
                body = response.read(limits.max_bytes + 1)
            if _is_challenge_page(body):
                raise SearchProviderError("DuckDuckGo returned an anti-bot challenge", "BLOCKED_BY_CAPTCHA")
            if status == 429:
                raise SearchProviderError("DuckDuckGo rate limited the request", "RATE_LIMITED")
            if status in {408, 425, 500, 502, 503, 504}:
                raise SearchProviderError(f"DuckDuckGo temporary HTTP failure ({status})", "RETRYABLE_ERROR")
            if status < 200 or status >= 300:
                raise SearchProviderError(f"DuckDuckGo HTTP failure ({status})", "UNAVAILABLE")
            if status != 200:
                raise SearchProviderError(f"DuckDuckGo unexpected HTTP status ({status})", "INVALID_RESPONSE")
            return "text/html", body[:limits.max_bytes]
        except SearchProviderError as exc:
            if exc.outcome not in {"RETRYABLE_ERROR", "RATE_LIMITED"} or attempt + 1 >= attempts:
                raise
            _retry_delay(limits, attempt)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise SearchProviderError("DuckDuckGo authentication failed", "AUTH_FAILED") from exc
            if exc.code == 403:
                body = b""
                try:
                    body = exc.read(limits.max_bytes + 1)
                except Exception:
                    pass
                outcome = "BLOCKED_BY_CAPTCHA" if _is_challenge_page(body) else "AUTH_FAILED"
                raise SearchProviderError(f"DuckDuckGo request rejected ({exc.code})", outcome) from exc
            retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt + 1 >= attempts:
                outcome = "RATE_LIMITED" if exc.code == 429 else "RETRYABLE_ERROR" if retryable else "UNAVAILABLE"
                raise SearchProviderError(f"DuckDuckGo HTTP failure ({exc.code})", outcome) from exc
            retry_after = getattr(exc.headers, "get", lambda _name: None)("Retry-After")
            try:
                retry_after_value = float(retry_after) if retry_after is not None else None
            except (TypeError, ValueError):
                retry_after_value = None
            _retry_delay(limits, attempt, retry_after_value)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt + 1 >= attempts:
                raise SearchProviderError("DuckDuckGo transport temporarily unavailable", "RETRYABLE_ERROR") from exc
            _retry_delay(limits, attempt)
    raise SearchProviderError("DuckDuckGo search exhausted retry budget", "RETRYABLE_ERROR")


def _normalize_provider_results(raw_results: list, limits: SearchLimits) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = _safe_result_url(str(item.get("url") or item.get("href") or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        results.append({
            "url": url,
            "title": " ".join(str(item.get("title", "")).split())[:500],
            "snippet": " ".join(str(item.get("description") or item.get("snippet") or item.get("body") or "").split())[:1000],
        })
        if len(results) >= limits.max_results:
            break
    return results


def _provider_search(query: str, limits: SearchLimits, mode: str) -> dict:
    """Call the configured Hermes provider with bounded, typed degradation."""
    del mode  # mode is part of the graph dedup key; providers receive the query.
    provider_name, provider = _active_provider()
    if provider is None:
        if _allow_legacy_html_fallback():
            content_type, body = _fetch_search(query, limits)
            return {
                "provider": "duckduckgo_html",
                "content_type": content_type,
                "body": body,
                "attempt_count": 1,
            }
        return {
            "provider": "none",
            "outcome": "UNAVAILABLE",
            "attempt_count": 0,
            "detail": "no allowlisted Hermes search provider is configured",
            "provider_ready": False,
        }

    attempts = min(3, max(1, int(limits.max_attempts)))
    deadline_at = time.monotonic() + min(120.0, max(1.0, float(limits.deadline_seconds)))
    for attempt in range(attempts):
        if deadline_at <= time.monotonic():
            return {
                "provider": provider_name,
                "outcome": "RETRYABLE_ERROR",
                "attempt_count": attempt,
                "detail": "search task deadline exceeded",
            }
        try:
            response = provider.search(query, min(20, max(1, int(limits.max_results))))
            if not isinstance(response, dict) or not isinstance(response.get("success"), bool):
                return {
                    "provider": provider_name,
                    "outcome": "INVALID_RESPONSE",
                    "attempt_count": attempt + 1,
                    "detail": "provider response envelope is invalid",
                }
            if not response.get("success"):
                detail = _safe_provider_detail(response.get("error", "provider returned failure"))
                outcome = _provider_failure_outcome(detail)
                if outcome in {"RETRYABLE_ERROR", "RATE_LIMITED"} and attempt + 1 < attempts:
                    _interruptible_delay(limits, attempt)
                    continue
                return {
                    "provider": provider_name,
                    "outcome": outcome,
                    "attempt_count": attempt + 1,
                    "detail": detail,
                }
            data = response.get("data")
            raw_results = data.get("web") if isinstance(data, dict) else None
            if not isinstance(raw_results, list):
                return {
                    "provider": provider_name,
                    "outcome": "INVALID_RESPONSE",
                    "attempt_count": attempt + 1,
                    "detail": "provider response has no web result list",
                }
            if any(not isinstance(item, dict) for item in raw_results):
                return {
                    "provider": provider_name,
                    "outcome": "INVALID_RESPONSE",
                    "attempt_count": attempt + 1,
                    "detail": "provider result list contains a non-object item",
                }
            return {
                "provider": provider_name,
                "raw_results": raw_results,
                "attempt_count": attempt + 1,
            }
        except SearchProviderError:
            raise
        except Exception as exc:
            outcome = _provider_failure_outcome(exc)
            detail = _safe_provider_detail(exc)
            if outcome in {"RETRYABLE_ERROR", "RATE_LIMITED"} and attempt + 1 < attempts:
                _interruptible_delay(limits, attempt)
                continue
            return {
                "provider": provider_name,
                "outcome": outcome,
                "attempt_count": attempt + 1,
                "detail": detail,
            }
    return {
        "provider": provider_name,
        "outcome": "RETRYABLE_ERROR",
        "attempt_count": attempts,
        "detail": "provider retry budget exhausted",
    }


def build_query(node_type: str, value: str, mode: str = "exact") -> str:
    normalized = " ".join(str(value).split())[:300]
    if mode == "maps":
        return f'"{normalized}" map'
    if mode == "marketplace":
        return f'("{normalized}") (site:prom.ua OR site:rozetka.com.ua OR site:olx.ua)'
    if mode == "document":
        if node_type == "domain":
            return f"site:{normalized} (filetype:pdf OR filetype:xls OR filetype:doc OR filetype:xml)"
        return f'"{normalized}" (filetype:pdf OR filetype:xls OR filetype:doc OR filetype:xml)'
    return f'"{normalized}"'


def _html_result(query: str, content_type: str, body: bytes, limits: SearchLimits) -> dict:
    if not content_type.startswith("text/html"):
        return {
            "outcome": "INVALID_RESPONSE",
            "query": query,
            "results": [],
            "provider": "duckduckgo_html",
            "attempt_count": limits.max_attempts,
            "error": _GENERIC_ERRORS["INVALID_RESPONSE"],
            "technical_error": "search response is not HTML",
        }
    if _is_challenge_page(body):
        return {
            "outcome": "BLOCKED_BY_CAPTCHA",
            "query": query,
            "results": [],
            "provider": "duckduckgo_html",
            "attempt_count": 1,
            "error": _GENERIC_ERRORS["BLOCKED_BY_CAPTCHA"],
            "technical_error": "DuckDuckGo returned an anti-bot challenge",
        }
    parser = _DuckDuckGoParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    results = []
    seen = set()
    for item in parser.results:
        url = _safe_result_url(item.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        results.append({
            "url": url,
            "title": " ".join(item.get("title", "").split())[:500],
            "snippet": " ".join(item.get("snippet", "").split())[:1000],
        })
        if len(results) >= limits.max_results:
            break
    return {
        "outcome": "COMPLETED_WITH_RESULTS" if results else "COMPLETED_ZERO_RESULTS",
        "query": query,
        "results": results,
        "provider": "duckduckgo_html",
        "attempt_count": 1,
        "error": None,
        "technical_error": None,
    }


def execute_public_search(
    node_type: str,
    value: str,
    *,
    fetcher=None,
    limits: SearchLimits | None = None,
    mode: str = "exact",
) -> dict:
    limits = configured_search_limits() if limits is None else limits
    query = build_query(node_type, value, mode=mode)
    if fetcher is None:
        try:
            provider_result = _provider_search(query, limits, mode)
        except SearchProviderError as exc:
            outcome = exc.outcome
            detail = _safe_provider_detail(exc)
            return {
                "outcome": outcome,
                "query": query,
                "results": [],
                "provider": active_provider_name(),
                "attempt_count": limits.max_attempts,
                "error": _GENERIC_ERRORS.get(outcome, _GENERIC_ERRORS["UNAVAILABLE"]),
                "technical_error": detail,
            }
        except Exception as exc:
            detail = _safe_provider_detail(exc)
            return {
                "outcome": _provider_failure_outcome(detail),
                "query": query,
                "results": [],
                "provider": active_provider_name(),
                "attempt_count": limits.max_attempts,
                "error": _GENERIC_ERRORS.get(_provider_failure_outcome(detail), _GENERIC_ERRORS["UNAVAILABLE"]),
                "technical_error": detail,
            }
        if provider_result.get("outcome"):
            outcome = str(provider_result["outcome"])
            detail = _safe_provider_detail(provider_result.get("detail", outcome))
            return {
                "outcome": outcome,
                "query": query,
                "results": [],
                "provider": provider_result.get("provider", "none"),
                "attempt_count": int(provider_result.get("attempt_count", 0)),
                "error": _GENERIC_ERRORS.get(outcome, _GENERIC_ERRORS["UNAVAILABLE"]),
                "technical_error": detail,
            }
        if "raw_results" in provider_result:
            results = _normalize_provider_results(provider_result["raw_results"], limits)
            return {
                "outcome": "COMPLETED_WITH_RESULTS" if results else "COMPLETED_ZERO_RESULTS",
                "query": query,
                "results": results,
                "provider": provider_result.get("provider", "none"),
                "attempt_count": int(provider_result.get("attempt_count", 1)),
                "error": None,
                "technical_error": None,
            }
        if "body" in provider_result:
            return _html_result(query, provider_result["content_type"], provider_result["body"], limits)
        return {
            "outcome": "UNAVAILABLE",
            "query": query,
            "results": [],
            "provider": "none",
            "attempt_count": 0,
            "error": _GENERIC_ERRORS["UNAVAILABLE"],
            "technical_error": "search provider adapter returned no result",
        }

    try:
        content_type, body = fetcher(query, limits)
        result = _html_result(query, content_type, body, limits)
        result["provider"] = "duckduckgo_html"
        return result
    except SearchProviderError as exc:
        outcome = exc.outcome
        return {
            "outcome": outcome,
            "query": query,
            "results": [],
            "provider": "duckduckgo_html",
            "attempt_count": limits.max_attempts,
            "error": _GENERIC_ERRORS.get(outcome, _GENERIC_ERRORS["UNAVAILABLE"]),
            "technical_error": _safe_provider_detail(exc),
        }
    except Exception as exc:
        outcome = _provider_failure_outcome(exc)
        return {
            "outcome": outcome,
            "query": query,
            "results": [],
            "provider": "duckduckgo_html",
            "attempt_count": limits.max_attempts,
            "error": _GENERIC_ERRORS.get(outcome, _GENERIC_ERRORS["UNAVAILABLE"]),
            "technical_error": _safe_provider_detail(exc),
        }
