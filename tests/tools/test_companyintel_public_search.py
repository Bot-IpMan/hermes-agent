import json
import urllib.error

import pytest
from unittest.mock import patch


def test_document_query_is_domain_level():
    from tools.companyintel_public_search import build_query

    assert build_query("domain", "bb-battery.com.ua", mode="document").startswith(
        "site:bb-battery.com.ua "
    )


def test_url_nodes_create_only_typed_document_extraction_pivot():
    from tools.companyintel_pivots import expand_pivots

    pivots = expand_pivots("url")
    assert len(pivots) == 1
    assert pivots[0].pivot_type == "document_extraction"
    assert pivots[0].worker == "document_extraction"
    assert pivots[0].implemented is True


def test_challenge_page_is_never_zero_results():
    from tools.companyintel_public_search import execute_public_search

    result = execute_public_search(
        "domain",
        "example.com",
        fetcher=lambda _query, _limits: (
            "text/html",
            b"<html><script src='anomaly.js'></script>challenge-form</html>",
        ),
    )

    assert result["outcome"] == "BLOCKED_BY_CAPTCHA"
    assert result["results"] == []
    assert result["outcome"] != "COMPLETED_ZERO_RESULTS"


def test_transport_reset_retries_three_times(monkeypatch):
    from tools import companyintel_public_search as module

    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b"<html></html>"

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise urllib.error.URLError("connection reset by peer")
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda _delay: None)
    content_type, body = module._fetch_search(
        "example",
        module.SearchLimits(max_attempts=3, retry_jitter_seconds=0),
    )

    assert content_type == "text/html"
    assert body == b"<html></html>"
    assert len(calls) == 3


def test_provider_results_are_normalized_and_private_urls_filtered(monkeypatch):
    from tools import companyintel_public_search as module

    class Provider:
        name = "brave-free"

        def supports_search(self):
            return True

        def search(self, _query, limit):
            assert limit == 10
            return {
                "success": True,
                "data": {
                    "web": [
                        {"url": "https://example.com/a", "title": " A ", "description": " desc "},
                        {"url": "https://example.com/a", "title": "duplicate"},
                        {"url": "http://127.0.0.1/private", "title": "private"},
                    ]
                },
            }

    monkeypatch.setattr(module, "_active_provider", lambda: ("brave-free", Provider()))
    monkeypatch.setattr(module, "_allow_legacy_html_fallback", lambda: False)
    result = module.execute_public_search("domain", "example.com")

    assert result["outcome"] == "COMPLETED_WITH_RESULTS"
    assert result["provider"] == "brave-free"
    assert result["attempt_count"] == 1
    assert result["results"] == [
        {"url": "https://example.com/a", "title": "A", "snippet": "desc"}
    ]


def test_provider_transport_error_is_bounded_and_typed(monkeypatch):
    from tools import companyintel_public_search as module

    calls = []

    class Provider:
        name = "tavily"

        def supports_search(self):
            return True

        def search(self, _query, limit):
            calls.append(limit)
            raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(module, "_active_provider", lambda: ("tavily", Provider()))
    monkeypatch.setattr(module, "_allow_legacy_html_fallback", lambda: False)
    monkeypatch.setattr(module.time, "sleep", lambda _delay: None)
    result = module.execute_public_search(
        "domain",
        "example.com",
        limits=module.SearchLimits(max_attempts=3, retry_jitter_seconds=0),
    )

    assert result["outcome"] == "RETRYABLE_ERROR"
    assert result["attempt_count"] == 3
    assert len(calls) == 3
    assert "connection reset" not in result["error"].lower()
    assert "connection reset" in result["technical_error"].lower()


def test_provider_auth_failure_is_not_retried(monkeypatch):
    from tools import companyintel_public_search as module

    calls = []

    class Provider:
        name = "tavily"

        def supports_search(self):
            return True

        def search(self, _query, _limit):
            calls.append(1)
            return {"success": False, "error": "401 Unauthorized"}

    monkeypatch.setattr(module, "_active_provider", lambda: ("tavily", Provider()))
    monkeypatch.setattr(module, "_allow_legacy_html_fallback", lambda: False)
    result = module.execute_public_search("domain", "example.com")

    assert result["outcome"] == "AUTH_FAILED"
    assert result["attempt_count"] == 1
    assert len(calls) == 1


def test_provider_readiness_distinguishes_tavily_401(monkeypatch):
    from tools import companyintel_public_search as module

    class Provider:
        name = "tavily"
        def is_available(self):
            return True
        def supports_search(self):
            return True
        def search(self, _query, _limit):
            return {"success": False, "error": "401 Unauthorized"}

    provider = Provider()
    monkeypatch.setattr(module, "_active_provider", lambda: ("tavily", provider))
    module._READINESS_CACHE.clear()
    readiness = module.provider_readiness()
    assert readiness["configured"] is True
    assert readiness["credentials_available"] is True
    assert readiness["authenticated"] is False
    assert readiness["reachable"] is True
    assert readiness["ready"] is False
    assert "401" not in readiness["reason"]


def test_provider_readiness_requires_live_probe(monkeypatch):
    from tools import companyintel_public_search as module

    class Provider:
        name = "tavily"
        def is_available(self):
            return True
        def supports_search(self):
            return True
        def search(self, _query, _limit):
            return {"success": True, "data": {"web": []}}

    provider = Provider()
    monkeypatch.setattr(module, "_active_provider", lambda: ("tavily", provider))
    module._READINESS_CACHE.clear()
    readiness = module.provider_readiness(probe=False)
    assert readiness["configured"] is True
    assert readiness["authenticated"] is False
    assert readiness["reachable"] is False
    assert readiness["ready"] is False


def test_no_provider_fails_closed_without_html_request(monkeypatch):
    from tools import companyintel_public_search as module

    monkeypatch.setattr(module, "_active_provider", lambda: ("", None))
    monkeypatch.setattr(module, "_allow_legacy_html_fallback", lambda: False)
    with patch.object(module.urllib.request, "urlopen", side_effect=AssertionError("legacy HTML called")):
        result = module.execute_public_search("domain", "example.com")

    assert result["outcome"] == "UNAVAILABLE"
    assert result["provider"] == "none"
    assert result["attempt_count"] == 0


def test_provider_typed_error_is_returned_as_stable_outcome(monkeypatch):
    from tools import companyintel_public_search as module

    class Provider:
        name = "searxng"

        def supports_search(self):
            return True

        def search(self, _query, _limit):
            raise module.SearchProviderError("provider challenge", "BLOCKED_BY_CAPTCHA")

    monkeypatch.setattr(module, "_active_provider", lambda: ("searxng", Provider()))
    monkeypatch.setattr(module, "_allow_legacy_html_fallback", lambda: False)
    result = module.execute_public_search("domain", "example.com")

    assert result["outcome"] == "BLOCKED_BY_CAPTCHA"
    assert result["provider"] == "searxng"
    assert "challenge" not in result["error"].lower()


def test_http_202_challenge_is_blocked_without_retry(monkeypatch):
    from tools import companyintel_public_search as module

    calls = []

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b"<html><script src=\"anomaly.js\"></script></html>"

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: (calls.append(1) or Response()))
    with pytest.raises(module.SearchProviderError) as error:
        module._fetch_search(
            "example",
            module.SearchLimits(max_attempts=3, retry_jitter_seconds=0),
        )

    assert error.value.outcome == "BLOCKED_BY_CAPTCHA"
    assert len(calls) == 1


def test_http_202_without_valid_search_body_is_invalid(monkeypatch):
    from tools import companyintel_public_search as module

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b"<html>accepted but not a search response</html>"

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(module.SearchProviderError) as error:
        module._fetch_search("example")

    assert error.value.outcome == "INVALID_RESPONSE"


def test_http_429_retry_after_is_bounded(monkeypatch):
    from email.message import Message
    from tools import companyintel_public_search as module

    calls = []
    delays = []
    headers = Message()
    headers["Retry-After"] = "9999"

    def fake_urlopen(*_args, **_kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(
            "https://html.duckduckgo.com/html/", 429, "rate", headers, None
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda delay: delays.append(delay))
    with pytest.raises(module.SearchProviderError) as error:
        module._fetch_search(
            "example",
            module.SearchLimits(max_attempts=3, retry_jitter_seconds=0),
        )

    assert error.value.outcome == "RATE_LIMITED"
    assert len(calls) == 3
    assert delays and all(0 <= delay <= 8 for delay in delays)


def test_configured_limits_are_loaded_and_hard_capped(monkeypatch):
    from tools import companyintel_public_search as module

    monkeypatch.setattr(module, "_load_search_config", lambda: {
        "timeout_seconds": 999,
        "max_bytes": 99999999,
        "max_results": 999,
        "max_attempts": 999,
        "deadline_seconds": 999,
        "retry_backoff_seconds": 999,
        "retry_jitter_seconds": 999,
    })
    limits = module.configured_search_limits()

    assert limits.timeout_seconds == 15
    assert limits.max_bytes == 512 * 1024
    assert limits.max_results == 20
    assert limits.max_attempts == 3
    assert limits.deadline_seconds == 120
    assert limits.retry_backoff_seconds == 8
    assert limits.retry_jitter_seconds == 1


def test_unofficial_ddgs_provider_is_not_allowed(monkeypatch):
    from tools import companyintel_public_search as module

    assert "ddgs" not in module._ALLOWED_PROVIDER_NAMES
