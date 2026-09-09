"""Shared, lifecycle-safe provider clients for Hermes tools.

CompanyIntel reasoning is routed through the active Hermes OpenAI provider
(currently Codex OAuth with ``gpt-5.6-luna``), rather than a separate
OpenRouter credential. The compatibility module name is retained because
older CompanyIntel callers import it directly.
"""

import inspect
import os
import re
import threading
from typing import Any

_client: Any = None
_client_mode: str | None = None
_client_route: tuple[str, str] | None = None
_cache_lock = threading.RLock()
_DEFAULT_REASONING_PROVIDER = "openai-codex"
_DEFAULT_REASONING_MODEL = "gpt-5.6-luna"
# Compatibility alias for integrations that imported the old constant.
_COMPANYINTEL_REASONING_MODEL = _DEFAULT_REASONING_MODEL
_ALLOWED_REASONING_PROVIDERS = frozenset({"openai-codex", "openai-api"})


def _normalize_reasoning_provider(value: Any) -> str:
    provider = str(value or "").strip().lower()
    return {"codex": "openai-codex", "openai": "openai-api"}.get(provider, provider)


def get_reasoning_route() -> tuple[str, str]:
    """Return the configured Hermes route used by CompanyIntel reasoning.

    The route is intentionally limited to OpenAI providers. An old OpenRouter
    setting therefore fails closed instead of silently changing billing, data
    residency, or model semantics for a CompanyIntel run.
    """
    provider = _DEFAULT_REASONING_PROVIDER
    model = _DEFAULT_REASONING_MODEL
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        model_cfg = config.get("model") if isinstance(config, dict) else {}
        if isinstance(model_cfg, dict):
            provider = _normalize_reasoning_provider(model_cfg.get("provider")) or provider
            configured_model = str(model_cfg.get("default") or "").strip()
            if configured_model:
                model = configured_model[:256]
    except Exception:
        # Keep a safe default for fresh profiles and isolated unit tests.
        pass

    provider = _normalize_reasoning_provider(provider) or _DEFAULT_REASONING_PROVIDER
    if provider not in _ALLOWED_REASONING_PROVIDERS:
        raise ValueError(
            "CompanyIntel reasoning requires an OpenAI Hermes provider "
            "(openai-codex or openai-api); refusing an unapproved provider"
        )
    if not model or any(ord(char) < 32 for char in model):
        raise ValueError("CompanyIntel reasoning model is invalid")
    return provider, model


def get_active_reasoning_route() -> tuple[str, str]:
    """Return the effective route for the cached CompanyIntel client."""
    with _cache_lock:
        if _client_route is not None:
            return _client_route
    return get_reasoning_route()


def _safe_text(value: Any, limit: int) -> str:
    """Bound and redact provider-shaped text before durable persistence."""
    text = " ".join(str(value or "").split())
    text = re.sub(
        r"(?i)(authorization|api[_ -]?key|token|secret|password|cookie|credential)(?:\s*[=:]\s*)(?:bearer\s+)?[^,;\s}]+",
        r"\1=<redacted>",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+[^,;\s}]+", "Bearer <redacted>", text)
    text = re.sub(r"(?i)([?&](?:key|token|secret|password|auth)[^=]*=)[^&\s]+", r"\1<redacted>", text)
    return text[:limit]


class OpenRouterClientError(RuntimeError):
    """Safe, structured error envelope for provider failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "REASONING_PROVIDER_ERROR",
        retryable: bool = False,
        attempt: int | None = None,
        operation: str | None = None,
        cause_class: str | None = None,
        retry_after: float | None = None,
        failure_stage: str | None = None,
        error_fingerprint: str | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        super().__init__(_safe_text(message, 512))
        self.code = _safe_text(code, 96)
        self.retryable = bool(retryable)
        self.attempt = attempt
        self.operation = _safe_text(operation, 96) if operation else None
        self.cause_class = _safe_text(cause_class, 96) if cause_class else None
        if retry_after is not None:
            try:
                self.retry_after = max(0.0, min(float(retry_after), 30.0))
            except (TypeError, ValueError):
                self.retry_after = None
        else:
            self.retry_after = None
        self.failure_stage = _safe_text(failure_stage, 64) if failure_stage else None
        self.error_fingerprint = _safe_text(error_fingerprint, 128) if error_fingerprint else None
        self.provider_request_id = _safe_text(provider_request_id, 128) if provider_request_id else None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "message": _safe_text(self, 512),
            "retryable": self.retryable,
        }
        if self.attempt is not None:
            value["attempt"] = self.attempt
        if self.operation:
            value["operation"] = self.operation
        if self.cause_class:
            value["cause_class"] = self.cause_class
        if self.retry_after is not None:
            value["retry_after"] = self.retry_after
        if self.failure_stage:
            value["failure_stage"] = self.failure_stage[:64]
        if self.error_fingerprint:
            value["error_fingerprint"] = self.error_fingerprint[:128]
        if self.provider_request_id:
            value["provider_request_id"] = self.provider_request_id[:128]
        return value


def _close_client(client: Any) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close_result = close()
        # Never run a second event loop during eviction. Sync CompanyIntel
        # reasoning only uses get_sync_client().
        if inspect.isawaitable(close_result):
            close_result.close()
    except Exception:
        pass


def reset_client() -> None:
    """Evict and close the cached client after a transport failure."""
    global _client, _client_mode, _client_route
    with _cache_lock:
        old = _client
        _client = None
        _client_mode = None
        _client_route = None
        _close_client(old)


def get_sync_client():
    """Return the synchronous client for the configured Hermes route."""
    global _client, _client_mode, _client_route
    provider, requested_model = get_reasoning_route()
    with _cache_lock:
        if _client is None or _client_mode != "sync" or _client_route != (provider, requested_model):
            if _client is not None:
                _close_client(_client)
            from agent.auxiliary_client import resolve_provider_client

            # Pin resolver metadata to the configured Hermes model. This
            # prevents auxiliary fallback policy from selecting a different
            # provider or model merely to construct the transport.
            client, _model = resolve_provider_client(
                provider,
                model=requested_model,
                async_mode=False,
                task="companyintel_reasoning",
            )
            if client is None:
                raise ValueError("configured Hermes reasoning credentials are unavailable")
            _client = client
            _client_mode = "sync"
            _client_route = (provider, str(_model or requested_model))
        return _client


def get_async_client():
    """Return an async client for the configured Hermes reasoning route."""
    global _client, _client_mode, _client_route
    provider, requested_model = get_reasoning_route()
    with _cache_lock:
        if _client is None or _client_mode != "async" or _client_route != (provider, requested_model):
            if _client is not None:
                _close_client(_client)
            from agent.auxiliary_client import resolve_provider_client

            client, _model = resolve_provider_client(
                provider,
                model=requested_model,
                async_mode=True,
                task="companyintel_reasoning",
            )
            if client is None:
                raise ValueError("configured Hermes reasoning credentials are unavailable")
            _client = client
            _client_mode = "async"
            _client_route = (provider, str(_model or requested_model))
        return _client


def check_api_key() -> bool:
    """Check whether the OpenRouter API key is present."""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            return bool(get_secret("OPENROUTER_API_KEY"))
        except UnscopedSecretError:
            pass
    except Exception:
        pass
    return bool(os.getenv("OPENROUTER_API_KEY"))
