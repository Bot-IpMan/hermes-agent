import json

import pytest


def _response(payload):
    class Message:
        content = json.dumps(payload)
    class Choice:
        message = Message()
    class Response:
        choices = [Choice()]
    return Response()


def test_reasoning_route_uses_configured_openai_codex_luna(monkeypatch):
    from tools import openrouter_client as module

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"provider": "openai-codex", "default": "gpt-5.6-luna"}},
    )
    module.reset_client()
    assert module.get_reasoning_route() == ("openai-codex", "gpt-5.6-luna")


def test_reasoning_route_rejects_non_openai_provider(monkeypatch):
    from tools import openrouter_client as module

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"provider": "openrouter", "default": "some-model"}},
    )
    with pytest.raises(ValueError, match="requires an OpenAI Hermes provider"):
        module.get_reasoning_route()


def test_transport_retry_evicts_client_and_uses_new_instance(monkeypatch):
    from tools import companyintel_openrouter_reasoning as module

    class FakeCompletions:
        def __init__(self, owner):
            self.owner = owner

        def create(self, **kwargs):
            self.owner.calls += 1
            if self.owner.fail:  # one poisoned transport per instance
                self.owner.fail = False
                raise ConnectionError("connection reset by peer")
            return _response({"ok": True})

    class FakeClient:
        def __init__(self, fail=True):
            self.fail = fail
            self.calls = 0
            self.chat = type("Chat", (), {"completions": FakeCompletions(self)})()

    clients = [FakeClient(fail=True), FakeClient(fail=False)]
    resets = []

    def get_client():
        return clients.pop(0)

    monkeypatch.setattr(module, "get_sync_client", get_client)
    monkeypatch.setattr(module, "reset_client", lambda: resets.append(True))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.random, "uniform", lambda _a, _b: 0.0)

    assert module.request_json(operation="probe", system="return JSON", payload={}) == {"ok": True}
    assert len(resets) == 1


def test_auth_failure_is_not_retried(monkeypatch):
    from tools import companyintel_openrouter_reasoning as module

    class AuthError(Exception):
        status_code = 401

    class Completions:
        def create(self, **kwargs):
            raise AuthError("unauthorized")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    calls = []
    monkeypatch.setattr(module, "get_sync_client", lambda: client)
    monkeypatch.setattr(module, "reset_client", lambda: calls.append("reset"))
    monkeypatch.setattr(module, "refresh_active_reasoning_credentials", lambda: False)
    with pytest.raises(module.OpenRouterClientError) as caught:
        module.request_json(operation="probe", system="return JSON", payload={}, max_attempts=3)
    assert caught.value.code == "REASONING_AUTH_FAILED"
    assert caught.value.retryable is False
    assert calls == []


def test_auth_failure_refreshes_codex_once_before_retry(monkeypatch):
    from tools import companyintel_openrouter_reasoning as module

    class AuthError(Exception):
        status_code = 401

    class DeniedCompletions:
        def create(self, **kwargs):
            raise AuthError("unauthorized")

    denied = type("Client", (), {"chat": type("Chat", (), {"completions": DeniedCompletions()})()})()
    clients = iter([denied, type("Client", (), {"chat": type("Chat", (), {"completions": type("Completions", (), {"create": lambda self, **kwargs: _response({"ok": True})})()})()})()])
    refreshes = []
    monkeypatch.setattr(module, "get_sync_client", lambda: next(clients))
    monkeypatch.setattr(module, "refresh_active_reasoning_credentials", lambda: refreshes.append(True) or True)

    assert module.request_json(operation="synthesize_report", system="return JSON", payload={}, max_attempts=2) == {"ok": True}
    assert refreshes == [True]


def test_hypothesis_schema_repair_is_bounded(monkeypatch):
    from tools.companyintel_reasoning_controller import CompanyIntelReasoningController

    calls = []
    def request(operation, payload):
        calls.append(operation)
        if operation == "generate_hypotheses":
            return {"hypotheses": [{"claim": "missing required fields"}]}
        return {"hypotheses": [{"kind": "ownership", "subject_node_ids": ["node-1"], "supporting_evidence_ids": ["e-1"], "claim": "valid", "confidence": 0.8}]}

    controller = CompanyIntelReasoningController(llm_request=request)
    result = controller.generate_hypotheses_with_llm(payload={"run_id": "run-1"}, evidence_ids={"e-1"})
    assert result[0]["kind"] == "ownership"
    assert calls == ["generate_hypotheses", "repair_generate_hypotheses"]


def test_missing_choices_is_structured_and_bounded_retry(monkeypatch):
    from tools import companyintel_openrouter_reasoning as module

    class Completions:
        def create(self, **kwargs):
            return type("Response", (), {"choices": None, "_request_id": "req-safe"})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    resets = []
    monkeypatch.setattr(module, "get_sync_client", lambda: client)
    monkeypatch.setattr(module, "reset_client", lambda: resets.append(True))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.random, "uniform", lambda _a, _b: 0.0)

    with pytest.raises(module.OpenRouterClientError) as caught:
        module.request_json(operation="synthesize_report", system="return JSON", payload={}, max_attempts=2)

    assert caught.value.code == "REASONING_MISSING_CHOICES"
    assert caught.value.failure_stage == "response_shape"
    assert caught.value.retryable is True
    assert caught.value.provider_request_id == "req-safe"
    assert len(resets) == 1


def test_generic_type_error_is_nonretryable_and_redacted(monkeypatch):
    from tools import companyintel_openrouter_reasoning as module

    class Completions:
        def create(self, **kwargs):
            raise TypeError("Authorization: Bearer super-secret-token")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    resets = []
    monkeypatch.setattr(module, "get_sync_client", lambda: client)
    monkeypatch.setattr(module, "reset_client", lambda: resets.append(True))

    with pytest.raises(module.OpenRouterClientError) as caught:
        module.request_json(operation="synthesize_report", system="return JSON", payload={}, max_attempts=3)

    assert caught.value.code == "REASONING_CLIENT_CONTRACT_ERROR"
    assert caught.value.failure_stage == "provider_request"
    assert caught.value.retryable is False
    assert "super-secret-token" not in str(caught.value)
    assert caught.value.error_fingerprint
    assert resets == []


def test_report_prompts_require_ukrainian_human_text(monkeypatch):
    from tools import companyintel_openrouter_reasoning as module

    calls = []
    monkeypatch.setattr(
        module,
        "request_json",
        lambda **kwargs: calls.append(kwargs) or {"report_text": "звіт", "claims": []},
    )
    module.synthesize_report({"run_id": "run-language"})
    module.review_report({"run_id": "run-language"})

    assert len(calls) == 2
    for call in calls:
        assert "українською мовою" in call["system"]
        assert "evidence_ids" in call["system"]
        assert "URLs" in call["system"]


def test_controller_uses_operation_specific_token_limits(monkeypatch):
    from tools.companyintel_reasoning_controller import CompanyIntelReasoningController

    calls = []
    def fake_request_json(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr("tools.companyintel_openrouter_reasoning.request_json", fake_request_json)
    controller = CompanyIntelReasoningController()
    controller._request("synthesize_report", {})
    controller._request("review_report", {})

    assert [call["max_tokens"] for call in calls] == [5000, 3000]


def test_oversized_request_is_rejected_before_provider_call(monkeypatch):
    from tools import companyintel_openrouter_reasoning as module

    calls = []
    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _response({"ok": True})

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setattr(module, "get_sync_client", lambda: client)

    with pytest.raises(module.OpenRouterClientError) as caught:
        module.request_json(operation="synthesize_report", system="return JSON", payload={"blob": "x" * 300_000}, max_attempts=1)

    assert caught.value.code == "REASONING_INPUT_TOO_LARGE"
    assert caught.value.failure_stage == "request_serialization"
    assert caught.value.retryable is False
    assert calls == []


@pytest.mark.parametrize(
    ("response", "code", "stage"),
    [
        (None, "REASONING_EMPTY_RESPONSE", "response_shape"),
        (type("Response", (), {"choices": None})(), "REASONING_MISSING_CHOICES", "response_shape"),
        (type("Response", (), {"choices": []})(), "REASONING_EMPTY_CHOICES", "response_shape"),
        (type("Response", (), {"choices": [None]})(), "REASONING_EMPTY_CHOICE", "response_shape"),
        (type("Response", (), {"choices": [type("Choice", (), {})()]})(), "REASONING_MISSING_MESSAGE", "response_shape"),
        (type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": ""})()})()]})(), "REASONING_EMPTY_CONTENT", "response_content"),
        (type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "not json"})()})()]})(), "REASONING_INVALID_RESPONSE", "json_decode"),
    ],
)
def test_response_shape_failures_are_structured(monkeypatch, response, code, stage):
    from tools import companyintel_openrouter_reasoning as module

    class Completions:
        def create(self, **kwargs):
            return response

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setattr(module, "get_sync_client", lambda: client)
    monkeypatch.setattr(module, "reset_client", lambda: None)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(module.OpenRouterClientError) as caught:
        module.request_json(operation="probe", system="return JSON", payload={}, max_attempts=1)
    assert caught.value.code == code
    assert caught.value.failure_stage == stage
    assert caught.value.attempt == 1


def test_output_token_limit_is_bounded(monkeypatch):
    from tools import companyintel_openrouter_reasoning as module

    captured = []
    class Completions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return _response({"ok": True})

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setattr(module, "get_sync_client", lambda: client)
    assert module.request_json(operation="probe", system="return JSON", payload={}, max_attempts=1, max_tokens=999999) == {"ok": True}
    assert captured[0]["max_tokens"] == 5000
    response_format = captured[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"].startswith("companyintel_")


def test_error_envelope_fields_are_bounded_and_redacted():
    from tools.openrouter_client import OpenRouterClientError

    error = OpenRouterClientError(
        "Authorization: Bearer super-secret " + "x" * 1000,
        code="C" * 200,
        operation="O" * 200,
        cause_class="K" * 200,
        failure_stage="S" * 200,
        provider_request_id="P" * 200,
    )
    data = error.as_dict()
    assert "super-secret" not in data["message"]
    assert len(data["message"]) <= 512
    assert len(data["code"]) <= 96
    assert len(data["operation"]) <= 96
    assert len(data["cause_class"]) <= 96
    assert len(data["failure_stage"]) <= 64
    assert len(data["provider_request_id"]) <= 128


def test_attempt_audit_event_contains_only_bounded_safe_fields(monkeypatch, caplog):
    from tools import companyintel_openrouter_reasoning as module

    class Completions:
        def create(self, **kwargs):
            raise TypeError("Authorization: Bearer super-secret-token")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setattr(module, "get_sync_client", lambda: client)
    with caplog.at_level("WARNING", logger=module._LOGGER.name):
        with pytest.raises(module.OpenRouterClientError):
            module.request_json(
                operation="probe",
                system="return JSON",
                payload={"run_id": "run_audit"},
                max_attempts=1,
            )
    lines = [record.getMessage() for record in caplog.records if "companyintel.reasoning_attempt" in record.getMessage()]
    assert len(lines) == 1
    assert "super-secret-token" not in lines[0]
    assert '"run_id":"run_audit"' in lines[0]
    assert '"result":"failure"' in lines[0]
    assert '"failure_stage":"provider_request"' in lines[0]



def test_review_lowercase_verdict_is_normalized():
    from tools.companyintel_reasoning_controller import CompanyIntelReasoningController

    controller = CompanyIntelReasoningController(llm_request=lambda operation, payload: {
        "verdict": "pass",
        "findings": [],
        "review_text": "approved",
    })
    result = controller.review_report_with_llm(report={"run_id": "run-1", "graph_revision": "rev"})
    assert result["verdict"] == "PASS"


def test_review_schema_repair_is_bounded():
    from tools.companyintel_reasoning_controller import CompanyIntelReasoningController

    calls = []
    def request(operation, payload):
        calls.append(operation)
        if operation == "review_report":
            return {"verdict": "FAIL", "findings": [], "review_text": "invalid verdict"}
        return {"verdict": "PASS_WITH_LIMITATIONS", "findings": [], "review_text": "bounded review"}

    controller = CompanyIntelReasoningController(llm_request=request)
    result = controller.review_report_with_llm(report={"graph_revision": "rev"})
    assert result["verdict"] == "PASS_WITH_LIMITATIONS"
    assert result["findings"] == []
    assert calls == ["review_report", "repair_review_report"]


def test_review_schema_failure_after_repair_is_nonretryable():
    from tools.companyintel_reasoning_controller import CompanyIntelReasoningController
    from tools.openrouter_client import OpenRouterClientError

    calls = []
    def request(operation, payload):
        calls.append(operation)
        return {"verdict": "FAIL", "findings": [], "review_text": "invalid"}

    controller = CompanyIntelReasoningController(llm_request=request)
    with pytest.raises(OpenRouterClientError) as caught:
        controller.review_report_with_llm(report={"graph_revision": "rev"})
    assert caught.value.code == "REASONING_SCHEMA_VALIDATION_FAILED"
    assert caught.value.operation == "repair_review_report"
    assert caught.value.retryable is False
    assert calls == ["review_report", "repair_review_report"]


def test_revise_schema_repair_is_bounded():
    from tools.companyintel_reasoning_controller import CompanyIntelReasoningController

    calls = []
    def request(operation, payload):
        calls.append(operation)
        if operation == "revise_report":
            return {"report_text": "invalid", "claims": [{"text": "unknown", "evidence_ids": ["missing"]}]}
        return {"report_text": "revised", "claims": [{"text": "supported", "evidence_ids": ["e-1"]}]}

    controller = CompanyIntelReasoningController(llm_request=request)
    result = controller.revise_report_with_llm(
        graph_digest={"graph_revision": "rev", "evidence_ids": ["e-1"]},
        report={"report_text": "draft", "claims": []},
        findings=[{"code": "C1", "severity": "WARNING", "message": "fix"}],
    )
    assert result["report_text"] == "revised"
    assert calls == ["revise_report", "repair_revise_report"]


def test_post_repair_schema_failure_is_structured_nonretryable():
    from tools.companyintel_reasoning_controller import CompanyIntelReasoningController
    from tools.openrouter_client import OpenRouterClientError

    calls = []
    def request(operation, payload):
        calls.append(operation)
        return {"report_text": "invalid", "claims": [{"text": "bad", "evidence_ids": ["missing"]}]}

    controller = CompanyIntelReasoningController(llm_request=request)
    with pytest.raises(OpenRouterClientError) as caught:
        controller.synthesize_report_with_llm(graph_digest={"graph_revision": "rev", "evidence_ids": []})
    assert caught.value.code == "REASONING_SCHEMA_VALIDATION_FAILED"
    assert caught.value.failure_stage == "schema_validation"
    assert caught.value.retryable is False
    assert calls == ["synthesize_report", "repair_synthesize_report"]
