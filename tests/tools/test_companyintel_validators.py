import pytest

from tools.validate_graph import validate_path
from tools.validate_report import validate_report


def test_graph_cli_validator_accepts_checked_in_fixture():
    result = validate_path("tests/fixtures/companyintel/url-phone-marketplace-legal-registry.v2.json")
    assert result["validated"] is True


def test_report_validator_is_fail_closed_for_completed_unvalidated_graph():
    with pytest.raises(ValueError, match="validated"):
        validate_report({"status": "COMPLETED", "projection_gate": {"validated": False}})


def test_report_validator_accepts_only_v2_validated_projection():
    result = validate_report({
        "status": "COMPLETED",
        "projection_gate": {"validated": True, "graph_schema": "corporate-intelligence-graph/v2"},
        "artifact": {"summary": "bounded"},
    })
    assert result["report_ready"] is True
