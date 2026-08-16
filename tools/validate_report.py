"""Fail-closed validator for CompanyIntel report artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_report(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("report must be an object")
    gate = document.get("projection_gate", document.get("gate", {}))
    if not isinstance(gate, dict):
        raise ValueError("report projection gate must be an object")
    validated = bool(gate.get("validated", gate.get("valid", False)))
    graph_schema = gate.get("graph_schema", gate.get("schema_version"))
    status = str(document.get("status", document.get("research_state", {}).get("status", ""))).upper()
    if status in {"COMPLETED", "COMPLETE", "SUCCESS"} and (not validated or graph_schema != "corporate-intelligence-graph/v2"):
        raise ValueError("completed report requires validated corporate-intelligence-graph/v2 projection")
    if document.get("report") is None and document.get("artifact") is None and status in {"COMPLETED", "COMPLETE", "SUCCESS"}:
        raise ValueError("completed report is missing report artifact")
    return {"validated": True, "status": status or "UNSPECIFIED", "graph_schema": graph_schema, "report_ready": validated and status in {"COMPLETED", "COMPLETE", "SUCCESS"}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json")
    args = parser.parse_args()
    try:
        result = validate_report(json.loads(Path(args.report_json).read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"validated": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
