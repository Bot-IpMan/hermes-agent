"""CLI validator for the canonical CompanyIntel graph/v2 projection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from companyintel.schema import adapt_legacy_graph, validate_graph


def validate_path(path: str | Path, *, allow_legacy: bool = False) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if allow_legacy:
        document = adapt_legacy_graph(document) if document.get("schema_version") != "corporate-intelligence-graph/v2" else document
    return validate_graph(document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_json")
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(validate_path(args.graph_json, allow_legacy=args.allow_legacy), sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"validated": False, "error": str(exc)}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
