"""Load the repository-owned CompanyIntel artifact contract for Hermes.

The policy implementation lives in lookdata-stack. Hermes imports this bridge
instead of maintaining a second approved-artifact allowlist."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_contract():
    module_path = Path(
        os.environ.get(
            "COMPANYINTEL_ARTIFACT_MODULE",
            "/home/lookdata/git/lookdata-stack/intelligence-worker/app/companyintel_artifacts.py",
        )
    ).resolve()
    if not module_path.is_file():
        raise RuntimeError(f"CompanyIntel artifact contract is unavailable: {module_path}")
    module_name = "lookdata_companyintel_artifacts"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("CompanyIntel artifact contract could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_CONTRACT = _load_contract()
ArtifactClass = _CONTRACT.ArtifactClass
ArtifactDecision = _CONTRACT.ArtifactDecision
ArtifactFailure = _CONTRACT.ArtifactFailure
DEFAULT_READER_PRINCIPAL = _CONTRACT.DEFAULT_READER_PRINCIPAL
classify_artifact = _CONTRACT.classify_artifact
publish_text = _CONTRACT.publish_text

__all__ = [
    "ArtifactClass",
    "ArtifactDecision",
    "ArtifactFailure",
    "DEFAULT_READER_PRINCIPAL",
    "classify_artifact",
    "publish_text",
]
