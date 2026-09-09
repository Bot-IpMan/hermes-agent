import pytest

from tools.companyintel_document_extraction import ExtractionError, _safe_url


def test_extractor_rejects_private_and_credential_urls():
    with pytest.raises(ExtractionError, match="UNSAFE_URL"):
        _safe_url("https://user:pass@example.com/")
    with pytest.raises(ExtractionError, match="SSRF_BLOCKED"):
        _safe_url("http://127.0.0.1:8787/")


def test_extractor_rejects_non_http_scheme():
    with pytest.raises(ExtractionError, match="UNSAFE_URL"):
        _safe_url("file:///etc/passwd")
