from companyintel.extractors.html import text_digest
from companyintel.extractors.javascript import public_identifiers
from companyintel.extractors.site import same_public_origin


def test_html_digest_removes_script_and_style_blocks():
    assert text_digest("<script>secret()</script><style>.x{}</style><h1>Acme</h1>") == "Acme"


def test_javascript_identifier_extractor_supports_common_assignments():
    assert public_identifiers("analytics_id = 'ga-123'; clientKey: \"key-1\"") == ["ga-123", "key-1"]


def test_site_origin_rejects_different_host():
    assert same_public_origin("https://example.com/a", "https://example.com/b")
    assert not same_public_origin("https://example.com", "https://evil.example")
