from tools.companyintel_source_classifier import classify_search_result, classify_source


def test_search_results_are_always_discovery_tier_c():
    result = classify_search_result("https://example.com/about", target_url="https://example.com")
    assert result.tier == "C"
    assert result.source_class == "search_snippet"


def test_target_origin_is_first_party_tier_b_only_for_bounded_retrieval():
    result = classify_source(
        "https://example.com/about", target_url="https://example.com",
        retrieval_method="bounded_http_extraction", content_type="text/html",
    )
    assert result.tier == "B"
    assert result.source_class == "first_party"
    assert result.authority_reason


def test_government_host_is_tier_a_but_fake_looking_host_is_not():
    registry = classify_source("https://data.gov.ua/resource/1", target_url="https://example.com", retrieval_method="bounded_http_extraction")
    fake = classify_source("https://example.gov.example/about", target_url="https://example.com", retrieval_method="bounded_http_extraction")
    generic = classify_source("https://example.gov/about", target_url="https://example.com", retrieval_method="bounded_http_extraction")
    assert registry.tier == "A"
    assert fake.tier == "C"
    assert generic.tier == "C"


def test_https_and_name_match_do_not_promote_external_source():
    result = classify_source(
        "https://example-directory.com/acme", target_url="https://acme.example",
        retrieval_method="bounded_http_extraction", content_type="text/html",
    )
    assert result.tier == "C"
