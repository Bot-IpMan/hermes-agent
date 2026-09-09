from companyintel.planners import plan_pivots
from companyintel.source_matrix import source_matrix


def test_phone_planner_emits_bounded_typed_variants():
    planned = plan_pivots("phone", "+380 50 123 4567")
    assert planned
    assert {item["pivot_type"] for item in planned} >= {"exact_search", "maps", "marketplaces"}
    assert len(planned) <= 12
    assert all(item["input_value"] for item in planned)


def test_domain_planner_deduplicates_www_variant():
    planned = plan_pivots("domain", "https://www.example.com")
    values = {item["input_value"] for item in planned}
    assert "example.com" in values
    assert len(values) <= 3


def test_wave5_matrix_is_explicitly_fail_closed():
    matrix = source_matrix()
    assert {item.name for item in matrix} >= {"marketplace", "maps", "social", "official_registry", "prozorro_spending"}
    assert all(not item.implemented and item.outcome == "CAPABILITY_UNAVAILABLE" for item in matrix)
