from __future__ import annotations


def test_companyintel_profile_exposes_registry_only_tools_to_api_server(monkeypatch, tmp_path):
    companyintel_home = tmp_path / "companyintel"
    companyintel_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(companyintel_home))
    monkeypatch.setenv("HERMES_PROFILE", "companyintel")

    # Import registration before resolving the platform surface, matching model startup.
    import tools.companyintel_graph_tool  # noqa: F401
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import _clear_tool_defs_cache, get_tool_definitions

    _clear_tool_defs_cache()
    enabled = sorted(_get_platform_tools({}, "api_server"))
    names = {
        item["function"]["name"]
        for item in get_tool_definitions(enabled, quiet_mode=True)
    }

    assert "companyintel" in enabled
    assert "companyintel_graph" in names
    assert "companyintel_init_run" in names
    assert "companyintel_validate" in names
