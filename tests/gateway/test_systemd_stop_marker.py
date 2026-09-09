from gateway.systemd_stop_marker import main


def test_marker_helper_rejects_missing_or_invalid_pid():
    assert main([]) == 2
    assert main(["not-a-pid"]) == 2
    assert main(["0"]) == 2


def test_marker_helper_delegates_to_status(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: pid == 1234,
    )
    assert main(["1234"]) == 0
    assert main(["4321"]) == 1
