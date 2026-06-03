from __future__ import annotations

from fastapi.testclient import TestClient

from kimi_cli.web.app import create_app


def test_system_status_endpoint_returns_machine_summary() -> None:
    app = create_app(session_token="test-token")
    with TestClient(app) as client:
        response = client.get(
            "/api/system/status",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["hostname"]
    assert payload["platform"]
    assert payload["cpu"]["logical_cores"] >= 1
    assert "load_average" in payload["cpu"]
    assert payload["memory"]["total_bytes"] >= 0
    assert isinstance(payload["gpus"], list)
