from __future__ import annotations

from fastapi.testclient import TestClient

from kimi_cli.web.app import create_app
from kimi_cli.web.runner.codex_process import CodexCLIRunner


def test_web_app_always_uses_codex_runner(monkeypatch) -> None:
    monkeypatch.setenv("EVOINFER_AGENT_BACKEND", "kimi")

    app = create_app()
    with TestClient(app):
        assert isinstance(app.state.runner, CodexCLIRunner)
