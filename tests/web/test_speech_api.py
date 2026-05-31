from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from starlette.testclient import TestClient

from kimi_cli.web.api import speech


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(speech.router)
    return TestClient(app)


def test_transcribe_upload_returns_text_and_removes_temp_file(monkeypatch) -> None:
    captured_paths: list[Path] = []

    def fake_transcribe(path: Path) -> tuple[str, str | None]:
        captured_paths.append(path)
        assert path.exists()
        assert path.read_bytes() == b"audio bytes"
        return "你好，开始写代码", "zh"

    monkeypatch.setattr(speech, "_transcribe_audio_file", fake_transcribe)

    response = _make_client().post(
        "/api/speech/transcribe",
        files={"file": ("voice.webm", b"audio bytes", "audio/webm")},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "你好，开始写代码"
    assert response.json()["language"] == "zh"
    assert isinstance(response.json()["duration_ms"], int)
    assert captured_paths
    assert not captured_paths[0].exists()


def test_transcribe_upload_rejects_empty_audio() -> None:
    response = _make_client().post(
        "/api/speech/transcribe",
        files={"file": ("empty.webm", b"", "audio/webm")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Audio upload is empty."


def test_transcribe_upload_rejects_large_audio(monkeypatch) -> None:
    monkeypatch.setattr(speech, "MAX_AUDIO_UPLOAD_SIZE", 4)

    response = _make_client().post(
        "/api/speech/transcribe",
        files={"file": ("large.webm", b"12345", "audio/webm")},
    )

    assert response.status_code == 413
