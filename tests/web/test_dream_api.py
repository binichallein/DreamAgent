from __future__ import annotations

import json

from fastapi import FastAPI
from starlette.testclient import TestClient

from kimi_cli.web.api import dream


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(dream.router)
    return TestClient(app)


def test_list_dream_memories_reads_memory_file(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".kimi"
    memory_file = memory_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "id": "m1",
                        "category": "optimization",
                        "title": "Chunked prefill improved TTFT",
                        "summary": "A100 + vLLM long-prompt win.",
                        "success": True,
                        "chosen": 4,
                        "useful_when_chosen": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KIMI_SHARE_DIR", str(memory_dir))

    response = _make_client().get("/api/dream/memories")

    assert response.status_code == 200
    data = response.json()
    assert data["memories"][0]["id"] == "m1"
    assert data["memories"][0]["category"] == "optimization"
    assert data["memories"][0]["useful_rate"] == 0.75


def test_list_dream_memories_filters_category(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".kimi"
    memory_file = memory_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            [
                {
                    "id": "opt",
                    "category": "optimization",
                    "title": "Optimization",
                },
                {
                    "id": "debug",
                    "category": "environment_debug",
                    "title": "Debug",
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KIMI_SHARE_DIR", str(memory_dir))

    response = _make_client().get("/api/dream/memories?category=environment_debug")

    assert response.status_code == 200
    data = response.json()
    assert [memory["id"] for memory in data["memories"]] == ["debug"]


def test_list_dream_memories_returns_empty_when_file_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / ".kimi"))

    response = _make_client().get("/api/dream/memories")

    assert response.status_code == 200
    assert response.json() == {"memories": []}


def test_create_optimization_memory_persists_entry(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".kimi"
    monkeypatch.setenv("KIMI_SHARE_DIR", str(memory_dir))

    response = _make_client().post(
        "/api/dream/memories/optimization",
        json={
            "title": "Paged attention improves long context throughput",
            "summary": "vLLM paged attention reduced KV cache fragmentation.",
            "environment": "A100 80GB",
            "model_type": "llm",
            "model_arch": "decoder-only transformer",
            "inference_backend": "vLLM",
            "metrics_before": {"tokens_per_second": 118},
            "metrics_after": {"tokens_per_second": 171},
            "success": True,
            "detail_description": "Enabled paged attention and raised max_num_batched_tokens.",
            "chosen": 2,
            "useful_when_chosen": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()["memory"]
    assert data["category"] == "optimization"
    assert data["id"]
    assert data["useful_rate"] == 0.5

    listed = _make_client().get("/api/dream/memories").json()["memories"]
    assert [memory["title"] for memory in listed] == [
        "Paged attention improves long context throughput"
    ]


def test_create_environment_debug_memory_persists_debug_schema(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".kimi"
    monkeypatch.setenv("KIMI_SHARE_DIR", str(memory_dir))

    response = _make_client().post(
        "/api/dream/memories/environment-debug",
        json={
            "title": "Force Whisper CPU backend when CUDA runtime is absent",
            "summary": "STT failed because faster-whisper tried to load libcuda.so.12.",
            "environment": "limx CPU-only browser STT",
            "debug_type": "runtime",
            "component": "speech-to-text",
            "hardware": "CPU",
            "os": "Ubuntu",
            "runtime": "Python 3.12",
            "dependency_stack": {"faster-whisper": "1.x", "ctranslate2": "4.x"},
            "issue_signature": "Library libcuda.so.12 is not found",
            "symptoms": "Voice input failed immediately after clicking the mic.",
            "root_cause": "The STT backend selected CUDA on a host without CUDA libraries.",
            "solution": "Set KIMI_WEB_STT_DEVICE=cpu and KIMI_WEB_STT_COMPUTE_TYPE=int8.",
            "verification": "Recorded voice input transcribed successfully through the GUI.",
            "commands": [
                "KIMI_WEB_STT_DEVICE=cpu KIMI_WEB_STT_COMPUTE_TYPE=int8 kimi web"
            ],
            "success": True,
        },
    )

    assert response.status_code == 200
    data = response.json()["memory"]
    assert data["category"] == "environment_debug"
    assert data["debug_type"] == "runtime"
    assert data["component"] == "speech-to-text"
    assert data["commands"] == [
        "KIMI_WEB_STT_DEVICE=cpu KIMI_WEB_STT_COMPUTE_TYPE=int8 kimi web"
    ]

    listed = _make_client().get("/api/dream/memories?category=environment_debug").json()[
        "memories"
    ]
    assert [memory["issue_signature"] for memory in listed] == [
        "Library libcuda.so.12 is not found"
    ]
