from __future__ import annotations

import json

import pytest

from kimi_cli.tools.dream_memory import (
    EnvironmentDebugMemoryParams,
    OptimizationMemoryParams,
    WriteEnvironmentDebugMemory,
    WriteOptimizationMemory,
)


@pytest.mark.asyncio
async def test_write_optimization_memory_tool_persists_memory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / ".kimi"))
    tool = WriteOptimizationMemory()

    result = await tool(
        OptimizationMemoryParams(
            title="FP8 KV cache improves H100 decode throughput",
            summary="Switching KV cache precision to FP8 improved decode throughput.",
            environment="H100 SXM",
            model_type="llm",
            model_arch="decoder-only transformer",
            inference_backend="TensorRT-LLM",
            precision={"weights": "fp16", "kv_cache": "fp8"},
            metrics_before={"tokens_per_second": 210},
            metrics_after={"tokens_per_second": 284},
            objective_metric="tokens_per_second",
            success=True,
            detail_description="Enabled FP8 KV cache after validating output quality.",
            chosen=3,
            useful_when_chosen=2,
        )
    )

    assert not result.is_error
    assert "optimization memory saved" in result.output.lower()

    data = json.loads((tmp_path / ".kimi" / "dream" / "memories.json").read_text())
    assert data["memories"][0]["category"] == "optimization"
    assert data["memories"][0]["useful_rate"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_write_environment_debug_memory_tool_persists_debug_schema(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / ".kimi"))
    tool = WriteEnvironmentDebugMemory()

    result = await tool(
        EnvironmentDebugMemoryParams(
            title="Pin torch build to matching CUDA runtime",
            summary="Torch imported but CUDA kernels failed at runtime.",
            environment="RTX 4090 workstation",
            debug_type="dependency",
            component="torch runtime",
            hardware="RTX 4090",
            os="Ubuntu 22.04",
            driver="NVIDIA 550",
            runtime="Python 3.11",
            dependency_stack={"torch": "2.4.0+cu121", "cuda": "12.1"},
            issue_signature="undefined symbol in libtorch_cuda.so",
            symptoms="Import succeeded, first CUDA tensor allocation crashed.",
            root_cause="Torch wheel CUDA runtime did not match installed extension builds.",
            solution="Reinstalled torch and extensions against the same cu121 wheel index.",
            verification="Ran CUDA tensor allocation and model warmup twice.",
            commands=["pip install --index-url https://download.pytorch.org/whl/cu121 torch"],
            success=True,
        )
    )

    assert not result.is_error
    assert "environment debug memory saved" in result.output.lower()

    data = json.loads((tmp_path / ".kimi" / "dream" / "memories.json").read_text())
    memory = data["memories"][0]
    assert memory["category"] == "environment_debug"
    assert memory["debug_type"] == "dependency"
    assert memory["component"] == "torch runtime"
    assert memory["commands"] == [
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch"
    ]


def test_dream_memory_tool_schemas_are_model_callable() -> None:
    opt_schema = WriteOptimizationMemory().base.parameters
    debug_schema = WriteEnvironmentDebugMemory().base.parameters

    assert "detail_description" in opt_schema["required"]
    assert opt_schema["properties"]["metrics_before"]["type"] == "object"
    assert opt_schema["properties"]["metrics_after"]["type"] == "object"

    assert "root_cause" in debug_schema["required"]
    assert "solution" in debug_schema["required"]
    assert debug_schema["properties"]["debug_type"]["enum"] == [
        "install",
        "build",
        "runtime",
        "dependency",
        "driver",
        "network",
        "auth",
        "filesystem",
        "performance",
        "other",
    ]
