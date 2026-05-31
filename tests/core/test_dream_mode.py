from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from kosong import StepResult
from kosong.message import Message, TextPart, ToolCall
from kosong.tooling import ToolResult

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul, StepOutcome, TurnOutcome
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.dream_memory import WriteEnvironmentDebugMemory, WriteOptimizationMemory


def _make_soul(runtime: Runtime, tmp_path: Path) -> tuple[KimiSoul, KimiToolset]:
    toolset = KimiToolset()
    toolset.add(WriteOptimizationMemory())
    toolset.add(WriteEnvironmentDebugMemory())
    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=toolset,
        runtime=runtime,
    )
    soul = KimiSoul(agent, context=Context(file_backend=tmp_path / "history.jsonl"))
    return soul, toolset


@pytest.mark.asyncio
async def test_dream_memory_tools_hidden_until_dream_mode_enabled(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul, toolset = _make_soul(runtime, tmp_path)

    assert "WriteOptimizationMemory" not in {tool.name for tool in toolset.tools}
    assert "WriteEnvironmentDebugMemory" not in {tool.name for tool in toolset.tools}

    await soul.set_dream_mode_from_manual(True)
    soul.sync_dream_memory_tool_visibility(client_supports_dream_mode=True)

    assert "WriteOptimizationMemory" in {tool.name for tool in toolset.tools}
    assert "WriteEnvironmentDebugMemory" in {tool.name for tool in toolset.tools}


@pytest.mark.asyncio
async def test_dream_memory_tools_stay_hidden_when_client_does_not_support_dream_mode(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul, toolset = _make_soul(runtime, tmp_path)

    await soul.set_dream_mode_from_manual(True)
    soul.sync_dream_memory_tool_visibility(client_supports_dream_mode=False)

    assert "WriteOptimizationMemory" not in {tool.name for tool in toolset.tools}
    assert "WriteEnvironmentDebugMemory" not in {tool.name for tool in toolset.tools}


@pytest.mark.asyncio
async def test_dream_mode_model_tool_calls_persist_specialist_memories(
    runtime: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / ".kimi"))
    soul, _toolset = _make_soul(runtime, tmp_path)
    await soul.set_dream_mode_from_manual(True)
    soul.sync_dream_memory_tool_visibility(client_supports_dream_mode=True)
    await soul.context.append_message(
        Message(
            role="user",
            content=[TextPart(text="We fixed a CUDA runtime mismatch during model deployment.")],
        )
    )

    captured_tool_names: set[str] = set()
    captured_history: list[Message] = []

    async def fake_kosong_step(
        _chat_provider,
        _system_prompt,
        toolset,
        history,
        **_kwargs,
    ) -> StepResult:
        captured_tool_names.update(tool.name for tool in toolset.tools)
        captured_history[:] = list(history)
        debug_tool_call = ToolCall(
            id="dream-debug-1",
            function=ToolCall.FunctionBody(
                name="WriteEnvironmentDebugMemory",
                arguments=json.dumps(
                    {
                        "title": "Pin torch and extension wheels to the same CUDA runtime",
                        "environment": "RTX 4090 workstation, Ubuntu 22.04",
                        "debug_type": "dependency",
                        "component": "torch CUDA extension stack",
                        "inference_backend": "pytorch",
                        "issue_signature": "libtorch_cuda.so undefined symbol",
                        "symptoms": "Model import worked but first CUDA inference crashed.",
                        "root_cause": "Torch and extension wheels targeted different CUDA runtimes.",
                        "solution": "Reinstalled torch and extensions from the same cu121 wheel index.",
                        "verification": "Ran CUDA tensor allocation and one model warmup successfully.",
                        "success": True,
                    }
                ),
            ),
        )
        optimization_tool_call = ToolCall(
            id="dream-opt-1",
            function=ToolCall.FunctionBody(
                name="WriteOptimizationMemory",
                arguments=json.dumps(
                    {
                        "title": "Reduce decode latency by switching KV cache to FP8",
                        "environment": "H100 SXM",
                        "model_type": "llm",
                        "model_arch": "decoder-only transformer",
                        "model_name": "test-llm",
                        "inference_backend": "TensorRT-LLM",
                        "precision": {"weights": "fp16", "kv_cache": "fp8"},
                        "metrics_before": {"tokens_per_second": 210},
                        "metrics_after": {"tokens_per_second": 284},
                        "objective_metric": "tokens_per_second",
                        "success": True,
                        "detail_description": (
                            "Enabled FP8 KV cache after verifying accuracy stayed acceptable."
                        ),
                    }
                ),
            ),
        )

        tool_result_futures: dict[str, asyncio.Future[ToolResult]] = {}
        for tool_call in (debug_tool_call, optimization_tool_call):
            handle_result = toolset.handle(tool_call)
            if isinstance(handle_result, ToolResult):
                future: asyncio.Future[ToolResult] = asyncio.get_running_loop().create_future()
                future.set_result(handle_result)
            else:
                future = handle_result
            tool_result_futures[tool_call.id] = future

        return StepResult(
            id="dream-step-1",
            message=Message(
                role="assistant",
                content=[TextPart(text="Saving reusable deployment/debug and optimization lessons.")],
                tool_calls=[debug_tool_call, optimization_tool_call],
            ),
            usage=None,
            tool_calls=[debug_tool_call, optimization_tool_call],
            _tool_result_futures=tool_result_futures,
        )

    monkeypatch.setattr("kimi_cli.soul.kimisoul.kosong.step", fake_kosong_step)
    monkeypatch.setattr("kimi_cli.soul.kimisoul.wire_send", lambda _msg: None)

    outcome = await soul._step()

    assert outcome is None
    assert "WriteEnvironmentDebugMemory" in captured_tool_names
    assert "WriteOptimizationMemory" in captured_tool_names
    assert any(
        "Dream mode is active for this session" in part.text
        for msg in captured_history
        for part in msg.content
        if isinstance(part, TextPart)
    )

    data = json.loads((tmp_path / ".kimi" / "dream" / "memories.json").read_text())
    memories = {memory["category"]: memory for memory in data["memories"]}
    debug_memory = memories["environment_debug"]
    assert debug_memory["debug_type"] == "dependency"
    assert debug_memory["component"] == "torch CUDA extension stack"
    optimization_memory = memories["optimization"]
    assert optimization_memory["model_type"] == "llm"
    assert optimization_memory["metrics_after"]["tokens_per_second"] == 284


@pytest.mark.asyncio
async def test_dream_mode_forces_post_turn_memory_hook(
    runtime: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / ".kimi"))
    soul, _toolset = _make_soul(runtime, tmp_path)
    await soul.set_dream_mode_from_manual(True)
    soul.sync_dream_memory_tool_visibility(client_supports_dream_mode=True)

    async def fake_turn(user_message: Message) -> TurnOutcome:
        await soul.context.append_message(user_message)
        assistant_message = Message(
            role="assistant",
            content=[
                TextPart(
                    text=(
                        "Deployment lesson: on RTX 4090, torch CUDA extension failed with "
                        "libtorch_cuda.so undefined symbol. Pinning torch and extensions to "
                        "the same cu121 wheel index fixed it."
                    )
                )
            ],
        )
        await soul.context.append_message(assistant_message)
        return TurnOutcome(
            stop_reason="no_tool_calls",
            final_message=assistant_message,
            step_count=1,
        )

    captured_tool_names: set[str] = set()
    captured_history: list[Message] = []

    async def fake_kosong_step(
        _chat_provider,
        _system_prompt,
        toolset,
        history,
        **_kwargs,
    ) -> StepResult:
        captured_tool_names.update(tool.name for tool in toolset.tools)
        captured_history[:] = list(history)
        tool_call = ToolCall(
            id="forced-dream-debug-1",
            function=ToolCall.FunctionBody(
                name="WriteEnvironmentDebugMemory",
                arguments=json.dumps(
                    {
                        "title": "Force hook stores cu121 wheel pinning lesson",
                        "environment": "RTX 4090 workstation",
                        "debug_type": "dependency",
                        "component": "torch CUDA extension stack",
                        "issue_signature": "libtorch_cuda.so undefined symbol",
                        "symptoms": "CUDA extension failed during deployment.",
                        "root_cause": "Torch and extension wheels used mismatched CUDA runtimes.",
                        "solution": "Pin torch and extension wheels to the same cu121 wheel index.",
                        "verification": "Assistant reported the deployment issue was fixed.",
                        "success": True,
                    }
                ),
            ),
        )
        handle_result = toolset.handle(tool_call)
        if isinstance(handle_result, ToolResult):
            future: asyncio.Future[ToolResult] = asyncio.get_running_loop().create_future()
            future.set_result(handle_result)
        else:
            future = handle_result
        return StepResult(
            id="forced-dream-hook-step",
            message=Message(
                role="assistant",
                content=[TextPart(text="Saved forced Dream memory.")],
                tool_calls=[tool_call],
            ),
            usage=None,
            tool_calls=[tool_call],
            _tool_result_futures={tool_call.id: future},
        )

    monkeypatch.setattr(soul, "_turn", fake_turn)
    monkeypatch.setattr("kimi_cli.soul.kimisoul.kosong.step", fake_kosong_step)
    monkeypatch.setattr("kimi_cli.soul.kimisoul.wire_send", lambda _msg: None)

    await soul.run("We fixed the deployment issue.")

    assert "WriteEnvironmentDebugMemory" in captured_tool_names
    assert "WriteOptimizationMemory" in captured_tool_names
    assert any(
        "Dream memory post-turn hook" in part.text
        for msg in captured_history
        for part in msg.content
        if isinstance(part, TextPart)
    )

    data = json.loads((tmp_path / ".kimi" / "dream" / "memories.json").read_text())
    [memory] = data["memories"]
    assert memory["category"] == "environment_debug"
    assert memory["title"] == "Force hook stores cu121 wheel pinning lesson"


@pytest.mark.asyncio
async def test_dream_mode_runs_memory_hook_every_ten_steps(
    runtime: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    soul, _toolset = _make_soul(runtime, tmp_path)
    await soul.set_dream_mode_from_manual(True)
    soul.sync_dream_memory_tool_visibility(client_supports_dream_mode=True)

    hook_windows: list[tuple[int, int]] = []

    async def fake_dream_hook(start_index: int) -> bool:
        hook_windows.append((start_index, len(soul.context.history)))
        return True

    step_calls = 0

    async def fake_step() -> StepOutcome | None:
        nonlocal step_calls
        step_calls += 1
        assistant_message = Message(
            role="assistant",
            content=[TextPart(text=f"step {step_calls}")],
        )
        await soul.context.append_message(assistant_message)
        if step_calls == 11:
            return StepOutcome(
                stop_reason="no_tool_calls",
                assistant_message=assistant_message,
            )
        return None

    monkeypatch.setattr(soul, "_run_dream_memory_post_turn_hook", fake_dream_hook)
    monkeypatch.setattr(soul, "_step", fake_step)
    monkeypatch.setattr("kimi_cli.soul.kimisoul.wire_send", lambda _msg: None)

    await soul.run("Run a long inference optimization task.")

    assert hook_windows == [
        (0, 11),  # user message + 10 assistant steps
        (11, 12),  # only the final assistant step after the periodic hook
    ]
