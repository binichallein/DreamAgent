from __future__ import annotations

import json
from types import MethodType
from uuid import uuid4

import pytest

from kimi_cli.web.runner.codex_adapter import CodexEventTranslator
from kimi_cli.web.runner.codex_process import CodexSessionProcess
from kimi_cli.wire.types import (
    ContentPart,
    StatusUpdate,
    StepBegin,
    ToolCall,
    ToolResult,
    TurnBegin,
    TurnEnd,
)


def test_codex_agent_delta_becomes_text_content_part() -> None:
    translator = CodexEventTranslator()

    messages = translator.translate_notification(
        {
            "method": "item/agentMessage/delta",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "itemId": "msg-1",
                "delta": "hello",
            },
        }
    )

    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ContentPart)
    assert msg.type == "text"
    assert msg.text == "hello"


def test_codex_turn_started_and_completed_become_wire_step_and_turn_end() -> None:
    translator = CodexEventTranslator()

    started = translator.translate_notification(
        {
            "method": "turn/started",
            "params": {"threadId": "thread-1", "turn": {"id": "turn-1"}},
        }
    )
    completed = translator.translate_notification(
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {"id": "turn-1", "status": "completed"},
            },
        }
    )

    assert started == [StepBegin(n=1)]
    assert completed == [TurnEnd()]


def test_codex_command_execution_becomes_tool_call_and_result() -> None:
    translator = CodexEventTranslator()

    started = translator.translate_notification(
        {
            "method": "item/started",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "id": "call-1",
                    "command": "/bin/bash -lc pwd",
                    "cwd": "/tmp/project",
                }
            },
        }
    )
    completed = translator.translate_notification(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "id": "call-1",
                    "command": "/bin/bash -lc pwd",
                    "cwd": "/tmp/project",
                    "aggregatedOutput": "/tmp/project\n",
                    "exitCode": 0,
                }
            },
        }
    )

    assert len(started) == 1
    tool_call = started[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.id == "call-1"
    assert tool_call.function.name == "Shell"
    assert tool_call.function.arguments == '{"command": "/bin/bash -lc pwd", "cwd": "/tmp/project"}'

    assert len(completed) == 1
    tool_result = completed[0]
    assert isinstance(tool_result, ToolResult)
    assert tool_result.tool_call_id == "call-1"
    assert tool_result.return_value.is_error is False
    assert tool_result.return_value.output == "/tmp/project\n"
    assert tool_result.return_value.display[0].type == "shell"


@pytest.mark.asyncio
async def test_codex_initialize_returns_frontend_safe_slash_commands() -> None:
    process = CodexSessionProcess(uuid4())
    broadcasts: list[str] = []

    async def start_noop(self: CodexSessionProcess) -> None:
        return None

    async def capture_broadcast(self: CodexSessionProcess, message: str) -> None:
        broadcasts.append(message)

    process.start = MethodType(start_noop, process)
    process._broadcast = MethodType(capture_broadcast, process)

    await process.send_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": "init-1",
                "params": {
                    "protocol_version": "1.9",
                    "client": {"name": "test"},
                    "capabilities": {
                        "supports_question": True,
                        "supports_plan_mode": True,
                        "supports_dream_mode": True,
                    },
                },
            }
        )
    )

    assert len(broadcasts) == 1
    response = json.loads(broadcasts[0])
    slash_commands = response["result"]["slash_commands"]

    assert [command["name"] for command in slash_commands] == [
        "init",
        "compact",
        "clear",
        "plan",
        "yolo",
    ]
    assert all(isinstance(command["aliases"], list) for command in slash_commands)


@pytest.mark.asyncio
async def test_codex_plan_slash_command_is_handled_locally() -> None:
    process = CodexSessionProcess(uuid4())
    broadcasts: list[str] = []
    wire_messages: list[object] = []

    async def start_noop(self: CodexSessionProcess) -> None:
        return None

    async def capture_broadcast(self: CodexSessionProcess, message: str) -> None:
        broadcasts.append(message)

    async def capture_wire_message(self: CodexSessionProcess, message: object) -> None:
        wire_messages.append(message)

    async def fail_codex_request(
        self: CodexSessionProcess,
        method: str,
        params: dict[str, object] | None,
    ) -> dict[str, object]:
        raise AssertionError(f"slash command unexpectedly called Codex {method}")

    process.start = MethodType(start_noop, process)
    process._broadcast = MethodType(capture_broadcast, process)
    process._emit_wire_message = MethodType(capture_wire_message, process)
    process._send_codex_request = MethodType(fail_codex_request, process)

    await process.send_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "prompt",
                "id": "prompt-1",
                "params": {"user_input": "/plan on"},
            }
        )
    )

    assert isinstance(wire_messages[0], TurnBegin)
    assert wire_messages[0].user_input == "/plan on"
    assert any(isinstance(message, StatusUpdate) and message.plan_mode for message in wire_messages)
    assert isinstance(wire_messages[-1], TurnEnd)

    response = next(item for item in map(json.loads, broadcasts) if item.get("id") == "prompt-1")
    assert response["id"] == "prompt-1"
    assert response["result"]["status"] == "finished"
