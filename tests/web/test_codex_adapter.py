from __future__ import annotations

from kimi_cli.web.runner.codex_adapter import CodexEventTranslator
from kimi_cli.wire.types import ContentPart, StepBegin, ToolCall, ToolResult, TurnEnd


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
