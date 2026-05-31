from __future__ import annotations

import json
from typing import Any

from kosong.message import TextPart, ThinkPart, ToolCall
from kosong.tooling import ToolResult, ToolReturnValue

from kimi_cli.tools.display import ShellDisplayBlock
from kimi_cli.wire.jsonrpc import JSONRPCSuccessResponse, Statuses
from kimi_cli.wire.serde import serialize_wire_message
from kimi_cli.wire.types import StepBegin, TurnEnd, WireMessage


def dump_wire_event(message: WireMessage) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "event",
            "params": serialize_wire_message(message),
        },
        ensure_ascii=False,
    )


def dump_success_response(message_id: str, result: dict[str, Any] | None = None) -> str:
    return JSONRPCSuccessResponse(id=message_id, result=result or {}).model_dump_json()


def dump_finished_response(message_id: str) -> str:
    return dump_success_response(message_id, {"status": Statuses.FINISHED})


class CodexEventTranslator:
    """Translate Codex app-server notifications into the existing web wire events."""

    def __init__(self) -> None:
        self._step = 0

    def translate_notification(self, notification: dict[str, Any]) -> list[WireMessage]:
        method = notification.get("method")
        params = notification.get("params")
        if not isinstance(params, dict):
            params = {}

        match method:
            case "turn/started":
                self._step += 1
                return [StepBegin(n=self._step)]
            case "turn/completed":
                self._step = 0
                return [TurnEnd()]
            case "item/agentMessage/delta":
                delta = params.get("delta")
                if isinstance(delta, str) and delta:
                    return [TextPart(text=delta)]
            case "item/reasoning/textDelta" | "item/reasoning/summaryTextDelta":
                delta = params.get("delta")
                if isinstance(delta, str) and delta:
                    return [ThinkPart(think=delta)]
            case "item/started":
                return self._translate_item_started(params)
            case "item/completed":
                return self._translate_item_completed(params)
        return []

    def _translate_item_started(self, params: dict[str, Any]) -> list[WireMessage]:
        item = params.get("item")
        if not isinstance(item, dict) or item.get("type") != "commandExecution":
            return []

        item_id = item.get("id")
        command = item.get("command")
        if not isinstance(item_id, str) or not isinstance(command, str):
            return []

        cwd = item.get("cwd")
        arguments = {"command": command}
        if isinstance(cwd, str):
            arguments["cwd"] = cwd

        return [
            ToolCall(
                id=item_id,
                function=ToolCall.FunctionBody(
                    name="Shell",
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        ]

    def _translate_item_completed(self, params: dict[str, Any]) -> list[WireMessage]:
        item = params.get("item")
        if not isinstance(item, dict) or item.get("type") != "commandExecution":
            return []

        item_id = item.get("id")
        command = item.get("command")
        if not isinstance(item_id, str) or not isinstance(command, str):
            return []

        output = item.get("aggregatedOutput")
        if not isinstance(output, str):
            output = ""

        exit_code = item.get("exitCode")
        is_error = isinstance(exit_code, int) and exit_code != 0
        message = f"Command exited with code {exit_code}" if isinstance(exit_code, int) else ""

        return [
            ToolResult(
                tool_call_id=item_id,
                return_value=ToolReturnValue(
                    is_error=is_error,
                    output=output,
                    message=message,
                    display=[ShellDisplayBlock(language="bash", command=command)],
                ),
            )
        ]
