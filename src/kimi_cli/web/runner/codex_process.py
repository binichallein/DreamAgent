"""Codex app-server backed runner for the EvoInfer web interface."""

from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from kosong.message import ContentPart, TextPart
from starlette.websockets import WebSocket, WebSocketState

from kimi_cli import logger
from kimi_cli.utils.subprocess_env import get_clean_env
from kimi_cli.web.models import (
    SessionNoticePayload,
    SessionState,
    SessionStatus,
)
from kimi_cli.web.runner.codex_adapter import (
    CodexEventTranslator,
    dump_finished_response,
    dump_success_response,
    dump_wire_event,
)
from kimi_cli.web.runner.messages import new_session_status_message
from kimi_cli.web.store.sessions import load_session_by_id
from kimi_cli.wire.file import WireFile
from kimi_cli.wire.jsonrpc import (
    ErrorCodes,
    JSONRPCCancelMessage,
    JSONRPCErrorObject,
    JSONRPCErrorResponse,
    JSONRPCInitializeMessage,
    JSONRPCInMessageAdapter,
    JSONRPCPromptMessage,
    JSONRPCSetDreamModeMessage,
    JSONRPCSetPlanModeMessage,
    JSONRPCSteerMessage,
    JSONRPCSuccessResponse,
)
from kimi_cli.wire.types import StatusUpdate, TurnBegin, WireMessage

CODEX_THREAD_FILE = "codex_thread.json"
ENV_CODEX_BIN = "EVOINFER_CODEX_BIN"
ENV_CODEX_MODEL = "EVOINFER_CODEX_MODEL"
ENV_CODEX_MODEL_PROVIDER = "EVOINFER_CODEX_MODEL_PROVIDER"
ENV_CODEX_SANDBOX = "EVOINFER_CODEX_SANDBOX"
ENV_CODEX_APPROVAL_POLICY = "EVOINFER_CODEX_APPROVAL_POLICY"


class CodexSessionProcess:
    """Manage one GUI session backed by a Codex app-server subprocess."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        self._in_flight_prompt_ids: set[str] = set()
        self._active_prompt_id: str | None = None
        self._active_turn_id: str | None = None
        self._active_assistant_parts: list[str] = []
        self._status_seq = 0
        self._worker_id: str | None = None
        self._status = SessionStatus(
            session_id=self.session_id,
            state="stopped",
            seq=self._status_seq,
            worker_id=self._worker_id,
            reason=None,
            detail=None,
            updated_at=datetime.now(UTC),
        )
        self._process: asyncio.subprocess.Process | None = None
        self._websockets: set[WebSocket] = set()
        self._websocket_count = 0
        self._replay_buffers: dict[WebSocket, list[str]] = {}
        self._read_task: asyncio.Task[None] | None = None
        self._expecting_exit = False
        self._lock = asyncio.Lock()
        self._ws_lock = asyncio.Lock()
        self._sent_files: set[str] = set()
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._translator = CodexEventTranslator()
        self._thread_id: str | None = None
        self._thread_path: str | None = None
        self._session_dir: Path | None = None
        self._context_file: Path | None = None
        self._wire_file: WireFile | None = None
        self._plan_mode = False
        self._dream_mode = False

    @property
    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.returncode is None

    @property
    def is_running(self) -> bool:
        return self.is_alive

    @property
    def is_busy(self) -> bool:
        return bool(self._in_flight_prompt_ids)

    @property
    def status(self) -> SessionStatus:
        return self._status

    @property
    def websocket_count(self) -> int:
        return self._websocket_count

    def clear_in_flight(self) -> None:
        self._in_flight_prompt_ids.clear()
        self._active_prompt_id = None
        self._active_turn_id = None
        self._active_assistant_parts.clear()

    async def send_status_snapshot(self, ws: WebSocket) -> None:
        await ws.send_text(new_session_status_message(self._status).model_dump_json())

    def _build_status(
        self,
        state: SessionState,
        reason: str | None,
        detail: str | None,
    ) -> SessionStatus | None:
        current = self._status
        if (
            current.state == state
            and current.reason == reason
            and current.detail == detail
            and current.worker_id == self._worker_id
        ):
            return None
        self._status_seq += 1
        status = SessionStatus(
            session_id=self.session_id,
            state=state,
            seq=self._status_seq,
            worker_id=self._worker_id,
            reason=reason,
            detail=detail,
            updated_at=datetime.now(UTC),
        )
        self._status = status
        return status

    async def _emit_status(
        self,
        state: SessionState,
        *,
        reason: str | None = None,
        detail: str | None = None,
    ) -> None:
        status = self._build_status(state, reason, detail)
        if status is not None:
            await self._broadcast(new_session_status_message(status).model_dump_json())

    async def start(
        self,
        *,
        reason: str | None = None,
        detail: str | None = None,
        restart_started_at: float | None = None,
    ) -> None:
        async with self._lock:
            if self.is_alive:
                if self._read_task is None or self._read_task.done():
                    self._read_task = asyncio.create_task(self._read_loop())
                return

            self.clear_in_flight()
            self._expecting_exit = False
            self._worker_id = str(uuid4())
            self._translator = CodexEventTranslator()
            self._load_session_paths()

            codex_bin = os.environ.get(ENV_CODEX_BIN) or shutil.which("codex")
            if not codex_bin:
                await self._emit_status(
                    "error",
                    reason="codex_not_found",
                    detail="codex executable was not found on PATH",
                )
                raise RuntimeError("codex executable was not found on PATH")

            self._process = await asyncio.create_subprocess_exec(
                codex_bin,
                "app-server",
                "--listen",
                "stdio://",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=16 * 1024 * 1024,
                env=get_clean_env(),
            )
            self._read_task = asyncio.create_task(self._read_loop())

            await self._initialize_codex()
            await self._ensure_thread()

            if restart_started_at is not None:
                elapsed_ms = int((time.perf_counter() - restart_started_at) * 1000)
                await self._emit_status(
                    "idle",
                    reason=reason or "start",
                    detail=f"restart_ms={elapsed_ms}",
                )
                await self._emit_restart_notice(reason=reason, restart_ms=elapsed_ms)
            else:
                await self._emit_status("idle", reason=reason or "start", detail=detail)

    async def stop(self) -> None:
        await self.stop_worker(reason="stop")
        await self._close_all_websockets()

    async def stop_worker(
        self,
        *,
        reason: str | None = None,
        emit_status: bool = True,
    ) -> None:
        async with self._lock:
            self._expecting_exit = True
            for future in self._pending_requests.values():
                if not future.done():
                    future.cancel()
            self._pending_requests.clear()

            if self._process is not None:
                if self._process.returncode is None:
                    self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=10.0)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
                self._process = None

            if self._read_task is not None:
                self._read_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._read_task
                self._read_task = None

            self.clear_in_flight()
            self._worker_id = None
            self._expecting_exit = False
            if emit_status:
                await self._emit_status("stopped", reason=reason or "stop")

    async def restart_worker(self, *, reason: str | None = None) -> None:
        started_at = time.perf_counter()
        await self._emit_status("restarting", reason=reason or "restart")
        await self.stop_worker(reason="restart", emit_status=False)
        await self.start(reason=reason or "restart", restart_started_at=started_at)

    async def _emit_restart_notice(self, *, reason: str | None, restart_ms: int) -> None:
        payload = SessionNoticePayload(
            text=f"Session restarted - {restart_ms}ms",
            kind="restart",
            reason=reason,
            restart_ms=restart_ms,
        )
        await self._broadcast(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "event",
                    "params": {
                        "type": "SessionNotice",
                        "payload": payload.model_dump(mode="json"),
                    },
                },
                ensure_ascii=False,
            )
        )

    def _load_session_paths(self) -> None:
        joint_session = load_session_by_id(self.session_id)
        if joint_session is None:
            raise ValueError(f"Session not found: {self.session_id}")
        session = joint_session.kimi_cli_session
        self._session_dir = session.dir
        self._context_file = session.context_file
        self._wire_file = session.wire_file

    @property
    def _metadata_path(self) -> Path:
        if self._session_dir is None:
            self._load_session_paths()
        assert self._session_dir is not None
        return self._session_dir / CODEX_THREAD_FILE

    def _load_codex_thread_metadata(self) -> dict[str, Any]:
        path = self._metadata_path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_codex_thread_metadata(self, result: dict[str, Any]) -> None:
        thread = result.get("thread")
        if not isinstance(thread, dict):
            return
        thread_id = thread.get("id")
        if not isinstance(thread_id, str):
            return
        self._thread_id = thread_id
        path = thread.get("path")
        self._thread_path = path if isinstance(path, str) else None
        payload = {
            "thread_id": self._thread_id,
            "thread_path": self._thread_path,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    async def _initialize_codex(self) -> None:
        response = await self._send_codex_request(
            "initialize",
            {
                "clientInfo": {
                    "name": "evoinfer-web",
                    "title": "EvoInfer",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "experimentalApi": True,
                    "requestAttestation": False,
                    "optOutNotificationMethods": [],
                },
            },
        )
        if "error" in response:
            raise RuntimeError(response["error"].get("message", "Codex initialize failed"))
        await self._send_codex_notification("initialized", None)

    async def _ensure_thread(self) -> None:
        metadata = self._load_codex_thread_metadata()
        thread_id = metadata.get("thread_id")
        joint_session = load_session_by_id(self.session_id)
        if joint_session is None:
            raise ValueError(f"Session not found: {self.session_id}")

        params = self._base_thread_params(str(joint_session.kimi_cli_session.work_dir))
        if isinstance(thread_id, str) and thread_id:
            params["threadId"] = thread_id
            response = await self._send_codex_request("thread/resume", params)
            if "result" in response:
                self._save_codex_thread_metadata(response["result"])
                return
            logger.warning(
                "Failed to resume Codex thread {thread_id}: {response}",
                thread_id=thread_id,
                response=response,
            )

        response = await self._send_codex_request("thread/start", params)
        if "error" in response:
            raise RuntimeError(response["error"].get("message", "Codex thread/start failed"))
        self._save_codex_thread_metadata(response["result"])

    def _base_thread_params(self, cwd: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "cwd": cwd,
            "approvalPolicy": os.environ.get(ENV_CODEX_APPROVAL_POLICY, "never"),
            "sandbox": os.environ.get(ENV_CODEX_SANDBOX, "workspace-write"),
            "personality": "pragmatic",
        }
        model = os.environ.get(ENV_CODEX_MODEL)
        if model:
            params["model"] = model
        model_provider = os.environ.get(ENV_CODEX_MODEL_PROVIDER)
        if model_provider:
            params["modelProvider"] = model_provider
        return params

    async def _send_codex_request(
        self,
        method: str,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Codex process is not running")
        request_id = str(uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[request_id] = future
        payload: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()
        try:
            return await asyncio.wait_for(future, timeout=120.0)
        finally:
            self._pending_requests.pop(request_id, None)

    async def _send_codex_notification(self, method: str, params: dict[str, Any] | None) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Codex process is not running")
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    if self._expecting_exit:
                        break
                    stderr = await self._process.stderr.read()
                    detail = stderr.decode("utf-8", errors="replace") or "Codex app-server exited"
                    self.clear_in_flight()
                    await self._broadcast(
                        JSONRPCErrorResponse(
                            id=str(uuid4()),
                            error=JSONRPCErrorObject(
                                code=self._process.returncode or -1,
                                message=detail,
                            ),
                        ).model_dump_json()
                    )
                    await self._emit_status("error", reason="process_exit", detail=detail)
                    break

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Invalid Codex app-server message: {line}", line=line)
                    continue

                await self._handle_codex_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Unexpected error in Codex read loop: {error}", error=exc)
            self.clear_in_flight()
            await self._emit_status("error", reason="read_loop_error", detail=str(exc))

    async def _handle_codex_message(self, message: dict[str, Any]) -> None:
        message_id = message.get("id")
        if isinstance(message_id, str) and ("result" in message or "error" in message):
            future = self._pending_requests.get(message_id)
            if future is not None and not future.done():
                future.set_result(message)
            return

        method = message.get("method")
        if isinstance(message_id, str) and isinstance(method, str):
            await self._handle_codex_server_request(message_id, method, message.get("params"))
            return

        if not isinstance(method, str):
            return

        params = message.get("params")
        if method == "item/agentMessage/delta" and isinstance(params, dict):
            delta = params.get("delta")
            if isinstance(delta, str):
                self._active_assistant_parts.append(delta)
        elif method == "turn/started" and isinstance(params, dict):
            turn = params.get("turn")
            if isinstance(turn, dict):
                turn_id = turn.get("id")
                if isinstance(turn_id, str):
                    self._active_turn_id = turn_id

        for wire_message in self._translator.translate_notification(message):
            await self._emit_wire_message(wire_message)

        if method == "thread/status/changed" and isinstance(params, dict):
            status = params.get("status")
            if isinstance(status, dict):
                status_type = status.get("type")
                if status_type == "active":
                    await self._emit_status("busy", reason="codex_active")
                elif status_type == "idle" and not self.is_busy:
                    await self._emit_status("idle", reason="codex_idle")
                elif status_type == "systemError":
                    await self._emit_status("error", reason="codex_system_error")

        if method == "thread/tokenUsage/updated" and isinstance(params, dict):
            await self._emit_token_usage(params)

        if method == "turn/completed":
            await self._complete_active_prompt()

        if method == "error":
            error_text = ""
            if isinstance(params, dict):
                error_text = str(params.get("message") or params)
            await self._fail_active_prompt(error_text or "Codex app-server error")

    async def _handle_codex_server_request(
        self,
        request_id: str,
        method: str,
        params: Any,
    ) -> None:
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
        }:
            result: dict[str, Any] = {"decision": "accept"}
        else:
            await self._send_codex_response(
                {
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unsupported Codex request: {method}"},
                }
            )
            return
        await self._send_codex_response({"id": request_id, "result": result})

    async def _send_codex_response(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()

    async def _emit_token_usage(self, params: dict[str, Any]) -> None:
        usage = params.get("tokenUsage")
        if not isinstance(usage, dict):
            return
        total = usage.get("total")
        if not isinstance(total, dict):
            return
        context_window = usage.get("modelContextWindow")
        context_tokens = total.get("inputTokens")
        context_usage: float | None = None
        if (
            isinstance(context_tokens, int)
            and isinstance(context_window, int)
            and context_window > 0
        ):
            context_usage = context_tokens / context_window
        await self._emit_wire_message(
            StatusUpdate(
                context_usage=context_usage,
                context_tokens=context_tokens if isinstance(context_tokens, int) else None,
                max_context_tokens=context_window if isinstance(context_window, int) else None,
            )
        )

    async def _complete_active_prompt(self) -> None:
        prompt_id = self._active_prompt_id
        assistant_text = "".join(self._active_assistant_parts)
        if assistant_text:
            await self._append_context({"role": "assistant", "content": assistant_text})
        if prompt_id is not None:
            self._in_flight_prompt_ids.discard(prompt_id)
            await self._broadcast(dump_finished_response(prompt_id))
        self._active_prompt_id = None
        self._active_turn_id = None
        self._active_assistant_parts.clear()
        await self._emit_status("idle", reason="prompt_complete")

    async def _fail_active_prompt(self, message: str) -> None:
        prompt_id = self._active_prompt_id or str(uuid4())
        self.clear_in_flight()
        await self._broadcast(
            JSONRPCErrorResponse(
                id=prompt_id,
                error=JSONRPCErrorObject(code=ErrorCodes.INTERNAL_ERROR, message=message),
            ).model_dump_json()
        )
        await self._emit_status("error", reason="codex_error", detail=message)

    async def _emit_wire_message(self, message: WireMessage) -> None:
        wire_file = self._wire_file
        if wire_file is None:
            self._load_session_paths()
            wire_file = self._wire_file
        assert wire_file is not None
        await wire_file.append_message(message)
        await self._broadcast(dump_wire_event(message))
        if self._context_file is not None:
            await asyncio.to_thread(self._context_file.touch)

    async def _append_context(self, payload: dict[str, Any]) -> None:
        context_file = self._context_file
        if context_file is None:
            self._load_session_paths()
            context_file = self._context_file
        assert context_file is not None
        context_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        await asyncio.to_thread(_append_text, context_file, line)

    async def _collect_uploaded_files(self) -> tuple[list[ContentPart], list[dict[str, Any]]]:
        session = load_session_by_id(self.session_id)
        if session is None:
            return [], []
        uploads_dir = session.kimi_cli_session.dir / "uploads"
        if not uploads_dir.exists():
            return [], []

        sent_marker = uploads_dir / ".sent"
        if sent_marker.exists():
            try:
                already_sent = json.loads(sent_marker.read_text(encoding="utf-8"))
                if isinstance(already_sent, list):
                    self._sent_files.update(str(item) for item in already_sent)
            except Exception:
                pass

        files = [
            file
            for file in sorted(uploads_dir.iterdir(), key=lambda path: path.name)
            if file.is_file() and file.name != ".sent" and file.name not in self._sent_files
        ]
        if not files:
            return [], []

        wire_parts: list[ContentPart] = []
        codex_inputs: list[dict[str, Any]] = []
        lines = ["<uploaded_files>"]
        for idx, file in enumerate(files, start=1):
            lines.append(f"{idx}. {file}")
        lines.append("</uploaded_files>")
        wire_parts.append(TextPart(text="\n".join(lines) + "\n\n"))

        text_extensions = {
            ".txt",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".css",
            ".js",
            ".ts",
            ".py",
            ".sh",
            ".csv",
            ".log",
            ".rst",
            ".toml",
            ".ini",
        }

        for file in files:
            mime_type, _ = mimetypes.guess_type(file.name)
            mime_type = mime_type or "application/octet-stream"
            if mime_type.startswith("image/"):
                codex_inputs.append({"type": "localImage", "path": str(file)})
            elif file.suffix.lower() in text_extensions or mime_type.startswith("text/"):
                try:
                    content = file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    content = ""
                if content:
                    wire_parts.append(
                        TextPart(
                            text=(
                                f'<document path="{file}" content_type="{mime_type}">\n'
                                f"{content}\n</document>\n\n"
                            )
                        )
                    )
            self._sent_files.add(file.name)

        sent_marker.write_text(
            json.dumps(sorted(self._sent_files), ensure_ascii=False),
            encoding="utf-8",
        )
        return wire_parts, codex_inputs

    async def _build_user_inputs(
        self,
        user_input: str | list[ContentPart],
    ) -> tuple[str | list[ContentPart], list[dict[str, Any]], str]:
        wire_parts, codex_inputs = await self._collect_uploaded_files()
        text_chunks: list[str] = []
        for part in wire_parts:
            if isinstance(part, TextPart):
                text_chunks.append(part.text)

        if isinstance(user_input, str):
            if user_input != "KIMI_FILE_UPLOAD_WITHOUT_MESSAGE":
                wire_parts.append(TextPart(text=user_input))
                text_chunks.append(user_input)
        else:
            wire_parts.extend(user_input)
            for part in user_input:
                if isinstance(part, TextPart):
                    text_chunks.append(part.text)

        text = "\n".join(chunk for chunk in text_chunks if chunk)
        codex_input = []
        if text:
            codex_input.append({"type": "text", "text": text, "text_elements": []})
        codex_input.extend(codex_inputs)
        if not codex_input:
            codex_input.append({"type": "text", "text": "", "text_elements": []})
        if not wire_parts:
            wire_input = ""
        elif len(wire_parts) == 1 and isinstance(wire_parts[0], TextPart):
            wire_input = wire_parts[0].text
        else:
            wire_input = wire_parts
        return wire_input, codex_input, text

    async def _broadcast(self, message: str) -> None:
        disconnected: set[WebSocket] = set()
        async with self._ws_lock:
            websockets = list(self._websockets)
            to_send: list[WebSocket] = []
            for ws in websockets:
                buffer = self._replay_buffers.get(ws)
                if buffer is not None:
                    buffer.append(message)
                else:
                    to_send.append(ws)

        for ws in to_send:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(message)
                else:
                    disconnected.add(ws)
            except Exception:
                disconnected.add(ws)

        if disconnected:
            async with self._ws_lock:
                self._websockets -= disconnected
                self._websocket_count = len(self._websockets)
                for ws in disconnected:
                    self._replay_buffers.pop(ws, None)

    async def add_websocket_and_begin_replay(self, ws: WebSocket) -> None:
        async with self._ws_lock:
            if ws not in self._websockets:
                self._websockets.add(ws)
                self._websocket_count = len(self._websockets)
            self._replay_buffers.setdefault(ws, [])

    async def end_replay(self, ws: WebSocket) -> None:
        while True:
            async with self._ws_lock:
                buffer = self._replay_buffers.get(ws)
                if buffer is None:
                    return
                if not buffer:
                    self._replay_buffers.pop(ws, None)
                    return
                chunk = buffer.copy()
                buffer.clear()
            for message in chunk:
                try:
                    await ws.send_text(message)
                except Exception:
                    async with self._ws_lock:
                        self._replay_buffers.pop(ws, None)
                    return

    async def _close_all_websockets(self) -> None:
        async with self._ws_lock:
            websockets = list(self._websockets)
            self._websockets.clear()
            self._websocket_count = 0
            self._replay_buffers.clear()
        for ws in websockets:
            with contextlib.suppress(Exception):
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.close(code=1001, reason="Session process exited")

    async def remove_websocket(self, ws: WebSocket) -> None:
        async with self._ws_lock:
            self._websockets.discard(ws)
            self._websocket_count = len(self._websockets)
            self._replay_buffers.pop(ws, None)

    async def send_message(self, message: str) -> None:
        await self.start()
        try:
            in_message = JSONRPCInMessageAdapter.validate_json(message)
        except ValueError as exc:
            logger.warning("Invalid GUI JSON-RPC message for Codex runner: {error}", error=exc)
            return

        match in_message:
            case JSONRPCInitializeMessage():
                await self._broadcast(
                    dump_success_response(
                        in_message.id,
                        {
                            "slash_commands": [
                                {
                                    "name": "compact",
                                    "description": "Compact context",
                                    "aliases": [],
                                },
                                {
                                    "name": "clear",
                                    "description": "Clear the visible chat",
                                    "aliases": [],
                                },
                            ]
                        },
                    )
                )
            case JSONRPCSetPlanModeMessage():
                self._plan_mode = in_message.params.enabled
                await self._emit_wire_message(StatusUpdate(plan_mode=self._plan_mode))
                await self._broadcast(dump_success_response(in_message.id))
            case JSONRPCSetDreamModeMessage():
                self._dream_mode = in_message.params.enabled
                await self._emit_wire_message(StatusUpdate(dream_mode=self._dream_mode))
                await self._broadcast(dump_success_response(in_message.id))
            case JSONRPCPromptMessage():
                await self._send_prompt(in_message)
            case JSONRPCSteerMessage():
                await self._send_steer(in_message)
            case JSONRPCCancelMessage():
                await self._cancel_turn(in_message)
            case JSONRPCSuccessResponse() | JSONRPCErrorResponse():
                await self._broadcast(dump_success_response(in_message.id))
            case _:
                fallback_id = getattr(in_message, "id", str(uuid4()))
                await self._broadcast(dump_success_response(fallback_id))

    async def _send_prompt(self, message: JSONRPCPromptMessage) -> None:
        if self.is_busy:
            await self._broadcast(
                JSONRPCErrorResponse(
                    id=message.id,
                    error=JSONRPCErrorObject(
                        code=ErrorCodes.INVALID_STATE,
                        message="Session is busy; wait for completion before sending a new prompt.",
                    ),
                ).model_dump_json()
            )
            return

        wire_input, codex_input, context_text = await self._build_user_inputs(
            message.params.user_input
        )
        self._in_flight_prompt_ids.add(message.id)
        self._active_prompt_id = message.id
        self._active_assistant_parts.clear()
        await self._emit_status("busy", reason="prompt")
        await self._emit_wire_message(TurnBegin(user_input=wire_input))
        if context_text:
            await self._append_context({"role": "user", "content": context_text})

        assert self._thread_id is not None
        params: dict[str, Any] = {
            "threadId": self._thread_id,
            "input": codex_input,
            "approvalPolicy": os.environ.get(ENV_CODEX_APPROVAL_POLICY, "never"),
        }
        response = await self._send_codex_request("turn/start", params)
        if "error" in response:
            error = response["error"].get("message", "Codex turn/start failed")
            await self._fail_active_prompt(error)

    async def _send_steer(self, message: JSONRPCSteerMessage) -> None:
        assert self._thread_id is not None
        _, codex_input, _ = await self._build_user_inputs(message.params.user_input)
        response = await self._send_codex_request(
            "turn/steer",
            {"threadId": self._thread_id, "input": codex_input},
        )
        if "error" in response:
            await self._broadcast(
                JSONRPCErrorResponse(
                    id=message.id,
                    error=JSONRPCErrorObject(
                        code=ErrorCodes.INTERNAL_ERROR,
                        message=response["error"].get("message", "Codex turn/steer failed"),
                    ),
                ).model_dump_json()
            )
        else:
            await self._broadcast(dump_success_response(message.id))

    async def _cancel_turn(self, message: JSONRPCCancelMessage) -> None:
        if self._thread_id is not None and self._active_turn_id is not None:
            await self._send_codex_request(
                "turn/interrupt",
                {"threadId": self._thread_id, "turnId": self._active_turn_id},
            )
        self.clear_in_flight()
        await self._broadcast(dump_success_response(message.id, {"status": "cancelled"}))
        await self._emit_status("idle", reason="cancelled")


class CodexCLIRunner:
    """Manage multiple Codex-backed GUI session processes."""

    def __init__(self) -> None:
        self._sessions: dict[UUID, CodexSessionProcess] = {}
        self._lock = asyncio.Lock()

    def start(self) -> None:
        pass

    async def stop(self) -> None:
        tasks = [asyncio.create_task(session.stop()) for session in self._sessions.values()]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=5.0)
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def get_or_create_session(self, session_id: UUID) -> CodexSessionProcess:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = CodexSessionProcess(session_id)
            return self._sessions[session_id]

    def get_session(self, session_id: UUID) -> CodexSessionProcess | None:
        return self._sessions.get(session_id)

    async def detach_websocket(self, ws: WebSocket, session_id: UUID) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                await session.remove_websocket(ws)

    async def restart_running_workers(self, *, reason: str, force: bool) -> RestartWorkersSummary:
        async with self._lock:
            running = [(sid, proc) for sid, proc in self._sessions.items() if proc.is_running]

        restarted: list[UUID] = []
        skipped_busy: list[UUID] = []
        tasks: list[asyncio.Task[None]] = []
        for session_id, proc in running:
            if proc.is_busy and not force:
                skipped_busy.append(session_id)
                continue
            restarted.append(session_id)
            tasks.append(asyncio.create_task(proc.restart_worker(reason=reason)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return RestartWorkersSummary(restarted, skipped_busy)


@dataclass(slots=True)
class RestartWorkersSummary:
    restarted_session_ids: list[UUID]
    skipped_busy_session_ids: list[UUID]


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
