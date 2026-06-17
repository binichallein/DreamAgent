from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

EvoInferHookClient = Literal["codex", "claude"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", choices=["codex", "claude"], required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--share-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--every-steps", type=int, required=True)
    args = parser.parse_args()

    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        event = {"hook_event_name": "unknown", "raw_stdin": raw[:2000]}

    output = run_dream_checkpoint_hook(
        event,
        client=args.client,
        every_steps=args.every_steps,
        session_dir=args.session_dir,
        share_dir=args.share_dir,
        state_file=args.state_file,
        context_file=args.context_file,
    )
    if output is not None:
        sys.stdout.write(json.dumps(output, ensure_ascii=False))
        sys.stdout.write("\n")


def run_dream_checkpoint_hook(
    event: dict[str, Any],
    *,
    client: EvoInferHookClient,
    every_steps: int,
    session_dir: Path,
    share_dir: Path,
    state_file: Path,
    context_file: Path,
) -> dict[str, Any] | None:
    """Run one Dream checkpoint hook event.

    Tool events increment a stable checkpoint counter. SessionStart and Stop
    return context immediately; tool checkpoints only inject context every N
    events to avoid noisy agent UX.
    """

    if every_steps < 1:
        raise ValueError("every_steps must be >= 1")

    event_name = str(event.get("hook_event_name") or "")
    session_dir = session_dir.expanduser().resolve()
    share_dir = share_dir.expanduser().resolve()
    state_file = state_file.expanduser().resolve()
    context_file = context_file.expanduser().resolve()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    context_file.parent.mkdir(parents=True, exist_ok=True)

    state = _read_state(state_file)
    should_inject = event_name == "SessionStart"
    trigger = "session_start"

    if _is_tool_checkpoint_event(event_name):
        state["tool_checkpoint_count"] = int(state.get("tool_checkpoint_count", 0)) + 1
        trigger = "periodic_tool_checkpoint"
        should_inject = state["tool_checkpoint_count"] % every_steps == 0
    elif event_name == "Stop":
        trigger = "stop"
        should_inject = True

    _write_json(state_file, state)
    if not should_inject:
        return None

    context = _build_context(
        event,
        trigger=trigger,
        share_dir=share_dir,
        context_file=context_file,
    )
    _append_context(context_file, context)
    return _hook_output(client=client, event_name=event_name, context=context)


def _build_context(
    event: dict[str, Any],
    *,
    trigger: str,
    share_dir: Path,
    context_file: Path,
) -> str:
    cwd = Path(str(event.get("cwd") or ".")).expanduser()
    query = _build_search_query(event, cwd=cwd, trigger=trigger)
    now = datetime.now(UTC).isoformat()
    sections = [
        f"## Dream checkpoint ({trigger})",
        f"- time: {now}",
        f"- cwd: {cwd}",
        f"- query: {query[:1000]}",
    ]

    search_text = _run_dream_search(query=query, share_dir=share_dir, cwd=cwd)
    if search_text:
        sections.extend(["", "### Retrieved Dream memories", search_text])
    else:
        sections.extend(["", "### Retrieved Dream memories", "No matching Dream memory found."])

    if trigger == "stop":
        write_text = _run_completion_extraction(share_dir=share_dir, cwd=cwd)
        if write_text:
            sections.extend(["", "### Completion memory extraction", write_text])

    sections.extend(
        [
            "",
            "Agent instruction:",
            (
                f"Read `{context_file}` as EvoInfer Dream context. Re-check route choices "
                "against retrieved positive and negative memories before continuing."
            ),
        ]
    )
    return "\n".join(sections).strip() + "\n"


def _run_dream_search(*, query: str, share_dir: Path, cwd: Path) -> str:
    from evoinfer_mcp.dream.mcp_server import dream_search_memories_tool

    with _temporary_env("EVOINFER_SHARE_DIR", str(share_dir)):
        try:
            return dream_search_memories_tool(
                query=query,
                category=None,
                tags=[],
                top_k=5,
                record_choice=True,
                render_mode="artifact_protocol",
                task_context=f"Hook cwd: {cwd}",
            )
        except Exception as exc:
            return f"Dream search failed: {type(exc).__name__}: {exc}"


def _run_completion_extraction(*, share_dir: Path, cwd: Path) -> str:
    from evoinfer_mcp.dream.mcp_server import dream_extract_and_write_memories_tool

    if not any((cwd / name).exists() for name in _ARTIFACT_FILES):
        return ""
    with _temporary_env("EVOINFER_SHARE_DIR", str(share_dir)):
        try:
            return dream_extract_and_write_memories_tool(workdir=str(cwd), dry_run=False)
        except Exception as exc:
            return f"Dream extraction failed: {type(exc).__name__}: {exc}"


def _build_search_query(event: dict[str, Any], *, cwd: Path, trigger: str) -> str:
    task_text = _read_small(cwd / "TASK.md", limit=3000)
    artifact_names = [name for name in _ARTIFACT_FILES if (cwd / name).exists()]
    compact_event = {
        "trigger": trigger,
        "hook_event_name": event.get("hook_event_name"),
        "tool_name": event.get("tool_name"),
        "tool_input": event.get("tool_input"),
        "tool_calls": _compact_tool_calls(event.get("tool_calls")),
        "artifact_files": artifact_names,
    }
    return "\n".join(
        part
        for part in [
            "EvoInfer inference optimization or environment-debug checkpoint.",
            f"Task file:\n{task_text}" if task_text else "",
            f"Hook event:\n{json.dumps(compact_event, ensure_ascii=False, default=str)[:4000]}",
        ]
        if part
    )


def _compact_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value[:10]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "tool_name": item.get("tool_name"),
                "tool_input": item.get("tool_input"),
            }
        )
    return compact


def _hook_output(
    *,
    client: EvoInferHookClient,
    event_name: str,
    context: str,
) -> dict[str, Any]:
    hook_event_name = event_name or ("PostToolUse" if client == "codex" else "PostToolBatch")
    return {
        "hookSpecificOutput": {
            "hookEventName": hook_event_name,
            "additionalContext": context,
        }
    }


def _is_tool_checkpoint_event(event_name: str) -> bool:
    return event_name in {"PostToolUse", "PostToolBatch", "PostToolUseFailure"}


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"tool_checkpoint_count": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"tool_checkpoint_count": 0}
    return payload if isinstance(payload, dict) else {"tool_checkpoint_count": 0}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_context(path: Path, context: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(context)


def _read_small(path: Path, *, limit: int) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:limit]


@contextmanager
def _temporary_env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


_ARTIFACT_FILES = (
    "benchmark_raw.json",
    "correctness_raw.json",
    "profiler_summary.json",
    "environment.json",
    "verifier_result.json",
    "dream_write_candidates.json",
)


if __name__ == "__main__":
    main()
