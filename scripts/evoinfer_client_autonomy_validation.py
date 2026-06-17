#!/usr/bin/env python3
"""Run client autonomy checks against EvoInfer Dream MCP.

The user prompt intentionally does not mention EvoInfer, Dream, or MCP. The
Dream protocol is supplied only through a dedicated mandatory Dream session
bundle. Passing this check means the client/agent followed that session protocol
and called Dream tools on its own.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from evoinfer_mcp.cli import build_evoinfer_force_session_bundle

CLIENTS = ("claude", "codex", "kimi")
PROMPT = (
    "Analyze TASK.md and the local benchmark/correctness/profiler artifacts. "
    "Create final_report.md with the recommended route, correctness status, "
    "and the evidence files you used."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp/evoinfer-client-autonomy-validation"))
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--client", choices=CLIENTS, action="append")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = tuple(args.client or CLIENTS)

    results: dict[str, Any] = {}
    for client in selected:
        case = prepare_case(root, client, python=args.python)
        if shutil.which(client) is None:
            results[client] = {
                "ok": False,
                "skipped": True,
                "reason": f"{client} command not found",
                **case_paths(case),
            }
            continue
        result = run_client(client, case, timeout_seconds=args.timeout_seconds)
        results[client] = result

    summary = {
        "root": str(root),
        "prompt": PROMPT,
        "results": results,
    }
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def prepare_case(root: Path, client: str, *, python: Path) -> dict[str, Path]:
    case_root = root / client
    share_dir = case_root / "share"
    workdir = case_root / "work"
    call_log = case_root / "mcp_calls.jsonl"
    mcp_config = case_root / "mcp.json"
    stdout_log = case_root / "stdout.jsonl"
    stderr_log = case_root / "stderr.log"
    final_log = case_root / "final.txt"

    if case_root.exists():
        shutil.rmtree(case_root)
    bundle = build_evoinfer_force_session_bundle(
        session_dir=case_root,
        share_dir=share_dir,
        workdir=workdir,
        command=str(python),
    )
    mcp_config = Path(str(bundle["mcp_config_path"]))
    call_log = Path(str(bundle["call_log_path"]))
    for stale in (call_log, stdout_log, stderr_log, final_log):
        if stale.exists():
            stale.unlink()

    seed_memory_store(share_dir)
    write_task_file(workdir)
    write_artifacts(workdir)
    return {
        "case_root": case_root,
        "share_dir": share_dir,
        "workdir": workdir,
        "call_log": call_log,
        "mcp_config": mcp_config,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "final_log": final_log,
    }


def seed_memory_store(share_dir: Path) -> None:
    write_json(
        share_dir / "dream" / "memories.json",
        {
            "version": 1,
            "memories": [
                {
                    "id": "opt_autonomy_cuda_softmax_float32_shared_reduce",
                    "category": "optimization",
                    "title": "CUDA row-wise softmax should use shared reduction for float32 rows",
                    "summary": (
                        "For RTX 3090 float32 row-wise softmax with seq=1024, a shared-memory "
                        "block reduction was faster than naive global-memory reduction."
                    ),
                    "tags": ["cuda", "softmax", "row-wise", "float32", "rtx3090"],
                    "environment": "RTX 3090, CUDA 12.x",
                    "model_type": "operator-kernel",
                    "model_arch": "row-wise softmax",
                    "inference_backend": "cuda",
                    "success": True,
                    "precision": {"dtype": "float32"},
                    "workload": {"batch": 64, "seq": 1024},
                    "metrics_before": {"latency_ms": 0.42},
                    "metrics_after": {"latency_ms": 0.19},
                    "objective_metric": "latency_ms",
                    "detail_description": (
                        "Use one block per row, subtract row max, exponentiate, reduce sum in "
                        "shared memory, and validate against torch.softmax."
                    ),
                    "artifacts": ["seed/baseline.json", "seed/candidate.json"],
                    "correctness_artifacts": ["seed/correctness.json"],
                    "profiler_artifacts": ["seed/profiler_summary.json"],
                    "status": "promoted",
                    "evidence_level": "verified",
                    "chosen": 0,
                    "useful_when_chosen": 0,
                }
            ],
        },
    )


def write_task_file(workdir: Path) -> None:
    (workdir / "TASK.md").write_text(
        """# Local Optimization Task

Hardware: RTX 3090
Backend: CUDA
Operator: row-wise softmax
DType: float32
Workload: batch=64, seq=1024

The current candidate already has standard artifacts in this directory. Review
the evidence and produce final_report.md with a concise recommendation.
""",
        encoding="utf-8",
    )


def write_artifacts(workdir: Path) -> None:
    write_json(
        workdir / "environment.json",
        {
            "gpu_name": "RTX 3090",
            "cuda_version": "12.4",
            "driver_version": "550.xx",
            "backend": "cuda",
            "model_type": "operator-kernel",
        },
    )
    write_json(
        workdir / "benchmark_raw.json",
        {
            "operator": "row-wise softmax",
            "backend": "cuda",
            "dtype": "float32",
            "workload": {"batch": 64, "seq": 1024},
            "baseline": {"latency_ms": 0.42},
            "candidate": {"latency_ms": 0.19},
            "unit": "ms",
        },
    )
    write_json(
        workdir / "correctness_raw.json",
        {
            "passed": True,
            "reference": "torch.softmax",
            "max_abs_error": 0.000001,
            "tolerance": 0.00001,
        },
    )
    write_json(
        workdir / "profiler_summary.json",
        {
            "bottleneck_type": "memory_bandwidth",
            "hotspots": [
                {
                    "kernel": "softmax_shared_reduce",
                    "latency_ms": 0.19,
                    "dram_pct": 61.0,
                }
            ],
        },
    )
    write_json(
        workdir / "verifier_result.json",
        {
            "status": "passed",
            "command": "python verify_softmax.py --dtype float32 --batch 64 --seq 1024",
        },
    )
    (workdir / "agent_trace.md").write_text(
        "Validated the existing row-wise softmax candidate from local artifacts.\n",
        encoding="utf-8",
    )


def run_client(client: str, case: dict[str, Path], *, timeout_seconds: float) -> dict[str, Any]:
    command = client_command(client, case)
    started = time.monotonic()
    stdout_log = case["stdout_log"]
    stderr_log = case["stderr_log"]
    exit_code: int | None = None
    timed_out = False
    exception: str | None = None
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=case["workdir"],
                text=True,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exception = str(exc)
            exit_code = 124
    duration_s = time.monotonic() - started
    calls = read_call_log(case["call_log"])
    memories = read_memory_store(case["share_dir"])
    tools = [str(call.get("tool")) for call in calls]
    protocol_called = "dream_get_agent_protocol" in tools
    search_called = bool({"dream_search_memories", "search_dream_memories"} & set(tools))
    write_called = bool(
        {
            "dream_extract_and_write_memories",
            "dream_write_optimization_memory",
            "dream_write_environment_debug_memory",
            "dream_stage_memory_candidate",
        }
        & set(tools)
    )
    final_report_exists = (case["workdir"] / "final_report.md").is_file()
    ok = exit_code == 0 and protocol_called and search_called and write_called
    return {
        "ok": ok,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "exception": exception,
        "duration_s": round(duration_s, 3),
        "protocol_called": protocol_called,
        "search_called": search_called,
        "write_called": write_called,
        "tool_calls": tools,
        "tool_call_count": len(tools),
        "memory_count": len(memories),
        "memory_ids": [memory.get("id") for memory in memories if isinstance(memory, dict)],
        "final_report_exists": final_report_exists,
        **case_paths(case),
        "command": redact_command(command),
    }


def client_command(client: str, case: dict[str, Path]) -> list[str]:
    if client == "claude":
        return [
            "claude",
            "-p",
            PROMPT,
            "--output-format",
            "stream-json",
            "--verbose",
            "--mcp-config",
            str(case["mcp_config"]),
            "--strict-mcp-config",
            "--permission-mode",
            "bypassPermissions",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
        ]
    if client == "kimi":
        return [
            "kimi",
            "--work-dir",
            str(case["workdir"]),
            "--mcp-config-file",
            str(case["mcp_config"]),
            "--print",
            "--output-format",
            "stream-json",
            "--yolo",
            "--max-steps-per-turn",
            "40",
            "--prompt",
            PROMPT,
        ]
    if client == "codex":
        server = json.loads(case["mcp_config"].read_text(encoding="utf-8"))["mcpServers"][
            "evoinfer-dream"
        ]
        command = [
            "codex",
            "exec",
            "--json",
            "--cd",
            str(case["workdir"]),
            "--skip-git-repo-check",
            "-s",
            "danger-full-access",
            "-c",
            f"mcp_servers.evoinfer-dream.command={toml_string(server['command'])}",
            "-c",
            'mcp_servers.evoinfer-dream.args=["-m","evoinfer_mcp.dream.mcp_server"]',
        ]
        env = server.get("env", {})
        if isinstance(env, dict):
            for key, value in sorted(env.items()):
                command.extend(
                    [
                        "-c",
                        f"mcp_servers.evoinfer-dream.env.{key}={toml_string(str(value))}",
                    ]
                )
        command.append(PROMPT)
        return command
    raise ValueError(f"unsupported client: {client}")


def read_call_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    calls: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            calls.append(payload)
    return calls


def read_memory_store(share_dir: Path) -> list[dict[str, Any]]:
    path = share_dir / "dream" / "memories.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    memories = payload.get("memories", payload)
    if not isinstance(memories, list):
        return []
    return [memory for memory in memories if isinstance(memory, dict)]


def case_paths(case: dict[str, Path]) -> dict[str, str]:
    return {
        "case_root": str(case["case_root"]),
        "workdir": str(case["workdir"]),
        "share_dir": str(case["share_dir"]),
        "call_log": str(case["call_log"]),
        "stdout_log": str(case["stdout_log"]),
        "stderr_log": str(case["stderr_log"]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def toml_string(value: str) -> str:
    return json.dumps(value)


def redact_command(command: list[str]) -> list[str]:
    return [part if not part.startswith("sk-") else "<redacted>" for part in command]


if __name__ == "__main__":
    main()
