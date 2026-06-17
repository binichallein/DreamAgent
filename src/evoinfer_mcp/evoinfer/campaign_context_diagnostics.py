"""Offline context diagnostics for EvoInfer campaign runs.

This module intentionally observes existing run JSON and artifacts only. It does
not change campaign execution, prompts, verifier behavior, or model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evoinfer_mcp.evoinfer.campaign import SessionRunResult
from evoinfer_mcp.evoinfer.campaign_analysis import _load_campaign_result


class RunContextDiagnostics(BaseModel):
    """Context-related observations for one campaign arm run."""

    arm_name: str
    session_id: str
    work_dir: str
    duration_seconds: float
    context_tokens: int | None = None
    input_prompt_chars: int = 0
    input_prompt_line_count: int = 0
    initial_dream_retrieval_chars: int = 0
    assistant_text_chars: int = 0
    verification_output_chars: int = 0
    tool_call_count: int = 0
    failed_tool_call_count: int = 0
    shell_call_count: int = 0
    shell_failure_count: int = 0
    benchmark_attempt_count: int = 0
    benchmark_failure_count: int = 0
    wire_event_count: int = 0
    wire_event_json_bytes: int = 0
    wire_event_payload_json_bytes: int = 0
    wire_event_text_chars: int = 0
    wire_event_think_chars: int = 0
    wire_event_tool_argument_chars: int = 0
    wire_event_tool_output_chars: int = 0
    artifact_file_count: int = 0
    artifact_total_bytes: int = 0
    agent_trace_bytes: int = 0
    route_decision_bytes: int = 0
    route_policy_audit_bytes: int = 0
    controller_route_decision_audit_bytes: int = 0
    controller_route_decision_initial_bytes: int = 0
    controller_route_decision_changed: bool | None = None
    selected_dtype_count: int = 0
    audit_dtype_count: int = 0
    avoid_dtype_count: int = 0
    candidate_unit_count: int = 0


class ArmContextDiagnostics(BaseModel):
    """Averaged context diagnostics for one arm."""

    arm_name: str
    run_count: int
    average_duration_seconds: float | None = None
    average_context_tokens: float | None = None
    average_input_prompt_chars: float | None = None
    average_initial_dream_retrieval_chars: float | None = None
    average_assistant_text_chars: float | None = None
    average_tool_call_count: float | None = None
    average_failed_tool_call_count: float | None = None
    average_shell_call_count: float | None = None
    average_benchmark_attempt_count: float | None = None
    average_wire_event_count: float | None = None
    average_wire_event_json_bytes: float | None = None
    average_wire_event_payload_json_bytes: float | None = None
    average_wire_event_text_chars: float | None = None
    average_wire_event_think_chars: float | None = None
    average_wire_event_tool_argument_chars: float | None = None
    average_wire_event_tool_output_chars: float | None = None
    average_artifact_total_bytes: float | None = None
    average_agent_trace_bytes: float | None = None
    average_route_decision_bytes: float | None = None
    average_controller_route_decision_changed_rate: float | None = None
    average_candidate_unit_count: float | None = None
    context_token_delta_vs_baseline: float | None = None
    duration_delta_vs_baseline_seconds: float | None = None
    prompt_char_delta_vs_baseline: float | None = None
    retrieval_char_delta_vs_baseline: float | None = None
    tool_call_delta_vs_baseline: float | None = None
    failed_tool_call_delta_vs_baseline: float | None = None
    wire_event_json_bytes_delta_vs_baseline: float | None = None
    wire_event_tool_output_chars_delta_vs_baseline: float | None = None
    candidate_unit_delta_vs_baseline: float | None = None


class CampaignContextDiagnostics(BaseModel):
    """Offline diagnostic report across campaign result files."""

    campaign_count: int
    run_count: int
    baseline_arm_name: str = "without_memory"
    source_files: list[str] = Field(default_factory=list)
    limitation: str = (
        "This report uses run JSON counters and artifact file sizes. It is not "
        "a tokenizer-level or per-turn wire-event attribution."
    )
    runs: list[RunContextDiagnostics] = Field(default_factory=list)
    arms: dict[str, ArmContextDiagnostics] = Field(default_factory=dict)


def analyze_campaign_context(
    paths: list[Path] | tuple[Path, ...],
    *,
    baseline_arm_name: str = "without_memory",
) -> CampaignContextDiagnostics:
    """Build an offline context-diagnostic report from campaign result JSON files."""

    run_reports: list[RunContextDiagnostics] = []
    for path in paths:
        result = _load_campaign_result(path)
        run_reports.extend(
            _diagnose_run(run, source_path=path, campaign_name=result.name)
            for run in result.runs
        )

    arms = _aggregate_arms(run_reports, baseline_arm_name=baseline_arm_name)
    return CampaignContextDiagnostics(
        campaign_count=len(paths),
        run_count=len(run_reports),
        baseline_arm_name=baseline_arm_name,
        source_files=[str(path) for path in paths],
        runs=run_reports,
        arms=arms,
    )


def render_context_diagnostics_markdown(report: CampaignContextDiagnostics) -> str:
    """Render context diagnostics as a markdown report."""

    lines = [
        "# EvoInfer Campaign Context Diagnostics",
        "",
        f"- Campaign files: {report.campaign_count}",
        f"- Runs: {report.run_count}",
        f"- Baseline arm: `{report.baseline_arm_name}`",
        f"- Limitation: {report.limitation}",
        "",
        "## Arm Summary",
        "",
        (
            "| arm | runs | avg_context_tokens | context_delta | avg_duration_s | "
            "duration_delta_s | prompt_chars | retrieval_chars | tool_calls | "
            "failed_tools | benchmarks | route_dtype_units | artifact_bytes | "
            "route_changed | agent_trace_bytes | wire_events | wire_json_bytes | "
            "wire_tool_output_chars |"
        ),
        _markdown_separator(18),
    ]
    for arm in report.arms.values():
        lines.append(
            "| "
            f"{arm.arm_name} | "
            f"{arm.run_count} | "
            f"{_fmt_float(arm.average_context_tokens, 1)} | "
            f"{_fmt_float(arm.context_token_delta_vs_baseline, 1)} | "
            f"{_fmt_float(arm.average_duration_seconds, 3)} | "
            f"{_fmt_float(arm.duration_delta_vs_baseline_seconds, 3)} | "
            f"{_fmt_float(arm.average_input_prompt_chars, 1)} | "
            f"{_fmt_float(arm.average_initial_dream_retrieval_chars, 1)} | "
            f"{_fmt_float(arm.average_tool_call_count, 2)} | "
            f"{_fmt_float(arm.average_failed_tool_call_count, 2)} | "
            f"{_fmt_float(arm.average_benchmark_attempt_count, 2)} | "
            f"{_fmt_float(arm.average_candidate_unit_count, 2)} | "
            f"{_fmt_float(arm.average_artifact_total_bytes, 1)} | "
            f"{_fmt_float(arm.average_controller_route_decision_changed_rate, 2)} | "
            f"{_fmt_float(arm.average_agent_trace_bytes, 1)} | "
            f"{_fmt_float(arm.average_wire_event_count, 2)} | "
            f"{_fmt_float(arm.average_wire_event_json_bytes, 1)} | "
            f"{_fmt_float(arm.average_wire_event_tool_output_chars, 1)} |"
        )

    lines.extend(["", "## Run-Level Observations", ""])
    lines.extend(
        [
            (
                "| arm | context_tokens | duration_s | prompt_chars | retrieval_chars | "
                "tool_calls | failed_tools | benchmarks | route_dtype_units | "
                "route_decision_bytes | route_changed | agent_trace_bytes | "
                "wire_json_bytes | wire_tool_output_chars |"
            ),
            _markdown_separator(14),
        ]
    )
    for run in report.runs:
        lines.append(
            "| "
            f"{run.arm_name} | "
            f"{run.context_tokens if run.context_tokens is not None else ''} | "
            f"{run.duration_seconds:.3f} | "
            f"{run.input_prompt_chars} | "
            f"{run.initial_dream_retrieval_chars} | "
            f"{run.tool_call_count} | "
            f"{run.failed_tool_call_count} | "
            f"{run.benchmark_attempt_count} | "
            f"{run.candidate_unit_count} | "
            f"{run.route_decision_bytes} | "
            f"{_fmt_bool(run.controller_route_decision_changed)} | "
            f"{run.agent_trace_bytes} | "
            f"{run.wire_event_json_bytes} | "
            f"{run.wire_event_tool_output_chars} |"
        )

    lines.extend(["", "## Reading Guide", ""])
    lines.append(
        "- `prompt_chars` and `retrieval_chars` measure initial input growth; they do "
        "not include later tool outputs or model reasoning."
    )
    lines.append(
        "- `route_dtype_units` is `selected + audit + avoid` dtypes from "
        "`route_decision.json`; it is a route-breadth proxy, not a kernel metric."
    )
    lines.append(
        "- `wire_*` fields are populated only for campaigns run after event-size "
        "summary capture was added; older run JSONs show zero for those columns."
    )
    lines.append(
        "- If context rises while route dtype units stay flat or fall, the next "
        "experiment should inspect per-turn wire events or reduce interaction "
        "loops instead of only compressing the initial memory text."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Campaign JSON result files.")
    parser.add_argument("--baseline-arm", default="without_memory")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    report = analyze_campaign_context(args.paths, baseline_arm_name=args.baseline_arm)
    markdown = render_context_diagnostics_markdown(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


def _diagnose_run(
    run: SessionRunResult,
    *,
    source_path: Path,
    campaign_name: str,
) -> RunContextDiagnostics:
    work_dir = _resolve_artifact_work_dir(
        run.work_dir,
        source_path=source_path,
        campaign_name=campaign_name,
    )
    artifact_sizes = _artifact_sizes(work_dir)
    route_decision = _load_json_object(work_dir / "route_decision.json")
    route_decision_path = work_dir / "route_decision.json"
    controller_audit = _load_json_object(work_dir / "controller_route_decision_audit.json")
    selected_count = _list_count(route_decision.get("selected_dtypes")) if route_decision else 0
    audit_count = _list_count(route_decision.get("audit_dtypes")) if route_decision else 0
    avoid_count = _list_count(route_decision.get("avoid_dtypes")) if route_decision else 0
    controller_route_changed = _controller_route_decision_changed(
        route_decision_path,
        controller_audit,
    )
    wire_sizes = _sum_wire_event_size_summaries(run)
    return RunContextDiagnostics(
        arm_name=run.arm_name,
        session_id=run.session_id,
        work_dir=str(work_dir),
        duration_seconds=run.duration_seconds,
        context_tokens=run.context_tokens,
        input_prompt_chars=run.input_prompt_chars,
        input_prompt_line_count=run.input_prompt_line_count,
        initial_dream_retrieval_chars=run.initial_dream_retrieval_chars,
        assistant_text_chars=len(run.assistant_text),
        verification_output_chars=len(run.verification_output),
        tool_call_count=run.tool_call_count,
        failed_tool_call_count=run.failed_tool_call_count,
        shell_call_count=run.shell_call_count,
        shell_failure_count=run.shell_failure_count,
        benchmark_attempt_count=run.benchmark_attempt_count,
        benchmark_failure_count=run.benchmark_failure_count,
        wire_event_count=wire_sizes["count"],
        wire_event_json_bytes=wire_sizes["json_bytes"],
        wire_event_payload_json_bytes=wire_sizes["payload_json_bytes"],
        wire_event_text_chars=wire_sizes["text_chars"],
        wire_event_think_chars=wire_sizes["think_chars"],
        wire_event_tool_argument_chars=wire_sizes["tool_argument_chars"],
        wire_event_tool_output_chars=wire_sizes["tool_output_chars"],
        artifact_file_count=artifact_sizes["file_count"],
        artifact_total_bytes=artifact_sizes["total_bytes"],
        agent_trace_bytes=_file_size(work_dir / "agent_trace.md"),
        route_decision_bytes=_file_size(work_dir / "route_decision.json"),
        route_policy_audit_bytes=_file_size(work_dir / "route_policy_audit.json"),
        controller_route_decision_audit_bytes=_file_size(
            work_dir / "controller_route_decision_audit.json"
        ),
        controller_route_decision_initial_bytes=_controller_route_decision_initial_bytes(
            controller_audit
        ),
        controller_route_decision_changed=controller_route_changed,
        selected_dtype_count=selected_count,
        audit_dtype_count=audit_count,
        avoid_dtype_count=avoid_count,
        candidate_unit_count=selected_count + audit_count + avoid_count,
    )


def _aggregate_arms(
    runs: list[RunContextDiagnostics],
    *,
    baseline_arm_name: str,
) -> dict[str, ArmContextDiagnostics]:
    by_arm: dict[str, list[RunContextDiagnostics]] = defaultdict(list)
    for run in runs:
        by_arm[run.arm_name].append(run)

    arms = {
        arm_name: _aggregate_arm(arm_name, arm_runs)
        for arm_name, arm_runs in sorted(by_arm.items())
    }
    baseline = arms.get(baseline_arm_name)
    if baseline is not None:
        for arm in arms.values():
            arm.context_token_delta_vs_baseline = _optional_delta(
                arm.average_context_tokens,
                baseline.average_context_tokens,
            )
            arm.duration_delta_vs_baseline_seconds = _optional_delta(
                arm.average_duration_seconds,
                baseline.average_duration_seconds,
            )
            arm.prompt_char_delta_vs_baseline = _optional_delta(
                arm.average_input_prompt_chars,
                baseline.average_input_prompt_chars,
            )
            arm.retrieval_char_delta_vs_baseline = _optional_delta(
                arm.average_initial_dream_retrieval_chars,
                baseline.average_initial_dream_retrieval_chars,
            )
            arm.tool_call_delta_vs_baseline = _optional_delta(
                arm.average_tool_call_count,
                baseline.average_tool_call_count,
            )
            arm.failed_tool_call_delta_vs_baseline = _optional_delta(
                arm.average_failed_tool_call_count,
                baseline.average_failed_tool_call_count,
            )
            arm.wire_event_json_bytes_delta_vs_baseline = _optional_delta(
                arm.average_wire_event_json_bytes,
                baseline.average_wire_event_json_bytes,
            )
            arm.wire_event_tool_output_chars_delta_vs_baseline = _optional_delta(
                arm.average_wire_event_tool_output_chars,
                baseline.average_wire_event_tool_output_chars,
            )
            arm.candidate_unit_delta_vs_baseline = _optional_delta(
                arm.average_candidate_unit_count,
                baseline.average_candidate_unit_count,
            )
    return arms


def _aggregate_arm(
    arm_name: str,
    runs: list[RunContextDiagnostics],
) -> ArmContextDiagnostics:
    return ArmContextDiagnostics(
        arm_name=arm_name,
        run_count=len(runs),
        average_duration_seconds=_mean(run.duration_seconds for run in runs),
        average_context_tokens=_mean(run.context_tokens for run in runs),
        average_input_prompt_chars=_mean(run.input_prompt_chars for run in runs),
        average_initial_dream_retrieval_chars=_mean(
            run.initial_dream_retrieval_chars for run in runs
        ),
        average_assistant_text_chars=_mean(run.assistant_text_chars for run in runs),
        average_tool_call_count=_mean(run.tool_call_count for run in runs),
        average_failed_tool_call_count=_mean(run.failed_tool_call_count for run in runs),
        average_shell_call_count=_mean(run.shell_call_count for run in runs),
        average_benchmark_attempt_count=_mean(run.benchmark_attempt_count for run in runs),
        average_wire_event_count=_mean(run.wire_event_count for run in runs),
        average_wire_event_json_bytes=_mean(run.wire_event_json_bytes for run in runs),
        average_wire_event_payload_json_bytes=_mean(
            run.wire_event_payload_json_bytes for run in runs
        ),
        average_wire_event_text_chars=_mean(run.wire_event_text_chars for run in runs),
        average_wire_event_think_chars=_mean(run.wire_event_think_chars for run in runs),
        average_wire_event_tool_argument_chars=_mean(
            run.wire_event_tool_argument_chars for run in runs
        ),
        average_wire_event_tool_output_chars=_mean(
            run.wire_event_tool_output_chars for run in runs
        ),
        average_artifact_total_bytes=_mean(run.artifact_total_bytes for run in runs),
        average_agent_trace_bytes=_mean(run.agent_trace_bytes for run in runs),
        average_route_decision_bytes=_mean(run.route_decision_bytes for run in runs),
        average_controller_route_decision_changed_rate=_mean(
            _bool_to_float(run.controller_route_decision_changed) for run in runs
        ),
        average_candidate_unit_count=_mean(run.candidate_unit_count for run in runs),
    )


def _artifact_sizes(work_dir: Path) -> dict[str, int]:
    if not work_dir.exists():
        return {"file_count": 0, "total_bytes": 0}
    file_count = 0
    total_bytes = 0
    for path in work_dir.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += _file_size(path)
    return {"file_count": file_count, "total_bytes": total_bytes}


def _sum_wire_event_size_summaries(run: Any) -> dict[str, int]:
    totals = {
        "count": 0,
        "json_bytes": 0,
        "payload_json_bytes": 0,
        "text_chars": 0,
        "think_chars": 0,
        "tool_argument_chars": 0,
        "tool_output_chars": 0,
    }
    for summary in getattr(run, "event_size_summaries", []) or []:
        totals["count"] += int(getattr(summary, "count", 0) or 0)
        totals["json_bytes"] += int(getattr(summary, "total_json_bytes", 0) or 0)
        totals["payload_json_bytes"] += int(
            getattr(summary, "total_payload_json_bytes", 0) or 0
        )
        totals["text_chars"] += int(getattr(summary, "total_text_chars", 0) or 0)
        totals["think_chars"] += int(getattr(summary, "total_think_chars", 0) or 0)
        totals["tool_argument_chars"] += int(
            getattr(summary, "total_tool_argument_chars", 0) or 0
        )
        totals["tool_output_chars"] += int(
            getattr(summary, "total_tool_output_chars", 0) or 0
        )
    return totals


def _resolve_artifact_work_dir(
    raw_work_dir: str,
    *,
    source_path: Path,
    campaign_name: str,
) -> Path:
    work_dir = Path(raw_work_dir).expanduser()
    if work_dir.exists():
        return work_dir

    remote_parts = Path(raw_work_dir).parts
    if len(remote_parts) < 2:
        return work_dir
    rep_name = remote_parts[-2]
    arm_name = remote_parts[-1]
    if not re.fullmatch(r"rep\d+", rep_name):
        return work_dir

    suite_stem = re.sub(r"-rep\d+$", "", campaign_name)
    relative = (
        Path("docs")
        / "evoinfer"
        / "campaign-artifacts"
        / f"{suite_stem}-suite-work"
        / rep_name
        / arm_name
    )
    for parent in [source_path.parent, *source_path.parents, Path.cwd(), *Path.cwd().parents]:
        candidate = parent / relative
        if candidate.exists():
            return candidate
    return work_dir


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _controller_route_decision_changed(
    route_decision_path: Path,
    controller_audit: dict[str, Any] | None,
) -> bool | None:
    if not controller_audit:
        return None
    initial_hash = controller_audit.get("route_decision_sha256")
    if not isinstance(initial_hash, str) or not initial_hash:
        return None
    current_hash = _file_sha256(route_decision_path)
    if current_hash is None:
        return None
    return current_hash != initial_hash


def _controller_route_decision_initial_bytes(
    controller_audit: dict[str, Any] | None,
) -> int:
    if not controller_audit:
        return 0
    value = controller_audit.get("route_decision_bytes")
    return int(value) if isinstance(value, int) else 0


def _list_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def _bool_to_float(value: bool | None) -> float | None:
    if value is None:
        return None
    return 1.0 if value else 0.0


def _mean(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _fmt_float(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "yes" if value else "no"


def _markdown_separator(column_count: int) -> str:
    if column_count <= 1:
        return "| --- |"
    return "| --- | " + " | ".join("---:" for _ in range(column_count - 1)) + " |"


if __name__ == "__main__":
    main()
