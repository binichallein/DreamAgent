"""Run repeated EvoInfer campaigns and aggregate w/wo memory results."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from evoinfer_mcp.evoinfer.campaign import (
    CampaignClient,
    CampaignResult,
    CampaignRunner,
    CampaignSpec,
    EvoInferWebClient,
    normalize_dream_retrieval_categories,
    render_campaign_markdown,
    _parse_campaign_arms,
)
from evoinfer_mcp.evoinfer.campaign_analysis import (
    analyze_campaign_results,
    render_campaign_analysis_markdown,
)
from evoinfer_mcp.evoinfer.protocol_suite_gate import (
    ProtocolSuiteGateResult,
    run_evoinfer_protocol_suite_gate,
)

ArmOrderPolicy = Literal["alternate", "fixed"]


class CampaignSuiteSpec(BaseModel):
    """Repeated campaign suite for paper-oriented w/wo memory experiments."""

    name: str
    base_campaign: CampaignSpec
    repeat_count: int = Field(default=1, ge=1)
    campaign_out_dir: str = "docs/evoinfer/campaign-runs"
    analysis_out_dir: str = "docs/evoinfer/campaign-analysis"
    suite_out_dir: str = "docs/evoinfer/campaign-suites"
    arm_order_policy: ArmOrderPolicy = "alternate"
    resume_existing: bool = False
    restore_dream_memory_after_arm: bool = False
    protocol_artifact_root: str | None = None
    min_protocol_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_strict_protocol_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class CampaignSuiteRun(BaseModel):
    """One repeated campaign produced by a suite."""

    index: int
    campaign_name: str
    work_dir: str
    arm_order: list[str]
    json_path: str
    markdown_path: str
    environment_json_path: str
    memory_snapshot_before_path: str
    memory_snapshot_after_path: str


class CampaignSuiteResult(BaseModel):
    """Manifest for a repeated EvoInfer campaign suite."""

    name: str
    repeat_count: int
    started_at: float
    ended_at: float
    duration_seconds: float
    runs: list[CampaignSuiteRun]
    analysis_json_path: str
    analysis_markdown_path: str
    protocol_gate: ProtocolSuiteGateResult | None = None


class CampaignSuiteProtocolGateError(RuntimeError):
    """Raised when an explicitly configured suite protocol gate fails."""

    def __init__(self, result: CampaignSuiteResult) -> None:
        gate = result.protocol_gate
        failures = gate.failures if gate is not None else ["protocol gate failed"]
        super().__init__(
            "EvoInfer campaign suite protocol gate failed: " + "; ".join(failures)
        )
        self.result = result


async def run_campaign_suite(
    client: CampaignClient,
    spec: CampaignSuiteSpec,
    *,
    delete_sessions: bool = False,
    progress_sink: Callable[[str], None] | None = None,
) -> CampaignSuiteResult:
    """Run a repeated campaign suite and write campaign plus aggregate artifacts."""

    started_at = time.time()
    campaign_out_dir = Path(spec.campaign_out_dir)
    analysis_out_dir = Path(spec.analysis_out_dir)
    suite_out_dir = Path(spec.suite_out_dir)
    campaign_out_dir.mkdir(parents=True, exist_ok=True)
    analysis_out_dir.mkdir(parents=True, exist_ok=True)
    suite_out_dir.mkdir(parents=True, exist_ok=True)
    state_path = suite_out_dir / f"{_slugify(spec.name)}-state.json"
    suite_state = _new_suite_state(spec, started_at)
    _write_suite_state(state_path, suite_state)

    suite_runs: list[CampaignSuiteRun] = []
    campaign_paths: list[Path] = []
    runner = CampaignRunner(client)

    for index in range(1, spec.repeat_count + 1):
        campaign_spec = _campaign_spec_for_repetition(
            spec.base_campaign,
            index,
            arm_order_policy=spec.arm_order_policy,
            restore_dream_memory_after_arm=spec.restore_dream_memory_after_arm,
        )
        existing = (
            _find_existing_campaign_result(campaign_spec.name, campaign_out_dir)
            if spec.resume_existing
            else None
        )
        if existing is not None:
            suite_state["active_campaign"] = campaign_spec.name
            _append_suite_state_event(
                suite_state,
                "resumed",
                campaign_name=campaign_spec.name,
                index=index,
            )
            result, json_path = existing
            markdown_path = json_path.with_suffix(".md")
            _report_progress(
                progress_sink,
                f"resume {campaign_spec.name} json={json_path}",
            )
            suite_state["active_campaign"] = None
        else:
            suite_state["active_campaign"] = campaign_spec.name
            _append_suite_state_event(
                suite_state,
                "started",
                campaign_name=campaign_spec.name,
                index=index,
            )
            _write_suite_state(state_path, suite_state)
            _report_progress(progress_sink, f"start {campaign_spec.name}")
            try:
                result = await runner.run(campaign_spec, delete_sessions=delete_sessions)
            except BaseException as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                suite_state["status"] = "error"
                suite_state["error"] = error
                _append_suite_state_event(
                    suite_state,
                    "error",
                    campaign_name=campaign_spec.name,
                    index=index,
                    error=error,
                )
                _write_suite_state(state_path, suite_state)
                _report_progress(
                    progress_sink,
                    f"end {campaign_spec.name} status=error error={type(exc).__name__}: {exc}",
                )
                raise
            json_path, markdown_path = _write_campaign_result(result, campaign_out_dir)
            result_status = _campaign_result_progress_status(result)
            _report_progress(
                progress_sink,
                f"end {campaign_spec.name} status={result_status} json={json_path}",
            )
        campaign_paths.append(json_path)
        suite_run = CampaignSuiteRun(
            index=index,
            campaign_name=result.name,
            work_dir=result.work_dir,
            arm_order=[run.arm_name for run in result.runs],
            json_path=str(json_path),
            markdown_path=str(markdown_path),
            environment_json_path=result.environment_json_path or "",
            memory_snapshot_before_path=result.memory_snapshot_before_path or "",
            memory_snapshot_after_path=result.memory_snapshot_after_path or "",
        )
        suite_runs.append(suite_run)
        suite_state["runs"].append(suite_run.model_dump(mode="json"))
        suite_state["completed_count"] = len(suite_runs)
        suite_state["active_campaign"] = None
        _append_suite_state_event(
            suite_state,
            "completed",
            campaign_name=campaign_spec.name,
            index=index,
            json_path=str(json_path),
            markdown_path=str(markdown_path),
        )
        _write_suite_state(state_path, suite_state)

    protocol_artifact_root = _optional_path(spec.protocol_artifact_root)
    analysis = analyze_campaign_results(
        campaign_paths,
        protocol_artifact_root=protocol_artifact_root,
    )
    stem = f"{int(started_at)}-{_slugify(spec.name)}-suite-analysis"
    analysis_json_path = analysis_out_dir / f"{stem}.json"
    analysis_markdown_path = analysis_out_dir / f"{stem}.md"
    analysis_json_path.write_text(
        json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    analysis_markdown_path.write_text(
        render_campaign_analysis_markdown(analysis),
        encoding="utf-8",
    )
    protocol_gate = _run_protocol_gate_if_requested(
        spec,
        campaign_paths=campaign_paths,
        artifact_root=protocol_artifact_root,
    )

    ended_at = time.time()
    suite_result = CampaignSuiteResult(
        name=spec.name,
        repeat_count=spec.repeat_count,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=ended_at - started_at,
        runs=suite_runs,
        analysis_json_path=str(analysis_json_path),
        analysis_markdown_path=str(analysis_markdown_path),
        protocol_gate=protocol_gate,
    )
    suite_stem = f"{int(started_at)}-{_slugify(spec.name)}-suite"
    suite_json_path = suite_out_dir / f"{suite_stem}.json"
    suite_markdown_path = suite_out_dir / f"{suite_stem}.md"
    suite_json_path.write_text(
        json.dumps(suite_result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    suite_markdown_path.write_text(render_campaign_suite_markdown(suite_result), encoding="utf-8")
    if protocol_gate is not None:
        suite_state["protocol_gate"] = protocol_gate.model_dump(mode="json")
        if protocol_gate.ok:
            _append_suite_state_event(
                suite_state,
                "protocol_gate_passed",
                checked_count=protocol_gate.checked_count,
                strict_protocol_pass_rate=protocol_gate.strict_protocol_pass_rate,
            )
        else:
            suite_state["status"] = "protocol_gate_failed"
            suite_state["active_campaign"] = None
            suite_state["analysis_json_path"] = str(analysis_json_path)
            suite_state["analysis_markdown_path"] = str(analysis_markdown_path)
            suite_state["suite_json_path"] = str(suite_json_path)
            suite_state["suite_markdown_path"] = str(suite_markdown_path)
            _append_suite_state_event(
                suite_state,
                "protocol_gate_failed",
                checked_count=protocol_gate.checked_count,
                strict_protocol_pass_rate=protocol_gate.strict_protocol_pass_rate,
                failures=protocol_gate.failures,
            )
            _write_suite_state(state_path, suite_state)
            raise CampaignSuiteProtocolGateError(suite_result)
    suite_state["status"] = "complete"
    suite_state["active_campaign"] = None
    suite_state["analysis_json_path"] = str(analysis_json_path)
    suite_state["analysis_markdown_path"] = str(analysis_markdown_path)
    suite_state["suite_json_path"] = str(suite_json_path)
    suite_state["suite_markdown_path"] = str(suite_markdown_path)
    _append_suite_state_event(
        suite_state,
        "complete",
        suite_json_path=str(suite_json_path),
        suite_markdown_path=str(suite_markdown_path),
    )
    _write_suite_state(state_path, suite_state)
    return suite_result


def render_campaign_suite_markdown(result: CampaignSuiteResult) -> str:
    """Render a repeated campaign suite manifest as markdown."""

    lines = [
        f"# EvoInfer Campaign Suite: {result.name}",
        "",
        f"- Repeats: {result.repeat_count}",
        f"- Duration: {result.duration_seconds:.3f}s",
        f"- Analysis JSON: `{result.analysis_json_path}`",
        f"- Analysis Markdown: `{result.analysis_markdown_path}`",
        "",
    ]
    if result.protocol_gate is not None:
        gate = result.protocol_gate
        lines.extend(
            [
                "## Dream Protocol Gate",
                "",
                f"- Status: `{'pass' if gate.ok else 'fail'}`",
                f"- Checked dream-enabled runs: {gate.checked_count}",
                f"- Protocol pass rate: {gate.protocol_pass_rate:.3f}",
                f"- Strict protocol pass rate: {gate.strict_protocol_pass_rate:.3f}",
                f"- Required protocol pass rate: {gate.min_protocol_pass_rate:.3f}",
                f"- Required strict protocol pass rate: {gate.min_strict_protocol_pass_rate:.3f}",
                "",
            ]
        )
        if gate.failures:
            lines.extend(["Failures:", ""])
            for failure in gate.failures:
                lines.append(f"- {failure}")
            lines.append("")
    lines.extend(
        [
        "## Campaign Runs",
        "",
        "| index | campaign | arm_order | work_dir | json | markdown |",
        "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for run in result.runs:
        arm_order = " -> ".join(run.arm_order)
        lines.append(
            "| "
            f"{run.index} | "
            f"{run.campaign_name} | "
            f"{arm_order} | "
            f"`{run.work_dir}` | "
            f"`{run.json_path}` | "
            f"`{run.markdown_path}` |"
        )
    return "\n".join(lines) + "\n"


def _new_suite_state(spec: CampaignSuiteSpec, started_at: float) -> dict:
    return {
        "name": spec.name,
        "repeat_count": spec.repeat_count,
        "started_at": started_at,
        "updated_at": started_at,
        "status": "running",
        "active_campaign": None,
        "completed_count": 0,
        "runs": [],
        "events": [],
        "error": None,
    }


def _append_suite_state_event(state: dict, status: str, **fields: object) -> None:
    timestamp = time.time()
    event = {"timestamp": timestamp, "status": status}
    event.update({key: value for key, value in fields.items() if value is not None})
    state["updated_at"] = timestamp
    state["events"].append(event)


def _write_suite_state(path: Path, state: dict) -> None:
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def run_campaign_suite_from_cli(args: argparse.Namespace) -> CampaignSuiteResult:
    base_campaign = CampaignSpec(
        name=args.name,
        prompt=args.prompt,
        work_dir=args.work_dir,
        create_dir=args.create_dir,
        timer_interval_seconds=args.timer_interval_seconds,
        observer_interval_seconds=args.observer_interval_seconds,
        timeout_seconds=args.timeout_seconds,
        isolate_work_dirs=not args.shared_work_dir,
        seed_dir=args.seed_dir,
        reset_work_dirs=args.reset_work_dirs,
        verifier_command=args.verifier_command,
        verifier_timeout_seconds=args.verifier_timeout_seconds,
        yolo_enabled=args.yolo,
        environment_command=args.environment_command,
        initial_dream_retrieval=(
            args.initial_dream_retrieval and not args.disable_initial_dream_retrieval
        ),
        dream_retrieval_query=args.dream_retrieval_query,
        dream_retrieval_tags=args.dream_retrieval_tag,
        dream_retrieval_categories=normalize_dream_retrieval_categories(
            args.dream_retrieval_category
        ),
        dream_retrieval_memory_ids=args.dream_retrieval_memory_id,
        dream_retrieval_top_k_per_category=args.dream_retrieval_top_k_per_category,
        dream_retrieval_render_mode=args.dream_retrieval_render_mode,
        arms=_parse_campaign_arms(args.arm),
    )
    spec = CampaignSuiteSpec(
        name=args.name,
        base_campaign=base_campaign,
        repeat_count=args.repeat_count,
        campaign_out_dir=str(args.campaign_out_dir),
        analysis_out_dir=str(args.analysis_out_dir),
        suite_out_dir=str(args.suite_out_dir),
        arm_order_policy=args.arm_order_policy,
        resume_existing=args.resume_existing,
        restore_dream_memory_after_arm=args.restore_dream_memory_after_arm,
        protocol_artifact_root=(
            str(protocol_artifact_root)
            if (
                protocol_artifact_root := getattr(
                    args,
                    "protocol_artifact_root",
                    None,
                )
            )
            else None
        ),
        min_protocol_pass_rate=getattr(args, "min_protocol_pass_rate", None),
        min_strict_protocol_pass_rate=getattr(
            args,
            "min_strict_protocol_pass_rate",
            None,
        ),
    )
    client = EvoInferWebClient(base_url=args.base_url, token=args.token)
    return await run_campaign_suite(
        client,
        spec,
        delete_sessions=args.delete_sessions,
        progress_sink=_print_progress,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5494")
    parser.add_argument("--token", default="evoinfer-local")
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument(
        "--arm-order-policy",
        choices=["alternate", "fixed"],
        default="alternate",
        help=(
            "Order campaign arms across repetitions. 'alternate' counterbalances "
            "without/with memory order on even repetitions."
        ),
    )
    parser.add_argument("--create-dir", action="store_true")
    parser.add_argument("--seed-dir")
    parser.add_argument("--reset-work-dirs", action="store_true")
    parser.add_argument("--shared-work-dir", action="store_true")
    parser.add_argument("--timer-interval-seconds", type=int)
    parser.add_argument(
        "--observer-interval-seconds",
        type=int,
        help=(
            "Controller-side non-invasive checkpoint sampling interval. "
            "Unlike --timer-interval-seconds, this does not send prompts to the agent."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--verifier-command")
    parser.add_argument("--verifier-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--environment-command",
        help=(
            "Optional command that prints JSON describing the experiment runtime. "
            "It is captured as an audit artifact and does not affect run metrics."
        ),
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help=(
            "Enable session-local YOLO auto-approval for campaign runs. "
            "This does not change global agent client settings."
        ),
    )
    parser.add_argument(
        "--initial-dream-retrieval",
        action="store_true",
        help=(
            "Explicitly prefetch Dream memories in the campaign runner before sending "
            "the prompt. The Codex runner already performs auditable prompt-start "
            "retrieval when Dream mode is enabled."
        ),
    )
    parser.add_argument(
        "--disable-initial-dream-retrieval",
        action="store_true",
        help="Compatibility flag: keep campaign-runner prefetch disabled.",
    )
    parser.add_argument(
        "--dream-retrieval-query",
        help="Override the query used for campaign-start Dream retrieval. Default: prompt.",
    )
    parser.add_argument(
        "--dream-retrieval-tag",
        action="append",
        default=[],
        help="Structured Dream retrieval tag. Can be repeated.",
    )
    parser.add_argument(
        "--dream-retrieval-category",
        action="append",
        choices=["optimization", "environment_debug"],
        default=None,
        help="Dream memory category to retrieve. Can be repeated.",
    )
    parser.add_argument(
        "--dream-retrieval-memory-id",
        action="append",
        default=[],
        help=(
            "Pin campaign-start Dream retrieval to a specific memory ID. "
            "Can be repeated for retrieval-controlled A/B suites."
        ),
    )
    parser.add_argument("--dream-retrieval-top-k-per-category", type=int, default=3)
    parser.add_argument(
        "--dream-retrieval-render-mode",
        choices=[
            "full",
            "route_policy",
            "route_policy_minimal",
            "route_policy_template",
            "route_policy_artifact",
            "route_policy_artifact_protocol",
        ],
        default="full",
        help=(
            "How campaign-start Dream retrieval is injected into prompts. "
            "Use route_policy to inject compact route constraints instead of full memory "
            "details; use route_policy_minimal for only route-decision fields; use "
            "route_policy_template for a compact route_decision.json scaffold; use "
            "route_policy_artifact to write controller-generated route_decision.json "
            "and inject only a short prompt notice; use route_policy_artifact_protocol "
            "to add stricter consumption instructions around the same artifact."
        ),
    )
    parser.add_argument(
        "--arm",
        action="append",
        help=(
            "JSON campaign arm config. Repeat to define custom arms. "
            'Example: {"name":"with_library_learning_memory","dream_enabled":true,'
            '"initial_dream_retrieval":true,"dream_retrieval_top_k_per_category":1}'
        ),
    )
    parser.add_argument("--delete-sessions", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Skip repetitions whose campaign JSON already exists in --campaign-out-dir. "
            "Useful after a long suite controller is interrupted."
        ),
    )
    parser.add_argument(
        "--restore-dream-memory-after-arm",
        action="store_true",
        help=(
            "Snapshot Dream memory before each campaign arm and restore it after the arm. "
            "This keeps A/B arms and repeats from seeing memories written by earlier arms."
        ),
    )
    parser.add_argument(
        "--protocol-artifact-root",
        type=Path,
        help=(
            "Optional local artifact root for Dream protocol checks when campaign "
            "JSON contains remote work_dir paths."
        ),
    )
    parser.add_argument(
        "--min-protocol-pass-rate",
        type=float,
        help=(
            "Enable suite-level Dream protocol gate and require this task-start "
            "protocol pass rate."
        ),
    )
    parser.add_argument(
        "--min-strict-protocol-pass-rate",
        type=float,
        help=(
            "Enable suite-level Dream protocol gate and require this strict pass "
            "rate with stuck/branch retrieval evidence."
        ),
    )
    parser.add_argument("--campaign-out-dir", type=Path, default=Path("docs/evoinfer/campaign-runs"))
    parser.add_argument(
        "--analysis-out-dir",
        type=Path,
        default=Path("docs/evoinfer/campaign-analysis"),
    )
    parser.add_argument(
        "--suite-out-dir",
        type=Path,
        default=Path("docs/evoinfer/campaign-suites"),
    )
    args = parser.parse_args(argv)

    result = asyncio.run(run_campaign_suite_from_cli(args))
    print(result.analysis_json_path)
    print(result.analysis_markdown_path)
    for run in result.runs:
        print(run.json_path)
        print(run.markdown_path)


def _campaign_spec_for_repetition(
    base: CampaignSpec,
    index: int,
    *,
    arm_order_policy: ArmOrderPolicy,
    restore_dream_memory_after_arm: bool = False,
) -> CampaignSpec:
    suffix = f"rep{index:02d}"
    arms = base.arms
    if arm_order_policy == "alternate" and index % 2 == 0:
        arms = tuple(reversed(base.arms))
    return base.model_copy(
        update={
            "name": f"{base.name}-{suffix}",
            "work_dir": str(Path(base.work_dir) / suffix),
            "arms": arms,
            "restore_dream_memory_after_arm": (
                base.restore_dream_memory_after_arm or restore_dream_memory_after_arm
            ),
        }
    )


def _write_campaign_result(result: CampaignResult, out_dir: Path) -> tuple[Path, Path]:
    stem = f"{int(result.started_at)}-{result.name}"
    json_path = out_dir / f"{stem}.json"
    markdown_path = out_dir / f"{stem}.md"
    sidecar_dir = out_dir / stem
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    environment_json_path = sidecar_dir / "environment.json"
    memory_snapshot_before_path = sidecar_dir / "memory_snapshot_before.json"
    memory_snapshot_after_path = sidecar_dir / "memory_snapshot_after.json"
    environment_json_path.write_text(
        json.dumps(result.environment_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    memory_snapshot_before_path.write_text(
        json.dumps(result.memory_snapshot_before, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    memory_snapshot_after_path.write_text(
        json.dumps(result.memory_snapshot_after, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.environment_json_path = str(environment_json_path)
    result.memory_snapshot_before_path = str(memory_snapshot_before_path)
    result.memory_snapshot_after_path = str(memory_snapshot_after_path)
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_campaign_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def _campaign_result_progress_status(result: CampaignResult) -> str:
    verifier_runs = [
        run for run in result.runs if run.verification_status is not None
    ]
    if any(run.verification_status != "passed" for run in verifier_runs):
        return "invalid_artifacts"
    return "ok"


def _find_existing_campaign_result(
    campaign_name: str,
    out_dir: Path,
) -> tuple[CampaignResult, Path] | None:
    matches = sorted(out_dir.glob(f"*-{campaign_name}.json"))
    if not matches:
        return None
    json_path = matches[-1]
    return CampaignResult.model_validate_json(
        json_path.read_text(encoding="utf-8")
    ), json_path


def _report_progress(
    progress_sink: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress_sink is None:
        return
    progress_sink(message)


def _run_protocol_gate_if_requested(
    spec: CampaignSuiteSpec,
    *,
    campaign_paths: list[Path],
    artifact_root: Path | None,
) -> ProtocolSuiteGateResult | None:
    if (
        spec.min_protocol_pass_rate is None
        and spec.min_strict_protocol_pass_rate is None
    ):
        return None
    return run_evoinfer_protocol_suite_gate(
        campaign_paths,
        artifact_root=artifact_root,
        min_protocol_pass_rate=spec.min_protocol_pass_rate or 0.0,
        min_strict_protocol_pass_rate=spec.min_strict_protocol_pass_rate or 0.0,
    )


def _optional_path(value: str | None) -> Path | None:
    if value is None or not value.strip():
        return None
    return Path(value)


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip(".-")
    return slug or "suite"


if __name__ == "__main__":
    main()
