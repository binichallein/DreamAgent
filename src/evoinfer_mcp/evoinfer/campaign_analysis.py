"""Aggregate EvoInfer campaign JSON results for w/wo memory experiments."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evoinfer_mcp.evoinfer.campaign import CampaignResult, SessionRunResult
from evoinfer_mcp.evoinfer.dream_protocol_verifier import (
    DreamProtocolVerificationError,
    verify_dream_protocol_run,
)


class ArmAnalysis(BaseModel):
    """Aggregate metrics for one campaign arm."""

    arm_name: str
    run_count: int = 0
    finished_count: int = 0
    verifier_pass_count: int = 0
    valid_artifact_count: int = 0
    finished_valid_artifact_count: int = 0
    timeout_valid_artifact_count: int = 0
    invalid_artifact_count: int = 0
    average_duration_seconds: float | None = None
    average_context_tokens: float | None = None
    average_checkpoint_count: float | None = None
    verification_pass_rate: float | None = None
    total_dream_chosen_delta: int = 0
    total_dream_useful_delta: int = 0
    total_dream_retrieval_count: int = 0
    dream_protocol_checked_count: int = 0
    dream_protocol_pass_count: int = 0
    dream_protocol_strict_pass_count: int = 0
    dream_protocol_pass_rate: float | None = None
    dream_protocol_strict_pass_rate: float | None = None
    memory_ids: list[str] = Field(default_factory=list)
    average_metrics: dict[str, float] = Field(default_factory=dict)
    average_path_metrics: dict[str, float] = Field(default_factory=dict)
    duration_delta_vs_baseline_seconds: float | None = None
    context_token_delta_vs_baseline: float | None = None


class CampaignAnalysis(BaseModel):
    """Aggregate report across multiple paired campaign result files."""

    campaign_count: int
    run_count: int
    baseline_arm_name: str = "without_memory"
    arms: dict[str, ArmAnalysis]
    source_files: list[str] = Field(default_factory=list)


def analyze_campaign_results(
    paths: list[Path] | tuple[Path, ...],
    *,
    baseline_arm_name: str = "without_memory",
    protocol_artifact_root: Path | None = None,
) -> CampaignAnalysis:
    """Load campaign JSON files and aggregate per-arm experimental metrics."""

    results = [_load_campaign_result(path) for path in paths]
    runs_by_arm: dict[str, list[SessionRunResult]] = defaultdict(list)
    for result in results:
        for run in result.runs:
            runs_by_arm[run.arm_name].append(run)

    arms = {
        arm_name: _analyze_arm(
            arm_name,
            runs,
            protocol_artifact_root=protocol_artifact_root,
        )
        for arm_name, runs in sorted(runs_by_arm.items())
    }
    baseline = arms.get(baseline_arm_name)
    if baseline is not None:
        for arm in arms.values():
            arm.duration_delta_vs_baseline_seconds = _optional_delta(
                arm.average_duration_seconds,
                baseline.average_duration_seconds,
            )
            arm.context_token_delta_vs_baseline = _optional_delta(
                arm.average_context_tokens,
                baseline.average_context_tokens,
            )

    return CampaignAnalysis(
        campaign_count=len(results),
        run_count=sum(len(result.runs) for result in results),
        baseline_arm_name=baseline_arm_name,
        arms=arms,
        source_files=[str(path) for path in paths],
    )


def render_campaign_analysis_markdown(analysis: CampaignAnalysis) -> str:
    """Render an aggregate campaign analysis as a compact markdown report."""

    lines = [
        "# EvoInfer Campaign Analysis",
        "",
        f"- Campaign files: {analysis.campaign_count}",
        f"- Runs: {analysis.run_count}",
        f"- Baseline arm: `{analysis.baseline_arm_name}`",
        "",
        "## Arm Summary",
        "",
        (
            "| arm | runs | avg_duration_s | avg_context_tokens | verifier_pass_rate | "
            "dream_retrievals | dream_chosen_delta | dream_useful_delta | "
            "duration_delta_s | context_delta |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in analysis.arms.values():
        lines.append(
            "| "
            f"{arm.arm_name} | "
            f"{arm.run_count} | "
            f"{_fmt_float(arm.average_duration_seconds, 3)} | "
            f"{_fmt_float(arm.average_context_tokens, 1)} | "
            f"{_fmt_float(arm.verification_pass_rate, 2)} | "
            f"{arm.total_dream_retrieval_count} | "
            f"{arm.total_dream_chosen_delta} | "
            f"{arm.total_dream_useful_delta} | "
            f"{_fmt_float(arm.duration_delta_vs_baseline_seconds, 3)} | "
            f"{_fmt_float(arm.context_token_delta_vs_baseline, 1)} |"
        )

    lines.extend(
        [
            "",
            "## Artifact Quality",
            "",
            (
                "| arm | runs | finished | verifier_pass | valid_artifacts | "
                "finished_valid | timeout_valid | invalid_artifacts | verifier_pass_rate |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in analysis.arms.values():
        lines.append(
            "| "
            f"{arm.arm_name} | "
            f"{arm.run_count} | "
            f"{arm.finished_count} | "
            f"{arm.verifier_pass_count} | "
            f"{arm.valid_artifact_count} | "
            f"{arm.finished_valid_artifact_count} | "
            f"{arm.timeout_valid_artifact_count} | "
            f"{arm.invalid_artifact_count} | "
            f"{_fmt_float(arm.verification_pass_rate, 2)} |"
        )

    lines.extend(
        [
            "",
            "## Dream Protocol Quality",
            "",
            (
                "| arm | checked | protocol_pass | strict_protocol_pass | "
                "protocol_pass_rate | strict_protocol_pass_rate |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in analysis.arms.values():
        lines.append(
            "| "
            f"{arm.arm_name} | "
            f"{arm.dream_protocol_checked_count} | "
            f"{arm.dream_protocol_pass_count} | "
            f"{arm.dream_protocol_strict_pass_count} | "
            f"{_fmt_float(arm.dream_protocol_pass_rate, 2)} | "
            f"{_fmt_float(arm.dream_protocol_strict_pass_rate, 2)} |"
        )

    lines.extend(["", "## Dream Memory IDs", ""])
    for arm in analysis.arms.values():
        if not arm.memory_ids:
            lines.append(f"- {arm.arm_name}: none")
            continue
        ids = ", ".join(f"`{memory_id}`" for memory_id in arm.memory_ids)
        lines.append(f"- {arm.arm_name}: {ids}")

    lines.extend(["", "## Parsed Metrics", ""])
    for arm in analysis.arms.values():
        if not arm.average_metrics:
            lines.append(f"- {arm.arm_name}: none")
            continue
        metrics = ", ".join(
            f"{name}={_fmt_metric_value(value)}"
            for name, value in sorted(arm.average_metrics.items())
        )
        lines.append(f"- {arm.arm_name}: {metrics}")

    lines.extend(["", "## Path Quality", ""])
    for arm in analysis.arms.values():
        if not arm.average_path_metrics:
            lines.append(f"- {arm.arm_name}: none")
            continue
        metrics = ", ".join(
            f"{name}={_fmt_metric_value(value)}"
            for name, value in sorted(arm.average_path_metrics.items())
        )
        lines.append(f"- {arm.arm_name}: {metrics}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Campaign JSON result files.")
    parser.add_argument("--baseline-arm", default="without_memory")
    parser.add_argument(
        "--protocol-artifact-root",
        type=Path,
        help=(
            "Optional local artifact root for Dream protocol checks when campaign "
            "JSON contains remote work_dir paths."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    analysis = analyze_campaign_results(
        args.paths,
        baseline_arm_name=args.baseline_arm,
        protocol_artifact_root=args.protocol_artifact_root,
    )
    markdown = render_campaign_analysis_markdown(analysis)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


def _load_campaign_result(path: Path) -> CampaignResult:
    result = CampaignResult.model_validate_json(path.read_text(encoding="utf-8"))
    _localize_synced_evoinfer_work_dirs(result, source_path=path)
    return result


def _localize_synced_evoinfer_work_dirs(
    result: CampaignResult,
    *,
    source_path: Path,
) -> None:
    """Map remote limx artifact paths to a locally synced docs/evoinfer tree if present."""

    for run in result.runs:
        if _artifact_dir(run.work_dir).exists():
            continue
        localized = _resolve_synced_evoinfer_path(run.work_dir, source_path=source_path)
        if localized is not None:
            run.work_dir = str(localized)


def _resolve_synced_evoinfer_path(raw_path: str, *, source_path: Path) -> Path | None:
    marker = "/docs/evoinfer/"
    if marker not in raw_path:
        return None
    suffix = Path("docs/evoinfer") / raw_path.split(marker, 1)[1]
    for parent in [source_path.parent, *source_path.parents, Path.cwd(), *Path.cwd().parents]:
        candidate = parent / suffix
        if _artifact_dir(str(candidate)).exists():
            return candidate
    return None


def _analyze_arm(
    arm_name: str,
    runs: list[SessionRunResult],
    *,
    protocol_artifact_root: Path | None,
) -> ArmAnalysis:
    memory_ids: list[str] = []
    seen_memory_ids: set[str] = set()
    for run in runs:
        for memory_id in run.dream_retrieved_memory_ids:
            if memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory_id)
            memory_ids.append(memory_id)

    metric_values: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for name, value in _extract_numeric_metrics(run).items():
            metric_values[name].append(value)

    path_metric_values: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for name, value in _extract_path_metrics(run).items():
            path_metric_values[name].append(value)

    verifier_runs = [run for run in runs if run.verification_status is not None]
    pass_rate = None
    if verifier_runs:
        pass_rate = sum(1 for run in verifier_runs if run.verification_status == "passed") / len(
            verifier_runs
        )
    verifier_pass_count = sum(1 for run in verifier_runs if run.verification_status == "passed")
    protocol_checked_count, protocol_pass_count, protocol_strict_pass_count = (
        _dream_protocol_counts(
            runs,
            protocol_artifact_root=protocol_artifact_root,
        )
    )

    return ArmAnalysis(
        arm_name=arm_name,
        run_count=len(runs),
        finished_count=sum(1 for run in runs if run.status == "finished"),
        verifier_pass_count=verifier_pass_count,
        valid_artifact_count=verifier_pass_count,
        finished_valid_artifact_count=sum(
            1
            for run in verifier_runs
            if run.status == "finished" and run.verification_status == "passed"
        ),
        timeout_valid_artifact_count=sum(
            1
            for run in verifier_runs
            if run.status == "timeout" and run.verification_status == "passed"
        ),
        invalid_artifact_count=sum(
            1 for run in verifier_runs if run.verification_status != "passed"
        ),
        average_duration_seconds=_mean(run.duration_seconds for run in runs),
        average_context_tokens=_mean(
            run.context_tokens for run in runs if run.context_tokens is not None
        ),
        average_checkpoint_count=_mean(run.checkpoint_count for run in runs),
        verification_pass_rate=pass_rate,
        total_dream_chosen_delta=sum(run.dream_chosen_delta for run in runs),
        total_dream_useful_delta=sum(run.dream_useful_delta for run in runs),
        total_dream_retrieval_count=sum(run.dream_retrieval_count for run in runs),
        dream_protocol_checked_count=protocol_checked_count,
        dream_protocol_pass_count=protocol_pass_count,
        dream_protocol_strict_pass_count=protocol_strict_pass_count,
        dream_protocol_pass_rate=_ratio(protocol_pass_count, protocol_checked_count),
        dream_protocol_strict_pass_rate=_ratio(
            protocol_strict_pass_count,
            protocol_checked_count,
        ),
        memory_ids=memory_ids,
        average_metrics={
            name: mean
            for name, values in sorted(metric_values.items())
            if (mean := _mean(values)) is not None
        },
        average_path_metrics={
            name: mean
            for name, values in sorted(path_metric_values.items())
            if (mean := _mean(values)) is not None
        },
    )


def _dream_protocol_counts(
    runs: list[SessionRunResult],
    *,
    protocol_artifact_root: Path | None,
) -> tuple[int, int, int]:
    checked_count = 0
    pass_count = 0
    strict_pass_count = 0
    for run in runs:
        if not run.dream_enabled:
            continue
        checked_count += 1
        payload = run.model_dump(mode="json", exclude_none=True)
        try:
            verify_dream_protocol_run(
                payload,
                artifact_root=protocol_artifact_root,
                require_stuck_retrieval=False,
            )
        except DreamProtocolVerificationError:
            pass
        else:
            pass_count += 1
        try:
            verify_dream_protocol_run(
                payload,
                artifact_root=protocol_artifact_root,
                require_stuck_retrieval=True,
            )
        except DreamProtocolVerificationError:
            pass
        else:
            strict_pass_count += 1
    return checked_count, pass_count, strict_pass_count


_METRIC_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def _extract_numeric_metrics(run: SessionRunResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for text in (run.assistant_text, run.verification_output):
        metrics.update(_extract_numeric_metrics_from_text(text))
    metrics.update(_extract_numeric_metrics_from_artifacts(run))
    return metrics


def _extract_numeric_metrics_from_artifacts(run: SessionRunResult) -> dict[str, float]:
    work_dir = _artifact_dir(run.work_dir)
    metrics: dict[str, float] = {}
    metrics.update(_extract_benchmark_raw_metrics(work_dir / "benchmark_raw.json"))
    metrics.update(_extract_correctness_raw_metrics(work_dir / "correctness_raw.json"))
    return metrics


def _artifact_dir(raw_path: str) -> Path:
    return Path(raw_path).expanduser()


def _extract_benchmark_raw_metrics(path: Path) -> dict[str, float]:
    payload = _load_json_object(path)
    if payload is None:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    metrics: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prefix = _entry_metric_prefix(entry)
        _copy_number_metric(metrics, entry, f"{prefix}_baseline_ms", "baseline_ms_mean")
        _copy_number_metric(metrics, entry, f"{prefix}_candidate_ms", "candidate_ms_mean")
        _copy_number_metric(metrics, entry, f"{prefix}_speedup", "speedup")
        _copy_number_metric(metrics, entry, f"{prefix}_first_call_ms", "first_call_ms")
    return metrics


def _extract_correctness_raw_metrics(path: Path) -> dict[str, float]:
    payload = _load_json_object(path)
    if payload is None:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    metrics: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prefix = _entry_metric_prefix(entry)
        _copy_number_metric(metrics, entry, f"{prefix}_max_abs_error", "max_abs_error")
        _copy_number_metric(metrics, entry, f"{prefix}_mean_abs_error", "mean_abs_error")
        passed = entry.get("passed")
        if isinstance(passed, bool):
            metrics[f"{prefix}_correctness_passed"] = 1.0 if passed else 0.0
    return metrics


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _entry_metric_prefix(entry: dict[str, Any]) -> str:
    raw = entry.get("operator") or entry.get("id") or "artifact"
    return re.sub(r"[^a-z0-9]+", "_", str(raw).lower()).strip("_") or "artifact"


def _copy_number_metric(
    metrics: dict[str, float],
    entry: dict[str, Any],
    metric_name: str,
    source_name: str,
) -> None:
    value = entry.get(source_name)
    if isinstance(value, int | float) and not isinstance(value, bool):
        metrics[metric_name] = float(value)


def _extract_path_metrics(run: SessionRunResult) -> dict[str, float]:
    fields = (
        "tool_call_count",
        "failed_tool_call_count",
        "shell_call_count",
        "shell_failure_count",
        "observer_checkpoint_count",
        "benchmark_attempt_count",
        "benchmark_failure_count",
        "edit_command_count",
        "candidate_metric_report_count",
        "valid_speedup_count",
        "failed_variant_count",
        "first_valid_speedup_benchmark_index",
    )
    metrics: dict[str, float] = {}
    for field in fields:
        value = getattr(run, field)
        if value is not None:
            metrics[field] = float(value)
    return metrics


def _extract_numeric_metrics_from_text(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw_value in _METRIC_RE.findall(text):
        normalized = name.strip().lower()
        if _is_report_metric_name(normalized):
            metrics[normalized] = float(raw_value)
    return metrics


def _is_report_metric_name(name: str) -> bool:
    return (
        name
        in {
            "speedup",
            "speedup_x",
            "baseline_ms",
            "candidate_ms",
            "max_abs_error",
            "max_row_sum_error",
        }
        or name.endswith("_ms")
        or name.endswith("_tokens_per_second")
        or name.endswith("_tok_s")
    )


def _mean(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _fmt_float(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _fmt_metric_value(value: float) -> str:
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
