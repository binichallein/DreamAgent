"""Dream protocol pass-rate gate for EvoInfer campaign suites."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from evoinfer_mcp.evoinfer.campaign_analysis import analyze_campaign_results


class ProtocolSuiteGateResult(BaseModel):
    """Aggregate Dream protocol gate result across campaign JSON files."""

    ok: bool
    campaign_count: int
    run_count: int
    checked_count: int
    protocol_pass_count: int
    strict_protocol_pass_count: int
    protocol_pass_rate: float
    strict_protocol_pass_rate: float
    min_protocol_pass_rate: float
    min_strict_protocol_pass_rate: float
    failures: list[str] = Field(default_factory=list)
    arms: dict[str, dict[str, object]] = Field(default_factory=dict)
    source_files: list[str] = Field(default_factory=list)


def run_evoinfer_protocol_suite_gate(
    campaign_results: list[Path],
    *,
    artifact_root: Path | None,
    min_protocol_pass_rate: float,
    min_strict_protocol_pass_rate: float,
) -> ProtocolSuiteGateResult:
    """Gate a set of campaign results by active Dream protocol evidence."""

    analysis = analyze_campaign_results(
        campaign_results,
        protocol_artifact_root=artifact_root,
    )
    checked_count = sum(
        arm.dream_protocol_checked_count for arm in analysis.arms.values()
    )
    protocol_pass_count = sum(
        arm.dream_protocol_pass_count for arm in analysis.arms.values()
    )
    strict_protocol_pass_count = sum(
        arm.dream_protocol_strict_pass_count for arm in analysis.arms.values()
    )
    protocol_pass_rate = _safe_rate(protocol_pass_count, checked_count)
    strict_protocol_pass_rate = _safe_rate(strict_protocol_pass_count, checked_count)
    failures: list[str] = []
    if checked_count == 0:
        failures.append("no dream-enabled runs were checked")
    if protocol_pass_rate < min_protocol_pass_rate:
        failures.append(
            "protocol_pass_rate "
            f"{protocol_pass_rate:.3f} below required {min_protocol_pass_rate:.3f}"
        )
    if strict_protocol_pass_rate < min_strict_protocol_pass_rate:
        failures.append(
            "strict_protocol_pass_rate "
            f"{strict_protocol_pass_rate:.3f} below required {min_strict_protocol_pass_rate:.3f}"
        )

    return ProtocolSuiteGateResult(
        ok=not failures,
        campaign_count=analysis.campaign_count,
        run_count=analysis.run_count,
        checked_count=checked_count,
        protocol_pass_count=protocol_pass_count,
        strict_protocol_pass_count=strict_protocol_pass_count,
        protocol_pass_rate=protocol_pass_rate,
        strict_protocol_pass_rate=strict_protocol_pass_rate,
        min_protocol_pass_rate=min_protocol_pass_rate,
        min_strict_protocol_pass_rate=min_strict_protocol_pass_rate,
        failures=failures,
        arms={
            arm_name: {
                "checked_count": arm.dream_protocol_checked_count,
                "protocol_pass_count": arm.dream_protocol_pass_count,
                "strict_protocol_pass_count": arm.dream_protocol_strict_pass_count,
                "protocol_pass_rate": arm.dream_protocol_pass_rate,
                "strict_protocol_pass_rate": arm.dream_protocol_strict_pass_rate,
            }
            for arm_name, arm in analysis.arms.items()
        },
        source_files=analysis.source_files,
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
