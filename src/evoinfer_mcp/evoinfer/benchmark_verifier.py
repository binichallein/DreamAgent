"""Run a benchmark command repeatedly and emit stable key=value metrics."""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_METRIC_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


class VerificationFailure(RuntimeError):
    """Raised when the repeated verifier cannot produce a valid result."""


@dataclass(frozen=True)
class CommandRun:
    """One shell command execution result."""

    command: str
    returncode: int
    output: str


@dataclass(frozen=True)
class BenchmarkVerification:
    """Aggregated benchmark verification output."""

    setup: CommandRun | None
    runs: list[CommandRun]
    metrics: dict[str, list[float]]

    def render(self) -> str:
        lines: list[str] = []
        if self.setup is not None:
            lines.extend(
                [
                    "setup_command=" + self.setup.command,
                    "--- setup output ---",
                    self.setup.output.rstrip(),
                ]
            )
        for index, run in enumerate(self.runs, start=1):
            lines.extend(
                [
                    f"--- benchmark run {index} command ---",
                    run.command,
                    f"--- benchmark run {index} output ---",
                    run.output.rstrip(),
                ]
            )
        lines.append("--- aggregate metrics ---")
        for name in sorted(self.metrics):
            values = self.metrics[name]
            primary = max(values) if _metric_uses_worst_value(name) else statistics.median(values)
            lines.append(f"{name}={_format_float(primary)}")
            lines.append(f"{name}_median={_format_float(statistics.median(values))}")
            lines.append(f"{name}_min={_format_float(min(values))}")
            lines.append(f"{name}_max={_format_float(max(values))}")
        return "\n".join(line for line in lines if line) + "\n"


def run_repeated_benchmark(
    *,
    command: str,
    repeat: int,
    cwd: Path | None = None,
    setup_command: str | None = None,
    timeout_seconds: float = 120.0,
    error_metric: str = "max_abs_error",
    max_error_threshold: float | None = 1e-3,
    required_metrics: tuple[str, ...] = (),
    metric_thresholds: dict[str, float] | None = None,
) -> BenchmarkVerification:
    """Run setup once, benchmark repeatedly, and aggregate parsed metrics."""

    if repeat <= 0:
        raise ValueError("repeat must be positive")

    setup_run = None
    if setup_command:
        setup_run = _run_shell(setup_command, cwd=cwd, timeout_seconds=timeout_seconds)
        if setup_run.returncode != 0:
            raise VerificationFailure(
                f"Setup command failed with code {setup_run.returncode}.\n{setup_run.output}"
            )

    runs: list[CommandRun] = []
    metrics_by_name: dict[str, list[float]] = {}
    for _ in range(repeat):
        run = _run_shell(command, cwd=cwd, timeout_seconds=timeout_seconds)
        runs.append(run)
        if run.returncode != 0:
            raise VerificationFailure(
                f"Benchmark command failed with code {run.returncode}.\n{run.output}"
            )
        metrics = _extract_metrics(run.output)
        for name, value in metrics.items():
            metrics_by_name.setdefault(name, []).append(value)

    complete_metrics = {
        name: values for name, values in metrics_by_name.items() if len(values) == repeat
    }
    if not complete_metrics:
        raise VerificationFailure("Benchmark output did not contain stable numeric metrics.")

    thresholds = {
        _normalize_metric_name(name): value
        for name, value in (metric_thresholds or {}).items()
    }
    required = {_normalize_metric_name(name) for name in required_metrics}
    required.update(thresholds)
    missing_metrics = sorted(name for name in required if name not in complete_metrics)
    if missing_metrics:
        raise VerificationFailure(
            "Missing required benchmark metrics: " + ", ".join(missing_metrics)
        )

    for metric_name, threshold in thresholds.items():
        worst_value = max(complete_metrics[metric_name])
        if worst_value > threshold:
            raise VerificationFailure(
                f"{metric_name}={worst_value:.9g} exceeded threshold {threshold:.9g}."
            )

    if max_error_threshold is not None and error_metric in complete_metrics:
        worst_error = max(complete_metrics[error_metric])
        if worst_error > max_error_threshold:
            raise VerificationFailure(
                f"{error_metric}={worst_error:.9g} exceeded threshold "
                f"{max_error_threshold:.9g}."
            )

    return BenchmarkVerification(setup=setup_run, runs=runs, metrics=complete_metrics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-command", help="Shell command to run once before repeats.")
    parser.add_argument("--command", required=True, help="Shell benchmark command to repeat.")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--error-metric", default="max_abs_error")
    parser.add_argument("--max-error-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--required-metric",
        action="append",
        default=[],
        help="Metric that must appear in every benchmark run. Can be repeated.",
    )
    parser.add_argument(
        "--metric-threshold",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Metric-specific maximum threshold. Can be repeated, for example "
            "max_abs_error=1e-4 max_row_sum_error=1e-5."
        ),
    )
    parser.add_argument(
        "--no-error-threshold",
        action="store_true",
        help="Do not fail based on the error metric.",
    )
    args = parser.parse_args(argv)

    try:
        metric_thresholds = _parse_metric_thresholds(args.metric_threshold)
        result = run_repeated_benchmark(
            command=args.command,
            repeat=args.repeat,
            cwd=args.cwd,
            setup_command=args.setup_command,
            timeout_seconds=args.timeout_seconds,
            error_metric=args.error_metric,
            max_error_threshold=None if args.no_error_threshold else args.max_error_threshold,
            required_metrics=tuple(args.required_metric),
            metric_thresholds=metric_thresholds,
        )
    except VerificationFailure as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    print(result.render(), end="")


def _run_shell(command: str, *, cwd: Path | None, timeout_seconds: float) -> CommandRun:
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        raise VerificationFailure(
            f"Command timed out after {timeout_seconds:.3f}s: {command}\n{output}"
        ) from exc
    return CommandRun(command=command, returncode=process.returncode, output=process.stdout)


def _extract_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw_value in _METRIC_RE.findall(text):
        metrics[_normalize_metric_name(name)] = float(raw_value)
    return metrics


def _format_float(value: float) -> str:
    return f"{value:.9g}"


def _normalize_metric_name(name: str) -> str:
    return name.strip().lower()


def _metric_uses_worst_value(name: str) -> bool:
    return "error" in name


def _parse_metric_thresholds(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw in values:
        name, separator, threshold = raw.partition("=")
        if not separator or not name.strip() or not threshold.strip():
            raise VerificationFailure(
                f"Invalid --metric-threshold value {raw!r}; expected NAME=VALUE."
            )
        try:
            thresholds[_normalize_metric_name(name)] = float(threshold)
        except ValueError as exc:
            raise VerificationFailure(
                f"Invalid threshold for metric {name.strip()!r}: {threshold!r}."
            ) from exc
    return thresholds


if __name__ == "__main__":
    main()
