from __future__ import annotations

from pathlib import Path

import pytest

from evoinfer_mcp.evoinfer.benchmark_verifier import (
    VerificationFailure,
    main,
    run_repeated_benchmark,
)


def test_repeated_benchmark_verifier_aggregates_metrics(tmp_path: Path) -> None:
    command = (
        "python3 -c 'from pathlib import Path; "
        "p=Path(\"counter.txt\"); "
        "i=int(p.read_text())+1 if p.exists() else 1; "
        "p.write_text(str(i)); "
        "print(f\"candidate_ms={i/10:.3f}\"); "
        "print(f\"speedup={10/i:.3f}\"); "
        "print(\"max_abs_error=1e-6\")'"
    )

    result = run_repeated_benchmark(command=command, repeat=3, cwd=tmp_path)
    rendered = result.render()

    assert result.metrics["candidate_ms"] == [0.1, 0.2, 0.3]
    assert "candidate_ms=0.2" in rendered
    assert "candidate_ms_min=0.1" in rendered
    assert "candidate_ms_max=0.3" in rendered
    assert "max_abs_error=1e-06" in rendered


def test_repeated_benchmark_verifier_fails_on_error_threshold(tmp_path: Path) -> None:
    command = "python3 -c 'print(\"candidate_ms=0.1\"); print(\"max_abs_error=0.2\")'"

    with pytest.raises(VerificationFailure, match="exceeded threshold"):
        run_repeated_benchmark(
            command=command,
            repeat=1,
            cwd=tmp_path,
            max_error_threshold=1e-3,
        )


def test_repeated_benchmark_verifier_requires_metrics_and_checks_named_thresholds(
    tmp_path: Path,
) -> None:
    missing_row_sum = (
        "python3 -c 'print(\"candidate_ms=0.1\"); print(\"max_abs_error=1e-6\")'"
    )

    with pytest.raises(VerificationFailure, match="Missing required benchmark metrics"):
        run_repeated_benchmark(
            command=missing_row_sum,
            repeat=1,
            cwd=tmp_path,
            required_metrics=("max_abs_error", "max_row_sum_error"),
            metric_thresholds={"max_abs_error": 1e-4, "max_row_sum_error": 1e-5},
        )

    bad_row_sum = (
        "python3 -c 'print(\"candidate_ms=0.1\"); "
        "print(\"max_abs_error=1e-6\"); print(\"max_row_sum_error=2e-4\")'"
    )

    with pytest.raises(VerificationFailure, match="max_row_sum_error"):
        run_repeated_benchmark(
            command=bad_row_sum,
            repeat=1,
            cwd=tmp_path,
            required_metrics=("max_abs_error", "max_row_sum_error"),
            metric_thresholds={"max_abs_error": 1e-4, "max_row_sum_error": 1e-5},
        )

    good = (
        "python3 -c 'print(\"candidate_ms=0.1\"); "
        "print(\"max_abs_error=1e-6\"); print(\"max_row_sum_error=3e-6\")'"
    )
    result = run_repeated_benchmark(
        command=good,
        repeat=2,
        cwd=tmp_path,
        required_metrics=("max_abs_error", "max_row_sum_error"),
        metric_thresholds={"max_abs_error": 1e-4, "max_row_sum_error": 1e-5},
    )

    assert result.metrics["max_row_sum_error"] == [3e-6, 3e-6]
    assert "max_row_sum_error=3e-06" in result.render()


def test_repeated_benchmark_verifier_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--command",
            "python3 -c 'print(\"candidate_ms=0.1\"); print(\"max_abs_error=0\")'",
            "--repeat",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert "candidate_ms=0.1" in captured.out
    assert "candidate_ms_median=0.1" in captured.out


def test_repeated_benchmark_verifier_cli_supports_metric_thresholds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--command",
            (
                "python3 -c 'print(\"candidate_ms=0.1\"); "
                "print(\"max_abs_error=1e-6\"); print(\"max_row_sum_error=2e-6\")'"
            ),
            "--repeat",
            "2",
            "--required-metric",
            "max_abs_error",
            "--required-metric",
            "max_row_sum_error",
            "--metric-threshold",
            "max_abs_error=1e-4",
            "--metric-threshold",
            "max_row_sum_error=1e-5",
        ]
    )

    captured = capsys.readouterr()
    assert "max_row_sum_error=2e-06" in captured.out
