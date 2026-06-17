from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoinfer_mcp.evoinfer.phase0_smoke_verifier import (
    Phase0SmokeVerificationError,
    verify_phase0_smoke_dir,
)


def _write_smoke_artifacts(root: Path, *, env: dict[str, object]) -> None:
    (root / "environment.json").write_text(json.dumps(env))
    (root / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operation_name": "elementwise_add",
                "warmup_count": 10,
                "repeat_count": 50,
                "elapsed_ms_mean": 0.12,
                "elapsed_ms_min": 0.11,
                "elapsed_ms_max": 0.13,
            }
        )
    )
    (root / "correctness_raw.json").write_text(
        json.dumps({"max_abs_error": 0.0, "passed": True})
    )
    (root / "agent_trace.md").write_text("ran smoke benchmark\n")
    (root / "dream_write_candidates.json").write_text("[]")


def test_phase0_smoke_verifier_accepts_version_suffixed_environment_keys(
    tmp_path: Path,
) -> None:
    _write_smoke_artifacts(
        tmp_path,
        env={
            "gpu": "NVIDIA GeForce RTX 3090",
            "driver": "580.126.18",
            "torch_version": "2.12.0+cu126",
            "triton_version": "3.7.0",
            "flashinfer_version": "0.6.12",
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 3090",
        },
    )

    result = verify_phase0_smoke_dir(tmp_path)

    assert result["env"]["torch_version"] == "2.12.0+cu126"
    assert result["bench"]["repeat_count"] == 50
    assert result["corr"]["passed"] is True


def test_phase0_smoke_verifier_rejects_memory_candidates(tmp_path: Path) -> None:
    _write_smoke_artifacts(
        tmp_path,
        env={
            "torch": "2.12.0+cu126",
            "triton": "3.7.0",
            "flashinfer": "0.6.12",
            "cuda_available": True,
        },
    )
    (tmp_path / "dream_write_candidates.json").write_text(
        json.dumps([{"kind": "optimization"}])
    )

    with pytest.raises(Phase0SmokeVerificationError, match="must be empty"):
        verify_phase0_smoke_dir(tmp_path)


def test_phase0_smoke_verifier_rejects_malformed_driver_field(tmp_path: Path) -> None:
    _write_smoke_artifacts(
        tmp_path,
        env={
            "gpu": "NVIDIA GeForce RTX 3090",
            "driver": 8,
            "torch": "2.12.0+cu126",
            "triton": "3.7.0",
            "flashinfer": "0.6.12",
            "cuda_available": True,
        },
    )

    with pytest.raises(Phase0SmokeVerificationError, match="driver"):
        verify_phase0_smoke_dir(tmp_path)
