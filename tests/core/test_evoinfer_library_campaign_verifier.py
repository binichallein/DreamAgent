from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoinfer_mcp.evoinfer.library_campaign_verifier import (
    LibraryCampaignVerificationError,
    verify_library_campaign_dir,
)


def _write_valid_library_artifacts(root: Path) -> None:
    (root / "environment.json").write_text(
        json.dumps(
            {
                "library": "flashinfer",
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "driver_version": "580.126.18",
                "cuda_available": True,
                "torch_version": "2.12.0+cu126",
                "flashinfer_version": "0.6.12",
            }
        )
    )
    (root / "api_inventory.json").write_text(
        json.dumps(
            {
                "library": "flashinfer",
                "tested_api": [
                    {
                        "name": "flashinfer.single_decode_with_kv_cache",
                        "status": "passed",
                    }
                ],
            }
        )
    )
    (root / "correctness_raw.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "fi_single_decode_fp16_t512_h8_d64",
                        "operator": "single_decode_with_kv_cache",
                        "dtype": "float16",
                        "shape": {"kv_len": 512, "num_heads": 8, "head_dim": 64},
                        "max_abs_error": 0.001,
                        "mean_abs_error": 0.0001,
                        "passed": True,
                    }
                ]
            }
        )
    )
    (root / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "fi_single_decode_fp16_t512_h8_d64",
                        "operator": "single_decode_with_kv_cache",
                        "dtype": "float16",
                        "shape": {"kv_len": 512, "num_heads": 8, "head_dim": 64},
                        "warmup_count": 10,
                        "repeat_count": 50,
                        "baseline_ms_mean": 0.05,
                        "candidate_ms_mean": 0.02,
                        "first_call_ms": 1200.0,
                    }
                ]
            }
        )
    )
    (root / "library_notes.md").write_text("FlashInfer single decode notes\n")
    (root / "agent_trace.md").write_text("commands and observations\n")
    (root / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "category": "candidate_optimization",
                    "library": "flashinfer",
                    "artifact_refs": [
                        "environment.json",
                        "correctness_raw.json",
                        "benchmark_raw.json",
                    ],
                }
            ]
        )
    )


def _write_environment_debug_artifacts(root: Path) -> None:
    (root / "environment.json").write_text(
        json.dumps(
            {
                "classification": "environment_debug",
                "library": "flashinfer",
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "driver_version": "580.126.18",
                "cuda_available": True,
                "torch_version": "2.12.0+cu126",
                "flashinfer_python_version": "0.6.12",
                "single_decode_smoke_exit_code": 1,
            }
        )
    )
    (root / "api_inventory.json").write_text(
        json.dumps(
            {
                "library": "flashinfer",
                "tested_api": [
                    {
                        "name": "flashinfer.single_decode_with_kv_cache",
                        "status": "failed_before_correctness",
                        "failure_stage": "JIT compile",
                    }
                ],
            }
        )
    )
    (root / "correctness_raw.json").write_text(
        json.dumps(
            {
                "status": "not_run",
                "passed": False,
                "reason": "candidate failed during JIT compilation",
            }
        )
    )
    (root / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "status": "not_run",
                "reason": "candidate failed during JIT compilation",
            }
        )
    )
    (root / "library_notes.md").write_text("JIT failed before correctness\n")
    (root / "agent_trace.md").write_text("JIT failed on binary headers\n")
    (root / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "category": "environment_debug",
                    "library": "flashinfer",
                    "artifact_refs": [
                        "environment.json",
                        "api_inventory.json",
                        "agent_trace.md",
                    ],
                }
            ]
        )
    )


def test_library_campaign_verifier_accepts_valid_library_artifacts(
    tmp_path: Path,
) -> None:
    _write_valid_library_artifacts(tmp_path)

    result = verify_library_campaign_dir(tmp_path)

    assert result["mode"] == "benchmark"
    assert result["benchmark_entry_count"] == 1
    assert result["correctness_entry_count"] == 1
    assert result["dream_candidate_count"] == 1


def test_library_campaign_verifier_rejects_missing_flashinfer_version(
    tmp_path: Path,
) -> None:
    _write_valid_library_artifacts(tmp_path)
    env = json.loads((tmp_path / "environment.json").read_text())
    env.pop("flashinfer_version")
    (tmp_path / "environment.json").write_text(json.dumps(env))

    with pytest.raises(LibraryCampaignVerificationError, match="flashinfer"):
        verify_library_campaign_dir(tmp_path)


def test_library_campaign_verifier_rejects_speedup_with_failed_correctness(
    tmp_path: Path,
) -> None:
    _write_valid_library_artifacts(tmp_path)
    correctness = json.loads((tmp_path / "correctness_raw.json").read_text())
    correctness["entries"][0]["passed"] = False
    (tmp_path / "correctness_raw.json").write_text(json.dumps(correctness))

    with pytest.raises(LibraryCampaignVerificationError, match="did not pass"):
        verify_library_campaign_dir(tmp_path)


def test_library_campaign_verifier_accepts_failed_correctness_without_benchmark(
    tmp_path: Path,
) -> None:
    _write_valid_library_artifacts(tmp_path)
    correctness = json.loads((tmp_path / "correctness_raw.json").read_text())
    correctness["entries"].append(
        {
            "id": "fi_single_decode_failed_candidate",
            "operator": "single_decode_with_kv_cache",
            "dtype": "float16",
            "shape": {"kv_len": 512, "num_heads": 8, "head_dim": 64},
            "max_abs_error": 0.25,
            "mean_abs_error": 0.02,
            "passed": False,
        }
    )
    (tmp_path / "correctness_raw.json").write_text(json.dumps(correctness))
    dream = json.loads((tmp_path / "dream_write_candidates.json").read_text())
    dream.append(
        {
            "category": "negative_optimization",
            "library": "flashinfer",
            "artifact_refs": ["correctness_raw.json", "agent_trace.md"],
        }
    )
    (tmp_path / "dream_write_candidates.json").write_text(json.dumps(dream))

    result = verify_library_campaign_dir(tmp_path)

    assert result["correctness_entry_count"] == 2
    assert result["failed_correctness_entry_count"] == 1
    assert result["benchmark_entry_count"] == 1


def test_library_campaign_verifier_rejects_dream_candidate_without_artifact_refs(
    tmp_path: Path,
) -> None:
    _write_valid_library_artifacts(tmp_path)
    (tmp_path / "dream_write_candidates.json").write_text(
        json.dumps([{"category": "candidate_optimization", "library": "flashinfer"}])
    )

    with pytest.raises(LibraryCampaignVerificationError, match="artifact_refs"):
        verify_library_campaign_dir(tmp_path)


def test_library_campaign_verifier_accepts_environment_debug_artifacts(
    tmp_path: Path,
) -> None:
    _write_environment_debug_artifacts(tmp_path)

    result = verify_library_campaign_dir(tmp_path)

    assert result["mode"] == "environment_debug"
    assert result["benchmark_entry_count"] == 0
    assert result["correctness_entry_count"] == 0
    assert result["dream_candidate_count"] == 1


def test_library_campaign_verifier_rejects_debug_with_optimization_memory(
    tmp_path: Path,
) -> None:
    _write_environment_debug_artifacts(tmp_path)
    (tmp_path / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "category": "candidate_optimization",
                    "library": "flashinfer",
                    "artifact_refs": ["agent_trace.md"],
                }
            ]
        )
    )

    with pytest.raises(LibraryCampaignVerificationError, match="environment_debug"):
        verify_library_campaign_dir(tmp_path)
