from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoinfer_mcp.evoinfer.operator_campaign_verifier import (
    OperatorCampaignVerificationError,
    verify_operator_campaign_dir,
)


def _write_operator_artifacts(root: Path) -> None:
    (root / "operator_smoke.py").write_text("# benchmark script\n")
    (root / "environment.json").write_text(
        json.dumps(
            {
                "gpu": "NVIDIA GeForce RTX 3090",
                "driver_version": "580.126.18",
                "torch_version": "2.12.0+cu126",
                "triton_version": "3.7.0",
                "cuda_available": True,
            }
        )
    )
    (root / "correctness_raw.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "rmsnorm_fp16_b8_h1024",
                        "operator": "rmsnorm",
                        "candidate": "triton",
                        "max_abs_error": 0.0005,
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
                        "id": "rmsnorm_fp16_b8_h1024",
                        "operator": "rmsnorm",
                        "dtype": "float16",
                        "batch": 8,
                        "hidden": 1024,
                        "warmup_count": 10,
                        "repeat_count": 50,
                        "baseline_ms_mean": 0.03,
                        "candidate_ms_mean": 0.02,
                        "speedup": 1.5,
                    }
                ]
            }
        )
    )
    (root / "agent_trace.md").write_text("commands and observations\n")
    (root / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "category": "candidate_optimization",
                    "operator": "rmsnorm",
                    "artifact_refs": [
                        "benchmark_raw.json",
                        "correctness_raw.json",
                        "operator_smoke.py",
                    ],
                }
            ]
        )
    )


def test_operator_campaign_verifier_accepts_valid_artifacts(tmp_path: Path) -> None:
    _write_operator_artifacts(tmp_path)

    result = verify_operator_campaign_dir(tmp_path)

    assert result["benchmark_entry_count"] == 1
    assert result["correctness_entry_count"] == 1
    assert result["dream_candidate_count"] == 1


def test_operator_campaign_verifier_rejects_benchmark_without_matching_correctness(
    tmp_path: Path,
) -> None:
    _write_operator_artifacts(tmp_path)
    (tmp_path / "correctness_raw.json").write_text(json.dumps({"entries": []}))

    with pytest.raises(OperatorCampaignVerificationError, match="correctness"):
        verify_operator_campaign_dir(tmp_path)


def test_operator_campaign_verifier_rejects_dream_candidate_without_artifact_refs(
    tmp_path: Path,
) -> None:
    _write_operator_artifacts(tmp_path)
    (tmp_path / "dream_write_candidates.json").write_text(
        json.dumps([{"category": "candidate_optimization"}])
    )

    with pytest.raises(OperatorCampaignVerificationError, match="artifact_refs"):
        verify_operator_campaign_dir(tmp_path)


def test_operator_campaign_verifier_rejects_malformed_driver_field(
    tmp_path: Path,
) -> None:
    _write_operator_artifacts(tmp_path)
    env = json.loads((tmp_path / "environment.json").read_text())
    env["driver"] = 82
    env.pop("driver_version", None)
    (tmp_path / "environment.json").write_text(json.dumps(env))

    with pytest.raises(OperatorCampaignVerificationError, match="driver"):
        verify_operator_campaign_dir(tmp_path)


def test_operator_campaign_verifier_rejects_passed_correctness_with_high_error(
    tmp_path: Path,
) -> None:
    _write_operator_artifacts(tmp_path)
    correctness = json.loads((tmp_path / "correctness_raw.json").read_text())
    correctness["entries"][0]["max_abs_error"] = 0.5
    correctness["entries"][0]["max_rel_error"] = 0.5
    correctness["entries"][0]["passed"] = True
    (tmp_path / "correctness_raw.json").write_text(json.dumps(correctness))

    with pytest.raises(OperatorCampaignVerificationError, match="max_abs_error"):
        verify_operator_campaign_dir(tmp_path)


def test_operator_campaign_verifier_rejects_high_row_sum_error(
    tmp_path: Path,
) -> None:
    _write_operator_artifacts(tmp_path)
    correctness = json.loads((tmp_path / "correctness_raw.json").read_text())
    correctness["entries"][0]["max_row_sum_error"] = 0.2
    (tmp_path / "correctness_raw.json").write_text(json.dumps(correctness))

    with pytest.raises(OperatorCampaignVerificationError, match="max_row_sum_error"):
        verify_operator_campaign_dir(tmp_path)


def test_operator_campaign_verifier_requires_masked_output_error_for_masked_softmax(
    tmp_path: Path,
) -> None:
    _write_operator_artifacts(tmp_path)
    correctness = json.loads((tmp_path / "correctness_raw.json").read_text())
    correctness["entries"][0]["operator"] = "masked_softmax"
    correctness["entries"][0]["max_row_sum_error"] = 0.0
    (tmp_path / "correctness_raw.json").write_text(json.dumps(correctness))

    benchmark = json.loads((tmp_path / "benchmark_raw.json").read_text())
    benchmark["entries"][0]["operator"] = "masked_softmax"
    benchmark["entries"][0]["seq_len"] = 1024
    benchmark["entries"][0].pop("hidden", None)
    (tmp_path / "benchmark_raw.json").write_text(json.dumps(benchmark))

    with pytest.raises(OperatorCampaignVerificationError, match="max_masked_output_abs"):
        verify_operator_campaign_dir(tmp_path)


def test_operator_campaign_verifier_rejects_high_masked_output_error(
    tmp_path: Path,
) -> None:
    _write_operator_artifacts(tmp_path)
    correctness = json.loads((tmp_path / "correctness_raw.json").read_text())
    correctness["entries"][0]["operator"] = "masked_softmax"
    correctness["entries"][0]["max_row_sum_error"] = 0.0
    correctness["entries"][0]["max_masked_output_abs"] = 0.2
    (tmp_path / "correctness_raw.json").write_text(json.dumps(correctness))

    benchmark = json.loads((tmp_path / "benchmark_raw.json").read_text())
    benchmark["entries"][0]["operator"] = "masked_softmax"
    benchmark["entries"][0]["seq_len"] = 1024
    benchmark["entries"][0].pop("hidden", None)
    (tmp_path / "benchmark_raw.json").write_text(json.dumps(benchmark))

    with pytest.raises(OperatorCampaignVerificationError, match="max_masked_output_abs"):
        verify_operator_campaign_dir(tmp_path)
