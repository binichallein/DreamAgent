from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evoinfer_mcp.evoinfer.dream_protocol_verifier import (
    DreamProtocolVerificationError,
    verify_dream_protocol_campaign_result,
)


def _write_campaign_result(
    path: Path,
    *,
    work_dir: Path,
    dream_retrieval_events: list[dict],
    verification_status: str = "passed",
) -> None:
    payload = {
        "name": "protocol-smoke",
        "prompt": "Run an inference optimization campaign.",
        "work_dir": str(work_dir.parent),
        "started_at": 1,
        "ended_at": 2,
        "duration_seconds": 1,
        "memory_before": {"memory_count": 1, "total_chosen": 0},
        "memory_after": {"memory_count": 1, "total_chosen": 2},
        "runs": [
            {
                "arm_name": "without_memory",
                "dream_enabled": False,
                "session_id": "without",
                "work_dir": str(work_dir.parent / "without_memory"),
                "status": "finished",
                "started_at": 1,
                "ended_at": 2,
                "duration_seconds": 1,
            },
            {
                "arm_name": "with_memory",
                "dream_enabled": True,
                "session_id": "with",
                "work_dir": str(work_dir),
                "status": "finished",
                "started_at": 1,
                "ended_at": 2,
                "duration_seconds": 1,
                "dream_retrieval_events": dream_retrieval_events,
                "dream_retrieval_count": len(dream_retrieval_events),
                "dream_retrieved_memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "verification_status": verification_status,
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_candidate_artifacts(work_dir: Path) -> None:
    work_dir.mkdir(parents=True)
    (work_dir / "benchmark_raw.json").write_text('{"entries": []}', encoding="utf-8")
    (work_dir / "correctness_raw.json").write_text('{"entries": []}', encoding="utf-8")
    (work_dir / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "opt_protocol_candidate",
                    "category": "candidate_optimization",
                    "title": "RMSNorm block-parallel CUDA route",
                    "artifact_refs": ["benchmark_raw.json", "correctness_raw.json"],
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_standard_optimization_artifacts(work_dir: Path) -> None:
    work_dir.mkdir(parents=True)
    (work_dir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 3090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "rmsnorm",
                "backend": "cuda",
                "baseline": {"latency_ms": 1.2},
                "candidate": {"latency_ms": 0.8},
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 0.0}),
        encoding="utf-8",
    )
    (work_dir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python verify.py"}),
        encoding="utf-8",
    )
    (work_dir / "agent_trace.md").write_text(
        "Agent verified RMSNorm candidate from standard artifacts.",
        encoding="utf-8",
    )


def _write_standard_environment_debug_artifacts(work_dir: Path) -> None:
    work_dir.mkdir(parents=True)
    (work_dir / "environment.json").write_text(
        json.dumps(
            {
                "classification": "environment_debug",
                "gpu_name": "RTX 3090",
                "driver_version": "580.126.18",
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "environment_debug.json").write_text(
        json.dumps(
            {
                "debug_type": "dependency",
                "component": "flashinfer",
                "issue_signature": "FlashInfer JIT consumed binary TSZ header",
                "symptoms": "single_decode failed before correctness.",
                "root_cause": "JIT resolved binary package resource as a header.",
                "solution": "Install matching cubin package and clear stale cache.",
                "verification": "show-config passed after reinstall.",
                "success": True,
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "diagnostic.log").write_text("TSZ# binary header token", encoding="utf-8")
    (work_dir / "verification.log").write_text("show-config ok", encoding="utf-8")
    (work_dir / "agent_trace.md").write_text(
        "Diagnosed FlashInfer JIT deployment issue.",
        encoding="utf-8",
    )


def test_dream_protocol_verifier_accepts_start_stuck_and_completion_artifacts(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "cuda rmsnorm",
                "categories": ["optimization"],
                "top_k_per_category": 3,
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 0,
            },
            {
                "trigger": "stuck",
                "query": "rmsnorm benchmark regressed",
                "categories": ["optimization"],
                "top_k_per_category": 3,
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 14,
            },
        ],
    )

    result = verify_dream_protocol_campaign_result(
        campaign_path,
        require_stuck_retrieval=True,
    )

    assert result["passed"] is True
    assert result["dream_enabled_run_count"] == 1
    assert result["retrieval_event_count"] == 2
    assert result["completion_candidate_count"] == 1
    assert result["candidate_artifact_ref_count"] == 2
    assert result["artifact_valid_success_count"] == 1


def test_dream_protocol_verifier_accepts_standard_artifact_completion_extraction(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_standard_optimization_artifacts(work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "cuda rmsnorm",
                "categories": ["optimization"],
                "top_k_per_category": 3,
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 0,
            },
            {
                "trigger": "stuck",
                "query": "rmsnorm route failed, search prior memory again",
                "categories": ["optimization"],
                "top_k_per_category": 3,
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 12,
            },
        ],
    )

    result = verify_dream_protocol_campaign_result(
        campaign_path,
        require_stuck_retrieval=True,
    )

    assert result["passed"] is True
    assert result["completion_candidate_count"] == 1
    assert result["candidate_artifact_ref_count"] == 5


def test_dream_protocol_verifier_accepts_environment_debug_standard_artifacts(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_environment_debug_memory"
    _write_standard_environment_debug_artifacts(work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "flashinfer jit deployment error",
                "categories": ["environment_debug"],
                "memory_ids": ["env_flashinfer_jit_cache"],
                "result_count": 1,
                "step_count": 0,
            },
            {
                "trigger": "stuck",
                "query": "same jit error after reinstall",
                "categories": ["environment_debug"],
                "memory_ids": ["env_flashinfer_jit_cache"],
                "result_count": 1,
                "step_count": 12,
            },
        ],
    )

    result = verify_dream_protocol_campaign_result(
        campaign_path,
        require_stuck_retrieval=True,
    )

    assert result["passed"] is True
    assert result["completion_candidate_count"] == 1
    assert result["candidate_artifact_ref_count"] == 5


def test_dream_protocol_verifier_rejects_missing_task_start_retrieval(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "periodic",
                "query": "cuda rmsnorm",
                "categories": ["optimization"],
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 20,
            }
        ],
    )

    with pytest.raises(DreamProtocolVerificationError, match="task-start retrieval"):
        verify_dream_protocol_campaign_result(campaign_path)


def test_dream_protocol_verifier_rejects_missing_stuck_retrieval_when_required(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "cuda rmsnorm",
                "categories": ["optimization"],
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 0,
            }
        ],
    )

    with pytest.raises(DreamProtocolVerificationError, match="stuck retrieval"):
        verify_dream_protocol_campaign_result(
            campaign_path,
            require_stuck_retrieval=True,
        )


def test_dream_protocol_verifier_rejects_candidate_with_missing_artifact_ref(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    candidates = json.loads((work_dir / "dream_write_candidates.json").read_text())
    candidates[0]["artifact_refs"].append("missing_profiler.json")
    (work_dir / "dream_write_candidates.json").write_text(json.dumps(candidates))
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "cuda rmsnorm",
                "categories": ["optimization"],
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 0,
            }
        ],
    )

    with pytest.raises(DreamProtocolVerificationError, match="missing artifact ref"):
        verify_dream_protocol_campaign_result(campaign_path)


def test_dream_protocol_verifier_resolves_relocated_artifact_root(
    tmp_path: Path,
) -> None:
    remote_work_dir = Path(
        "/remote/evoinfer-campaign-work/fla-route-suite/rep01/with_memory"
    )
    artifact_root = tmp_path / "campaign-artifacts" / "fla-route-suite-work"
    local_work_dir = artifact_root / "rep01" / "with_memory"
    _write_candidate_artifacts(local_work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=remote_work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "fla route policy",
                "categories": ["optimization"],
                "memory_ids": ["opt_fla_route_policy"],
                "result_count": 1,
                "step_count": 0,
            }
        ],
    )

    result = verify_dream_protocol_campaign_result(
        campaign_path,
        artifact_root=artifact_root,
    )

    assert result["passed"] is True
    assert result["completion_candidate_count"] == 1
    assert result["resolved_work_dirs"] == [str(local_work_dir)]


def test_dream_protocol_verifier_accepts_route_decision_with_retrieved_skip_evidence(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    (work_dir / "route_decision.json").write_text(
        json.dumps(
            {
                "selection_policy": "memory_route_policy",
                "selected_dtypes": ["float32"],
                "audit_dtypes": ["bfloat16"],
                "avoid_dtypes": ["float16"],
                "selected_memory_ids": ["opt_fla_float16_negative"],
                "skip_evidence": {
                    "float16": ["opt_fla_float16_negative"],
                },
                "reason": "Retrieved negative dtype-boundary memory.",
            }
        ),
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "fla float16 dtype boundary",
                "categories": ["optimization"],
                "memory_ids": ["opt_fla_float16_negative"],
                "result_count": 1,
                "step_count": 0,
            }
        ],
    )

    result = verify_dream_protocol_campaign_result(
        campaign_path,
        require_transfer_safety=True,
    )

    assert result["passed"] is True
    assert result["transfer_safety_checked_count"] == 1


def test_dream_protocol_verifier_rejects_route_decision_without_skip_evidence(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    (work_dir / "route_decision.json").write_text(
        json.dumps(
            {
                "selection_policy": "memory_route_policy",
                "selected_dtypes": ["float32"],
                "audit_dtypes": [],
                "avoid_dtypes": ["float16"],
                "selected_memory_ids": ["opt_fla_float16_negative"],
                "skip_evidence": {},
                "reason": "Claimed memory route without per-dtype evidence.",
            }
        ),
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "fla float16 dtype boundary",
                "categories": ["optimization"],
                "memory_ids": ["opt_fla_float16_negative"],
                "result_count": 1,
                "step_count": 0,
            }
        ],
    )

    with pytest.raises(DreamProtocolVerificationError, match="missing skip evidence"):
        verify_dream_protocol_campaign_result(
            campaign_path,
            require_transfer_safety=True,
        )


def test_dream_protocol_verifier_accepts_artifact_memory_write_evidence(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "cuda softmax",
                "categories": ["optimization"],
                "memory_ids": ["opt_cuda_softmax_prior"],
                "result_count": 1,
                "step_count": 0,
            }
        ],
    )
    payload = json.loads(campaign_path.read_text(encoding="utf-8"))
    payload["runs"][1]["dream_auto_write_count"] = 1
    payload["runs"][1]["dream_written_memory_ids"] = ["opt_artifact_with_memory"]
    campaign_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dream_protocol_campaign_result(
        campaign_path,
        require_artifact_memory_write=True,
    )

    assert result["passed"] is True
    assert result["artifact_memory_write_count"] == 1
    assert result["artifact_memory_write_blocker_count"] == 0


def test_dream_protocol_verifier_rejects_missing_artifact_memory_write_evidence(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "cuda softmax",
                "categories": ["optimization"],
                "memory_ids": ["opt_cuda_softmax_prior"],
                "result_count": 1,
                "step_count": 0,
            }
        ],
    )

    with pytest.raises(
        DreamProtocolVerificationError,
        match="missing artifact memory write evidence",
    ):
        verify_dream_protocol_campaign_result(
            campaign_path,
            require_artifact_memory_write=True,
        )


def test_dream_protocol_verifier_script_prints_machine_readable_pass(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "with_memory"
    _write_candidate_artifacts(work_dir)
    campaign_path = tmp_path / "campaign.json"
    _write_campaign_result(
        campaign_path,
        work_dir=work_dir,
        dream_retrieval_events=[
            {
                "trigger": "campaign_start",
                "query": "cuda rmsnorm",
                "categories": ["optimization"],
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 0,
            },
            {
                "trigger": "stuck",
                "query": "cuda rmsnorm failed variant",
                "categories": ["optimization"],
                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                "result_count": 1,
                "step_count": 12,
            },
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evoinfer_dream_protocol_verifier.py",
            str(campaign_path),
            "--require-stuck-retrieval",
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DREAM_PROTOCOL_VERIFIER_PASS" in completed.stdout
    assert '"retrieval_event_count": 2' in completed.stdout
