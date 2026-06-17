from __future__ import annotations

import asyncio
import json
import sys
from urllib.parse import quote

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from evoinfer_mcp.dream.mcp_server import mcp
from evoinfer_mcp.dream.mcp_server import dream_export_memory_store_tool
from evoinfer_mcp.dream.mcp_server import dream_extract_and_write_memories_tool
from evoinfer_mcp.dream.mcp_server import dream_extract_memory_candidates_tool
from evoinfer_mcp.dream.mcp_server import dream_get_memory_tool
from evoinfer_mcp.dream.mcp_server import dream_get_agent_protocol_tool
from evoinfer_mcp.dream.mcp_server import dream_import_memory_store_tool
from evoinfer_mcp.dream.mcp_server import dream_list_memories_tool
from evoinfer_mcp.dream.mcp_server import dream_promote_memory_tool
from evoinfer_mcp.dream.mcp_server import dream_record_feedback_tool
from evoinfer_mcp.dream.mcp_server import dream_reject_memory_tool
from evoinfer_mcp.dream.mcp_server import dream_search_memories_tool
from evoinfer_mcp.dream.mcp_server import dream_stage_memory_candidate_tool
from evoinfer_mcp.dream.mcp_server import dream_write_environment_debug_memory_tool
from evoinfer_mcp.dream.mcp_server import dream_write_optimization_memory_tool
from evoinfer_mcp.dream.mcp_server import record_dream_memory_feedback_tool
from evoinfer_mcp.dream.mcp_server import search_dream_memories_tool


def test_search_dream_memories_tool_returns_ranked_text_and_records_choice(
    tmp_path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    memory_file = memory_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "id": "opt_rmsnorm",
                        "category": "optimization",
                        "title": "CUDA RMSNorm block reduction",
                        "summary": "Use one block per row for hidden=4096 RMSNorm.",
                        "tags": ["cuda", "rmsnorm", "reduction"],
                        "environment": "limx",
                        "model_type": "llm",
                        "inference_backend": "cuda",
                        "success": True,
                        "detail_description": "Parallelize row reduction and strided writeback.",
                        "operation_semantics": [
                            "RMSNorm is a row-wise normalization reduction",
                        ],
                        "correctness_invariants": [
                            "max_abs_error <= 1e-3",
                        ],
                        "safe_transfer_notes": [
                            "reuse block-level row reduction",
                        ],
                        "unsafe_transfer_notes": [
                            "do not reuse RMSNorm final formula for Softmax",
                        ],
                        "chosen": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))

    text = search_dream_memories_tool(
        query="optimize CUDA RMSNorm reduction kernel",
        category="optimization",
        tags=["cuda", "rmsnorm"],
        top_k=3,
        record_choice=True,
    )

    assert "opt_rmsnorm" in text
    assert "CUDA RMSNorm block reduction" in text
    assert "Parallelize row reduction" in text
    assert "operation_semantics" in text
    assert "RMSNorm is a row-wise normalization reduction" in text
    assert "correctness_invariants" in text
    assert "safe_transfer_notes" in text
    assert "unsafe_transfer_notes" in text
    assert "do not reuse RMSNorm final formula for Softmax" in text

    persisted = json.loads(memory_file.read_text(encoding="utf-8"))
    assert persisted["memories"][0]["chosen"] == 1


@pytest.fixture()
def dream_memory_dir(tmp_path, monkeypatch):
    memory_dir = tmp_path / ".evoinfer"
    memory_file = memory_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "id": "opt_flashinfer_decode",
                        "category": "optimization",
                        "title": "FlashInfer decode attention baseline",
                        "summary": "Use FlashInfer decode kernels as a baseline for LLM attention.",
                        "tags": ["flashinfer", "decode", "attention"],
                        "environment": "rtx3090",
                        "model_type": "llm",
                        "inference_backend": "flashinfer",
                        "success": True,
                        "detail_description": "Compare custom decode attention with FlashInfer.",
                        "chosen": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))
    return memory_file


def test_dream_mcp_resource_template_can_be_read_and_records_choice(dream_memory_dir) -> None:
    async def read_template() -> str:
        templates = await mcp.list_resource_templates()
        assert any(template.uriTemplate == "dream://search/{category}/{query}" for template in templates)
        contents = await mcp.read_resource(
            f"dream://search/optimization/{quote('FlashInfer decode attention baseline')}"
        )
        return "\n".join(str(content.content) for content in contents)

    text = asyncio.run(read_template())

    assert "opt_flashinfer_decode" in text
    assert "FlashInfer decode attention baseline" in text
    persisted = json.loads(dream_memory_dir.read_text(encoding="utf-8"))
    assert persisted["memories"][0]["chosen"] == 1


def test_dream_mcp_feedback_tool_records_usefulness(dream_memory_dir) -> None:
    text = record_dream_memory_feedback_tool(
        memory_ids=["opt_flashinfer_decode", "missing"],
        useful=True,
        reason="Verifier passed after applying the retrieved FlashInfer baseline.",
        evidence_artifacts=["runs/flashinfer_decode_verifier.json"],
        source_session_id="session-1",
    )

    assert "opt_flashinfer_decode" in text
    assert "missing=missing" in text
    persisted = json.loads(dream_memory_dir.read_text(encoding="utf-8"))
    memory = persisted["memories"][0]
    assert memory["chosen"] == 1
    assert memory["useful_when_chosen"] == 1
    assert memory["useful_rate"] == 1.0


def test_dream_mcp_prefixed_search_and_feedback_aliases(dream_memory_dir) -> None:
    search_text = dream_search_memories_tool(
        query="FlashInfer decode attention baseline",
        category="optimization",
        tags=["flashinfer"],
        top_k=1,
        record_choice=True,
    )

    assert "opt_flashinfer_decode" in search_text

    feedback_text = dream_record_feedback_tool(
        memory_ids=["opt_flashinfer_decode"],
        useful=True,
        reason="Benchmark and verifier passed.",
        evidence_artifacts=["runs/flashinfer_decode_benchmark.json"],
        source_session_id="session-prefixed",
    )

    assert "opt_flashinfer_decode" in feedback_text
    persisted = json.loads(dream_memory_dir.read_text(encoding="utf-8"))
    memory = persisted["memories"][0]
    assert memory["chosen"] == 1
    assert memory["useful_when_chosen"] == 1
    assert memory["useful_rate"] == 1.0


def test_dream_mcp_feedback_requires_artifact_evidence(dream_memory_dir) -> None:
    with pytest.raises(ValueError, match="useful Dream memory feedback must include evidence artifact"):
        dream_record_feedback_tool(
            memory_ids=["opt_flashinfer_decode"],
            useful=True,
            reason="Verifier passed, but no artifact reference was supplied.",
            source_session_id="session-no-evidence-artifact",
        )

    persisted = json.loads(dream_memory_dir.read_text(encoding="utf-8"))
    memory = persisted["memories"][0]
    assert memory["chosen"] == 0
    assert memory.get("useful_when_chosen", 0) == 0


def test_dream_mcp_search_supports_agent_render_modes(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".evoinfer"
    memory_file = memory_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "id": "opt_cuda_softmax",
                        "category": "optimization",
                        "title": "CUDA Softmax shared-memory row reduction",
                        "summary": "One block computes one row and preserves probability sums.",
                        "tags": ["cuda", "softmax", "shared_memory"],
                        "environment": "RTX 3090",
                        "model_type": "operator-kernel",
                        "inference_backend": "cuda",
                        "success": True,
                        "metrics_before": {"latency_ms": 0.42},
                        "metrics_after": {"latency_ms": 0.08},
                        "benchmark_command": "./softmax_bench",
                        "correctness_artifacts": ["runs/softmax/correctness.json"],
                        "detail_description": "Use shared-memory row max/sum reductions.",
                        "operation_semantics": ["row-wise Softmax probability distribution"],
                        "correctness_invariants": ["row sums must remain 1.0"],
                        "safe_transfer_notes": ["reuse row-wise max/sum reduction pattern"],
                        "unsafe_transfer_notes": ["do not drop exp normalization"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))

    compact_text = dream_search_memories_tool(
        query="optimize cuda softmax",
        category="optimization",
        tags=["softmax"],
        top_k=1,
        record_choice=False,
        render_mode="compact",
        task_context="batch=32, seq=2048, fp16",
    )
    assert "Dream memory compact results:" in compact_text
    assert "opt_cuda_softmax" in compact_text
    assert "details=" not in compact_text
    assert "task_context=batch=32, seq=2048, fp16" in compact_text

    actionable_text = dream_search_memories_tool(
        query="optimize cuda softmax",
        category="optimization",
        tags=["softmax"],
        top_k=1,
        record_choice=False,
        render_mode="agent_actionable",
    )
    assert "Apply as hypothesis:" in actionable_text
    assert "Validate with:" in actionable_text
    assert "./softmax_bench" in actionable_text
    assert "row sums must remain 1.0" in actionable_text

    protocol_text = dream_search_memories_tool(
        query="optimize cuda softmax",
        category="optimization",
        tags=["softmax"],
        top_k=1,
        record_choice=False,
        render_mode="artifact_protocol",
    )
    assert "Artifact protocol:" in protocol_text
    assert "Do not claim success without benchmark and correctness artifacts." in protocol_text
    assert "recommended_action=apply_as_hypothesis" in protocol_text
    assert "runs/softmax/correctness.json" in protocol_text


def test_dream_mcp_agent_protocol_guides_active_memory_lifecycle() -> None:
    payload = json.loads(
        dream_get_agent_protocol_tool(
            task_type="optimization",
            workdir="/workspace/campaign",
        )
    )

    assert payload["identity"] == "EvoInfer Dream is an MCP memory manager"
    assert payload["task_type"] == "optimization"
    assert payload["workdir"] == "/workspace/campaign"
    assert [phase["phase"] for phase in payload["lifecycle"]] == [
        "task_start",
        "stuck_or_branch_point",
        "completion",
        "promotion",
        "feedback",
    ]
    tool_sequence = [
        tool for phase in payload["lifecycle"] for tool in phase["required_tools"]
    ]
    assert "dream_search_memories" in tool_sequence
    assert "dream_stage_memory_candidate" in tool_sequence
    assert "dream_extract_memory_candidates" in tool_sequence
    assert "dream_extract_and_write_memories" in tool_sequence
    assert "dream_promote_memory" in tool_sequence
    assert "dream_record_feedback" in tool_sequence
    completion_phase = next(
        phase for phase in payload["lifecycle"] if phase["phase"] == "completion"
    )
    assert "dream_extract_memory_candidates" in completion_phase["required_tools"]
    assert "dream_extract_and_write_memories" in completion_phase["required_tools"]
    assert "Never promote a successful optimization without correctness artifacts." in payload["gates"]
    assert "Do not optimize for token/context reduction as the primary outcome." in payload["metrics"]


def test_dream_mcp_agent_protocol_reports_mandatory_session(monkeypatch) -> None:
    monkeypatch.setenv("EVOINFER_DREAM_SESSION_ID", "session-abc")
    monkeypatch.setenv("EVOINFER_DREAM_MANDATORY", "1")

    payload = json.loads(
        dream_get_agent_protocol_tool(
            task_type="optimization",
            workdir="/workspace/campaign",
        )
    )

    assert payload["mode"] == "mandatory-session"
    assert payload["session_id"] == "session-abc"
    assert payload["mandatory"] is True
    assert "Call Dream before task-local exploration." in payload["mandatory_rules"]


def test_dream_mcp_write_list_get_export_import_roundtrip(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))

    opt_text = dream_write_optimization_memory_tool(
        memory_json=json.dumps(
            {
                "id": "opt_roundtrip",
                "title": "CUDA Softmax row-wise shared-memory reduction",
                "summary": "Use row-wise shared-memory reduction for Softmax.",
                "tags": ["cuda", "softmax"],
                "environment": "RTX 3090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
                "metrics_before": {"latency_ms": 0.42},
                "metrics_after": {"latency_ms": 0.08},
                "success": True,
                "detail_description": "One CUDA block computes one row and normalizes probabilities.",
                "artifacts": ["runs/softmax/benchmark.json"],
                "correctness_artifacts": ["runs/softmax/correctness.json"],
            }
        )
    )
    debug_text = dream_write_environment_debug_memory_tool(
        memory_json=json.dumps(
            {
                "id": "env_roundtrip",
                "title": "Force CPU STT when libcuda is absent",
                "summary": "Avoid CUDA STT backend on CPU-only machine.",
                "tags": ["stt", "cuda"],
                "environment": "limx",
                "debug_type": "runtime",
                "component": "speech-to-text",
                "issue_signature": "Library libcuda.so.12 is not found",
                "symptoms": "Voice input failed.",
                "root_cause": "The STT backend selected CUDA without CUDA runtime libraries.",
                "solution": "Set the STT device to cpu and compute type to int8.",
                "verification": "Voice input transcribed successfully.",
                "diagnostic_artifacts": ["logs/stt-libcuda-error.txt"],
                "verification_artifacts": ["logs/stt-cpu-transcription-ok.txt"],
                "success": True,
            }
        )
    )

    assert "opt_roundtrip" in opt_text
    assert "env_roundtrip" in debug_text

    listed = json.loads(dream_list_memories_tool())
    assert [memory["id"] for memory in listed["memories"]] == [
        "opt_roundtrip",
        "env_roundtrip",
    ]

    listed_debug = json.loads(dream_list_memories_tool(category="environment_debug"))
    assert [memory["id"] for memory in listed_debug["memories"]] == ["env_roundtrip"]

    fetched = json.loads(dream_get_memory_tool(memory_id="opt_roundtrip"))
    assert fetched["memory"]["title"] == "CUDA Softmax row-wise shared-memory reduction"

    exported = dream_export_memory_store_tool()
    exported_payload = json.loads(exported)
    assert [memory["id"] for memory in exported_payload["memories"]] == [
        "opt_roundtrip",
        "env_roundtrip",
    ]

    import_preview = json.loads(
        dream_import_memory_store_tool(
            memory_store_json=json.dumps(
                {
                    "memories": [
                        {
                            "id": "env_imported",
                            "category": "environment_debug",
                            "title": "Imported debug memory",
                            "environment": "limx",
                            "debug_type": "runtime",
                            "component": "test",
                            "issue_signature": "import smoke",
                            "symptoms": "dry run",
                            "root_cause": "test",
                            "solution": "test",
                            "verification": "test",
                            "verification_artifacts": ["logs/imported-ok.txt"],
                            "success": True,
                        }
                    ]
                }
            ),
            dry_run=True,
        )
    )
    assert import_preview == {
        "dry_run": True,
        "imported_count": 1,
        "memory_ids": ["env_imported"],
    }
    assert "env_imported" not in dream_list_memories_tool()

    import_result = json.loads(
        dream_import_memory_store_tool(
            memory_store_json=json.dumps(exported_payload),
            dry_run=False,
        )
    )
    assert import_result["dry_run"] is False
    assert import_result["imported_count"] == 2


def test_dream_mcp_import_memory_store_rejects_unverified_memory(
    tmp_path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))

    with pytest.raises(ValueError, match="optimization memories must include"):
        dream_import_memory_store_tool(
            memory_store_json=json.dumps(
                {
                    "memories": [
                        {
                            "id": "opt_import_no_evidence",
                            "category": "optimization",
                            "title": "Imported optimization without evidence",
                            "summary": "Import must not bypass durable write gates.",
                            "environment": "RTX 3090",
                            "model_type": "operator-kernel",
                            "inference_backend": "cuda",
                            "success": True,
                            "detail_description": "No benchmark or correctness artifacts.",
                        }
                    ]
                }
            ),
            dry_run=True,
        )

    assert json.loads(dream_list_memories_tool()) == {"memories": []}


def test_dream_mcp_environment_debug_write_requires_artifact_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))

    with pytest.raises(ValueError, match="environment debug memories must include"):
        dream_write_environment_debug_memory_tool(
            memory_json=json.dumps(
                {
                    "id": "env_no_evidence",
                    "title": "CPU STT fallback without evidence",
                    "summary": "Do not persist plain chat summaries as debug memories.",
                    "tags": ["stt", "cuda"],
                    "environment": "limx",
                    "debug_type": "runtime",
                    "component": "speech-to-text",
                    "issue_signature": "Library libcuda.so.12 is not found",
                    "symptoms": "Voice input failed.",
                    "root_cause": "The STT backend selected CUDA without CUDA runtime libraries.",
                    "solution": "Set the STT device to cpu and compute type to int8.",
                    "verification": "Voice input transcribed successfully.",
                    "success": True,
                }
            )
        )

    assert json.loads(dream_list_memories_tool()) == {"memories": []}


def test_dream_mcp_extracts_memory_candidates_from_artifacts(tmp_path) -> None:
    workdir = tmp_path / "campaign" / "rep01" / "with_memory"
    workdir.mkdir(parents=True)
    (workdir / "benchmark_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "correctness_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "category": "candidate_optimization",
                    "title": "FLA route policy can avoid bad dtype branches",
                    "summary": "Use route evidence before expanding every dtype candidate.",
                    "library": "fla",
                    "operator": "linear_attn_generation",
                    "backend": "fla.fused_recurrent_linear_attn",
                    "artifact_refs": [
                        "benchmark_raw.json",
                        "correctness_raw.json",
                        "missing_profile.txt",
                    ],
                },
                {
                    "category": "negative_optimization",
                    "title": "Do not skip dtype candidates without evidence",
                    "lesson": "Skipping without retrieved memory evidence is unsafe.",
                    "artifact_refs": ["agent_trace.md"],
                },
            ]
        ),
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_memory_candidates_tool(workdir=str(workdir)))

    assert [candidate["status"] for candidate in payload["candidates"]] == [
        "candidate",
        "negative",
    ]
    assert payload["candidates"][0]["category"] == "optimization"
    assert payload["candidates"][0]["artifact_refs"] == [
        "benchmark_raw.json",
        "correctness_raw.json",
    ]
    assert payload["candidates"][0]["missing_artifact_refs"] == ["missing_profile.txt"]
    assert payload["candidates"][0]["tags"] == [
        "fla",
        "linear_attn_generation",
        "fla.fused_recurrent_linear_attn",
    ]


def test_dream_mcp_extracts_candidate_from_standard_artifacts_without_candidate_file(
    tmp_path,
) -> None:
    workdir = tmp_path / "campaign" / "rep01" / "with_memory"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 3090",
                "cuda_available": True,
                "model_type": "operator-kernel",
                "inference_backend": "triton",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "rmsnorm",
                "backend": "triton",
                "baseline": {"latency_ms": 1.2},
                "candidate": {"latency_ms": 0.7},
            }
        ),
        encoding="utf-8",
    )
    (workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 0.0}),
        encoding="utf-8",
    )
    (workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python verify.py"}),
        encoding="utf-8",
    )
    (workdir / "agent_trace.md").write_text(
        "Changed Triton block size and verified RMSNorm correctness.",
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_memory_candidates_tool(workdir=str(workdir)))
    [candidate] = payload["candidates"]

    assert candidate["extraction_source"] == "standard_artifacts"
    assert candidate["promotion_ready"] is True
    assert candidate["artifact_refs"] == [
        "environment.json",
        "benchmark_raw.json",
        "correctness_raw.json",
        "verifier_result.json",
        "agent_trace.md",
    ]
    promotion_input = candidate["promotion_input"]
    assert promotion_input["success"] is True
    assert promotion_input["environment"] == "RTX 3090"
    assert promotion_input["model_type"] == "operator-kernel"
    assert promotion_input["inference_backend"] == "triton"
    assert promotion_input["metrics_before"] == {"latency_ms": 1.2}
    assert promotion_input["metrics_after"] == {"latency_ms": 0.7}
    assert promotion_input["correctness_artifacts"] == ["correctness_raw.json"]
    assert promotion_input["benchmark_command"] == "python verify.py"


def test_dream_mcp_extract_and_write_requires_profiler_for_positive_optimization(
    tmp_path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))
    workdir = tmp_path / "campaign" / "rep01" / "with_memory"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 4090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "softmax",
                "backend": "cuda",
                "baseline": {"latency_ms": 0.42},
                "candidate": {"latency_ms": 0.08, "speedup": 5.25},
            }
        ),
        encoding="utf-8",
    )
    (workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 1e-6}),
        encoding="utf-8",
    )
    (workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python verify.py"}),
        encoding="utf-8",
    )
    (workdir / "agent_trace.md").write_text(
        "Changed softmax reduction and verified correctness.",
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_and_write_memories_tool(workdir=str(workdir)))

    assert payload["written_count"] == 0
    assert payload["written_memory_ids"] == []
    assert payload["rejected_count"] == 1
    assert "profiler or source-level bottleneck evidence" in payload["rejected_candidates"][0]["blockers"][0]
    assert json.loads(dream_list_memories_tool()) == {"memories": []}


def test_dream_mcp_extract_and_write_persists_candidate_from_verified_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))
    workdir = tmp_path / "campaign" / "rep01" / "with_memory"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 4090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "softmax",
                "backend": "cuda",
                "baseline": {"latency_ms": 0.42},
                "candidate": {"latency_ms": 0.08, "speedup": 5.25},
            }
        ),
        encoding="utf-8",
    )
    (workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 1e-6}),
        encoding="utf-8",
    )
    (workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python verify.py"}),
        encoding="utf-8",
    )
    (workdir / "profiler_summary.json").write_text(
        json.dumps(
            {
                "tool": "ncu",
                "bottleneck_type": "memory_bandwidth",
                "top_kernels": ["softmax_kernel"],
                "signals": ["shared-memory row reduction lowered global reads"],
            }
        ),
        encoding="utf-8",
    )
    (workdir / "agent_trace.md").write_text(
        "Changed softmax reduction and verified correctness.",
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_and_write_memories_tool(workdir=str(workdir)))

    assert payload["written_count"] == 1
    assert payload["rejected_count"] == 0
    assert payload["written_memory_ids"] == ["opt_artifact_with_memory"]
    [memory] = json.loads(dream_list_memories_tool())["memories"]
    assert memory["id"] == "opt_artifact_with_memory"
    assert memory["status"] == "candidate"
    assert memory["evidence_level"] == "smoke"
    assert memory["success"] is True
    assert memory["profiler_artifacts"] == ["profiler_summary.json"]
    assert memory["bottleneck_type"] == "memory_bandwidth"
    assert memory["correctness_artifacts"] == ["correctness_raw.json"]


def test_dream_mcp_extract_and_write_preserves_flat_artifact_workload_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))
    workdir = tmp_path / "campaign" / "rep01" / "match"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 4090",
                "python": "3.12.4",
                "platform": "linux-x86_64",
                "model_type": "operator-kernel",
                "inference_backend": "python-numpy",
                "dtype": "float32",
                "batch": 64,
                "hidden": 512,
            }
        ),
        encoding="utf-8",
    )
    (workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "row_sum",
                "backend": "python-numpy",
                "dtype": "float32",
                "workload": {"batch": 64, "hidden": 512},
                "baseline_ms": 7.015,
                "candidate_ms": 0.205,
                "speedup": 34.18,
            }
        ),
        encoding="utf-8",
    )
    (workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 0.0, "threshold": 1e-6}),
        encoding="utf-8",
    )
    (workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python benchmark.py"}),
        encoding="utf-8",
    )
    (workdir / "profiler_summary.json").write_text(
        json.dumps(
            {
                "tool": "source",
                "bottleneck_type": "python_loop_overhead",
                "evidence": "Python loop over rows dominated runtime.",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "agent_trace.md").write_text(
        "Replaced Python row loop with vectorized numpy row_sum and verified correctness.",
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_and_write_memories_tool(workdir=str(workdir)))

    assert payload["written_count"] == 1
    [memory] = json.loads(dream_list_memories_tool())["memories"]
    assert memory["title"] == "Artifact-backed row_sum optimization candidate"
    assert memory["environment"] == "RTX 4090"
    assert memory["inference_backend"] == "python-numpy"
    assert memory["precision"] == {"dtype": "float32"}
    assert memory["workload"] == {"batch": 64, "hidden": 512}
    assert memory["metrics_before"] == {"latency_ms": 7.015}
    assert memory["metrics_after"] == {
        "latency_ms": 0.205,
        "speedup": 34.18,
        "max_abs_error": 0.0,
    }
    assert "float32" in memory["tags"]
    assert "batch=64" in memory["applicability"]
    assert "hidden=512" in memory["applicability"]


def test_dream_mcp_standard_artifact_environment_falls_back_to_runtime_context(
    tmp_path,
) -> None:
    workdir = tmp_path / "campaign" / "rep01" / "cpu"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "python": "3.12.4",
                "platform": "Linux-x86_64",
                "backend": "python-numpy",
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "backend": "python-numpy",
                "baseline_ms": 1.0,
                "candidate_ms": 0.5,
                "speedup": 2.0,
            }
        ),
        encoding="utf-8",
    )
    (workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    (workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_memory_candidates_tool(workdir=str(workdir)))
    [candidate] = payload["candidates"]

    assert candidate["promotion_input"]["environment"] == "python-numpy on Linux-x86_64"


def test_dream_mcp_extracts_bottleneck_from_realistic_ncu_raw_artifact(
    tmp_path,
) -> None:
    workdir = tmp_path / "campaign" / "rep01" / "with_memory"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 4090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "softmax",
                "backend": "cuda",
                "baseline": {"latency_ms": 0.42},
                "candidate": {"latency_ms": 0.08, "speedup": 5.25},
            }
        ),
        encoding="utf-8",
    )
    (workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 1e-6}),
        encoding="utf-8",
    )
    (workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python verify.py"}),
        encoding="utf-8",
    )
    (workdir / "ncu_report.json").write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "kernelName": "softmax_kernel",
                        "metrics": {
                            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 87.5,
                            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 34.0,
                            "gpu__time_duration.sum": 0.08,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_memory_candidates_tool(workdir=str(workdir)))
    [candidate] = payload["candidates"]

    assert candidate["promotion_input"]["profiler_artifacts"] == ["ncu_report.json"]
    assert candidate["promotion_input"]["bottleneck_type"] == "memory_bandwidth"


def test_dream_mcp_extracts_bottleneck_from_realistic_nsys_raw_artifact(
    tmp_path,
) -> None:
    workdir = tmp_path / "campaign" / "rep01" / "with_memory"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 4090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "decode",
                "backend": "cuda",
                "baseline": {"latency_ms": 1.1},
                "candidate": {"latency_ms": 0.9},
            }
        ),
        encoding="utf-8",
    )
    (workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 1e-6}),
        encoding="utf-8",
    )
    (workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python verify.py"}),
        encoding="utf-8",
    )
    (workdir / "nsys_report.json").write_text(
        json.dumps(
            {
                "cuda_api": [
                    {"name": "cudaLaunchKernel", "time_ms": 6.2},
                    {"name": "cudaMemcpyAsync", "time_ms": 0.2},
                ],
                "gpu_kernels": [
                    {"name": "decode_kernel", "time_ms": 0.9},
                    {"name": "small_kernel", "time_ms": 0.02},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_memory_candidates_tool(workdir=str(workdir)))
    [candidate] = payload["candidates"]

    assert candidate["promotion_input"]["profiler_artifacts"] == ["nsys_report.json"]
    assert candidate["promotion_input"]["bottleneck_type"] == "launch_overhead"


@pytest.mark.asyncio
async def test_dream_mcp_stdio_extracts_standard_artifact_candidate_without_candidate_file(
    tmp_path,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    workdir = tmp_path / "campaign" / "rep01" / "with_memory"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 4090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "softmax",
                "backend": "cuda",
                "baseline": {"latency_ms": 2.4},
                "candidate": {"latency_ms": 1.1},
            }
        ),
        encoding="utf-8",
    )
    (workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 0.0}),
        encoding="utf-8",
    )
    (workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python verify_softmax.py"}),
        encoding="utf-8",
    )

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(memory_dir)},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "dream_extract_memory_candidates",
                arguments={"workdir": str(workdir)},
            )

    payload = json.loads(_tool_text(result))
    [candidate] = payload["candidates"]
    assert candidate["extraction_source"] == "standard_artifacts"
    assert candidate["promotion_ready"] is True
    assert candidate["promotion_input"]["benchmark_command"] == "python verify_softmax.py"
    assert candidate["promotion_input"]["correctness_artifacts"] == [
        "correctness_raw.json"
    ]


@pytest.mark.asyncio
async def test_dream_mcp_stdio_extract_and_write_artifact_memories(
    tmp_path,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    opt_workdir = tmp_path / "campaign" / "opt"
    opt_workdir.mkdir(parents=True)
    (opt_workdir / "environment.json").write_text(
        json.dumps(
            {
                "gpu_name": "RTX 4090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
            }
        ),
        encoding="utf-8",
    )
    (opt_workdir / "benchmark_raw.json").write_text(
        json.dumps(
            {
                "operator": "softmax",
                "backend": "cuda",
                "baseline": {"latency_ms": 2.4},
                "candidate": {"latency_ms": 1.1},
            }
        ),
        encoding="utf-8",
    )
    (opt_workdir / "correctness_raw.json").write_text(
        json.dumps({"passed": True, "max_abs_error": 0.0}),
        encoding="utf-8",
    )
    (opt_workdir / "verifier_result.json").write_text(
        json.dumps({"status": "passed", "command": "python verify_softmax.py"}),
        encoding="utf-8",
    )
    (opt_workdir / "profiler_summary.json").write_text(
        json.dumps({"tool": "ncu", "bottleneck_type": "memory_bandwidth"}),
        encoding="utf-8",
    )

    env_workdir = tmp_path / "campaign" / "env_debug"
    env_workdir.mkdir(parents=True)
    (env_workdir / "environment.json").write_text(
        json.dumps(
            {
                "classification": "environment_debug",
                "gpu_name": "RTX 3090",
                "driver_version": "580.126.18",
                "flashinfer_version": "0.6.12",
            }
        ),
        encoding="utf-8",
    )
    (env_workdir / "environment_debug.json").write_text(
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
    (env_workdir / "diagnostic.log").write_text("TSZ# binary header token", encoding="utf-8")
    (env_workdir / "verification.log").write_text("show-config ok", encoding="utf-8")

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(memory_dir)},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            opt_result = await session.call_tool(
                "dream_extract_and_write_memories",
                arguments={"workdir": str(opt_workdir)},
            )
            env_result = await session.call_tool(
                "dream_extract_and_write_memories",
                arguments={"workdir": str(env_workdir), "category_hint": "environment_debug"},
            )
            listed = await session.call_tool("dream_list_memories", arguments={})

    assert json.loads(_tool_text(opt_result))["written_memory_ids"] == [
        "opt_artifact_opt"
    ]
    assert json.loads(_tool_text(env_result))["written_memory_ids"] == [
        "env_artifact_env_debug"
    ]
    memories = json.loads(_tool_text(listed))["memories"]
    assert [memory["id"] for memory in memories] == [
        "opt_artifact_opt",
        "env_artifact_env_debug",
    ]
    assert memories[0]["profiler_artifacts"] == ["profiler_summary.json"]
    assert memories[1]["category"] == "environment_debug"


def test_dream_mcp_extracts_environment_debug_candidate_from_standard_artifacts(
    tmp_path,
) -> None:
    workdir = tmp_path / "campaign" / "rep01" / "with_environment_debug_memory"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps(
            {
                "classification": "environment_debug",
                "gpu_name": "RTX 3090",
                "driver_version": "580.126.18",
                "cuda_available": True,
                "flashinfer_version": "0.6.12",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "environment_debug.json").write_text(
        json.dumps(
            {
                "debug_type": "dependency",
                "component": "flashinfer",
                "issue_signature": "FlashInfer JIT consumed binary TSZ header",
                "symptoms": "single_decode_with_kv_cache failed before correctness.",
                "root_cause": "The JIT include path resolved binary package resources as headers.",
                "solution": "Install the matching flashinfer-cubin package and clear stale JIT cache.",
                "verification": "FlashInfer import and show-config passed after reinstall.",
                "success": True,
                "commands": [
                    "pip install --force-reinstall flashinfer-python flashinfer-cubin"
                ],
                "error_messages": ["TSZ# binary header token"],
            }
        ),
        encoding="utf-8",
    )
    (workdir / "diagnostic.log").write_text(
        "fatal error: TSZ# binary header token",
        encoding="utf-8",
    )
    (workdir / "verification.log").write_text(
        "flashinfer show-config ok",
        encoding="utf-8",
    )
    (workdir / "agent_trace.md").write_text(
        "Diagnosed FlashInfer JIT resource path and reinstalled matching cubin wheel.",
        encoding="utf-8",
    )

    payload = json.loads(
        dream_extract_memory_candidates_tool(
            workdir=str(workdir),
            category_hint="environment_debug",
        )
    )
    [candidate] = payload["candidates"]

    assert candidate["category"] == "environment_debug"
    assert candidate["extraction_source"] == "standard_environment_debug_artifacts"
    assert candidate["promotion_ready"] is True
    assert candidate["artifact_refs"] == [
        "environment.json",
        "environment_debug.json",
        "diagnostic.log",
        "verification.log",
        "agent_trace.md",
    ]
    promotion_input = candidate["promotion_input"]
    assert promotion_input["debug_type"] == "dependency"
    assert promotion_input["component"] == "flashinfer"
    assert promotion_input["environment"] == "RTX 3090"
    assert promotion_input["issue_signature"] == (
        "FlashInfer JIT consumed binary TSZ header"
    )
    assert promotion_input["diagnostic_artifacts"] == [
        "environment_debug.json",
        "diagnostic.log",
        "agent_trace.md",
    ]
    assert promotion_input["verification_artifacts"] == ["verification.log"]
    assert promotion_input["success"] is True


@pytest.mark.asyncio
async def test_dream_mcp_stdio_extracts_environment_debug_candidate_from_artifacts(
    tmp_path,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    workdir = tmp_path / "campaign" / "rep01" / "with_environment_debug_memory"
    workdir.mkdir(parents=True)
    (workdir / "environment.json").write_text(
        json.dumps({"classification": "environment_debug", "gpu_name": "limx RTX 3090"}),
        encoding="utf-8",
    )
    (workdir / "environment_debug.json").write_text(
        json.dumps(
            {
                "debug_type": "runtime",
                "component": "speech-to-text",
                "issue_signature": "Library libcuda.so.12 is not found",
                "symptoms": "Voice input failed in the browser.",
                "root_cause": "The STT backend selected CUDA on a CPU-only runtime.",
                "solution": "Force STT device=cpu and compute_type=int8.",
                "verification": "Voice input produced text after restart.",
                "success": True,
            }
        ),
        encoding="utf-8",
    )
    (workdir / "diagnostic.log").write_text("libcuda.so.12 missing", encoding="utf-8")
    (workdir / "verification.log").write_text("cpu transcription ok", encoding="utf-8")

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(memory_dir)},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "dream_extract_memory_candidates",
                arguments={
                    "workdir": str(workdir),
                    "category_hint": "environment_debug",
                },
            )

    payload = json.loads(_tool_text(result))
    [candidate] = payload["candidates"]
    assert candidate["promotion_ready"] is True
    assert candidate["promotion_input"]["component"] == "speech-to-text"
    assert candidate["promotion_input"]["verification_artifacts"] == [
        "verification.log"
    ]


def test_dream_mcp_extract_marks_candidate_not_ready_when_schema_gate_fails(
    tmp_path,
) -> None:
    workdir = tmp_path / "campaign" / "rep01" / "with_memory"
    workdir.mkdir(parents=True)
    (workdir / "benchmark_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "correctness_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "opt_missing_correctness_artifacts",
                    "category": "candidate_optimization",
                    "title": "Candidate with raw refs but incomplete durable schema",
                    "summary": "Extraction must not mark this ready for durable write.",
                    "environment": "RTX 3090",
                    "model_type": "operator-kernel",
                    "inference_backend": "cuda",
                    "success": True,
                    "detail_description": "Has raw refs but omits correctness_artifacts.",
                    "artifact_refs": ["benchmark_raw.json", "correctness_raw.json"],
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = json.loads(dream_extract_memory_candidates_tool(workdir=str(workdir)))
    [candidate] = payload["candidates"]

    assert candidate["promotion_ready"] is False
    assert any(
        "successful optimization memories must include correctness evidence" in blocker
        for blocker in candidate["promotion_blockers"]
    )


def test_dream_mcp_stages_memory_candidate_without_writing_memory_store(
    tmp_path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))
    workdir = tmp_path / "campaign" / "rep01"
    workdir.mkdir(parents=True)
    (workdir / "benchmark_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "correctness_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "existing_candidate",
                    "category": "negative_optimization",
                    "title": "Existing negative transfer note",
                    "artifact_refs": ["benchmark_raw.json"],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = json.loads(
        dream_stage_memory_candidate_tool(
            workdir=str(workdir),
            candidate_json=json.dumps(
                {
                    "id": "opt_fla_route",
                    "category": "candidate_optimization",
                    "title": "FLA route policy reduces local search",
                    "summary": "Use route evidence before expanding every dtype branch.",
                    "tags": ["fla", "route_policy"],
                    "artifact_refs": ["benchmark_raw.json", "correctness_raw.json"],
                }
            ),
        )
    )

    assert result["candidate"]["id"] == "opt_fla_route"
    assert result["candidate"]["artifact_refs"] == [
        "benchmark_raw.json",
        "correctness_raw.json",
    ]
    assert result["candidate_count"] == 2

    staged = json.loads((workdir / "dream_write_candidates.json").read_text(encoding="utf-8"))
    assert [candidate["id"] for candidate in staged] == [
        "existing_candidate",
        "opt_fla_route",
    ]
    assert json.loads(dream_list_memories_tool()) == {"memories": []}

    extracted = json.loads(dream_extract_memory_candidates_tool(workdir=str(workdir)))
    assert [candidate["title"] for candidate in extracted["candidates"]] == [
        "Existing negative transfer note",
        "FLA route policy reduces local search",
    ]


def test_dream_mcp_staging_candidate_requires_existing_relative_artifacts(tmp_path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir()
    (workdir / "benchmark_raw.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_refs"):
        dream_stage_memory_candidate_tool(
            workdir=str(workdir),
            candidate_json=json.dumps(
                {
                    "category": "candidate_optimization",
                    "title": "Missing evidence",
                    "artifact_refs": [],
                }
            ),
        )

    with pytest.raises(ValueError, match="relative paths"):
        dream_stage_memory_candidate_tool(
            workdir=str(workdir),
            candidate_json=json.dumps(
                {
                    "category": "candidate_optimization",
                    "title": "Path traversal evidence",
                    "artifact_refs": ["../outside.json"],
                }
            ),
        )

    with pytest.raises(ValueError, match="existing artifact"):
        dream_stage_memory_candidate_tool(
            workdir=str(workdir),
            candidate_json=json.dumps(
                {
                    "category": "candidate_optimization",
                    "title": "Nonexistent evidence",
                    "artifact_refs": ["missing.json"],
                }
            ),
        )


def test_dream_mcp_promotes_and_rejects_existing_memories(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))
    dream_write_optimization_memory_tool(
        memory_json=json.dumps(
            {
                "id": "opt_promote_me",
                "title": "Promotable optimization memory",
                "summary": "Has benchmark and correctness evidence.",
                "environment": "RTX 3090",
                "model_type": "operator-kernel",
                "inference_backend": "cuda",
                "metrics_before": {"latency_ms": 1.0},
                "metrics_after": {"latency_ms": 0.5},
                "success": True,
                "detail_description": "Validated optimization.",
                "artifacts": ["runs/bench.json"],
                "correctness_artifacts": ["runs/correctness.json"],
            }
        )
    )

    promoted = json.loads(
        dream_promote_memory_tool(
            memory_id="opt_promote_me",
            reason="3-repeat verifier passed with lower wall-clock time.",
            evidence_artifacts=["runs/correctness.json"],
            evidence_level="verified",
        )
    )
    assert promoted["memory"]["status"] == "promoted"
    assert promoted["memory"]["promotion_reason"] == (
        "3-repeat verifier passed with lower wall-clock time."
    )
    assert promoted["memory"]["promotion_artifacts"] == ["runs/correctness.json"]
    assert promoted["memory"]["evidence_level"] == "verified"

    rejected = json.loads(
        dream_reject_memory_tool(
            memory_id="opt_promote_me",
            reason="Later correctness regression invalidated transfer.",
            evidence_artifacts=["runs/correctness-regression.json"],
            negative=True,
        )
    )
    assert rejected["memory"]["status"] == "negative"
    assert rejected["memory"]["rejection_reason"] == (
        "Later correctness regression invalidated transfer."
    )
    assert rejected["memory"]["rejection_artifacts"] == [
        "runs/correctness-regression.json"
    ]


def test_dream_mcp_promote_reject_require_decision_artifacts(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".evoinfer"
    memory_file = memory_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "id": "opt_decision_gate",
                        "category": "optimization",
                        "title": "Decision gate candidate",
                        "summary": "Promote/reject should be evidence-backed.",
                        "environment": "RTX 3090",
                        "model_type": "operator-kernel",
                        "inference_backend": "cuda",
                        "success": True,
                        "detail_description": "Candidate with correctness evidence.",
                        "artifacts": ["runs/bench.json"],
                        "correctness_artifacts": ["runs/correctness.json"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))

    with pytest.raises(ValueError, match="promotion evidence_artifacts are required"):
        dream_promote_memory_tool(
            memory_id="opt_decision_gate",
            reason="Looks good from the chat summary only.",
        )

    with pytest.raises(ValueError, match="rejection evidence_artifacts are required"):
        dream_reject_memory_tool(
            memory_id="opt_decision_gate",
            reason="Looks unsafe from the chat summary only.",
            negative=True,
        )

    persisted = json.loads(memory_file.read_text(encoding="utf-8"))
    memory = persisted["memories"][0]
    assert memory.get("status", "candidate") == "candidate"
    assert memory.get("promotion_artifacts", []) == []
    assert memory.get("rejection_artifacts", []) == []


@pytest.mark.asyncio
async def test_dream_mcp_stdio_exposes_open_box_tool_names(dream_memory_dir) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(dream_memory_dir.parents[1])},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

    tool_names = {tool.name for tool in tools.tools}
    assert {
        "dream_search_memories",
        "dream_get_agent_protocol",
        "dream_get_memory",
        "dream_list_memories",
        "dream_write_optimization_memory",
        "dream_write_environment_debug_memory",
        "dream_record_feedback",
        "dream_export_memory_store",
        "dream_import_memory_store",
        "dream_stage_memory_candidate",
        "dream_extract_memory_candidates",
        "dream_extract_and_write_memories",
        "dream_promote_memory",
        "dream_reject_memory",
    } <= tool_names


@pytest.mark.asyncio
async def test_dream_mcp_stdio_roundtrips_agent_protocol(tmp_path) -> None:
    memory_dir = tmp_path / ".evoinfer"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(memory_dir)},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "dream_get_agent_protocol",
                arguments={"task_type": "environment_debug", "workdir": "/tmp/evoinfer-run"},
            )

    payload = json.loads(_tool_text(result))
    assert payload["task_type"] == "environment_debug"
    assert payload["workdir"] == "/tmp/evoinfer-run"
    assert payload["lifecycle"][0]["phase"] == "task_start"
    assert "dream_search_memories" in payload["lifecycle"][0]["required_tools"]
    assert "dream_stage_memory_candidate" in payload["lifecycle"][2]["required_tools"]


@pytest.mark.asyncio
async def test_dream_mcp_stdio_lifecycle_promotes_staged_candidate_and_records_feedback(
    tmp_path,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    memory_file = memory_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "memories": [
                    {
                        "id": "opt_prior_route",
                        "category": "optimization",
                        "title": "Use route evidence before expanding dtype branches",
                        "summary": "A prior FLA route memory reduced local candidate search.",
                        "tags": ["fla", "route_policy"],
                        "environment": "RTX 3090",
                        "model_type": "operator-kernel",
                        "inference_backend": "fla",
                        "success": True,
                        "artifacts": ["prior/baseline.json", "prior/candidate.json"],
                        "correctness_artifacts": ["prior/correctness.json"],
                        "chosen": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    workdir = tmp_path / "campaign" / "rep01"
    workdir.mkdir(parents=True)
    (workdir / "baseline.json").write_text('{"latency_ms": 1.0}', encoding="utf-8")
    (workdir / "candidate.json").write_text('{"latency_ms": 0.5}', encoding="utf-8")
    (workdir / "correctness.json").write_text('{"max_abs_error": 0.0}', encoding="utf-8")
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(memory_dir)},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            protocol = await session.call_tool(
                "dream_get_agent_protocol",
                arguments={"task_type": "optimization", "workdir": str(workdir)},
            )
            assert "dream_stage_memory_candidate" in _tool_text(protocol)

            search = await session.call_tool(
                "dream_search_memories",
                arguments={
                    "query": "FLA route policy dtype branch expansion",
                    "category": "optimization",
                    "tags": ["fla", "route_policy"],
                    "top_k": 1,
                    "record_choice": True,
                    "render_mode": "artifact_protocol",
                },
            )
            assert "opt_prior_route" in _tool_text(search)

            stage = await session.call_tool(
                "dream_stage_memory_candidate",
                arguments={
                    "workdir": str(workdir),
                    "candidate_json": json.dumps(
                        {
                            "id": "opt_lifecycle_candidate",
                            "category": "candidate_optimization",
                            "title": "FLA route policy halves dtype branch search time",
                            "summary": "Route memory selected a narrower candidate set.",
                            "tags": ["fla", "route_policy", "dtype"],
                            "environment": "RTX 3090",
                            "model_type": "operator-kernel",
                            "inference_backend": "fla",
                            "metrics_before": {"wall_clock_s": 1.0},
                            "metrics_after": {"wall_clock_s": 0.5},
                            "success": True,
                            "detail_description": (
                                "The agent used retrieved route evidence to avoid broad dtype "
                                "branch expansion, then verified correctness."
                            ),
                            "artifact_refs": [
                                "baseline.json",
                                "candidate.json",
                                "correctness.json",
                            ],
                            "artifacts": ["baseline.json", "candidate.json"],
                            "correctness_artifacts": ["correctness.json"],
                        }
                    ),
                },
            )
            assert "opt_lifecycle_candidate" in _tool_text(stage)

            extract = await session.call_tool(
                "dream_extract_memory_candidates",
                arguments={"workdir": str(workdir)},
            )
            extracted = json.loads(_tool_text(extract))
            [candidate] = extracted["candidates"]
            assert candidate["promotion_ready"] is True
            promotion_input = candidate["promotion_input"]
            assert promotion_input["id"] == "opt_lifecycle_candidate"
            assert promotion_input["artifact_refs"] == [
                "baseline.json",
                "candidate.json",
                "correctness.json",
            ]
            assert promotion_input["correctness_artifacts"] == ["correctness.json"]

            write = await session.call_tool(
                "dream_write_optimization_memory",
                arguments={"memory_json": json.dumps(promotion_input)},
            )
            written = json.loads(_tool_text(write))
            assert written["memory"]["id"] == "opt_lifecycle_candidate"

            promote = await session.call_tool(
                "dream_promote_memory",
                arguments={
                    "memory_id": "opt_lifecycle_candidate",
                    "reason": "Lifecycle smoke verifier artifacts passed.",
                    "evidence_artifacts": [
                        str(workdir / "baseline.json"),
                        str(workdir / "candidate.json"),
                        str(workdir / "correctness.json"),
                    ],
                    "evidence_level": "verified",
                },
            )
            promoted = json.loads(_tool_text(promote))
            assert promoted["memory"]["status"] == "promoted"

            feedback = await session.call_tool(
                "dream_record_feedback",
                arguments={
                    "memory_ids": ["opt_prior_route"],
                    "useful": True,
                    "reason": "Lifecycle smoke used prior route memory and verifier artifacts passed.",
                    "evidence_artifacts": [
                        str(workdir / "baseline.json"),
                        str(workdir / "candidate.json"),
                        str(workdir / "correctness.json"),
                    ],
                    "source_session_id": "stdio-lifecycle-smoke",
                },
            )
            assert "opt_prior_route" in _tool_text(feedback)

            listed = await session.call_tool("dream_list_memories", arguments={})

    memories = {
        memory["id"]: memory
        for memory in json.loads(_tool_text(listed))["memories"]
    }
    assert memories["opt_prior_route"]["chosen"] == 1
    assert memories["opt_prior_route"]["useful_when_chosen"] == 1
    assert memories["opt_lifecycle_candidate"]["status"] == "promoted"
    assert memories["opt_lifecycle_candidate"]["promotion_reason"] == (
        "Lifecycle smoke verifier artifacts passed."
    )


@pytest.mark.asyncio
async def test_dream_mcp_stdio_roundtrips_write_list_get_export_import(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / ".evoinfer"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(memory_dir))
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(memory_dir)},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            write_result = await session.call_tool(
                "dream_write_optimization_memory",
                arguments={
                    "memory_json": json.dumps(
                        {
                            "id": "opt_stdio",
                            "title": "MCP stdio optimization memory",
                            "summary": "Written through MCP stdio.",
                            "tags": ["mcp", "cuda"],
                            "environment": "limx",
                            "model_type": "operator-kernel",
                            "inference_backend": "cuda",
                            "metrics_before": {"latency_ms": 1.0},
                            "metrics_after": {"latency_ms": 0.5},
                            "success": True,
                            "detail_description": "A stdio write smoke memory.",
                            "artifacts": ["runs/stdio/benchmark.json"],
                            "correctness_artifacts": ["runs/stdio/correctness.json"],
                        }
                    )
                },
            )
            assert "opt_stdio" in _tool_text(write_result)

            list_result = await session.call_tool("dream_list_memories", arguments={})
            assert "opt_stdio" in _tool_text(list_result)

            get_result = await session.call_tool(
                "dream_get_memory",
                arguments={"memory_id": "opt_stdio"},
            )
            assert "MCP stdio optimization memory" in _tool_text(get_result)

            export_result = await session.call_tool(
                "dream_export_memory_store",
                arguments={},
            )
            exported = _tool_text(export_result)
            assert "opt_stdio" in exported

            import_result = await session.call_tool(
                "dream_import_memory_store",
                arguments={"memory_store_json": exported, "dry_run": True},
            )
            import_payload = json.loads(_tool_text(import_result))
            assert import_payload["dry_run"] is True
            assert import_payload["memory_ids"] == ["opt_stdio"]


@pytest.mark.asyncio
async def test_dream_mcp_stdio_rejects_environment_debug_write_without_evidence(
    tmp_path,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(memory_dir)},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "dream_write_environment_debug_memory",
                arguments={
                    "memory_json": json.dumps(
                        {
                            "id": "env_stdio_no_evidence",
                            "title": "No evidence environment debug memory",
                            "summary": "Plain chat summary must not be persisted.",
                            "environment": "limx",
                            "debug_type": "runtime",
                            "component": "speech-to-text",
                            "issue_signature": "Library libcuda.so.12 is not found",
                            "symptoms": "Voice input failed.",
                            "root_cause": "CUDA backend selected on CPU-only host.",
                            "solution": "Use CPU STT backend.",
                            "verification": "Voice input transcribed successfully.",
                            "success": True,
                        }
                    )
                },
            )
            list_result = await session.call_tool("dream_list_memories", arguments={})

    assert result.isError is True
    assert "environment debug memories must include" in _tool_text(result)
    assert json.loads(_tool_text(list_result)) == {"memories": []}


@pytest.mark.asyncio
async def test_dream_mcp_stdio_import_rejects_memory_without_evidence(
    tmp_path,
) -> None:
    memory_dir = tmp_path / ".evoinfer"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(memory_dir)},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            import_result = await session.call_tool(
                "dream_import_memory_store",
                arguments={
                    "memory_store_json": json.dumps(
                        {
                            "memories": [
                                {
                                    "id": "opt_stdio_import_no_evidence",
                                    "category": "optimization",
                                    "title": "Unverified imported memory",
                                    "summary": "Import should not bypass write gates.",
                                    "environment": "RTX 3090",
                                    "model_type": "operator-kernel",
                                    "inference_backend": "cuda",
                                    "success": True,
                                    "detail_description": "No raw evidence.",
                                }
                            ]
                        }
                    ),
                    "dry_run": True,
                },
            )
            list_result = await session.call_tool("dream_list_memories", arguments={})

    assert import_result.isError is True
    assert "optimization memories must include" in _tool_text(import_result)
    assert json.loads(_tool_text(list_result)) == {"memories": []}


@pytest.mark.asyncio
async def test_dream_mcp_stdio_exposes_search_resource_template(dream_memory_dir) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(dream_memory_dir.parents[1])},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            templates = await session.list_resource_templates()
            assert any(
                template.uriTemplate == "dream://search/{category}/{query}"
                for template in templates.resourceTemplates
            )
            result = await session.read_resource(
                f"dream://search/optimization/{quote('FlashInfer decode attention baseline')}"
            )

    text = "\n".join(str(content.text) for content in result.contents)
    assert "opt_flashinfer_decode" in text
    persisted = json.loads(dream_memory_dir.read_text(encoding="utf-8"))
    assert persisted["memories"][0]["chosen"] == 1


def _tool_text(result) -> str:
    return "\n".join(str(content.text) for content in result.content)
