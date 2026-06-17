"""MCP server exposing EvoInfer Dream memory retrieval to Codex."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from evoinfer_mcp.dream.memory import (
    DreamMemoryCategory,
    DreamMemoryEvidenceLevel,
    DreamMemoryFeedbackInput,
    DreamMemorySnapshotInput,
    DreamMemorySearchInput,
    DreamMemorySearchResult,
    EnvironmentDebugMemoryInput,
    OptimizationMemoryInput,
    create_environment_debug_memory,
    create_optimization_memory,
    list_dream_memories,
    record_dream_memory_feedback,
    restore_dream_memory_snapshot,
    search_dream_memories,
)
from evoinfer_mcp.utils.io import atomic_json_write

DreamSearchRenderMode = Literal["full", "compact", "agent_actionable", "artifact_protocol"]
DreamAgentTaskType = Literal["optimization", "environment_debug", "mixed"]

mcp = FastMCP(
    "evoinfer-dream",
    instructions=(
        "Retrieve EvoInfer Dream memories for inference optimization and "
        "environment deployment/debug tasks. Validate every retrieved memory with "
        "real environment checks, benchmarks, profiling, and correctness tests. "
        "Stage new lessons as artifact-backed candidates before promoting durable memories."
    ),
)


@mcp.tool(
    name="dream_get_agent_protocol",
    description=(
        "Return the EvoInfer Dream lifecycle protocol for agents. Call this at the "
        "start of an inference optimization or environment-debug task to learn when "
        "to search, stage candidates, extract, promote, and record feedback."
    ),
)
def dream_get_agent_protocol_tool(
    task_type: DreamAgentTaskType = "optimization",
    workdir: str | None = None,
) -> str:
    """Return a machine-readable protocol for active Dream memory use."""

    return _json_response(
        {
            "identity": "EvoInfer Dream is an MCP memory manager",
            "task_type": task_type,
            "workdir": workdir,
            "lifecycle": [
                {
                    "phase": "task_start",
                    "goal": (
                        "Retrieve prior specialist experience before local exploration."
                    ),
                    "required_tools": ["dream_search_memories"],
                    "tool_arguments": {
                        "render_mode": "artifact_protocol",
                        "record_choice": True,
                        "task_context": (
                            "Summarize hardware, backend, model/operator, workload, and "
                            "current failure or optimization goal."
                        ),
                    },
                },
                {
                    "phase": "stuck_or_branch_point",
                    "goal": (
                        "Search again when local exploration stalls, when profiler evidence "
                        "changes the suspected bottleneck, or before switching libraries/backends."
                    ),
                    "required_tools": ["dream_search_memories"],
                    "trigger_examples": [
                        "No artifact-valid progress after 10-20 agent steps.",
                        "Benchmark regressed or correctness failed.",
                        "Environment error blocks deployment or profiling.",
                        "Choosing between CUDA, Triton, PyTorch, FlashInfer, FLA, or another route.",
                    ],
                },
                {
                    "phase": "completion",
                    "goal": (
                        "Extract artifact-backed candidate lessons without polluting the durable "
                        "memory store. Use standard campaign artifacts when available; stage a "
                        "candidate only when the standard artifacts are insufficient."
                    ),
                    "required_tools": [
                        "dream_stage_memory_candidate",
                        "dream_extract_memory_candidates",
                        "dream_extract_and_write_memories",
                    ],
                    "artifact_requirements": [
                        "benchmark artifacts or raw command output for performance claims",
                        "correctness artifacts for successful optimization claims",
                        "profiler/source/log artifacts when they support the diagnosis",
                        "environment snapshot, commands, logs, or verification artifacts for debug claims",
                    ],
                },
                {
                    "phase": "promotion",
                    "goal": (
                        "Promote or reject only evidence-backed durable memories after reviewing extracted candidates."
                    ),
                    "required_tools": [
                        "dream_write_optimization_memory",
                        "dream_write_environment_debug_memory",
                        "dream_promote_memory",
                        "dream_reject_memory",
                    ],
                    "artifact_requirements": [
                        "dream_promote_memory requires evidence_artifacts reviewed for the promotion decision",
                        "dream_reject_memory requires evidence_artifacts reviewed for the rejection or negative constraint decision",
                    ],
                },
                {
                    "phase": "feedback",
                    "goal": (
                        "Record whether retrieved memories helped after verified task evidence exists."
                    ),
                    "required_tools": ["dream_record_feedback"],
                    "artifact_requirements": [
                        "useful=true requires evidence_artifacts that point to supporting benchmark, verifier, profiler, correctness, log, or resolved-debug artifacts",
                        "do not increment useful_when_chosen from conversation-only summaries",
                    ],
                },
            ],
            "gates": [
                "Treat retrieved memories as hypotheses, not authority.",
                "Never promote a successful optimization without correctness artifacts.",
                "Never claim speedup without comparable baseline and candidate benchmark artifacts.",
                "Record failed attempts as negative/rejected only when the failure condition is precise.",
                "Do not store unrelated personal facts, generic coding tips, or unverified summaries.",
            ],
            "metrics": [
                "Primary: wall-clock task time.",
                "Primary: artifact-valid success rate.",
                "Primary: repeated-known-error reduction.",
                "Do not optimize for token/context reduction as the primary outcome.",
                "Secondary: token/context cost for mechanism analysis.",
            ],
        }
    )


@mcp.tool(
    name="dream_search_memories",
    description=(
        "Search EvoInfer Dream memories by query, category, and tags. Use this "
        "when an inference optimization or environment-debug task is stalled, or "
        "before applying a known kernel/backend optimization pattern."
    ),
)
def dream_search_memories_tool(
    query: str,
    category: Literal["optimization", "environment_debug"] = "optimization",
    tags: list[str] | None = None,
    top_k: int = 5,
    record_choice: bool = True,
    render_mode: DreamSearchRenderMode = "full",
    task_context: str | None = None,
) -> str:
    """Return a compact, model-readable Dream memory search result."""

    search_query = query
    if task_context:
        search_query = f"{query}\n\nTask context: {task_context}"
    request = DreamMemorySearchInput(
        query=search_query,
        category=_normalize_category(category),
        tags=tags or [],
        top_k=max(1, min(20, top_k)),
        record_choice=record_choice,
    )
    response = search_dream_memories(request)
    if not response.results:
        return "No Dream memories matched the query."

    lines = _search_header(render_mode, task_context=task_context)
    for index, result in enumerate(response.results, start=1):
        lines.extend(_format_result(index, result, render_mode=render_mode))
    return "\n".join(lines)


@mcp.tool(
    name="dream_record_feedback",
    description=(
        "Record whether retrieved EvoInfer Dream memories were useful after a "
        "verified optimization/debug outcome. Use useful=true only with concrete "
        "benchmark, profiler, correctness, verifier, or resolved-debug evidence; "
        "useful=true requires evidence_artifacts."
    ),
)
def dream_record_feedback_tool(
    memory_ids: list[str],
    useful: bool = True,
    reason: str | None = None,
    evidence_artifacts: list[str] | None = None,
    source_session_id: str | None = None,
) -> str:
    """Record useful/not-useful feedback for selected memories."""

    response = record_dream_memory_feedback(
        DreamMemoryFeedbackInput(
            memory_ids=memory_ids,
            useful=useful,
            reason=reason,
            evidence_artifacts=evidence_artifacts or [],
            source_session_id=source_session_id,
        )
    )
    updated_ids = [memory.id for memory in response.memories]
    parts = [
        f"Updated Dream memory feedback for {len(updated_ids)} memor{'y' if len(updated_ids) == 1 else 'ies'}.",
    ]
    if updated_ids:
        parts.append("updated=" + ", ".join(updated_ids))
    if response.missing_memory_ids:
        parts.append("missing=" + ", ".join(response.missing_memory_ids))
    return "\n".join(parts)


@mcp.tool(
    name="dream_write_optimization_memory",
    description=(
        "Write or update an EvoInfer optimization Dream memory from JSON. The JSON "
        "must satisfy the optimization memory schema and include artifact evidence."
    ),
)
def dream_write_optimization_memory_tool(memory_json: str) -> str:
    payload = _load_json_object(memory_json)
    memory = create_optimization_memory(OptimizationMemoryInput.model_validate(payload))
    return _json_response({"memory": memory.model_dump(mode="json", exclude_none=True)})


@mcp.tool(
    name="dream_write_environment_debug_memory",
    description=(
        "Write or update an EvoInfer environment/debug Dream memory from JSON. "
        "Use this for verified deployment, dependency, runtime, driver, auth, "
        "filesystem, network, or inference environment lessons."
    ),
)
def dream_write_environment_debug_memory_tool(memory_json: str) -> str:
    payload = _load_json_object(memory_json)
    memory = create_environment_debug_memory(
        EnvironmentDebugMemoryInput.model_validate(payload)
    )
    return _json_response({"memory": memory.model_dump(mode="json", exclude_none=True)})


@mcp.tool(
    name="dream_list_memories",
    description="List EvoInfer Dream memories, optionally filtered by category.",
)
def dream_list_memories_tool(
    category: Literal["optimization", "environment_debug"] | None = None,
) -> str:
    memories = list_dream_memories(_normalize_category(category) if category else None)
    return _json_response(
        {
            "memories": [
                memory.model_dump(mode="json", exclude_none=True) for memory in memories
            ]
        }
    )


@mcp.tool(
    name="dream_get_memory",
    description="Get a single EvoInfer Dream memory by stable memory id.",
)
def dream_get_memory_tool(memory_id: str) -> str:
    for memory in list_dream_memories():
        if memory.id == memory_id:
            return _json_response(
                {"memory": memory.model_dump(mode="json", exclude_none=True)}
            )
    return _json_response({"memory": None, "missing_memory_id": memory_id})


@mcp.tool(
    name="dream_export_memory_store",
    description="Export the complete EvoInfer Dream memory store as JSON.",
)
def dream_export_memory_store_tool() -> str:
    memories = list_dream_memories()
    return _json_response(
        {
            "version": 1,
            "memories": [
                memory.model_dump(mode="json", exclude_none=True) for memory in memories
            ],
        }
    )


@mcp.tool(
    name="dream_import_memory_store",
    description=(
        "Import an EvoInfer Dream memory store JSON snapshot. Use dry_run=true to "
        "validate without writing."
    ),
)
def dream_import_memory_store_tool(memory_store_json: str, dry_run: bool = True) -> str:
    payload = _load_json_object(memory_store_json)
    raw_memories = payload.get("memories", payload)
    snapshot = DreamMemorySnapshotInput.model_validate({"memories": raw_memories})
    memory_ids = [memory.id for memory in snapshot.memories]
    if dry_run:
        return _json_response(
            {"dry_run": True, "imported_count": len(memory_ids), "memory_ids": memory_ids}
        )
    memories = restore_dream_memory_snapshot(snapshot)
    return _json_response(
        {
            "dry_run": False,
            "imported_count": len(memories),
            "memory_ids": [memory.id for memory in memories],
        }
    )


@mcp.tool(
    name="dream_stage_memory_candidate",
    description=(
        "Append or upsert one artifact-backed candidate lesson into "
        "workdir/dream_write_candidates.json without writing the durable Dream memory "
        "store. Use this at task completion before dream_extract_memory_candidates "
        "and dream_promote_memory."
    ),
)
def dream_stage_memory_candidate_tool(workdir: str, candidate_json: str) -> str:
    root = Path(workdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = _sanitize_staged_candidate(root, _load_json_object(candidate_json))
    candidates_path = root / "dream_write_candidates.json"
    candidates = _read_candidate_list(candidates_path)

    candidate_id = candidate.get("id")
    if isinstance(candidate_id, str) and candidate_id:
        for index, existing in enumerate(candidates):
            if existing.get("id") == candidate_id:
                candidates[index] = candidate
                break
        else:
            candidates.append(candidate)
    else:
        candidates.append(candidate)

    atomic_json_write(candidates, candidates_path)
    return _json_response(
        {
            "workdir": str(root),
            "candidate": candidate,
            "candidate_count": len(candidates),
        }
    )


@mcp.tool(
    name="dream_extract_memory_candidates",
    description=(
        "Extract candidate Dream memories from a campaign/session workdir without "
        "writing them to the memory store. Reads dream_write_candidates.json when "
        "present; otherwise extracts conservative candidates from standard "
        "benchmark/correctness/verifier artifacts."
    ),
)
def dream_extract_memory_candidates_tool(
    workdir: str,
    category_hint: Literal["optimization", "environment_debug"] | None = None,
) -> str:
    root = Path(workdir)
    candidates_path = root / "dream_write_candidates.json"
    raw = _read_candidate_list(candidates_path)
    if not raw:
        raw = _standard_artifact_candidates(root, category_hint=category_hint)
    candidates: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        artifact_refs = [ref for ref in item.get("artifact_refs", []) if isinstance(ref, str)]
        existing_refs = [ref for ref in artifact_refs if (root / ref).exists()]
        missing_refs = [ref for ref in artifact_refs if not (root / ref).exists()]
        category = _candidate_category(item, category_hint)
        status = _candidate_status(item)
        promotion_input = _candidate_promotion_input(
            item,
            category=category,
            status=status,
            artifact_refs=existing_refs,
            source_workdir=str(root),
        )
        promotion_blockers = _promotion_blockers(
            promotion_input,
            category=category,
        )
        candidates.append(
            {
                "category": category,
                "status": status,
                "evidence_level": "smoke",
                "title": item.get("title", ""),
                "summary": item.get("summary") or item.get("lesson", ""),
                "tags": _candidate_tags(item),
                "extraction_source": item.get("extraction_source", "staged_candidate"),
                "artifact_refs": existing_refs,
                "missing_artifact_refs": missing_refs,
                "source_workdir": str(root),
                "promotion_ready": (
                    bool(existing_refs)
                    and not missing_refs
                    and not promotion_blockers
                ),
                "promotion_blockers": promotion_blockers,
                "promotion_input": promotion_input,
            }
        )
    return _json_response({"workdir": str(root), "candidates": candidates})


@mcp.tool(
    name="dream_extract_and_write_memories",
    description=(
        "Extract artifact-backed candidate memories from a workdir, verify the "
        "artifact evidence, and automatically write durable candidate memories. "
        "This never promotes memories; use dream_promote_memory after separate "
        "evidence review."
    ),
)
def dream_extract_and_write_memories_tool(
    workdir: str,
    category_hint: Literal["optimization", "environment_debug"] | None = None,
    dry_run: bool = False,
) -> str:
    extraction = json.loads(
        dream_extract_memory_candidates_tool(
            workdir=workdir,
            category_hint=category_hint,
        )
    )
    candidates = extraction.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    written_memories: list[dict[str, Any]] = []
    planned_memory_inputs: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        category = _normalize_category(str(candidate.get("category") or "optimization"))
        promotion_input = candidate.get("promotion_input")
        if not isinstance(promotion_input, dict):
            rejected_candidates.append(
                {
                    "title": candidate.get("title", ""),
                    "blockers": ["candidate missing promotion_input"],
                }
            )
            continue
        write_input = _prepare_auto_write_input(
            promotion_input,
            category=category,
            candidate_status=str(candidate.get("status") or "candidate"),
        )
        blockers = list(candidate.get("promotion_blockers") or [])
        blockers.extend(
            _auto_write_blockers(
                write_input,
                category=category,
                source_workdir=Path(str(extraction.get("workdir") or workdir)),
            )
        )
        blockers = list(dict.fromkeys(str(blocker) for blocker in blockers if blocker))
        if blockers:
            rejected_candidates.append(
                {
                    "title": candidate.get("title", ""),
                    "category": category,
                    "blockers": blockers,
                    "artifact_refs": candidate.get("artifact_refs", []),
                }
            )
            continue

        planned_memory_inputs.append(write_input)
        if dry_run:
            continue
        if category == "optimization":
            memory = create_optimization_memory(
                OptimizationMemoryInput.model_validate(write_input)
            )
        else:
            memory = create_environment_debug_memory(
                EnvironmentDebugMemoryInput.model_validate(write_input)
            )
        written_memories.append(memory.model_dump(mode="json", exclude_none=True))

    written_memory_ids = [
        str(memory.get("id"))
        for memory in written_memories
        if isinstance(memory.get("id"), str)
    ]
    return _json_response(
        {
            "workdir": str(extraction.get("workdir") or workdir),
            "dry_run": dry_run,
            "candidate_count": len(candidates),
            "written_count": len(written_memories),
            "written_memory_ids": written_memory_ids,
            "written_memories": written_memories,
            "planned_memory_inputs": planned_memory_inputs,
            "rejected_count": len(rejected_candidates),
            "rejected_candidates": rejected_candidates,
            "blockers": [
                blocker
                for candidate in rejected_candidates
                for blocker in candidate.get("blockers", [])
                if isinstance(blocker, str)
            ],
        }
    )


@mcp.tool(
    name="dream_promote_memory",
    description=(
        "Promote an existing Dream memory after evidence review. Optimization "
        "memories must already contain correctness evidence. The promotion call "
        "must include evidence_artifacts reviewed for the decision."
    ),
)
def dream_promote_memory_tool(
    memory_id: str,
    reason: str,
    evidence_artifacts: list[str] | None = None,
    evidence_level: DreamMemoryEvidenceLevel = "verified",
) -> str:
    if not reason.strip():
        raise ValueError("promotion reason is required")
    decision_artifacts = _require_decision_artifacts(
        evidence_artifacts,
        error_message="promotion evidence_artifacts are required",
    )

    def promote(memory):
        if memory.category == "optimization" and not memory.correctness_artifacts:
            raise ValueError("cannot promote optimization memory without correctness evidence")
        return memory.model_copy(
            update={
                "status": "promoted",
                "evidence_level": evidence_level,
                "promotion_reason": reason,
                "promotion_decision": "promoted",
                "promotion_artifacts": decision_artifacts,
            }
        )

    memory = _update_memory(memory_id, promote)
    return _json_response({"memory": memory.model_dump(mode="json", exclude_none=True)})


@mcp.tool(
    name="dream_reject_memory",
    description=(
        "Reject an existing Dream memory or mark it as a negative constraint. "
        "Requires evidence_artifacts reviewed for the decision."
    ),
)
def dream_reject_memory_tool(
    memory_id: str,
    reason: str,
    evidence_artifacts: list[str] | None = None,
    negative: bool = False,
) -> str:
    if not reason.strip():
        raise ValueError("rejection reason is required")
    decision_artifacts = _require_decision_artifacts(
        evidence_artifacts,
        error_message="rejection evidence_artifacts are required",
    )
    status = "negative" if negative else "rejected"

    def reject(memory):
        return memory.model_copy(
            update={
                "status": status,
                "rejection_reason": reason,
                "promotion_decision": "rejected",
                "rejection_artifacts": decision_artifacts,
            }
        )

    memory = _update_memory(memory_id, reject)
    return _json_response({"memory": memory.model_dump(mode="json", exclude_none=True)})


def _require_decision_artifacts(
    evidence_artifacts: list[str] | None,
    *,
    error_message: str,
) -> list[str]:
    artifacts = [artifact.strip() for artifact in evidence_artifacts or [] if artifact.strip()]
    if not artifacts:
        raise ValueError(error_message)
    return list(dict.fromkeys(artifacts))


@mcp.tool(
    name="search_dream_memories",
    description=(
        "Compatibility alias for dream_search_memories. Prefer dream_search_memories "
        "in new MCP clients."
    ),
)
def search_dream_memories_tool(
    query: str,
    category: Literal["optimization", "environment_debug"] = "optimization",
    tags: list[str] | None = None,
    top_k: int = 5,
    record_choice: bool = True,
) -> str:
    return dream_search_memories_tool(
        query=query,
        category=category,
        tags=tags,
        top_k=top_k,
        record_choice=record_choice,
    )


@mcp.tool(
    name="record_dream_memory_feedback",
    description=(
        "Compatibility alias for dream_record_feedback. Prefer dream_record_feedback "
        "in new MCP clients."
    ),
)
def record_dream_memory_feedback_tool(
    memory_ids: list[str],
    useful: bool = True,
    reason: str | None = None,
    evidence_artifacts: list[str] | None = None,
    source_session_id: str | None = None,
) -> str:
    return dream_record_feedback_tool(
        memory_ids=memory_ids,
        useful=useful,
        reason=reason,
        evidence_artifacts=evidence_artifacts,
        source_session_id=source_session_id,
    )


@mcp.resource(
    "dream://search/{category}/{query}",
    name="search_dream_memories",
    title="Search EvoInfer Dream memories",
    description=(
        "Search EvoInfer Dream memories as a readable resource. Category must be "
        "optimization or environment_debug. Use this when direct MCP tools are not "
        "available but MCP resource templates are discoverable."
    ),
    mime_type="text/plain",
)
def search_dream_memories_resource(category: str, query: str) -> str:
    """Resource-template wrapper for Codex clients that discover MCP resources."""

    return search_dream_memories_tool(
        query=unquote(query),
        category="environment_debug" if category == "environment_debug" else "optimization",
        top_k=5,
        record_choice=True,
    )


def _normalize_category(category: str) -> DreamMemoryCategory:
    if category == "environment_debug":
        return "environment_debug"
    return "optimization"


def _load_json_object(raw: str) -> dict:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _json_response(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _read_candidate_list(candidates_path: Path) -> list[dict]:
    if not candidates_path.exists():
        return []
    raw = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("dream_write_candidates.json must contain a list")
    return [item for item in raw if isinstance(item, dict)]


def _standard_artifact_candidates(
    root: Path,
    *,
    category_hint: Literal["optimization", "environment_debug"] | None,
) -> list[dict]:
    environment = _read_json_artifact(root / "environment.json")
    if (
        category_hint == "environment_debug"
        or (root / "environment_debug.json").is_file()
        or environment.get("classification") == "environment_debug"
    ):
        return _standard_environment_debug_candidates(root, environment=environment)

    artifact_refs = [
        name
        for name in (
            "environment.json",
            "benchmark_raw.json",
            "correctness_raw.json",
            "verifier_result.json",
            "profiler_summary.json",
            "profiler_raw.json",
            "ncu_report.json",
            "nsys_report.json",
            "torch_profiler.json",
            "agent_trace.md",
        )
        if (root / name).is_file()
    ]
    if "benchmark_raw.json" not in artifact_refs and "correctness_raw.json" not in artifact_refs:
        return []

    benchmark = _read_json_artifact(root / "benchmark_raw.json")
    correctness = _read_json_artifact(root / "correctness_raw.json")
    verifier = _read_json_artifact(root / "verifier_result.json")
    profiler_refs = _profiler_artifact_refs(artifact_refs)
    profiler = _first_profiler_artifact(root, profiler_refs)

    operator = _first_string(benchmark, "operator", "op", "kernel", default="inference")
    backend = _first_string(
        benchmark,
        "backend",
        default=_first_string(
            environment,
            "inference_backend",
            "backend",
            "runtime",
            default="unknown",
        ),
    )
    model_type = _first_string(
        environment,
        "model_type",
        default=_first_string(benchmark, "model_type", default="operator-kernel"),
    )
    env_name = _environment_name(environment)
    correctness_passed = _artifact_passed(correctness)
    verifier_passed = str(verifier.get("status", "")).lower() == "passed" if verifier else False
    success = bool(correctness_passed and verifier_passed)
    precision = _precision_from_artifacts(environment, benchmark)
    workload = _workload_from_artifacts(environment, benchmark)
    baseline_metrics = _benchmark_baseline_metrics(benchmark)
    candidate_metrics = _benchmark_candidate_metrics(benchmark, correctness)
    bottleneck_type = _profiler_bottleneck_type(profiler)

    candidate: dict[str, Any] = {
        "id": f"opt_artifact_{root.name}",
        "category": "candidate_optimization",
        "title": f"Artifact-backed {operator} optimization candidate",
        "summary": (
            "Extracted from standard campaign artifacts; review benchmark, "
            "correctness, verifier, and trace evidence before promotion."
        ),
        "environment": env_name,
        "model_type": model_type,
        "inference_backend": backend,
        "precision": precision,
        "workload": workload,
        "success": success,
        "detail_description": (
            "Artifact-driven extraction from benchmark_raw.json, "
            "correctness_raw.json, verifier_result.json, and agent_trace.md."
        ),
        "artifact_refs": artifact_refs,
        "artifacts": [ref for ref in artifact_refs if ref != "correctness_raw.json"],
        "correctness_artifacts": (
            ["correctness_raw.json"] if "correctness_raw.json" in artifact_refs else []
        ),
        "profiler_artifacts": profiler_refs,
        "bottleneck_type": bottleneck_type,
        "metrics_before": baseline_metrics,
        "metrics_after": candidate_metrics,
        "objective_metric": _objective_metric_from_metrics(baseline_metrics, candidate_metrics),
        "applicability": _applicability_from_artifacts(
            operator=operator,
            backend=backend,
            precision=precision,
            workload=workload,
            bottleneck_type=bottleneck_type,
        ),
        "benchmark_command": _first_string(verifier, "command", default=None),
        "tags": _optimization_artifact_tags(
            operator=operator,
            backend=backend,
            precision=precision,
            workload=workload,
            bottleneck_type=bottleneck_type,
        ),
        "extraction_source": "standard_artifacts",
    }
    if not success:
        candidate["failure_reason"] = _artifact_failure_reason(correctness, verifier)
    return [candidate]


def _standard_environment_debug_candidates(
    root: Path,
    *,
    environment: dict[str, Any],
) -> list[dict]:
    debug = _read_json_artifact(root / "environment_debug.json")
    if not debug:
        return []

    artifact_refs = [
        name
        for name in (
            "environment.json",
            "environment_debug.json",
            "diagnostic.log",
            "verification.log",
            "verifier_result.json",
            "agent_trace.md",
            "api_inventory.json",
            "library_notes.md",
        )
        if (root / name).is_file()
    ]
    verifier = _read_json_artifact(root / "verifier_result.json")
    diagnostic_artifacts = [
        ref
        for ref in (
            "environment_debug.json",
            "diagnostic.log",
            "agent_trace.md",
            "api_inventory.json",
            "library_notes.md",
        )
        if ref in artifact_refs
    ]
    verification_artifacts = [
        ref for ref in ("verification.log", "verifier_result.json") if ref in artifact_refs
    ]
    verifier_failed = verifier and str(verifier.get("status", "")).lower() != "passed"
    success = bool(debug.get("success")) and not verifier_failed
    component = _first_string(debug, "component", default="environment") or "environment"
    issue_signature = _first_string(debug, "issue_signature", "signature", default="")
    debug_type = _normalize_debug_type(_first_string(debug, "debug_type", "type", default="other"))

    candidate: dict[str, Any] = {
        "id": f"env_artifact_{root.name}",
        "category": "environment_debug",
        "title": _first_string(
            debug,
            "title",
            default=f"Artifact-backed {component} environment debug candidate",
        ),
        "summary": _first_string(
            debug,
            "summary",
            default="Extracted from structured environment debug artifacts.",
        ),
        "environment": _environment_name(environment),
        "debug_type": debug_type,
        "component": component,
        "hardware": _first_string(environment, "gpu_name", "gpu", default=None),
        "driver": _first_string(environment, "driver_version", "driver", default=None),
        "runtime": _first_string(environment, "runtime", "cuda_version", default=None),
        "dependency_stack": _dependency_stack_from_environment(environment),
        "inference_backend": _first_string(
            debug,
            "inference_backend",
            default=_first_string(environment, "inference_backend", "library", default=None),
        ),
        "issue_signature": issue_signature,
        "symptoms": _first_string(debug, "symptoms", default=""),
        "root_cause": _first_string(debug, "root_cause", default=""),
        "solution": _first_string(debug, "solution", default=""),
        "verification": _first_string(debug, "verification", default=""),
        "commands": _string_list(debug.get("commands")),
        "error_messages": _string_list(debug.get("error_messages")),
        "diagnostic_steps": _string_list(debug.get("diagnostic_steps")),
        "success": success,
        "artifact_refs": artifact_refs,
        "artifacts": artifact_refs,
        "diagnostic_artifacts": diagnostic_artifacts,
        "verification_artifacts": verification_artifacts,
        "environment_snapshot": environment,
        "tags": [tag for tag in (component, debug_type) if tag],
        "extraction_source": "standard_environment_debug_artifacts",
    }
    if debug.get("resolved_by_command"):
        candidate["resolved_by_command"] = debug.get("resolved_by_command")
    return [candidate]


def _profiler_artifact_refs(artifact_refs: list[str]) -> list[str]:
    profiler_names = {
        "profiler_summary.json",
        "profiler_raw.json",
        "ncu_report.json",
        "nsys_report.json",
        "torch_profiler.json",
    }
    return [ref for ref in artifact_refs if ref in profiler_names]


def _first_profiler_artifact(root: Path, profiler_refs: list[str]) -> dict[str, Any]:
    for ref in profiler_refs:
        profiler = _read_json_artifact(root / ref)
        if profiler:
            return profiler
    return {}


def _profiler_bottleneck_type(profiler: dict[str, Any]) -> str | None:
    explicit = _first_string(profiler, "bottleneck_type", "bottleneck", default=None)
    if explicit:
        return explicit
    ncu = _infer_ncu_bottleneck(profiler)
    if ncu:
        return ncu
    nsys = _infer_nsys_bottleneck(profiler)
    if nsys:
        return nsys
    torch = _infer_torch_profiler_bottleneck(profiler)
    if torch:
        return torch
    return None


def _infer_ncu_bottleneck(profiler: dict[str, Any]) -> str | None:
    reports = profiler.get("reports")
    if not isinstance(reports, list):
        return None
    max_dram = 0.0
    max_sm = 0.0
    for report in reports:
        if not isinstance(report, dict):
            continue
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            continue
        max_dram = max(
            max_dram,
            _numeric_metric(metrics, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        )
        max_sm = max(
            max_sm,
            _numeric_metric(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        )
    if max_dram >= 70 and max_dram >= max_sm * 1.2:
        return "memory_bandwidth"
    if max_sm >= 70 and max_sm >= max_dram * 1.2:
        return "compute"
    if max_dram or max_sm:
        return "mixed_gpu"
    return None


def _infer_nsys_bottleneck(profiler: dict[str, Any]) -> str | None:
    cuda_api = profiler.get("cuda_api")
    gpu_kernels = profiler.get("gpu_kernels")
    if not isinstance(cuda_api, list) and not isinstance(gpu_kernels, list):
        return None
    launch_time = 0.0
    api_time = 0.0
    if isinstance(cuda_api, list):
        for event in cuda_api:
            if not isinstance(event, dict):
                continue
            time_ms = _numeric_metric(event, "time_ms", "duration_ms")
            api_time += time_ms
            name = str(event.get("name") or "").lower()
            if "launch" in name:
                launch_time += time_ms
    kernel_time = 0.0
    if isinstance(gpu_kernels, list):
        for event in gpu_kernels:
            if isinstance(event, dict):
                kernel_time += _numeric_metric(event, "time_ms", "duration_ms")
    if launch_time > 0 and launch_time >= max(0.1, kernel_time * 2):
        return "launch_overhead"
    if api_time > 0 and api_time >= max(0.1, kernel_time * 2):
        return "runtime_api_overhead"
    if kernel_time > 0:
        return "gpu_kernel"
    return None


def _infer_torch_profiler_bottleneck(profiler: dict[str, Any]) -> str | None:
    events = profiler.get("events")
    if not isinstance(events, list):
        return None
    cpu_time = 0.0
    cuda_time = 0.0
    for event in events:
        if not isinstance(event, dict):
            continue
        cpu_time += _numeric_metric(event, "cpu_time_total", "self_cpu_time_total")
        cuda_time += _numeric_metric(event, "cuda_time_total", "self_cuda_time_total")
    if cpu_time > cuda_time * 2 and cpu_time > 0:
        return "cpu_overhead"
    if cuda_time > 0:
        return "gpu_kernel"
    return None


def _numeric_metric(values: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def _read_json_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_string(
    values: dict[str, Any],
    *keys: str,
    default: str | None = "",
) -> str | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _environment_name(environment: dict[str, Any]) -> str:
    gpu = _first_string(environment, "gpu_name", "gpu", default=None)
    if gpu:
        return gpu
    explicit = _first_string(environment, "name", "environment", default=None)
    if explicit:
        return explicit
    backend = _first_string(
        environment,
        "inference_backend",
        "backend",
        "runtime",
        "library",
        default=None,
    )
    platform = _first_string(environment, "platform", "os", default=None)
    if backend and platform:
        return f"{backend} on {platform}"
    if backend:
        return backend
    if platform:
        return platform
    return "unknown"


def _artifact_passed(payload: dict[str, Any]) -> bool:
    value = payload.get("passed")
    if isinstance(value, bool):
        return value
    status = payload.get("status")
    return isinstance(status, str) and status.lower() == "passed"


def _normalize_debug_type(value: str | None) -> str:
    allowed = {
        "install",
        "build",
        "runtime",
        "dependency",
        "driver",
        "network",
        "auth",
        "filesystem",
        "performance",
        "other",
    }
    normalized = (value or "other").strip().lower()
    return normalized if normalized in allowed else "other"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dependency_stack_from_environment(environment: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "torch_version",
        "flashinfer_version",
        "flashinfer_python_version",
        "triton_version",
        "cuda_version",
        "cuda_home",
        "nvcc_version",
    )
    return {key: environment[key] for key in keys if key in environment}


def _metric_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _benchmark_baseline_metrics(benchmark: dict[str, Any]) -> dict[str, Any]:
    nested = _metric_dict(benchmark.get("baseline"))
    if nested:
        return nested
    metrics: dict[str, Any] = {}
    latency_ms = _first_numeric(
        benchmark,
        "baseline_ms",
        "baseline_latency_ms",
        "before_ms",
        "before_latency_ms",
    )
    if latency_ms is not None:
        metrics["latency_ms"] = latency_ms
    latency_s = _first_numeric(
        benchmark,
        "baseline_s",
        "baseline_seconds",
        "before_s",
        "before_seconds",
    )
    if latency_s is not None and "latency_ms" not in metrics:
        metrics["latency_ms"] = latency_s * 1000.0
    return metrics


def _benchmark_candidate_metrics(
    benchmark: dict[str, Any],
    correctness: dict[str, Any],
) -> dict[str, Any]:
    nested_candidate = _metric_dict(benchmark.get("candidate"))
    metrics = dict(nested_candidate)
    latency_ms = _first_numeric(
        benchmark,
        "candidate_ms",
        "optimized_ms",
        "after_ms",
        "candidate_latency_ms",
        "after_latency_ms",
    )
    if latency_ms is not None:
        metrics.setdefault("latency_ms", latency_ms)
    latency_s = _first_numeric(
        benchmark,
        "candidate_s",
        "optimized_s",
        "after_s",
        "candidate_seconds",
        "after_seconds",
    )
    if latency_s is not None and "latency_ms" not in metrics:
        metrics["latency_ms"] = latency_s * 1000.0
    speedup = _first_numeric(benchmark, "speedup", "speedup_x")
    if speedup is not None:
        metrics.setdefault("speedup", speedup)
    if not nested_candidate:
        for key in (
            "max_abs_error",
            "max_relative_error",
            "max_rel_error",
            "max_row_sum_error",
            "mean_abs_error",
        ):
            value = _first_numeric(correctness, key)
            if value is not None:
                metrics.setdefault(key, value)
    return metrics


def _precision_from_artifacts(
    environment: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    for values in (benchmark, environment):
        precision = values.get("precision")
        if isinstance(precision, dict):
            return {str(key): value for key, value in precision.items() if value not in ("", None)}
        if isinstance(precision, str) and precision.strip():
            return {"dtype": precision.strip()}

    result: dict[str, Any] = {}
    for output_key, candidates in {
        "dtype": ("dtype", "data_type", "torch_dtype"),
        "compute_dtype": ("compute_dtype",),
        "accum_dtype": ("accum_dtype", "accumulator_dtype"),
        "quantization": ("quantization", "quantization_dtype"),
    }.items():
        value = _first_string(benchmark, *candidates, default=None) or _first_string(
            environment,
            *candidates,
            default=None,
        )
        if value:
            result[output_key] = value
    return result


def _workload_from_artifacts(
    environment: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    workload = benchmark.get("workload")
    if isinstance(workload, dict):
        return {str(key): value for key, value in workload.items() if value not in ("", None)}

    result: dict[str, Any] = {}
    for output_key, candidates in {
        "batch": ("batch", "batch_size", "b"),
        "seq": ("seq", "sequence", "seq_len", "sequence_length", "t"),
        "hidden": ("hidden", "hidden_size", "h"),
        "head_dim": ("head_dim", "headdim", "d"),
        "num_heads": ("num_heads", "heads"),
        "vocab_size": ("vocab_size",),
    }.items():
        value = _first_existing(benchmark, *candidates)
        if value is None:
            value = _first_existing(environment, *candidates)
        if value not in ("", None):
            result[output_key] = value
    return result


def _objective_metric_from_metrics(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> str | None:
    if "latency_ms" in baseline_metrics or "latency_ms" in candidate_metrics:
        return "latency_ms"
    if "tokens_per_second" in baseline_metrics or "tokens_per_second" in candidate_metrics:
        return "tokens_per_second"
    return None


def _applicability_from_artifacts(
    *,
    operator: str,
    backend: str,
    precision: dict[str, Any],
    workload: dict[str, Any],
    bottleneck_type: str | None,
) -> str:
    parts = [f"operator={operator}", f"backend={backend}"]
    dtype = precision.get("dtype")
    if dtype:
        parts.append(f"dtype={dtype}")
    parts.extend(
        f"{key}={value}"
        for key, value in workload.items()
        if value not in ("", None)
    )
    if bottleneck_type:
        parts.append(f"bottleneck={bottleneck_type}")
    return "Applies only after revalidating " + ", ".join(parts) + "."


def _optimization_artifact_tags(
    *,
    operator: str,
    backend: str,
    precision: dict[str, Any],
    workload: dict[str, Any],
    bottleneck_type: str | None,
) -> list[str]:
    tags = [tag for tag in (operator, backend, bottleneck_type) if tag and tag != "unknown"]
    dtype = precision.get("dtype")
    if isinstance(dtype, str) and dtype:
        tags.append(dtype)
    for key in workload:
        tags.append(str(key))
    return list(dict.fromkeys(tags))


def _first_numeric(values: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _first_existing(values: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _artifact_failure_reason(
    correctness: dict[str, Any],
    verifier: dict[str, Any],
) -> str:
    reasons: list[str] = []
    if correctness and not _artifact_passed(correctness):
        reasons.append("correctness artifact did not pass")
    if not verifier:
        reasons.append("missing verifier_result.json")
    elif str(verifier.get("status", "")).lower() != "passed":
        reasons.append(f"verifier status={verifier.get('status')}")
    return "; ".join(reasons) or "artifact evidence did not prove success"


def _sanitize_staged_candidate(root: Path, candidate: dict) -> dict:
    title = candidate.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("candidate title is required")

    refs = candidate.get("artifact_refs")
    if not isinstance(refs, list) or not any(isinstance(ref, str) and ref.strip() for ref in refs):
        raise ValueError("candidate artifact_refs are required")

    normalized_refs = _normalize_candidate_artifact_refs(root, refs)
    existing_refs = [ref for ref in normalized_refs if (root / ref).exists()]
    if not existing_refs:
        raise ValueError("candidate must reference at least one existing artifact")

    next_candidate = dict(candidate)
    next_candidate["artifact_refs"] = normalized_refs
    next_candidate.setdefault("category", "candidate_optimization")
    next_candidate["source_workdir"] = str(root)
    next_candidate["staged_at"] = datetime.now(UTC).isoformat()
    return next_candidate


def _normalize_candidate_artifact_refs(root: Path, refs: list[object]) -> list[str]:
    normalized: list[str] = []
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            continue
        path = Path(ref)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("candidate artifact_refs must be relative paths inside workdir")
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "candidate artifact_refs must be relative paths inside workdir"
            ) from exc
        normalized.append(path.as_posix())
    if not normalized:
        raise ValueError("candidate artifact_refs are required")
    return list(dict.fromkeys(normalized))


def _candidate_category(
    item: dict,
    category_hint: Literal["optimization", "environment_debug"] | None,
) -> DreamMemoryCategory:
    if category_hint:
        return category_hint
    category = str(item.get("category", ""))
    if "debug" in category or "environment" in category:
        return "environment_debug"
    return "optimization"


def _candidate_status(item: dict) -> str:
    category = str(item.get("category", ""))
    if category.startswith("negative"):
        return "negative"
    if category.startswith("rejected"):
        return "rejected"
    if category.startswith("promoted"):
        return "promoted"
    return "candidate"


def _candidate_tags(item: dict) -> list[str]:
    tags: list[str] = []
    for key in ("library", "operator", "backend", "component"):
        value = item.get(key)
        if isinstance(value, str) and value:
            tags.append(value)
    explicit_tags = item.get("tags")
    if isinstance(explicit_tags, list):
        tags.extend(tag for tag in explicit_tags if isinstance(tag, str) and tag)
    return list(dict.fromkeys(tags))


def _candidate_promotion_input(
    item: dict,
    *,
    category: DreamMemoryCategory,
    status: str,
    artifact_refs: list[str],
    source_workdir: str,
) -> dict:
    promotion_input = {
        key: value
        for key, value in item.items()
        if key
        not in {
            "category",
            "missing_artifact_refs",
            "source_workdir",
            "staged_at",
            "extraction_source",
        }
    }
    promotion_input["category"] = category
    promotion_input["status"] = status
    promotion_input["artifact_refs"] = artifact_refs
    promotion_input["source_workdir"] = source_workdir
    promotion_input["tags"] = _candidate_tags(item)
    if not promotion_input.get("summary") and isinstance(item.get("lesson"), str):
        promotion_input["summary"] = item["lesson"]
    if category == "optimization" and not promotion_input.get("inference_backend"):
        backend = item.get("backend")
        if isinstance(backend, str) and backend:
            promotion_input["inference_backend"] = backend
    return promotion_input


def _promotion_blockers(
    promotion_input: dict,
    *,
    category: DreamMemoryCategory,
) -> list[str]:
    try:
        if category == "optimization":
            OptimizationMemoryInput.model_validate(promotion_input)
        else:
            EnvironmentDebugMemoryInput.model_validate(promotion_input)
    except ValidationError as exc:
        return [
            str(error.get("msg", "")).removeprefix("Value error, ")
            for error in exc.errors()
            if error.get("msg")
        ]
    return []


def _prepare_auto_write_input(
    promotion_input: dict[str, Any],
    *,
    category: DreamMemoryCategory,
    candidate_status: str,
) -> dict[str, Any]:
    write_input = dict(promotion_input)
    write_input.pop("source_workdir", None)
    write_input.pop("category", None)
    if category == "optimization" and write_input.get("success") is False:
        write_input["status"] = "negative"
    elif candidate_status in {"candidate", "negative", "rejected"}:
        write_input["status"] = candidate_status
    else:
        write_input["status"] = "candidate"
    if write_input["status"] == "promoted":
        write_input["status"] = "candidate"
    write_input.setdefault("evidence_level", "smoke")
    return write_input


def _auto_write_blockers(
    write_input: dict[str, Any],
    *,
    category: DreamMemoryCategory,
    source_workdir: Path,
) -> list[str]:
    blockers: list[str] = []
    try:
        if category == "optimization":
            OptimizationMemoryInput.model_validate(write_input)
        else:
            EnvironmentDebugMemoryInput.model_validate(write_input)
    except ValidationError as exc:
        blockers.extend(
            str(error.get("msg", "")).removeprefix("Value error, ")
            for error in exc.errors()
            if error.get("msg")
        )

    if category == "optimization":
        blockers.extend(_optimization_auto_write_blockers(write_input, source_workdir))
    else:
        blockers.extend(_environment_debug_auto_write_blockers(write_input))
    return blockers


def _optimization_auto_write_blockers(
    write_input: dict[str, Any],
    source_workdir: Path,
) -> list[str]:
    success = write_input.get("success") is True
    if not success:
        return []

    blockers: list[str] = []
    artifact_refs = _string_list(write_input.get("artifact_refs"))
    if "verifier_result.json" not in artifact_refs:
        blockers.append("positive optimization auto-write requires verifier_result.json")
    else:
        verifier = _read_json_artifact(source_workdir / "verifier_result.json")
        if str(verifier.get("status", "")).lower() != "passed":
            blockers.append("positive optimization auto-write requires verifier pass")

    profiler_artifacts = _string_list(write_input.get("profiler_artifacts"))
    source_evidence_artifacts = [
        ref
        for ref in artifact_refs
        if Path(ref).name
        in {
            "source_evidence.json",
            "source_diff.patch",
            "kernel_diff.patch",
            "patch.diff",
        }
    ]
    if not profiler_artifacts and not source_evidence_artifacts:
        blockers.append(
            "positive optimization auto-write requires profiler or source-level bottleneck evidence"
        )
    return blockers


def _environment_debug_auto_write_blockers(write_input: dict[str, Any]) -> list[str]:
    if write_input.get("success") is not True:
        return []
    verification_artifacts = _string_list(write_input.get("verification_artifacts"))
    if not verification_artifacts:
        return ["resolved environment debug auto-write requires verification artifacts"]
    return []


def _update_memory(memory_id: str, updater):
    memories = list_dream_memories()
    updated = None
    next_memories = []
    for memory in memories:
        if memory.id == memory_id:
            updated = updater(memory)
            next_memories.append(updated)
        else:
            next_memories.append(memory)
    if updated is None:
        raise ValueError(f"Dream memory not found: {memory_id}")
    restore_dream_memory_snapshot(DreamMemorySnapshotInput(memories=next_memories))
    return updated


def _search_header(
    render_mode: DreamSearchRenderMode,
    *,
    task_context: str | None,
) -> list[str]:
    if render_mode == "compact":
        lines = ["Dream memory compact results:"]
    elif render_mode == "agent_actionable":
        lines = ["Dream memory actionable results:"]
    elif render_mode == "artifact_protocol":
        lines = [
            "Dream memory artifact-protocol results:",
            "",
            "Artifact protocol:",
            "- Treat retrieved memories as hypotheses, not authority.",
            "- Reproduce baseline before applying the memory.",
            "- Validate correctness on the same workload after changes.",
            "- Do not claim success without benchmark and correctness artifacts.",
        ]
    else:
        lines = [
            "Dream memory search results:",
            "",
            "Use these as prior experience only. Confirm applicability with real checks, "
            "benchmarks, profiling, and correctness validation before applying.",
        ]
    if task_context:
        lines.extend(["", f"task_context={task_context}"])
    return lines


def _format_result(
    index: int,
    result: DreamMemorySearchResult,
    *,
    render_mode: DreamSearchRenderMode,
) -> list[str]:
    if render_mode == "agent_actionable":
        return _format_actionable_result(index, result)
    if render_mode == "artifact_protocol":
        return _format_artifact_protocol_result(index, result)

    memory = result.memory
    lines = [
        "",
        f"{index}. [{memory.category}] {memory.title} ({memory.id})",
        f"   score={result.score:.3f}; reasons={', '.join(result.reasons) or 'semantic'}",
        f"   recommended_action={result.recommended_action}; status={memory.status}; evidence_level={memory.evidence_level}",
    ]
    if memory.summary:
        lines.append(f"   summary={memory.summary}")
    if memory.tags:
        lines.append(f"   tags={', '.join(memory.tags)}")
    if render_mode == "full":
        details = _actionable_details(result)
        if details:
            lines.append(f"   details={details}")
    return lines


def _format_actionable_result(index: int, result: DreamMemorySearchResult) -> list[str]:
    memory = result.memory
    lines = [
        "",
        f"{index}. [{memory.category}] {memory.title} ({memory.id})",
        f"   score={result.score:.3f}; reasons={', '.join(result.reasons) or 'semantic'}",
        f"   recommended_action={result.recommended_action}; status={memory.status}; evidence_level={memory.evidence_level}",
    ]
    if memory.summary:
        lines.append(f"   summary={memory.summary}")
    if memory.category == "optimization":
        hypothesis = memory.safe_transfer_notes or [memory.detail_description or memory.summary]
        validation = list(memory.correctness_invariants)
        if memory.benchmark_command:
            validation.append(memory.benchmark_command)
        if memory.correctness_artifacts:
            validation.extend(memory.correctness_artifacts)
        caveats = memory.unsafe_transfer_notes or ([memory.caveats] if memory.caveats else [])
    else:
        hypothesis = [memory.solution or memory.summary]
        validation = [memory.verification] if memory.verification else []
        validation.extend(memory.verification_artifacts)
        caveats = [memory.risk] if memory.risk else []
    lines.append(f"   Apply as hypothesis: {_join_nonempty(hypothesis)}")
    lines.append(f"   Validate with: {_join_nonempty(validation)}")
    if caveats:
        lines.append(f"   Do not blindly transfer: {_join_nonempty(caveats)}")
    return lines


def _format_artifact_protocol_result(index: int, result: DreamMemorySearchResult) -> list[str]:
    memory = result.memory
    lines = [
        "",
        f"{index}. [{memory.category}] {memory.title} ({memory.id})",
        f"   score={result.score:.3f}; reasons={', '.join(result.reasons) or 'semantic'}",
        f"   recommended_action={result.recommended_action}; status={memory.status}; evidence_level={memory.evidence_level}",
    ]
    artifact_refs = list(memory.artifacts)
    if memory.category == "optimization":
        artifact_refs.extend(memory.baseline_artifacts)
        artifact_refs.extend(memory.optimized_artifacts)
        artifact_refs.extend(memory.profiler_artifacts)
        artifact_refs.extend(memory.correctness_artifacts)
        if memory.benchmark_command:
            lines.append(f"   benchmark_command={memory.benchmark_command}")
        if memory.correctness_invariants:
            lines.append(
                f"   correctness_invariants={_join_nonempty(memory.correctness_invariants)}"
            )
    else:
        artifact_refs.extend(memory.diagnostic_artifacts)
        artifact_refs.extend(memory.verification_artifacts)
        if memory.issue_signature:
            lines.append(f"   issue_signature={memory.issue_signature}")
        if memory.verification:
            lines.append(f"   verification={memory.verification}")
    if artifact_refs:
        lines.append(f"   artifact_refs={_join_nonempty(artifact_refs)}")
    return lines


def _join_nonempty(values: list[str | None]) -> str:
    return "; ".join(value for value in values if value) or "no explicit detail recorded"


def _actionable_details(result: DreamMemorySearchResult) -> str:
    memory = result.memory
    if memory.category == "optimization":
        details = {
            "environment": memory.environment,
            "backend": memory.inference_backend,
            "workload": memory.workload or None,
            "metrics_before": memory.metrics_before or None,
            "metrics_after": memory.metrics_after or None,
            "benchmark_command": memory.benchmark_command,
            "correctness_artifacts": memory.correctness_artifacts or None,
            "bottleneck": memory.bottleneck_type,
            "description": memory.detail_description or None,
            "applicability": memory.applicability,
            "caveats": memory.caveats,
            "operation_semantics": memory.operation_semantics or None,
            "correctness_invariants": memory.correctness_invariants or None,
            "safe_transfer_notes": memory.safe_transfer_notes or None,
            "unsafe_transfer_notes": memory.unsafe_transfer_notes or None,
            "chosen": memory.chosen,
            "useful_rate": memory.useful_rate,
        }
    else:
        details = {
            "environment": memory.environment,
            "component": memory.component,
            "issue_signature": memory.issue_signature,
            "symptoms": memory.symptoms,
            "root_cause": memory.root_cause,
            "solution": memory.solution,
            "verification": memory.verification,
            "commands": memory.commands or None,
            "chosen": memory.chosen,
            "useful_rate": memory.useful_rate,
        }
    return json.dumps(
        {key: value for key, value in details.items() if value not in (None, [], {})},
        ensure_ascii=False,
        sort_keys=True,
    )


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
