"""Dream memory schemas and persistence."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from evoinfer_mcp import logger
from evoinfer_mcp.dream.embedding import score_texts_with_optional_embedding
from evoinfer_mcp.share import get_share_dir
from evoinfer_mcp.utils.io import atomic_json_write

DreamMemoryCategory = Literal["optimization", "environment_debug"]
DreamMemoryStatus = Literal["candidate", "promoted", "negative", "rejected"]
DreamMemoryEvidenceLevel = Literal["smoke", "repeated", "verified", "deprecated"]
DreamMemoryRecommendedAction = Literal[
    "apply_as_hypothesis",
    "avoid_as_negative_constraint",
    "review_evidence_before_use",
]
EnvironmentDebugType = Literal[
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
]


class DreamMemory(BaseModel):
    """A single specialist memory shown in the Dream page."""

    id: str = Field(description="Stable memory ID")
    category: DreamMemoryCategory = Field(description="Memory category")
    title: str = Field(description="Short display title")
    summary: str = Field(default="", description="One-line summary")
    tags: list[str] = Field(default_factory=list, description="Search and graph tags")

    # Shared / optimization schema
    environment: str | None = Field(default=None, description="Compute/runtime environment")
    model_type: str | None = Field(default=None, description="Model family, e.g. llm/vlm/vla")
    model_arch: str | None = Field(default=None, description="Model architecture")
    model_name: str | None = Field(default=None, description="Concrete model name")
    model_size: str | None = Field(default=None, description="Model size or parameter count")
    inference_backend: str | None = Field(default=None, description="Inference backend")
    serving_framework: str | None = Field(default=None, description="Serving framework")
    precision: dict[str, Any] = Field(default_factory=dict, description="Precision settings")
    workload: dict[str, Any] = Field(default_factory=dict, description="Benchmark workload")
    metrics_before: dict[str, Any] = Field(
        default_factory=dict, description="Metrics before optimization"
    )
    metrics_after: dict[str, Any] = Field(
        default_factory=dict, description="Metrics after optimization"
    )
    objective_metric: str | None = Field(default=None, description="Primary optimized metric")
    success: bool | None = Field(default=None, description="Whether the attempt succeeded")
    detail_description: str = Field(default="", description="Detailed optimization/debug note")
    applicability: str | None = Field(default=None, description="When this memory applies")
    caveats: str | None = Field(default=None, description="Known caveats")
    operation_semantics: list[str] = Field(
        default_factory=list,
        description="Target-operation semantics that must be preserved when transferring",
    )
    correctness_invariants: list[str] = Field(
        default_factory=list,
        description="Operation-specific correctness conditions that must be rechecked",
    )
    safe_transfer_notes: list[str] = Field(
        default_factory=list,
        description="Parts of the experience that are safe to reuse after validation",
    )
    unsafe_transfer_notes: list[str] = Field(
        default_factory=list,
        description="Parts of the experience that must not be blindly transferred",
    )
    failure_reason: str | None = Field(default=None, description="Failure reason if any")
    benchmark_command: str | None = Field(default=None, description="Benchmark command or method")
    baseline_artifacts: list[str] = Field(
        default_factory=list, description="Baseline benchmark artifacts"
    )
    optimized_artifacts: list[str] = Field(
        default_factory=list, description="Optimized candidate benchmark artifacts"
    )
    profiler_artifacts: list[str] = Field(default_factory=list, description="Profiler artifacts")
    correctness_artifacts: list[str] = Field(
        default_factory=list, description="Correctness validation artifacts"
    )
    bottleneck_type: str | None = Field(default=None, description="Diagnosed bottleneck type")
    promotion_decision: str | None = Field(
        default=None, description="Promotion/rejection decision for the attempt"
    )
    rejection_reason: str | None = Field(
        default=None, description="Reason a candidate was rejected"
    )

    # Environment deployment/debug schema
    debug_type: EnvironmentDebugType | None = Field(
        default=None, description="Environment/debug issue type"
    )
    component: str | None = Field(default=None, description="Affected component")
    hardware: str | None = Field(default=None, description="Hardware details")
    os: str | None = Field(default=None, description="Operating system")
    driver: str | None = Field(default=None, description="Driver stack")
    runtime: str | None = Field(default=None, description="Runtime stack")
    dependency_stack: dict[str, Any] = Field(
        default_factory=dict, description="Relevant packages and versions"
    )
    issue_signature: str | None = Field(default=None, description="Searchable error signature")
    symptoms: str | None = Field(default=None, description="Observed symptoms")
    root_cause: str | None = Field(default=None, description="Root cause")
    solution: str | None = Field(default=None, description="Fix or workaround")
    verification: str | None = Field(default=None, description="How the fix was verified")
    related_backend: str | None = Field(default=None, description="Related backend or subsystem")
    risk: str | None = Field(default=None, description="Risks of applying this memory")
    commands: list[str] = Field(default_factory=list, description="Commands used to debug/fix")
    error_messages: list[str] = Field(default_factory=list, description="Important errors")
    diagnostic_steps: list[str] = Field(default_factory=list, description="Debug steps taken")
    prevention: str | None = Field(default=None, description="How to prevent recurrence")
    diagnostic_artifacts: list[str] = Field(
        default_factory=list, description="Diagnostic logs or artifacts"
    )
    verification_artifacts: list[str] = Field(
        default_factory=list, description="Verification logs or artifacts"
    )
    resolved_by_command: str | None = Field(
        default=None, description="Primary command that resolved the issue"
    )
    environment_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Environment snapshot captured during debugging"
    )

    # Experiment bookkeeping
    artifacts: list[str] = Field(default_factory=list, description="Related files or URLs")
    artifact_refs: list[str] = Field(
        default_factory=list,
        description="Canonical evidence artifact references used for gating and export",
    )
    status: DreamMemoryStatus = Field(
        default="candidate",
        description="Memory lifecycle status",
    )
    evidence_level: DreamMemoryEvidenceLevel = Field(
        default="smoke",
        description="Evidence strength for this memory",
    )
    promotion_reason: str | None = Field(
        default=None,
        description="Evidence-backed reason for promoting this memory",
    )
    promotion_artifacts: list[str] = Field(
        default_factory=list,
        description="Artifacts reviewed when promoting this memory",
    )
    rejection_artifacts: list[str] = Field(
        default_factory=list,
        description="Artifacts reviewed when rejecting or marking this memory negative",
    )
    chosen: int = Field(default=0, ge=0, description="Times selected/reused")
    useful_when_chosen: int = Field(default=0, ge=0, description="Times useful when selected")
    useful_rate: float = Field(default=0, ge=0, le=1, description="useful_when_chosen / chosen")
    time: str | None = Field(default=None, description="Elapsed optimization/debug time")
    token_used: int | None = Field(default=None, ge=0, description="Tokens spent")
    source_session_id: str | None = Field(default=None, description="Session that wrote it")
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def derive_useful_rate(self) -> Self:
        if self.chosen > 0 and self.useful_when_chosen > 0 and self.useful_rate == 0:
            self.useful_rate = min(1, self.useful_when_chosen / self.chosen)
        return self


class OptimizationMemoryInput(BaseModel):
    """Input schema for writing an inference optimization memory."""

    id: str | None = Field(default=None, description="Optional stable ID for upsert")
    title: str = Field(description="Short title for this optimization experience", min_length=1)
    summary: str = Field(default="", description="One-line summary")
    tags: list[str] = Field(default_factory=list, description="Search and graph tags")
    environment: str = Field(description="Compute/runtime environment")
    model_type: str = Field(description="Model family, e.g. llm/vlm/vla/classical-ml")
    model_arch: str | None = Field(default=None, description="Model architecture")
    model_name: str | None = Field(default=None, description="Concrete model name")
    model_size: str | None = Field(default=None, description="Model size or parameter count")
    inference_backend: str = Field(
        description="Inference backend, e.g. cuda/triton/pytorch/tensorrt"
    )
    serving_framework: str | None = Field(default=None, description="Serving framework")
    precision: dict[str, Any] = Field(default_factory=dict, description="Precision settings")
    workload: dict[str, Any] = Field(default_factory=dict, description="Benchmark workload")
    metrics_before: dict[str, Any] = Field(
        default_factory=dict, description="Metrics before optimization"
    )
    metrics_after: dict[str, Any] = Field(
        default_factory=dict, description="Metrics after optimization"
    )
    objective_metric: str | None = Field(default=None, description="Primary optimized metric")
    success: bool = Field(description="Whether the optimization succeeded")
    detail_description: str = Field(description="What changed and why", min_length=1)
    applicability: str | None = Field(default=None, description="When this technique applies")
    caveats: str | None = Field(default=None, description="Known caveats")
    operation_semantics: list[str] = Field(
        default_factory=list,
        description="Target-operation semantics that must be preserved when transferring",
    )
    correctness_invariants: list[str] = Field(
        default_factory=list,
        description="Operation-specific correctness conditions that must be rechecked",
    )
    safe_transfer_notes: list[str] = Field(
        default_factory=list,
        description="Parts of the experience that are safe to reuse after validation",
    )
    unsafe_transfer_notes: list[str] = Field(
        default_factory=list,
        description="Parts of the experience that must not be blindly transferred",
    )
    failure_reason: str | None = Field(default=None, description="Failure reason if unsuccessful")
    artifacts: list[str] = Field(default_factory=list, description="Related files or URLs")
    artifact_refs: list[str] = Field(
        default_factory=list,
        description="Canonical evidence artifact references used for gating and export",
    )
    benchmark_command: str | None = Field(default=None, description="Benchmark command or method")
    baseline_artifacts: list[str] = Field(
        default_factory=list, description="Baseline benchmark artifacts"
    )
    optimized_artifacts: list[str] = Field(
        default_factory=list, description="Optimized candidate benchmark artifacts"
    )
    profiler_artifacts: list[str] = Field(default_factory=list, description="Profiler artifacts")
    correctness_artifacts: list[str] = Field(
        default_factory=list, description="Correctness validation artifacts"
    )
    bottleneck_type: str | None = Field(default=None, description="Diagnosed bottleneck type")
    promotion_decision: str | None = Field(
        default=None, description="Promotion/rejection decision for the attempt"
    )
    rejection_reason: str | None = Field(
        default=None, description="Reason a candidate was rejected"
    )
    status: DreamMemoryStatus = Field(
        default="candidate",
        description="Memory lifecycle status",
    )
    evidence_level: DreamMemoryEvidenceLevel = Field(
        default="smoke",
        description="Evidence strength for this memory",
    )
    promotion_reason: str | None = Field(
        default=None,
        description="Evidence-backed reason for promoting this memory",
    )
    chosen: int = Field(default=0, ge=0, description="Times selected/reused")
    useful_when_chosen: int = Field(default=0, ge=0, description="Times useful when selected")
    useful_rate: float = Field(default=0, ge=0, le=1, description="Useful rate if known")
    time: str | None = Field(default=None, description="Elapsed optimization time")
    token_used: int | None = Field(default=None, ge=0, description="Tokens spent")
    source_session_id: str | None = Field(default=None, description="Session that wrote it")

    @model_validator(mode="after")
    def require_artifact_evidence(self) -> Self:
        evidence_fields = (
            self.artifacts,
            self.artifact_refs,
            self.baseline_artifacts,
            self.optimized_artifacts,
            self.profiler_artifacts,
            self.correctness_artifacts,
        )
        if not any(evidence_fields):
            raise ValueError("optimization memories must include at least one artifact reference")
        if self.success and not self.correctness_artifacts:
            raise ValueError("successful optimization memories must include correctness evidence")
        return self


class EnvironmentDebugMemoryInput(BaseModel):
    """Input schema for writing environment deployment/debug memory."""

    id: str | None = Field(default=None, description="Optional stable ID for upsert")
    title: str = Field(description="Short title for this deployment/debug experience", min_length=1)
    summary: str = Field(default="", description="One-line summary")
    tags: list[str] = Field(default_factory=list, description="Search and graph tags")
    environment: str = Field(description="Compute/runtime environment")
    debug_type: EnvironmentDebugType = Field(description="Environment/debug issue type")
    component: str = Field(description="Affected component, package, service, or subsystem")
    hardware: str | None = Field(default=None, description="Hardware details")
    os: str | None = Field(default=None, description="Operating system")
    driver: str | None = Field(default=None, description="Driver stack")
    runtime: str | None = Field(default=None, description="Runtime stack")
    dependency_stack: dict[str, Any] = Field(
        default_factory=dict, description="Relevant packages and versions"
    )
    inference_backend: str | None = Field(default=None, description="Inference backend if relevant")
    related_backend: str | None = Field(default=None, description="Related backend or subsystem")
    issue_signature: str = Field(description="Stable searchable error signature", min_length=1)
    symptoms: str = Field(description="Observed failure symptoms", min_length=1)
    root_cause: str = Field(description="Root cause", min_length=1)
    solution: str = Field(description="Fix or workaround", min_length=1)
    verification: str = Field(description="How the fix was verified", min_length=1)
    commands: list[str] = Field(default_factory=list, description="Commands used to debug/fix")
    error_messages: list[str] = Field(default_factory=list, description="Important errors")
    diagnostic_steps: list[str] = Field(default_factory=list, description="Debug steps taken")
    prevention: str | None = Field(default=None, description="How to prevent recurrence")
    risk: str | None = Field(default=None, description="Risks of applying this memory")
    caveats: str | None = Field(default=None, description="Known caveats")
    diagnostic_artifacts: list[str] = Field(
        default_factory=list, description="Diagnostic logs or artifacts"
    )
    verification_artifacts: list[str] = Field(
        default_factory=list, description="Verification logs or artifacts"
    )
    resolved_by_command: str | None = Field(
        default=None, description="Primary command that resolved the issue"
    )
    environment_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Environment snapshot captured during debugging"
    )
    success: bool = Field(description="Whether the debug/deployment fix succeeded")
    artifacts: list[str] = Field(default_factory=list, description="Related files or URLs")
    artifact_refs: list[str] = Field(
        default_factory=list,
        description="Canonical evidence artifact references used for gating and export",
    )
    status: DreamMemoryStatus = Field(
        default="candidate",
        description="Memory lifecycle status",
    )
    evidence_level: DreamMemoryEvidenceLevel = Field(
        default="smoke",
        description="Evidence strength for this memory",
    )
    promotion_reason: str | None = Field(
        default=None,
        description="Evidence-backed reason for promoting this memory",
    )
    chosen: int = Field(default=0, ge=0, description="Times selected/reused")
    useful_when_chosen: int = Field(default=0, ge=0, description="Times useful when selected")
    useful_rate: float = Field(default=0, ge=0, le=1, description="Useful rate if known")
    time: str | None = Field(default=None, description="Elapsed debug/deployment time")
    token_used: int | None = Field(default=None, ge=0, description="Tokens spent")
    source_session_id: str | None = Field(default=None, description="Session that wrote it")

    @model_validator(mode="after")
    def require_artifact_evidence(self) -> Self:
        evidence_fields = (
            self.artifacts,
            self.artifact_refs,
            self.diagnostic_artifacts,
            self.verification_artifacts,
        )
        if not any(evidence_fields):
            raise ValueError(
                "environment debug memories must include diagnostic or "
                "verification artifact evidence"
            )
        return self


class DreamMemoriesResponse(BaseModel):
    """List response for Dream memories."""

    memories: list[DreamMemory]


class DreamMemorySnapshotInput(BaseModel):
    """Complete Dream memory snapshot used for campaign A/B isolation."""

    memories: list[DreamMemory] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_importable_memory_evidence(self) -> Self:
        for memory in self.memories:
            _validate_durable_memory_evidence(memory)
        return self


def _validate_durable_memory_evidence(memory: DreamMemory) -> None:
    if memory.category == "optimization":
        evidence_fields = (
            memory.artifacts,
            memory.artifact_refs,
            memory.baseline_artifacts,
            memory.optimized_artifacts,
            memory.profiler_artifacts,
            memory.correctness_artifacts,
        )
        if not any(evidence_fields):
            raise ValueError("optimization memories must include at least one artifact reference")
        if memory.success is True and not memory.correctness_artifacts:
            raise ValueError("successful optimization memories must include correctness evidence")
        return

    evidence_fields = (
        memory.artifacts,
        memory.artifact_refs,
        memory.diagnostic_artifacts,
        memory.verification_artifacts,
    )
    if not any(evidence_fields):
        raise ValueError(
            "environment debug memories must include diagnostic or verification artifact evidence"
        )


class DreamMemoryWriteResponse(BaseModel):
    """Write response for Dream memories."""

    memory: DreamMemory


class DreamMemorySearchInput(BaseModel):
    """Input schema for retrieving useful Dream memories."""

    query: str = Field(description="Search query describing the current optimization/debug task")
    category: DreamMemoryCategory = Field(
        default="optimization",
        description="Memory category to search",
    )
    tags: list[str] = Field(default_factory=list, description="Structured tags to match")
    memory_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional fixed Dream memory IDs for retrieval-controlled experiments. "
            "When provided, ranking uses this exact order for memories in the requested category."
        ),
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum memories to return")
    record_choice: bool = Field(
        default=True,
        description="Increment chosen count for returned memories",
    )


class DreamMemorySearchResult(BaseModel):
    """A ranked Dream memory search result."""

    memory: DreamMemory
    score: float = Field(ge=0, description="Combined tag and semantic score")
    reasons: list[str] = Field(default_factory=list, description="Why this memory matched")
    recommended_action: DreamMemoryRecommendedAction = Field(
        description="How an agent should treat this memory during transfer"
    )


class DreamMemorySearchResponse(BaseModel):
    """Search response for Dream memories."""

    results: list[DreamMemorySearchResult]


class DreamMemoryFeedbackInput(BaseModel):
    """Feedback that retrieved Dream memories helped an evidence-backed outcome."""

    memory_ids: list[str] = Field(
        description="Dream memory IDs that were selected or retrieved for the task",
        min_length=1,
    )
    useful: bool = Field(
        default=True,
        description="Whether the selected memories were useful for the verified outcome",
    )
    reason: str | None = Field(
        default=None,
        description="Short evidence note, e.g. verifier/benchmark/profiler result",
    )
    evidence_artifacts: list[str] = Field(
        default_factory=list,
        description=(
            "Benchmark, verifier, profiler, correctness, or debug artifacts "
            "supporting useful=true feedback"
        ),
    )
    source_session_id: str | None = Field(
        default=None,
        description="Session or campaign run that produced the feedback",
    )

    @model_validator(mode="after")
    def require_evidence_reason_for_useful_feedback(self) -> Self:
        if self.useful and not (self.reason and self.reason.strip()):
            raise ValueError("useful Dream memory feedback must include evidence reason")
        if self.useful and not any(artifact.strip() for artifact in self.evidence_artifacts):
            raise ValueError(
                "useful Dream memory feedback must include evidence artifact references"
            )
        return self


class DreamMemoryFeedbackResponse(BaseModel):
    """Result of recording Dream memory usefulness feedback."""

    memories: list[DreamMemory]
    missing_memory_ids: list[str] = Field(default_factory=list)


def dream_memory_file() -> Path:
    return get_share_dir() / "dream" / "memories.json"


def dream_memory_file_for_share_dir(share_dir: Path) -> Path:
    return share_dir.expanduser() / "dream" / "memories.json"


@contextlib.contextmanager
def _locked_file(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def _read_memories_unlocked(memory_file: Path) -> list[DreamMemory]:
    if not memory_file.exists():
        return []

    try:
        raw = json.loads(memory_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read Dream memories from {}: {}", memory_file, exc)
        return []

    items = raw.get("memories", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        logger.warning("Dream memory file must contain a list or an object with a memories list")
        return []

    memories: list[DreamMemory] = []
    for item in items:
        try:
            memories.append(DreamMemory.model_validate(item))
        except ValidationError as exc:
            logger.warning("Skipping invalid Dream memory entry: {}", exc)
    return memories


def list_dream_memories(category: DreamMemoryCategory | None = None) -> list[DreamMemory]:
    memory_file = dream_memory_file()
    with _locked_file(memory_file):
        memories = _read_memories_unlocked(memory_file)
    if category is not None:
        memories = [memory for memory in memories if memory.category == category]
    return memories


def restore_dream_memory_snapshot(snapshot: DreamMemorySnapshotInput) -> list[DreamMemory]:
    """Replace the Dream memory store with a schema-validated snapshot."""

    memory_file = dream_memory_file()
    memories = list(snapshot.memories)
    with _locked_file(memory_file):
        _write_memories_unlocked(memory_file, memories)
    return memories


def load_packaged_seed_memories() -> list[DreamMemory]:
    """Load the versioned seed memories shipped with EvoInfer Dream."""

    seed_text = (
        resources.files("evoinfer_mcp.dream")
        .joinpath("seed_memories.json")
        .read_text(encoding="utf-8")
    )
    payload = json.loads(seed_text)
    raw_memories = payload.get("memories", payload)
    snapshot = DreamMemorySnapshotInput.model_validate({"memories": raw_memories})
    return list(snapshot.memories)


def ensure_packaged_seed_memories(share_dir: Path | None = None) -> dict[str, Any]:
    """Seed a cold durable store with packaged EvoInfer memories.

    This is the lazy MCP startup path. It only writes when the target store is
    missing or empty, so existing user memories, local edits, and campaign
    isolation stores are not implicitly merged on every tool call. Use
    `merge_packaged_seed_memories` for explicit seed upgrades.
    """

    memory_file = dream_memory_file_for_share_dir(share_dir) if share_dir else dream_memory_file()
    if os.getenv("EVOINFER_DISABLE_SEED_MEMORY") == "1":
        return {
            "share_dir": str(memory_file.parents[1]),
            "memory_file": str(memory_file),
            "seed_count": 0,
            "imported_count": 0,
            "existing_count": 0,
            "memory_ids": [],
            "disabled": True,
        }

    seed_memories = load_packaged_seed_memories()
    with _locked_file(memory_file):
        existing_memories = _read_memories_unlocked(memory_file)
        if existing_memories:
            return {
                "share_dir": str(memory_file.parents[1]),
                "memory_file": str(memory_file),
                "seed_count": len(seed_memories),
                "imported_count": 0,
                "existing_count": len(existing_memories),
                "memory_ids": [],
                "already_initialized": True,
            }
        if seed_memories:
            _write_memories_unlocked(memory_file, seed_memories)
    return {
        "share_dir": str(memory_file.parents[1]),
        "memory_file": str(memory_file),
        "seed_count": len(seed_memories),
        "imported_count": len(seed_memories),
        "existing_count": 0,
        "memory_ids": [memory.id for memory in seed_memories],
    }


def merge_packaged_seed_memories(share_dir: Path | None = None) -> dict[str, Any]:
    """Merge packaged seed memories into a user's durable store without overwriting.

    Seed memories prevent cold-start retrieval, but the user's local store remains
    authoritative. Existing IDs are preserved so local feedback counters,
    promotions, and edits are not reset by package upgrades.
    """

    memory_file = dream_memory_file_for_share_dir(share_dir) if share_dir else dream_memory_file()
    seed_memories = load_packaged_seed_memories()
    with _locked_file(memory_file):
        existing_memories = _read_memories_unlocked(memory_file)
        existing_ids = {memory.id for memory in existing_memories}
        imported = [memory for memory in seed_memories if memory.id not in existing_ids]
        if imported:
            _write_memories_unlocked(memory_file, [*existing_memories, *imported])
    return {
        "share_dir": str(memory_file.parents[1]),
        "memory_file": str(memory_file),
        "seed_count": len(seed_memories),
        "imported_count": len(imported),
        "existing_count": len(existing_memories),
        "memory_ids": [memory.id for memory in imported],
    }


def _write_memories_unlocked(memory_file: Path, memories: list[DreamMemory]) -> None:
    payload = {
        "version": 1,
        "memories": [memory.model_dump(mode="json", exclude_none=True) for memory in memories],
    }
    atomic_json_write(payload, memory_file)


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:+-]+")
_EMBEDDING_DIMS = 128
_OPERATION_TAGS = {
    "attention",
    "decode",
    "embedding",
    "gemm",
    "kv_cache",
    "kv-cache",
    "layernorm",
    "matmul",
    "prefill",
    "rmsnorm",
    "softmax",
}
_CORRECTNESS_METRIC_THRESHOLDS = {
    "max_abs_error": 1e-4,
    "max_row_sum_error": 1e-5,
}
_NEGATIVE_SEARCH_INTENT_TAGS = {
    "bad",
    "bug",
    "correctness_failure",
    "failed",
    "failure",
    "fail",
    "fails",
    "fp16_failure",
    "negative",
    "negative_evidence",
    "negative_optimization",
    "reject",
    "rejected",
    "unsafe",
}


def _normalize_tag(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.strip()]


def _stable_token_bucket(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _EMBEDDING_DIMS


def _hashed_embedding(text: str) -> list[float]:
    vector = [0.0] * _EMBEDDING_DIMS
    for token in _tokenize(text):
        vector[_stable_token_bucket(token)] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return max(0.0, sum(a * b for a, b in zip(left, right, strict=False)))


def _memory_text(memory: DreamMemory) -> str:
    structured_values: list[Any] = [
        memory.title,
        memory.summary,
        *memory.tags,
        memory.environment,
        memory.model_type,
        memory.model_arch,
        memory.model_name,
        memory.model_size,
        memory.inference_backend,
        memory.serving_framework,
        memory.objective_metric,
        memory.detail_description,
        memory.applicability,
        memory.caveats,
        *memory.operation_semantics,
        *memory.correctness_invariants,
        *memory.safe_transfer_notes,
        *memory.unsafe_transfer_notes,
        memory.failure_reason,
        memory.benchmark_command,
        memory.bottleneck_type,
        memory.promotion_decision,
        memory.rejection_reason,
        memory.debug_type,
        memory.component,
        memory.hardware,
        memory.os,
        memory.driver,
        memory.runtime,
        memory.issue_signature,
        memory.symptoms,
        memory.root_cause,
        memory.solution,
        memory.verification,
        memory.related_backend,
        memory.risk,
        *memory.commands,
        *memory.error_messages,
        *memory.diagnostic_steps,
        memory.prevention,
    ]
    json_values: list[Any] = [
        memory.precision,
        memory.workload,
        memory.metrics_before,
        memory.metrics_after,
        memory.dependency_stack,
        memory.environment_snapshot,
    ]
    parts = [str(value) for value in structured_values if value]
    parts.extend(
        json.dumps(value, ensure_ascii=False, sort_keys=True) for value in json_values if value
    )
    return "\n".join(parts)


def _memory_tags(memory: DreamMemory) -> set[str]:
    raw_tags: list[str] = [
        *memory.tags,
        memory.category,
        memory.environment or "",
        memory.model_type or "",
        memory.model_arch or "",
        memory.model_name or "",
        memory.inference_backend or "",
        memory.serving_framework or "",
        memory.bottleneck_type or "",
        memory.debug_type or "",
        memory.component or "",
        memory.related_backend or "",
    ]
    tags: set[str] = set()
    for raw in raw_tags:
        normalized = _normalize_tag(str(raw))
        if normalized:
            tags.add(normalized)
            tags.update(_tokenize(normalized))
    return tags


def _score_memory(
    memory: DreamMemory,
    *,
    query: str,
    query_embedding: list[float],
    embedding_model_score: float | None,
    requested_tags: set[str],
) -> tuple[float, list[str]]:
    memory_text = _memory_text(memory)
    memory_embedding = _hashed_embedding(memory_text)
    hashed_embedding_score = _cosine_similarity(query_embedding, memory_embedding)
    semantic_score = (
        embedding_model_score if embedding_model_score is not None else hashed_embedding_score
    )

    query_tokens = set(_tokenize(query))
    memory_tokens = set(_tokenize(memory_text))
    lexical_matches = sorted(query_tokens & memory_tokens)
    lexical_score = min(1.0, len(lexical_matches) / max(1, len(query_tokens)))

    memory_tags = _memory_tags(memory)
    tag_matches = sorted(requested_tags & memory_tags)
    tag_score = len(tag_matches) / max(1, len(requested_tags)) if requested_tags else 0.0
    workload_score, workload_reasons = _workload_match_score(memory, query)
    precision_score, precision_reasons = _precision_match_score(memory, query)
    negative_constraint_bonus = _negative_constraint_bonus(
        memory,
        workload_reasons=workload_reasons,
        precision_reasons=precision_reasons,
    )

    success_bonus = 0.1 if memory.success is True else 0.0
    useful_bonus = min(0.2, memory.useful_rate * 0.2)
    score = (
        (tag_score * 2.0)
        + semantic_score
        + (lexical_score * 0.75)
        + workload_score
        + precision_score
        + negative_constraint_bonus
        + success_bonus
        + useful_bonus
    )

    reasons: list[str] = []
    reasons.extend(f"tag:{tag}" for tag in tag_matches)
    reasons.extend(workload_reasons)
    reasons.extend(precision_reasons)
    if negative_constraint_bonus:
        reasons.append("negative_constraint_match")
    reasons.extend(f"term:{term}" for term in lexical_matches[:8])
    if embedding_model_score is not None and embedding_model_score > 0:
        reasons.append(f"embedding_model:{embedding_model_score:.3f}")
    elif hashed_embedding_score > 0:
        reasons.append(f"embedding:{hashed_embedding_score:.3f}")
    if memory.useful_rate > 0:
        reasons.append(f"useful_rate:{memory.useful_rate:.3f}")
    return round(score, 6), reasons


def _workload_match_score(memory: DreamMemory, query: str) -> tuple[float, list[str]]:
    if not memory.workload:
        return 0.0, []
    query_numbers = _query_numeric_hints(query)
    if not query_numbers:
        return 0.0, []

    matches = 0
    mismatches = 0
    reasons: list[str] = []
    for raw_key, raw_value in memory.workload.items():
        key = _normalize_tag(str(raw_key))
        value = _as_float(raw_value)
        if not key or value is None:
            continue
        aliases = _workload_key_aliases(key)
        for alias in aliases:
            expected = query_numbers.get(alias)
            if expected is None:
                continue
            tolerance = max(1e-9, abs(expected) * 0.02)
            if abs(value - expected) <= tolerance:
                matches += 1
                reasons.append(f"workload_match:{alias}={_format_number(value)}")
            else:
                mismatches += 1
                reasons.append(
                    "workload_mismatch:"
                    f"{alias}={_format_number(value)}!={_format_number(expected)}"
                )
            break
    return min(1.5, matches * 0.5) - min(1.2, mismatches * 0.4), reasons


def _precision_match_score(memory: DreamMemory, query: str) -> tuple[float, list[str]]:
    query_precision = _query_precision_hints(query)
    if not query_precision:
        return 0.0, []

    memory_precision = _memory_precision_hints(memory)
    if not memory_precision:
        return 0.0, []

    matches = 0
    mismatches = 0
    reasons: list[str] = []
    for key, expected in query_precision.items():
        value = _precision_value_for_key(memory_precision, key)
        if value is None:
            continue
        if value == expected:
            matches += 1
            reasons.append(f"precision_match:{key}={expected}")
        else:
            mismatches += 1
            reasons.append(f"precision_mismatch:{key}={value}!={expected}")
    return min(1.2, matches * 0.6) - min(1.5, mismatches * 0.75), reasons


def _negative_constraint_bonus(
    memory: DreamMemory,
    *,
    workload_reasons: list[str],
    precision_reasons: list[str],
) -> float:
    if not _is_negative_constraint(memory):
        return 0.0
    if any(reason.startswith("precision_match:") for reason in precision_reasons):
        return 0.9
    if any(reason.startswith("workload_match:") for reason in workload_reasons) and any(
        reason.startswith("precision_mismatch:") for reason in precision_reasons
    ):
        return 0.45
    return 0.0


def _is_negative_constraint(memory: DreamMemory) -> bool:
    return (
        memory.status in {"negative", "rejected"}
        or memory.success is False
        or bool(_correctness_metric_violations(memory.metrics_after))
    )


def _query_numeric_hints(query: str) -> dict[str, float]:
    hints: dict[str, float] = {}
    for key, value in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_-]*)\s*[:=]\s*(\d+(?:\.\d+)?)",
        query,
    ):
        normalized = _normalize_tag(key)
        if normalized:
            hints[normalized] = float(value)
    for key, value in re.findall(
        r"\b(batch|seq|sequence|head_dim|headdim|hidden|hidden_size|b|t|n)\s*(\d+(?:\.\d+)?)\b",
        query,
        flags=re.IGNORECASE,
    ):
        normalized = _normalize_tag(key)
        if normalized:
            hints[normalized] = float(value)
    return hints


def _query_precision_hints(query: str) -> dict[str, str]:
    hints: dict[str, str] = {}
    for key, value in re.findall(
        r"\b(dtype|precision|compute_dtype|accum_dtype|accumulator_dtype|quantization)"
        r"\s*[:=]\s*([A-Za-z0-9_+.-]+)",
        query,
        flags=re.IGNORECASE,
    ):
        normalized_key = _normalize_precision_key(key)
        normalized_value = _normalize_precision_value(value)
        if normalized_key and normalized_value:
            hints[normalized_key] = normalized_value

    for token in _tokenize(query):
        dtype = _normalize_precision_value(token)
        if dtype in _KNOWN_DTYPE_ALIASES.values():
            hints.setdefault("dtype", dtype)
    return hints


def _memory_precision_hints(memory: DreamMemory) -> dict[str, str]:
    hints: dict[str, str] = {}
    for raw_key, raw_value in memory.precision.items():
        key = _normalize_precision_key(str(raw_key))
        if not key:
            continue
        value = _normalize_precision_value(str(raw_value))
        if value:
            hints[key] = value
    return hints


def _precision_value_for_key(memory_precision: dict[str, str], key: str) -> str | None:
    if key in memory_precision:
        return memory_precision[key]
    if key == "dtype":
        for alias in ("precision", "compute_dtype"):
            if alias in memory_precision:
                return memory_precision[alias]
    if key == "accum_dtype":
        return memory_precision.get("accumulator_dtype")
    return None


def _normalize_precision_key(key: str) -> str:
    normalized = _normalize_tag(key)
    if normalized == "precision":
        return "dtype"
    if normalized == "accumulator_dtype":
        return "accum_dtype"
    return normalized


_KNOWN_DTYPE_ALIASES = {
    "fp16": "float16",
    "f16": "float16",
    "float16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "f32": "float32",
    "float32": "float32",
    "single": "float32",
    "tf32": "tf32",
    "fp64": "float64",
    "f64": "float64",
    "float64": "float64",
    "int8": "int8",
    "uint8": "uint8",
    "fp8": "fp8",
}


def _normalize_precision_value(value: str) -> str:
    normalized = _normalize_tag(value)
    return _KNOWN_DTYPE_ALIASES.get(normalized, normalized)


def _workload_key_aliases(key: str) -> set[str]:
    aliases = {
        key,
    }
    if key == "batch":
        aliases.update({"b"})
    elif key == "seq":
        aliases.update({"sequence", "t", "n"})
    elif key == "sequence":
        aliases.update({"seq", "t", "n"})
    elif key == "head_dim":
        aliases.update({"headdim"})
    elif key == "hidden":
        aliases.update({"hidden_size"})
    return aliases


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _requested_operation_tags(query: str, requested_tags: set[str]) -> set[str]:
    query_tokens = {_normalize_tag(token) for token in _tokenize(query)}
    return (requested_tags | query_tokens) & _OPERATION_TAGS


def _memory_matches_any_operation(memory: DreamMemory, operation_tags: set[str]) -> bool:
    return bool(operation_tags & _memory_tags(memory))


def _filter_cross_operator_results(
    ranked: list[DreamMemorySearchResult],
    *,
    query: str,
    requested_tags: set[str],
) -> list[DreamMemorySearchResult]:
    """Prefer operation-specific memories while preserving generic fallback behavior."""

    operation_tags = _requested_operation_tags(query, requested_tags)
    if not operation_tags:
        return ranked

    operation_matches = [
        result for result in ranked if _memory_matches_any_operation(result.memory, operation_tags)
    ]
    return operation_matches or ranked


def _has_negative_search_intent(query: str, requested_tags: set[str]) -> bool:
    query_tokens = {_normalize_tag(token) for token in _tokenize(query)}
    return bool((query_tokens | requested_tags) & _NEGATIVE_SEARCH_INTENT_TAGS)


def _prefer_successful_optimization_results(
    ranked: list[DreamMemorySearchResult],
    *,
    query: str,
    requested_tags: set[str],
) -> list[DreamMemorySearchResult]:
    if _has_negative_search_intent(query, requested_tags):
        return ranked

    exact_negative_constraints = [
        result
        for result in ranked
        if result.recommended_action == "avoid_as_negative_constraint"
        and any(reason.startswith("precision_match:") for reason in result.reasons)
    ]

    successful = [
        result
        for result in ranked
        if result.memory.category == "optimization"
        and result.memory.success is True
        and not _correctness_metric_violations(result.memory.metrics_after)
    ]
    if not successful:
        return ranked
    successful_ids = {result.memory.id for result in successful}
    exact_negative_ids = {result.memory.id for result in exact_negative_constraints}
    return (
        exact_negative_constraints
        + [
            result
            for result in successful
            if result.memory.id not in exact_negative_ids
        ]
        + [
            result
            for result in ranked
            if result.memory.id not in successful_ids | exact_negative_ids
        ]
    )


def _recommended_action(memory: DreamMemory) -> DreamMemoryRecommendedAction:
    if _is_negative_constraint(memory):
        return "avoid_as_negative_constraint"
    if memory.category == "optimization":
        if memory.success is True and memory.correctness_artifacts:
            return "apply_as_hypothesis"
        return "review_evidence_before_use"
    if memory.success is True:
        return "apply_as_hypothesis"
    return "review_evidence_before_use"


def _upsert_memory(memory: DreamMemory) -> DreamMemory:
    memory_file = dream_memory_file()
    with _locked_file(memory_file):
        memories = _read_memories_unlocked(memory_file)
        for index, existing in enumerate(memories):
            if existing.id == memory.id:
                if memory.created_at is None:
                    memory.created_at = existing.created_at
                memories[index] = memory
                break
        else:
            memories.append(memory)
        _write_memories_unlocked(memory_file, memories)
    return memory


def _increment_chosen_unlocked(memories: list[DreamMemory], ids: set[str]) -> None:
    now = _now()
    for memory in memories:
        if memory.id not in ids:
            continue
        memory.chosen += 1
        memory.useful_rate = (
            min(1.0, memory.useful_when_chosen / memory.chosen) if memory.chosen > 0 else 0.0
        )
        memory.updated_at = now


def search_dream_memories(params: DreamMemorySearchInput) -> DreamMemorySearchResponse:
    """Retrieve memories by combining structured tags and lightweight semantic matching."""

    ensure_packaged_seed_memories()
    memory_file = dream_memory_file()
    requested_tags = {_normalize_tag(tag) for tag in params.tags if _normalize_tag(tag)}
    query_embedding = _hashed_embedding(params.query)

    with _locked_file(memory_file):
        all_memories = _read_memories_unlocked(memory_file)
        memories = [memory for memory in all_memories if memory.category == params.category]
        pinned_ids = list(dict.fromkeys(memory_id for memory_id in params.memory_ids if memory_id))
        if pinned_ids:
            memory_by_id = {memory.id: memory for memory in memories}
            ranked = [
                DreamMemorySearchResult(
                    memory=memory_by_id[memory_id],
                    score=1.0,
                    reasons=["pinned_id"],
                    recommended_action=_recommended_action(memory_by_id[memory_id]),
                )
                for memory_id in pinned_ids
                if memory_id in memory_by_id
            ]
            ranked = ranked[: params.top_k]
        else:
            ranked: list[DreamMemorySearchResult] = []
            memory_texts = [_memory_text(memory) for memory in memories]
            embedding_model_scores = score_texts_with_optional_embedding(
                params.query,
                memory_texts,
            )
            score_by_memory_id = (
                {
                    memory.id: score
                    for memory, score in zip(memories, embedding_model_scores, strict=False)
                }
                if embedding_model_scores is not None
                else {}
            )
            for memory in memories:
                score, reasons = _score_memory(
                    memory,
                    query=params.query,
                    query_embedding=query_embedding,
                    embedding_model_score=score_by_memory_id.get(memory.id),
                    requested_tags=requested_tags,
                )
                if score <= 0:
                    continue
                ranked.append(
                    DreamMemorySearchResult(
                        memory=memory,
                        score=score,
                        reasons=reasons,
                        recommended_action=_recommended_action(memory),
                    )
                )

            ranked.sort(
                key=lambda result: (
                    result.score,
                    result.memory.useful_rate,
                    result.memory.chosen,
                    result.memory.updated_at
                    or result.memory.created_at
                    or datetime.min.replace(tzinfo=UTC),
                ),
                reverse=True,
            )
            ranked = _filter_cross_operator_results(
                ranked,
                query=params.query,
                requested_tags=requested_tags,
            )
            ranked = _prefer_successful_optimization_results(
                ranked,
                query=params.query,
                requested_tags=requested_tags,
            )
            ranked = ranked[: params.top_k]
        if params.record_choice and ranked:
            selected_ids = {result.memory.id for result in ranked}
            _increment_chosen_unlocked(all_memories, selected_ids)
            memory_by_id = {memory.id: memory for memory in all_memories}
            ranked = [
                result.model_copy(
                    update={
                        "memory": memory_by_id[result.memory.id],
                        "recommended_action": _recommended_action(memory_by_id[result.memory.id]),
                    }
                )
                for result in ranked
            ]
            _write_memories_unlocked(memory_file, all_memories)
    return DreamMemorySearchResponse(results=ranked)


def record_dream_memory_feedback(
    params: DreamMemoryFeedbackInput,
) -> DreamMemoryFeedbackResponse:
    """Record whether selected memories were useful after an evidence-backed result."""

    requested_ids = list(dict.fromkeys(params.memory_ids))
    requested_set = set(requested_ids)
    memory_file = dream_memory_file()
    updated: list[DreamMemory] = []
    found_ids: set[str] = set()
    now = _now()

    with _locked_file(memory_file):
        memories = _read_memories_unlocked(memory_file)
        for memory in memories:
            if memory.id not in requested_set:
                continue
            found_ids.add(memory.id)
            if params.useful:
                if memory.chosen == 0:
                    memory.chosen = 1
                if memory.useful_when_chosen < memory.chosen:
                    memory.useful_when_chosen += 1
            memory.useful_rate = (
                min(1.0, memory.useful_when_chosen / memory.chosen) if memory.chosen > 0 else 0.0
            )
            memory.updated_at = now
            updated.append(memory)
        if updated:
            _write_memories_unlocked(memory_file, memories)

    return DreamMemoryFeedbackResponse(
        memories=updated,
        missing_memory_ids=[memory_id for memory_id in requested_ids if memory_id not in found_ids],
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _memory_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _apply_optimization_correctness_gate(values: dict[str, Any]) -> None:
    violations = _correctness_metric_violations(values.get("metrics_after"))
    if not violations:
        return

    reason = "Correctness metrics failed: " + "; ".join(violations)
    values["success"] = False
    values["useful_when_chosen"] = 0
    values["useful_rate"] = 0.0
    if not values.get("failure_reason"):
        values["failure_reason"] = reason
    if not values.get("rejection_reason"):
        values["rejection_reason"] = reason


def _correctness_metric_violations(metrics_after: Any) -> list[str]:
    if not isinstance(metrics_after, dict):
        return []
    violations: list[str] = []
    for metric, threshold in _CORRECTNESS_METRIC_THRESHOLDS.items():
        value = _as_float(metrics_after.get(metric))
        if value is not None and value > threshold:
            violations.append(f"{metric}={value} > {threshold}")
    return violations


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def create_optimization_memory(params: OptimizationMemoryInput) -> DreamMemory:
    now = _now()
    values = params.model_dump(exclude={"id"})
    _apply_optimization_correctness_gate(values)
    memory = DreamMemory(
        **values,
        id=params.id or _memory_id("opt"),
        category="optimization",
        created_at=now,
        updated_at=now,
    )
    return _upsert_memory(memory)


def create_environment_debug_memory(params: EnvironmentDebugMemoryInput) -> DreamMemory:
    now = _now()
    values = params.model_dump(exclude={"id"})
    memory = DreamMemory(
        **values,
        id=params.id or _memory_id("env"),
        category="environment_debug",
        created_at=now,
        updated_at=now,
        detail_description=params.solution,
    )
    return _upsert_memory(memory)
