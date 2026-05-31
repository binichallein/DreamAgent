"""Dream memory schemas and persistence."""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from kimi_cli import logger
from kimi_cli.share import get_share_dir
from kimi_cli.utils.io import atomic_json_write

DreamMemoryCategory = Literal["optimization", "environment_debug"]
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
    failure_reason: str | None = Field(default=None, description="Failure reason if any")

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

    # Experiment bookkeeping
    artifacts: list[str] = Field(default_factory=list, description="Related files or URLs")
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
    environment: str = Field(description="Compute/runtime environment")
    model_type: str = Field(description="Model family, e.g. llm/vlm/vla/classical-ml")
    model_arch: str | None = Field(default=None, description="Model architecture")
    model_name: str | None = Field(default=None, description="Concrete model name")
    model_size: str | None = Field(default=None, description="Model size or parameter count")
    inference_backend: str = Field(description="Inference backend, e.g. cuda/triton/pytorch/vllm")
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
    failure_reason: str | None = Field(default=None, description="Failure reason if unsuccessful")
    artifacts: list[str] = Field(default_factory=list, description="Related files or URLs")
    chosen: int = Field(default=0, ge=0, description="Times selected/reused")
    useful_when_chosen: int = Field(default=0, ge=0, description="Times useful when selected")
    useful_rate: float = Field(default=0, ge=0, le=1, description="Useful rate if known")
    time: str | None = Field(default=None, description="Elapsed optimization time")
    token_used: int | None = Field(default=None, ge=0, description="Tokens spent")
    source_session_id: str | None = Field(default=None, description="Session that wrote it")


class EnvironmentDebugMemoryInput(BaseModel):
    """Input schema for writing environment deployment/debug memory."""

    id: str | None = Field(default=None, description="Optional stable ID for upsert")
    title: str = Field(description="Short title for this deployment/debug experience", min_length=1)
    summary: str = Field(default="", description="One-line summary")
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
    success: bool = Field(description="Whether the debug/deployment fix succeeded")
    artifacts: list[str] = Field(default_factory=list, description="Related files or URLs")
    chosen: int = Field(default=0, ge=0, description="Times selected/reused")
    useful_when_chosen: int = Field(default=0, ge=0, description="Times useful when selected")
    useful_rate: float = Field(default=0, ge=0, le=1, description="Useful rate if known")
    time: str | None = Field(default=None, description="Elapsed debug/deployment time")
    token_used: int | None = Field(default=None, ge=0, description="Tokens spent")
    source_session_id: str | None = Field(default=None, description="Session that wrote it")


class DreamMemoriesResponse(BaseModel):
    """List response for Dream memories."""

    memories: list[DreamMemory]


class DreamMemoryWriteResponse(BaseModel):
    """Write response for Dream memories."""

    memory: DreamMemory


def dream_memory_file() -> Path:
    return get_share_dir() / "dream" / "memories.json"


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


def _write_memories_unlocked(memory_file: Path, memories: list[DreamMemory]) -> None:
    payload = {
        "version": 1,
        "memories": [memory.model_dump(mode="json", exclude_none=True) for memory in memories],
    }
    atomic_json_write(payload, memory_file)


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


def _now() -> datetime:
    return datetime.now(UTC)


def _memory_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def create_optimization_memory(params: OptimizationMemoryInput) -> DreamMemory:
    now = _now()
    values = params.model_dump(exclude={"id"})
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
