"""Dream memory API routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from kimi_cli.dream.memory import (
    DreamMemoriesResponse,
    DreamMemoryCategory,
    DreamMemoryWriteResponse,
    EnvironmentDebugMemoryInput,
    OptimizationMemoryInput,
    create_environment_debug_memory,
    create_optimization_memory,
    list_dream_memories,
)

router = APIRouter(prefix="/api/dream", tags=["dream"])


@router.get("/memories", summary="List Dream memories")
async def list_memories(
    category: DreamMemoryCategory | None = Query(default=None),
) -> DreamMemoriesResponse:
    """List specialist Dream memories."""

    return DreamMemoriesResponse(memories=list_dream_memories(category))


@router.post("/memories/optimization", summary="Create or update an optimization memory")
async def write_optimization_memory(
    request: OptimizationMemoryInput,
) -> DreamMemoryWriteResponse:
    """Persist an inference optimization memory."""

    return DreamMemoryWriteResponse(memory=create_optimization_memory(request))


@router.post("/memories/environment-debug", summary="Create or update an environment debug memory")
async def write_environment_debug_memory(
    request: EnvironmentDebugMemoryInput,
) -> DreamMemoryWriteResponse:
    """Persist an environment deployment/debug memory."""

    return DreamMemoryWriteResponse(memory=create_environment_debug_memory(request))
