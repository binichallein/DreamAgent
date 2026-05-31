from __future__ import annotations

from pathlib import Path
from typing import override

from kosong.tooling import CallableTool2, ToolReturnValue

from kimi_cli.dream.memory import (
    EnvironmentDebugMemoryInput,
    OptimizationMemoryInput,
    create_environment_debug_memory,
    create_optimization_memory,
)
from kimi_cli.tools.utils import load_desc

OptimizationMemoryParams = OptimizationMemoryInput
EnvironmentDebugMemoryParams = EnvironmentDebugMemoryInput


class WriteOptimizationMemory(CallableTool2[OptimizationMemoryParams]):
    name: str = "WriteOptimizationMemory"
    description: str = load_desc(Path(__file__).parent / "write_optimization_memory.md")
    params: type[OptimizationMemoryParams] = OptimizationMemoryParams

    @override
    async def __call__(self, params: OptimizationMemoryParams) -> ToolReturnValue:
        memory = create_optimization_memory(params)
        return ToolReturnValue(
            is_error=False,
            output=f"Optimization memory saved: {memory.id}",
            message=f"Optimization memory saved: {memory.id}",
            display=[],
        )


class WriteEnvironmentDebugMemory(CallableTool2[EnvironmentDebugMemoryParams]):
    name: str = "WriteEnvironmentDebugMemory"
    description: str = load_desc(Path(__file__).parent / "write_environment_debug_memory.md")
    params: type[EnvironmentDebugMemoryParams] = EnvironmentDebugMemoryParams

    @override
    async def __call__(self, params: EnvironmentDebugMemoryParams) -> ToolReturnValue:
        memory = create_environment_debug_memory(params)
        return ToolReturnValue(
            is_error=False,
            output=f"Environment debug memory saved: {memory.id}",
            message=f"Environment debug memory saved: {memory.id}",
            display=[],
        )


__all__ = (
    "EnvironmentDebugMemoryParams",
    "OptimizationMemoryParams",
    "WriteEnvironmentDebugMemory",
    "WriteOptimizationMemory",
)
