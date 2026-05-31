from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from kosong.message import Message, TextPart

from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_TURN_INTERVAL = 6


class DreamModeInjectionProvider(DynamicInjectionProvider):
    """Injects memory-writing guidance while Dream mode is active."""

    async def get_injections(
        self,
        history: Sequence[Message],
        soul: KimiSoul,
    ) -> list[DynamicInjection]:
        if not soul.dream_mode:
            return []

        if soul.consume_pending_dream_activation_injection():
            return [DynamicInjection(type="dream_mode", content=_full_reminder())]

        turns_since_last = 0
        found_previous = False
        for msg in reversed(history):
            if msg.role == "user" and _has_dream_reminder(msg):
                found_previous = True
                break
            if msg.role == "assistant":
                turns_since_last += 1

        if not found_previous:
            return [DynamicInjection(type="dream_mode", content=_full_reminder())]
        if turns_since_last < _TURN_INTERVAL:
            return []
        return [DynamicInjection(type="dream_mode", content=_sparse_reminder())]


def _has_dream_reminder(msg: Message) -> bool:
    keys = (
        _full_reminder().split("\n")[0],
        _sparse_reminder().split(".")[0],
    )
    for part in msg.content:
        if isinstance(part, TextPart) and any(key in part.text for key in keys):
            return True
    return False


def _full_reminder() -> str:
    return "\n".join(
        [
            "Dream mode is active for this session.",
            "",
            "Your specialist memory scope is strictly inference optimization and "
            "environment deployment/debug experience.",
            "",
            "When working on inference optimization, behave like a rigorous inference "
            "performance engineer:",
            "",
            "- Start from measurement. Establish a clear baseline before changing anything.",
            "- Use real profiling and benchmarking tools when applicable, such as nsys, ncu, "
            "torch profiler, Triton profiler, backend-specific tracing tools, or server-side "
            "latency/throughput metrics.",
            "- Identify the bottleneck before optimizing: CPU overhead, GPU kernel time, "
            "memory bandwidth, launch overhead, synchronization, data transfer, KV cache, "
            "attention, prefill/decode imbalance, batching, scheduling, quantization, or a "
            "specific operator.",
            "- Prefer controlled experiments. Change one major factor at a time when possible.",
            "- Consider backend-level changes when justified, such as PyTorch eager/compile, "
            "CUDA custom kernels, Triton kernels, CUTLASS/CuTe-style kernels, TensorRT-LLM, "
            "vLLM, SGLang, llama.cpp, ONNX Runtime, or other serving/runtime backends.",
            "- Consider narrow operator-level optimization when the profiler points to one "
            "hotspot: rewrite one op, fuse kernels, change memory layout, tune block sizes, "
            "reduce synchronization, improve cache locality, or specialize for shape/precision.",
            "- Always compare against the baseline with the same workload, hardware, input "
            "shapes, batch size, concurrency, precision, warmup, and measurement method.",
            "- Validate output correctness or acceptable numerical drift after optimization. "
            "Do not treat speedup as valid if inference results changed unexpectedly.",
            "- Do not fake benchmarks, cherry-pick measurements, hide failed runs, or hack "
            "numbers to claim SOTA performance. Record honest results, including negative or "
            "failed attempts when they teach a reusable lesson.",
            "- Capture enough detail for future reuse: hardware, model type, model architecture, "
            "backend, precision, workload, benchmark command or method, before/after metrics, "
            "correctness check, bottleneck diagnosis, and the actual change that caused the "
            "result.",
            "",
            "When the conversation yields reusable experience, write it with:",
            "",
            "- WriteOptimizationMemory for model inference optimization lessons, including "
            "hardware, model type/architecture, backend, precision, workload, profiler "
            "findings, before/after metrics, correctness validation, success, and detailed "
            "optimization description.",
            "- WriteEnvironmentDebugMemory for deployment/debug lessons, including "
            "environment, debug_type, component, dependency stack, issue_signature, symptoms, "
            "root_cause, solution, verification, commands, and success.",
            "",
            "Do not store unrelated personal facts, generic programming tips, product "
            "preferences, or broad conversation summaries. Write memories after the evidence "
            "is clear, not before. Failed optimization/debug attempts can be useful if the "
            "failure reason and conditions are precise.",
        ]
    )


def _sparse_reminder() -> str:
    return (
        "Dream mode is still active. If this turn produced a reusable inference optimization "
        "or environment deployment/debug lesson, persist it with the appropriate Dream memory "
        "tool. Prefer evidence-backed memories: profiler/bottleneck, baseline, changed variable, "
        "before/after benchmark, correctness check, and honest success or failure."
    )
