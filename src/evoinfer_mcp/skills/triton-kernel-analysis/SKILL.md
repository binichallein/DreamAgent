---
name: triton-kernel-analysis
description: Tune Triton kernels with evidence-backed benchmark and correctness validation.
---

# Triton Kernel Analysis

Use this skill when implementing or tuning Triton kernels.

Analysis checklist:
- Establish a baseline implementation and benchmark command.
- Define target shapes, precision, memory layout, and correctness tolerance.
- Tune block sizes, num warps, num stages, memory layout, fusion boundaries, and autotune search space.
- Avoid broad changes without profiler or benchmark evidence.
- Compare candidates under the same workload and warmup.
- Validate outputs against a trusted reference implementation.

Output expected from this skill:
- Triton candidate configuration.
- Benchmark before/after table.
- Correctness result and tolerance.
- Profiler or benchmark evidence artifacts.
- Promotion or rejection decision.
