---
name: cuda-kernel-analysis
description: Analyze CUDA kernel performance using baseline measurements, profiler evidence, and correctness checks.
---

# CUDA Kernel Analysis

Use this skill when profiler evidence points to CUDA kernels or custom kernel work.

Analysis checklist:
- Confirm the baseline kernel, input shapes, precision, and expected output tolerance.
- Inspect occupancy, registers, shared memory, memory coalescing, memory bandwidth, cache behavior, warp divergence, synchronization, and launch overhead.
- Prefer one controlled kernel change at a time.
- Compare against the baseline with identical workload and measurement method.
- Validate correctness after every candidate change.
- Save evidence: profiler output, benchmark output, patch or kernel source, and correctness logs.

Output expected from this skill:
- Kernel bottleneck diagnosis.
- Candidate kernel change.
- Benchmark before/after table.
- Correctness evidence.
- Promotion or rejection decision with reason.
