---
name: profiler-triage
description: Triage inference bottlenecks using profiler evidence before choosing an optimization.
---

# Profiler Triage

Use this skill when performance is unclear or an optimization target has not been proven.

Triage process:
- Start from a benchmark baseline and collect profiler evidence before changing major code paths.
- Use available tools such as nsys, ncu, torch profiler, Triton profiler, runtime traces, or server-side latency/throughput metrics.
- Classify the bottleneck: CPU overhead, GPU kernel time, memory bandwidth, launch overhead, synchronization, host-device transfer, attention, KV cache, batching, scheduling, quantization, compile overhead, or environment issues.
- Separate prefill, decode, end-to-end, data loading, and initialization when the workload supports it.
- Record hot kernels, blocking synchronization points, memory pressure, and overlap opportunities.

Output expected from this skill:
- Bottleneck classification.
- Profiler command and artifact paths.
- Evidence table of hotspots.
- Recommended next skill or optimization level.
- Unknowns that need more measurement.
