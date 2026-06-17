---
name: benchmark-design
description: Design fair, reproducible inference benchmarks with baseline, workload, metrics, and evidence.
---

# Benchmark Design

Use this skill when defining or reviewing an inference benchmark.

Benchmark contract:
- Record the baseline command before changing code or configuration.
- Keep hardware, model, input shape, batch size, sequence length, concurrency, precision, warmup, and measurement method comparable.
- Select objective metrics explicitly: latency, throughput, tokens/s, time-to-first-token, decode tokens/s, memory, or accuracy-related metrics.
- Run enough repeats to reduce noise, and report variance when possible.
- Save raw evidence: benchmark command, logs, CSV/JSON outputs, environment snapshot, and commit or patch identity.
- Do not claim improvement without before/after measurements from the same workload.

Output expected from this skill:
- Baseline command and optimized command.
- Workload definition.
- Metric table.
- Evidence artifact paths.
- Fairness risks and missing measurements.
