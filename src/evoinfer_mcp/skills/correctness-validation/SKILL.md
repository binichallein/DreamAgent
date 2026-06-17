---
name: correctness-validation
description: Validate inference output correctness before promoting any optimization.
---

# Correctness Validation

Use this skill before promoting an optimization or writing a successful optimization memory.

Validation checklist:
- Identify the trusted baseline output or reference implementation.
- Define acceptable tolerance for exact match, max absolute error, relative error, top-k agreement, task metric, or deterministic output.
- Compare representative inputs, edge shapes, and the optimized workload.
- Check precision changes and quantization effects explicitly.
- Save evidence: validation command, output diff, tolerance, failure cases, and logs.
- If correctness cannot be validated, do not promote the optimization as successful.

Output expected from this skill:
- Validation command.
- Tolerance definition.
- Pass/fail result.
- Correctness artifact paths.
- Remaining risk.
