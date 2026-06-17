---
name: environment-debug
description: Debug inference environment failures with issue signatures, commands, root cause, and verification evidence.
---

# Environment Debug

Use this skill when inference work is blocked by environment or deployment problems.

Debug process:
- Capture an environment snapshot: hardware, OS, driver, runtime, Python, packages, paths, and relevant environment variables.
- Record the issue signature, exact error messages, symptoms, and reproduction command.
- Isolate root cause across driver, CUDA/runtime, package ABI, build toolchain, filesystem, auth, network, remote session, or service process boundaries.
- Apply the smallest fix that addresses the root cause.
- Verify with a command that exercises the failing path.
- Save evidence: diagnostic logs, commands, environment snapshot, and verification artifacts.

Output expected from this skill:
- Issue signature.
- Root cause.
- Fix command or workaround.
- Verification command and result.
- Reuse caveats and prevention notes.
