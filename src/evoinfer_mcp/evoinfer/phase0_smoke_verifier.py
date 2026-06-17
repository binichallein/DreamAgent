from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class Phase0SmokeVerificationError(RuntimeError):
    pass


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Phase0SmokeVerificationError(f"{path.name} is not valid JSON: {exc}") from exc


def _has_any(mapping: dict[str, Any], *keys: str) -> bool:
    return any(bool(mapping.get(key)) for key in keys)


def _optional_string(mapping: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping.get(key)
        return isinstance(value, str) and bool(value.strip())
    return True


def verify_phase0_smoke_dir(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    required = [
        "environment.json",
        "benchmark_raw.json",
        "correctness_raw.json",
        "agent_trace.md",
        "dream_write_candidates.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise Phase0SmokeVerificationError(f"missing required artifacts: {missing}")

    env = _load_json(root / "environment.json")
    bench = _load_json(root / "benchmark_raw.json")
    corr = _load_json(root / "correctness_raw.json")
    dream = _load_json(root / "dream_write_candidates.json")

    if not isinstance(env, dict):
        raise Phase0SmokeVerificationError("environment.json must be an object")
    if not isinstance(bench, dict):
        raise Phase0SmokeVerificationError("benchmark_raw.json must be an object")
    if not isinstance(corr, dict):
        raise Phase0SmokeVerificationError("correctness_raw.json must be an object")

    errors: list[str] = []
    if env.get("cuda_available") is not True:
        errors.append(f"cuda_available={env.get('cuda_available')!r}")
    if not _has_any(env, "gpu", "device", "device_name"):
        errors.append("missing gpu/device name")
    if not _optional_string(env, "driver", "driver_version"):
        errors.append(f"driver field is malformed: {env.get('driver', env.get('driver_version'))!r}")
    if not _has_any(env, "torch", "torch_version"):
        errors.append("missing torch version")
    if not _has_any(env, "triton", "triton_version"):
        errors.append("missing triton version")
    if not _has_any(env, "flashinfer", "flashinfer_version"):
        errors.append("missing flashinfer version")
    if bench.get("repeat_count", 0) < 50:
        errors.append(f"repeat_count={bench.get('repeat_count')!r}")
    if bench.get("warmup_count", 0) < 10:
        errors.append(f"warmup_count={bench.get('warmup_count')!r}")
    elapsed = bench.get("elapsed_ms_mean")
    if not isinstance(elapsed, int | float) or elapsed <= 0:
        errors.append(f"elapsed_ms_mean={elapsed!r}")
    if corr.get("passed") is not True:
        errors.append(f"correctness passed={corr.get('passed')!r}")
    if corr.get("max_abs_error", 1) > 1e-6:
        errors.append(f"max_abs_error={corr.get('max_abs_error')!r}")
    if dream != []:
        errors.append("dream_write_candidates.json must be empty for Phase 0 smoke")

    if errors:
        raise Phase0SmokeVerificationError("; ".join(errors))
    return {"env": env, "bench": bench, "corr": corr}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify EvoInfer Phase 0 smoke artifacts")
    parser.add_argument("root", nargs="?", default=".", help="artifact directory to verify")
    args = parser.parse_args(argv)

    result = verify_phase0_smoke_dir(Path(args.root))
    print("PHASE0_SMOKE_VERIFIER_PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
