from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class OperatorCampaignVerificationError(RuntimeError):
    pass


ALLOWED_MEMORY_CATEGORIES = {
    "promoted_optimization",
    "candidate_optimization",
    "negative_optimization",
    "environment_debug",
    "debug",
}
DEFAULT_MAX_ABS_ERROR = 5e-3
DEFAULT_MAX_REL_ERROR = 5e-3
DEFAULT_MAX_ROW_SUM_ERROR = 5e-3
DEFAULT_MAX_MASKED_OUTPUT_ABS = 5e-3


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise OperatorCampaignVerificationError(
            f"{path.name} is not valid JSON: {exc}"
        ) from exc


def _entries(payload: Any, filename: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise OperatorCampaignVerificationError(f"{filename} must be an object")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise OperatorCampaignVerificationError(f"{filename} must contain entries")
    if not all(isinstance(entry, dict) for entry in entries):
        raise OperatorCampaignVerificationError(f"{filename} entries must be objects")
    return entries


def _has_any(mapping: dict[str, Any], *keys: str) -> bool:
    return any(bool(mapping.get(key)) for key in keys)


def _optional_string(mapping: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping.get(key)
        return isinstance(value, str) and bool(value.strip())
    return True


def verify_operator_campaign_dir(root: Path | str) -> dict[str, Any]:
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
        raise OperatorCampaignVerificationError(f"missing required artifacts: {missing}")
    if not any((root / name).is_file() for name in ("operator_smoke.py", "benchmark.py")):
        raise OperatorCampaignVerificationError(
            "missing executable benchmark source: operator_smoke.py or benchmark.py"
        )

    env = _load_json(root / "environment.json")
    if not isinstance(env, dict):
        raise OperatorCampaignVerificationError("environment.json must be an object")
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

    correctness_entries = _entries(
        _load_json(root / "correctness_raw.json"), "correctness_raw.json"
    )
    benchmark_entries = _entries(
        _load_json(root / "benchmark_raw.json"), "benchmark_raw.json"
    )
    correctness_by_id = {
        str(entry.get("id")): entry for entry in correctness_entries if entry.get("id")
    }
    for entry in correctness_entries:
        if entry.get("passed") is not True:
            errors.append(
                f"correctness entry {entry.get('id', '<missing-id>')} did not pass"
            )
        abs_error = entry.get("max_abs_error")
        if not isinstance(abs_error, int | float):
            errors.append(
                f"correctness entry {entry.get('id', '<missing-id>')} missing max_abs_error"
            )
        elif abs_error > DEFAULT_MAX_ABS_ERROR:
            errors.append(
                f"correctness entry {entry.get('id', '<missing-id>')} max_abs_error={abs_error!r} exceeds {DEFAULT_MAX_ABS_ERROR}"
            )
        rel_error = entry.get("max_rel_error")
        if rel_error is not None:
            if not isinstance(rel_error, int | float):
                errors.append(
                    f"correctness entry {entry.get('id', '<missing-id>')} invalid max_rel_error={rel_error!r}"
                )
            elif rel_error > DEFAULT_MAX_REL_ERROR:
                errors.append(
                    f"correctness entry {entry.get('id', '<missing-id>')} max_rel_error={rel_error!r} exceeds {DEFAULT_MAX_REL_ERROR}"
                )
        row_sum_error = entry.get("max_row_sum_error")
        if row_sum_error is not None:
            if not isinstance(row_sum_error, int | float):
                errors.append(
                    f"correctness entry {entry.get('id', '<missing-id>')} invalid max_row_sum_error={row_sum_error!r}"
                )
            elif row_sum_error > DEFAULT_MAX_ROW_SUM_ERROR:
                errors.append(
                    f"correctness entry {entry.get('id', '<missing-id>')} max_row_sum_error={row_sum_error!r} exceeds {DEFAULT_MAX_ROW_SUM_ERROR}"
                )
        operator_name = str(entry.get("operator") or "")
        if "masked" in operator_name:
            masked_error = entry.get("max_masked_output_abs")
            if not isinstance(masked_error, int | float):
                errors.append(
                    f"correctness entry {entry.get('id', '<missing-id>')} missing max_masked_output_abs"
                )
            elif masked_error > DEFAULT_MAX_MASKED_OUTPUT_ABS:
                errors.append(
                    f"correctness entry {entry.get('id', '<missing-id>')} max_masked_output_abs={masked_error!r} exceeds {DEFAULT_MAX_MASKED_OUTPUT_ABS}"
                )

    for entry in benchmark_entries:
        entry_id = str(entry.get("id") or "")
        if not entry_id:
            errors.append("benchmark entry missing id")
            continue
        if entry_id not in correctness_by_id:
            errors.append(f"benchmark entry {entry_id} has no matching correctness")
        if not entry.get("operator"):
            errors.append(f"benchmark entry {entry_id} missing operator")
        if not entry.get("dtype"):
            errors.append(f"benchmark entry {entry_id} missing dtype")
        if not _has_any(entry, "shape", "hidden", "seq_len"):
            errors.append(f"benchmark entry {entry_id} missing shape/hidden/seq_len")
        if entry.get("warmup_count", 0) < 10:
            errors.append(f"benchmark entry {entry_id} warmup_count too low")
        if entry.get("repeat_count", 0) < 20:
            errors.append(f"benchmark entry {entry_id} repeat_count too low")
        for metric in ("baseline_ms_mean", "candidate_ms_mean"):
            value = entry.get(metric)
            if not isinstance(value, int | float) or value <= 0:
                errors.append(f"benchmark entry {entry_id} invalid {metric}={value!r}")

    dream = _load_json(root / "dream_write_candidates.json")
    if not isinstance(dream, list) or not dream:
        errors.append("dream_write_candidates.json must contain at least one candidate")
    elif not all(isinstance(candidate, dict) for candidate in dream):
        errors.append("dream_write_candidates.json candidates must be objects")
    else:
        for index, candidate in enumerate(dream):
            category = candidate.get("category")
            if category not in ALLOWED_MEMORY_CATEGORIES:
                errors.append(f"dream candidate {index} invalid category={category!r}")
            refs = candidate.get("artifact_refs")
            if not isinstance(refs, list) or not refs:
                errors.append(f"dream candidate {index} missing artifact_refs")

    if errors:
        raise OperatorCampaignVerificationError("; ".join(errors))

    return {
        "environment": env,
        "correctness_entry_count": len(correctness_entries),
        "benchmark_entry_count": len(benchmark_entries),
        "dream_candidate_count": len(dream),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify EvoInfer controlled operator campaign artifacts"
    )
    parser.add_argument("root", nargs="?", default=".", help="artifact directory")
    args = parser.parse_args(argv)
    result = verify_operator_campaign_dir(args.root)
    print("OPERATOR_CAMPAIGN_VERIFIER_PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
