from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class LibraryCampaignVerificationError(RuntimeError):
    pass


ALLOWED_MEMORY_CATEGORIES = {
    "promoted_optimization",
    "candidate_optimization",
    "negative_optimization",
    "environment_debug",
    "library_pattern",
    "debug",
}
DEBUG_ONLY_CATEGORIES = {"environment_debug", "debug"}
DEFAULT_MAX_ABS_ERROR = 5e-2
DEFAULT_MEAN_ABS_ERROR = 5e-3


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise LibraryCampaignVerificationError(
            f"{path.name} is not valid JSON: {exc}"
        ) from exc


def _entries(payload: Any, filename: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise LibraryCampaignVerificationError(f"{filename} must be an object")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LibraryCampaignVerificationError(f"{filename} must contain entries")
    if not all(isinstance(entry, dict) for entry in entries):
        raise LibraryCampaignVerificationError(f"{filename} entries must be objects")
    return entries


def _has_any(mapping: dict[str, Any], *keys: str) -> bool:
    return any(bool(mapping.get(key)) for key in keys)


def _is_environment_debug(env: dict[str, Any], correctness: Any, benchmark: Any) -> bool:
    if env.get("classification") == "environment_debug":
        return True
    if isinstance(correctness, dict) and correctness.get("status") == "not_run":
        return True
    return isinstance(benchmark, dict) and benchmark.get("status") == "not_run"


def _validate_environment(env: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(env, dict):
        return ["environment.json must be an object"]
    if env.get("cuda_available") is not True:
        errors.append(f"cuda_available={env.get('cuda_available')!r}")
    if not _has_any(env, "gpu", "gpu_name", "device", "device_name"):
        errors.append("missing gpu/device name")
    driver = env.get("driver") if "driver" in env else env.get("driver_version")
    if driver is not None and not isinstance(driver, str):
        errors.append(f"driver field is malformed: {driver!r}")
    if not _has_any(env, "torch", "torch_version"):
        errors.append("missing torch version")
    library = str(env.get("library") or "").lower()
    if library == "flashinfer" or _has_any(
        env, "flashinfer_version", "flashinfer_python_version"
    ):
        if not _has_any(env, "flashinfer_version", "flashinfer_python_version"):
            errors.append("missing flashinfer version")
    elif not _has_any(env, "library_version", "package_version"):
        errors.append("missing library/package version")
    return errors


def _validate_api_inventory(api_inventory: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(api_inventory, dict):
        return ["api_inventory.json must be an object"]
    tested_api = api_inventory.get("tested_api")
    if not isinstance(tested_api, list) or not tested_api:
        errors.append("api_inventory.json must contain tested_api")
    elif not all(isinstance(entry, dict) and entry.get("name") for entry in tested_api):
        errors.append("tested_api entries must be objects with name")
    return errors


def _validate_dream_candidates(dream: Any, debug_mode: bool) -> tuple[int, list[str]]:
    errors: list[str] = []
    if not isinstance(dream, list) or not dream:
        return 0, ["dream_write_candidates.json must contain at least one candidate"]
    if not all(isinstance(candidate, dict) for candidate in dream):
        return 0, ["dream_write_candidates.json candidates must be objects"]
    for index, candidate in enumerate(dream):
        category = candidate.get("category")
        if category not in ALLOWED_MEMORY_CATEGORIES:
            errors.append(f"dream candidate {index} invalid category={category!r}")
        if debug_mode and category not in DEBUG_ONLY_CATEGORIES:
            errors.append(
                f"dream candidate {index} must be environment_debug/debug for environment_debug campaign"
            )
        refs = candidate.get("artifact_refs")
        if not isinstance(refs, list) or not refs:
            errors.append(f"dream candidate {index} missing artifact_refs")
    return len(dream), errors


def _validate_debug_payload(correctness: Any, benchmark: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(correctness, dict) or correctness.get("status") != "not_run":
        errors.append("environment_debug correctness_raw.json must have status='not_run'")
    if not isinstance(benchmark, dict) or benchmark.get("status") != "not_run":
        errors.append("environment_debug benchmark_raw.json must have status='not_run'")
    return errors


def _validate_benchmark_payload(
    correctness: Any, benchmark: Any
) -> tuple[int, int, int, list[str]]:
    errors: list[str] = []
    correctness_entries = _entries(correctness, "correctness_raw.json")
    benchmark_entries = _entries(benchmark, "benchmark_raw.json")
    correctness_by_id = {
        str(entry.get("id")): entry for entry in correctness_entries if entry.get("id")
    }
    failed_correctness_count = 0
    for entry in correctness_entries:
        entry_id = entry.get("id", "<missing-id>")
        passed = entry.get("passed") is True
        if not passed:
            failed_correctness_count += 1
        abs_error = entry.get("max_abs_error")
        if not isinstance(abs_error, int | float):
            errors.append(f"correctness entry {entry_id} missing max_abs_error")
        elif passed and abs_error > DEFAULT_MAX_ABS_ERROR:
            errors.append(
                f"correctness entry {entry_id} max_abs_error={abs_error!r} exceeds {DEFAULT_MAX_ABS_ERROR}"
            )
        mean_error = entry.get("mean_abs_error")
        if mean_error is not None:
            if not isinstance(mean_error, int | float):
                errors.append(
                    f"correctness entry {entry_id} invalid mean_abs_error={mean_error!r}"
                )
            elif passed and mean_error > DEFAULT_MEAN_ABS_ERROR:
                errors.append(
                    f"correctness entry {entry_id} mean_abs_error={mean_error!r} exceeds {DEFAULT_MEAN_ABS_ERROR}"
                )

    for entry in benchmark_entries:
        entry_id = str(entry.get("id") or "")
        if not entry_id:
            errors.append("benchmark entry missing id")
            continue
        if entry_id not in correctness_by_id:
            errors.append(f"benchmark entry {entry_id} has no matching correctness")
        elif correctness_by_id[entry_id].get("passed") is not True:
            errors.append(f"benchmark entry {entry_id} correctness did not pass")
        if not entry.get("operator"):
            errors.append(f"benchmark entry {entry_id} missing operator")
        if not entry.get("dtype"):
            errors.append(f"benchmark entry {entry_id} missing dtype")
        if not _has_any(entry, "shape", "hidden", "seq_len"):
            errors.append(f"benchmark entry {entry_id} missing shape/hidden/seq_len")
        if entry.get("warmup_count", 0) < 5:
            errors.append(f"benchmark entry {entry_id} warmup_count too low")
        if entry.get("repeat_count", 0) < 10:
            errors.append(f"benchmark entry {entry_id} repeat_count too low")
        for metric in ("baseline_ms_mean", "candidate_ms_mean"):
            value = entry.get(metric)
            if not isinstance(value, int | float) or value <= 0:
                errors.append(f"benchmark entry {entry_id} invalid {metric}={value!r}")
        first_call = entry.get("first_call_ms")
        if first_call is not None and (
            not isinstance(first_call, int | float) or first_call < 0
        ):
            errors.append(f"benchmark entry {entry_id} invalid first_call_ms={first_call!r}")

    return len(correctness_entries), failed_correctness_count, len(benchmark_entries), errors


def verify_library_campaign_dir(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    required = [
        "environment.json",
        "api_inventory.json",
        "correctness_raw.json",
        "benchmark_raw.json",
        "library_notes.md",
        "agent_trace.md",
        "dream_write_candidates.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise LibraryCampaignVerificationError(f"missing required artifacts: {missing}")

    env = _load_json(root / "environment.json")
    api_inventory = _load_json(root / "api_inventory.json")
    correctness = _load_json(root / "correctness_raw.json")
    benchmark = _load_json(root / "benchmark_raw.json")
    dream = _load_json(root / "dream_write_candidates.json")

    errors: list[str] = []
    errors.extend(_validate_environment(env))
    errors.extend(_validate_api_inventory(api_inventory))
    debug_mode = isinstance(env, dict) and _is_environment_debug(env, correctness, benchmark)

    if debug_mode:
        errors.extend(_validate_debug_payload(correctness, benchmark))
        correctness_entry_count = 0
        failed_correctness_entry_count = 0
        benchmark_entry_count = 0
    else:
        try:
            (
                correctness_entry_count,
                failed_correctness_entry_count,
                benchmark_entry_count,
                benchmark_errors,
            ) = _validate_benchmark_payload(correctness, benchmark)
        except LibraryCampaignVerificationError as exc:
            errors.append(str(exc))
            correctness_entry_count = 0
            failed_correctness_entry_count = 0
            benchmark_entry_count = 0
        else:
            errors.extend(benchmark_errors)

    dream_candidate_count, dream_errors = _validate_dream_candidates(dream, debug_mode)
    errors.extend(dream_errors)

    if errors:
        raise LibraryCampaignVerificationError("; ".join(errors))

    return {
        "mode": "environment_debug" if debug_mode else "benchmark",
        "environment": env,
        "api_count": len(api_inventory.get("tested_api", [])),
        "correctness_entry_count": correctness_entry_count,
        "failed_correctness_entry_count": failed_correctness_entry_count,
        "benchmark_entry_count": benchmark_entry_count,
        "dream_candidate_count": dream_candidate_count,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify EvoInfer library-learning campaign artifacts"
    )
    parser.add_argument("root", nargs="?", default=".", help="artifact directory")
    args = parser.parse_args(argv)
    result = verify_library_campaign_dir(args.root)
    print("LIBRARY_CAMPAIGN_VERIFIER_PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
