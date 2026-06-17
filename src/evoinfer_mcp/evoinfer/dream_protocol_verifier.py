from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class DreamProtocolVerificationError(RuntimeError):
    pass


START_RETRIEVAL_TRIGGERS = {
    "campaign_start",
    "task_start",
    "prompt_start",
    "initial",
    "initial_dream_retrieval",
}


def verify_dream_protocol_run(
    run: dict[str, Any],
    *,
    artifact_root: Path | str | None = None,
    require_stuck_retrieval: bool = False,
    require_completion_candidates: bool = True,
    require_artifact_valid_success: bool = True,
    require_transfer_safety: bool = False,
    require_artifact_memory_write: bool = False,
) -> dict[str, Any]:
    """Verify active Dream protocol evidence for one dream-enabled campaign run."""

    artifact_root_path = Path(artifact_root).expanduser().resolve() if artifact_root else None
    arm_name = str(run.get("arm_name") or run.get("session_id") or "<unknown>")
    events = _dream_retrieval_events(run)
    errors: list[str] = []
    if not any(_is_start_retrieval_event(event) for event in events):
        errors.append(f"{arm_name}: missing task-start retrieval")
    if require_stuck_retrieval and not any(_is_stuck_retrieval_event(event) for event in events):
        errors.append(f"{arm_name}: missing stuck retrieval")

    artifact_valid_success_count = 1 if run.get("verification_status") == "passed" else 0
    if require_artifact_valid_success and artifact_valid_success_count == 0:
        errors.append(f"{arm_name}: missing artifact-valid success")

    completion_candidate_count = 0
    candidate_artifact_ref_count = 0
    transfer_safety_checked_count = 0
    artifact_memory_write_count = 0
    artifact_memory_write_blocker_count = 0
    resolved_work_dir: str | None = None
    work_dir: Path | None = None
    if require_completion_candidates or require_transfer_safety:
        work_dir = _run_work_dir(run, artifact_root=artifact_root_path)
        if work_dir is None:
            errors.append(f"{arm_name}: missing work_dir")
        else:
            resolved_work_dir = str(work_dir)

    if require_completion_candidates and work_dir is not None:
        try:
            candidate_result = _verify_completion_candidates(work_dir)
        except DreamProtocolVerificationError as exc:
            errors.append(f"{arm_name}: {exc}")
        else:
            completion_candidate_count = candidate_result["candidate_count"]
            candidate_artifact_ref_count = candidate_result["artifact_ref_count"]

    if require_transfer_safety and work_dir is not None:
        try:
            transfer_safety_checked_count = _verify_route_decision_transfer_safety(
                work_dir,
                run=run,
                events=events,
            )
        except DreamProtocolVerificationError as exc:
            errors.append(f"{arm_name}: {exc}")

    artifact_memory_write_count, artifact_memory_write_blocker_count = (
        _artifact_memory_write_evidence_counts(run)
    )
    if (
        require_artifact_memory_write
        and artifact_memory_write_count == 0
        and artifact_memory_write_blocker_count == 0
    ):
        errors.append(f"{arm_name}: missing artifact memory write evidence")

    if errors:
        raise DreamProtocolVerificationError("; ".join(errors))

    return {
        "passed": True,
        "arm_name": arm_name,
        "retrieval_event_count": len(events),
        "completion_candidate_count": completion_candidate_count,
        "candidate_artifact_ref_count": candidate_artifact_ref_count,
        "transfer_safety_checked_count": transfer_safety_checked_count,
        "artifact_memory_write_count": artifact_memory_write_count,
        "artifact_memory_write_blocker_count": artifact_memory_write_blocker_count,
        "artifact_valid_success_count": artifact_valid_success_count,
        "resolved_work_dir": resolved_work_dir,
        "require_stuck_retrieval": require_stuck_retrieval,
        "require_completion_candidates": require_completion_candidates,
        "require_artifact_valid_success": require_artifact_valid_success,
        "require_transfer_safety": require_transfer_safety,
        "require_artifact_memory_write": require_artifact_memory_write,
    }


def verify_dream_protocol_campaign_result(
    campaign_result_path: Path | str,
    *,
    artifact_root: Path | str | None = None,
    require_stuck_retrieval: bool = False,
    require_completion_candidates: bool = True,
    require_artifact_valid_success: bool = True,
    require_transfer_safety: bool = False,
    require_artifact_memory_write: bool = False,
) -> dict[str, Any]:
    """Verify that a campaign result contains active Dream protocol evidence."""

    campaign_path = Path(campaign_result_path)
    artifact_root_path = Path(artifact_root).expanduser().resolve() if artifact_root else None
    payload = _load_json_object(campaign_path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise DreamProtocolVerificationError("campaign result must contain runs")

    errors: list[str] = []
    dream_enabled_runs = [run for run in runs if isinstance(run, dict) and run.get("dream_enabled")]
    if not dream_enabled_runs:
        raise DreamProtocolVerificationError("campaign result has no dream-enabled runs")

    retrieval_event_count = 0
    completion_candidate_count = 0
    candidate_artifact_ref_count = 0
    transfer_safety_checked_count = 0
    artifact_memory_write_count = 0
    artifact_memory_write_blocker_count = 0
    artifact_valid_success_count = 0
    resolved_work_dirs: list[str] = []

    for run in dream_enabled_runs:
        try:
            result = verify_dream_protocol_run(
                run,
                artifact_root=artifact_root_path,
                require_stuck_retrieval=require_stuck_retrieval,
                require_completion_candidates=require_completion_candidates,
                require_artifact_valid_success=False,
                require_transfer_safety=require_transfer_safety,
                require_artifact_memory_write=require_artifact_memory_write,
            )
        except DreamProtocolVerificationError as exc:
            errors.append(str(exc))
            continue
        retrieval_event_count += int(result["retrieval_event_count"])
        completion_candidate_count += int(result["completion_candidate_count"])
        candidate_artifact_ref_count += int(result["candidate_artifact_ref_count"])
        transfer_safety_checked_count += int(result["transfer_safety_checked_count"])
        artifact_memory_write_count += int(result["artifact_memory_write_count"])
        artifact_memory_write_blocker_count += int(
            result["artifact_memory_write_blocker_count"]
        )
        artifact_valid_success_count += int(result["artifact_valid_success_count"])
        resolved_work_dir = result.get("resolved_work_dir")
        if isinstance(resolved_work_dir, str):
            resolved_work_dirs.append(resolved_work_dir)

    if require_artifact_valid_success and artifact_valid_success_count == 0:
        errors.append("missing artifact-valid success: no dream-enabled run passed verification")

    if errors:
        raise DreamProtocolVerificationError("; ".join(errors))

    return {
        "passed": True,
        "campaign": payload.get("name"),
        "dream_enabled_run_count": len(dream_enabled_runs),
        "retrieval_event_count": retrieval_event_count,
        "completion_candidate_count": completion_candidate_count,
        "candidate_artifact_ref_count": candidate_artifact_ref_count,
        "transfer_safety_checked_count": transfer_safety_checked_count,
        "artifact_memory_write_count": artifact_memory_write_count,
        "artifact_memory_write_blocker_count": artifact_memory_write_blocker_count,
        "artifact_valid_success_count": artifact_valid_success_count,
        "resolved_work_dirs": resolved_work_dirs,
        "require_stuck_retrieval": require_stuck_retrieval,
        "require_completion_candidates": require_completion_candidates,
        "require_artifact_valid_success": require_artifact_valid_success,
        "require_transfer_safety": require_transfer_safety,
        "require_artifact_memory_write": require_artifact_memory_write,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DreamProtocolVerificationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DreamProtocolVerificationError(f"{path} must contain a JSON object")
    return payload


def _dream_retrieval_events(run: dict[str, Any]) -> list[dict[str, Any]]:
    events = run.get("dream_retrieval_events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _is_start_retrieval_event(event: dict[str, Any]) -> bool:
    trigger = str(event.get("trigger") or "").strip().lower()
    return trigger in START_RETRIEVAL_TRIGGERS


def _is_stuck_retrieval_event(event: dict[str, Any]) -> bool:
    trigger = str(event.get("trigger") or "").strip().lower()
    if trigger in START_RETRIEVAL_TRIGGERS:
        return False
    if trigger in {"stuck", "branch_point", "route_change", "periodic"}:
        return True
    step_count = event.get("step_count")
    return isinstance(step_count, int) and step_count > 0


def _verify_route_decision_transfer_safety(
    work_dir: Path,
    *,
    run: dict[str, Any],
    events: list[dict[str, Any]],
) -> int:
    route_decision_path = work_dir / "route_decision.json"
    if not route_decision_path.is_file():
        return 0

    route_decision = _load_json_object(route_decision_path)
    avoid_dtypes = _string_list(route_decision.get("avoid_dtypes"))
    selected_memory_ids = _string_list(route_decision.get("selected_memory_ids"))
    retrieved_memory_ids = _retrieved_memory_ids(run, events)

    for memory_id in selected_memory_ids:
        if memory_id not in retrieved_memory_ids:
            raise DreamProtocolVerificationError(
                f"route_decision selected memory id was not retrieved: {memory_id}"
            )

    if not avoid_dtypes:
        return 1

    skip_evidence = route_decision.get("skip_evidence")
    if not isinstance(skip_evidence, dict):
        raise DreamProtocolVerificationError(
            "route_decision missing skip evidence for avoided dtypes"
        )

    for dtype in avoid_dtypes:
        evidence_ids = _string_list(skip_evidence.get(dtype))
        if not evidence_ids:
            raise DreamProtocolVerificationError(
                f"route_decision missing skip evidence for avoided dtype {dtype}"
            )
        for memory_id in evidence_ids:
            if memory_id not in retrieved_memory_ids:
                raise DreamProtocolVerificationError(
                    "route_decision skip evidence for avoided dtype "
                    f"{dtype} was not retrieved: {memory_id}"
                )

    return 1


def _retrieved_memory_ids(
    run: dict[str, Any],
    events: list[dict[str, Any]],
) -> set[str]:
    memory_ids = set(_string_list(run.get("dream_retrieved_memory_ids")))
    for event in events:
        memory_ids.update(_string_list(event.get("memory_ids")))
    return memory_ids


def _artifact_memory_write_evidence_counts(run: dict[str, Any]) -> tuple[int, int]:
    written_ids = _string_list(run.get("dream_written_memory_ids"))
    write_count = _as_non_negative_int(run.get("dream_auto_write_count"))
    artifact_memory_write_count = max(write_count, len(written_ids))

    blockers = _string_list(run.get("dream_auto_write_blockers"))
    rejected_count = _as_non_negative_int(run.get("dream_auto_write_rejected_count"))
    artifact_memory_write_blocker_count = max(rejected_count, len(blockers))
    return artifact_memory_write_count, artifact_memory_write_blocker_count


def _as_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    return 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _run_work_dir(
    run: dict[str, Any],
    *,
    artifact_root: Path | None,
) -> Path | None:
    value = run.get("work_dir")
    if not isinstance(value, str) or not value.strip():
        return None
    work_dir = Path(value)
    if work_dir.exists():
        return work_dir.resolve()
    if artifact_root is None:
        return work_dir
    return _resolve_relocated_work_dir(
        original_work_dir=work_dir,
        artifact_root=artifact_root,
        arm_name=str(run.get("arm_name") or ""),
    )


def _resolve_relocated_work_dir(
    *,
    original_work_dir: Path,
    artifact_root: Path,
    arm_name: str,
) -> Path:
    candidates: list[Path] = []
    parts = original_work_dir.parts
    for index, part in enumerate(parts):
        if part.startswith("rep") and part[3:].isdigit():
            candidates.append(artifact_root.joinpath(*parts[index:]))
            break
    if arm_name:
        candidates.append(artifact_root / arm_name)
    candidates.append(artifact_root / original_work_dir.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else artifact_root.resolve()


def _verify_completion_candidates(work_dir: Path) -> dict[str, int]:
    candidate_path = work_dir / "dream_write_candidates.json"
    if not candidate_path.is_file():
        return _verify_standard_artifact_completion_candidates(work_dir)
    try:
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DreamProtocolVerificationError(
            f"dream_write_candidates.json is not valid JSON: {exc}"
        ) from exc
    if not isinstance(candidates, list) or not candidates:
        raise DreamProtocolVerificationError(
            "dream_write_candidates.json must contain at least one candidate"
        )

    artifact_ref_count = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise DreamProtocolVerificationError(
                f"dream candidate {index} must be an object"
            )
        refs = candidate.get("artifact_refs")
        if not isinstance(refs, list) or not refs:
            raise DreamProtocolVerificationError(
                f"dream candidate {index} missing artifact_refs"
            )
        artifact_ref_count += _verify_candidate_artifact_refs(work_dir, refs, index=index)

    return {"candidate_count": len(candidates), "artifact_ref_count": artifact_ref_count}


def _verify_standard_artifact_completion_candidates(work_dir: Path) -> dict[str, int]:
    optimization_refs = [
        name
        for name in (
            "environment.json",
            "benchmark_raw.json",
            "correctness_raw.json",
            "verifier_result.json",
            "agent_trace.md",
        )
        if (work_dir / name).is_file()
    ]
    if {"benchmark_raw.json", "correctness_raw.json"}.issubset(optimization_refs):
        return {
            "candidate_count": 1,
            "artifact_ref_count": len(optimization_refs),
        }

    debug_refs = [
        name
        for name in (
            "environment.json",
            "environment_debug.json",
            "diagnostic.log",
            "verification.log",
            "verifier_result.json",
            "agent_trace.md",
            "api_inventory.json",
            "library_notes.md",
        )
        if (work_dir / name).is_file()
    ]
    if "environment_debug.json" in debug_refs and any(
        ref in debug_refs for ref in ("diagnostic.log", "verification.log", "agent_trace.md")
    ):
        return {
            "candidate_count": 1,
            "artifact_ref_count": len(debug_refs),
        }

    raise DreamProtocolVerificationError(
        "missing completion candidates: no dream_write_candidates.json or standard extraction artifacts"
    )


def _verify_candidate_artifact_refs(
    work_dir: Path,
    refs: list[Any],
    *,
    index: int,
) -> int:
    root = work_dir.resolve()
    valid_count = 0
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            raise DreamProtocolVerificationError(
                f"dream candidate {index} has malformed artifact ref"
            )
        ref_path = Path(ref)
        if ref_path.is_absolute() or ".." in ref_path.parts:
            raise DreamProtocolVerificationError(
                f"dream candidate {index} has artifact ref outside workdir: {ref}"
            )
        resolved = (root / ref_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise DreamProtocolVerificationError(
                f"dream candidate {index} has artifact ref outside workdir: {ref}"
            ) from exc
        if not resolved.is_file():
            raise DreamProtocolVerificationError(
                f"dream candidate {index} missing artifact ref: {ref}"
            )
        valid_count += 1
    return valid_count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify active EvoInfer Dream protocol evidence in a campaign result."
    )
    parser.add_argument("campaign_result", help="Campaign result JSON path")
    parser.add_argument(
        "--require-stuck-retrieval",
        action="store_true",
        help="Require a non-start Dream retrieval event, such as stuck/periodic/branch-point.",
    )
    parser.add_argument(
        "--artifact-root",
        help=(
            "Local root for relocated campaign artifacts. Useful when campaign JSON "
            "contains remote work_dir paths copied into docs/evoinfer/campaign-artifacts."
        ),
    )
    parser.add_argument(
        "--no-completion-candidates",
        action="store_true",
        help="Do not require workdir/dream_write_candidates.json.",
    )
    parser.add_argument(
        "--no-artifact-valid-success",
        action="store_true",
        help="Do not require a dream-enabled run with verification_status='passed'.",
    )
    parser.add_argument(
        "--require-transfer-safety",
        action="store_true",
        help=(
            "Require route_decision.json avoided routes to cite skip_evidence memory IDs "
            "that were retrieved in the run."
        ),
    )
    parser.add_argument(
        "--require-artifact-memory-write",
        action="store_true",
        help=(
            "Require dream-enabled runs to show artifact-driven memory auto-write "
            "evidence or explicit write blockers."
        ),
    )
    args = parser.parse_args(argv)
    result = verify_dream_protocol_campaign_result(
        args.campaign_result,
        artifact_root=args.artifact_root,
        require_stuck_retrieval=args.require_stuck_retrieval,
        require_completion_candidates=not args.no_completion_candidates,
        require_artifact_valid_success=not args.no_artifact_valid_success,
        require_transfer_safety=args.require_transfer_safety,
        require_artifact_memory_write=args.require_artifact_memory_write,
    )
    print("DREAM_PROTOCOL_VERIFIER_PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
