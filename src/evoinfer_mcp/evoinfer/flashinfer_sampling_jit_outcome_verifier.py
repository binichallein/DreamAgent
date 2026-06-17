from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class FlashInferSamplingJitOutcomeError(RuntimeError):
    pass


REQUIRED_SAMPLING_PROBABILITY_PROBES = {
    "flashinfer.softmax",
    "flashinfer.sampling_from_probs",
    "flashinfer.top_k_sampling_from_probs",
    "flashinfer.top_p_sampling_from_probs",
}
CONTROL_PROBE = "flashinfer.top_k_control"
FLAGHEADS_SIGNATURE = "CUB BlockAdjacentDifference missing FlagHeads"


def verify_flashinfer_sampling_jit_outcome(
    root: Path | str,
    *,
    expected: str,
    require_repair_patch: bool = False,
) -> dict[str, Any]:
    """Verify the outcome state of a FlashInfer sampling JIT debug artifact.

    This verifier is intentionally outcome-specific. It does not replace the
    library campaign verifier; it checks whether the environment-debug campaign
    reproduced or repaired the known sm86 sampling JIT blocker.
    """

    if expected not in {"blocked", "repaired"}:
        raise ValueError("expected must be 'blocked' or 'repaired'")
    root = Path(root)
    audit_path = root / "sampling_jit_debug_audit.json"
    if not audit_path.is_file():
        raise FlashInferSamplingJitOutcomeError("missing sampling_jit_debug_audit.json")
    audit = _load_json(audit_path)
    if not isinstance(audit, dict):
        raise FlashInferSamplingJitOutcomeError("sampling_jit_debug_audit.json must be an object")
    probes = audit.get("probes")
    if not isinstance(probes, list) or not probes:
        raise FlashInferSamplingJitOutcomeError("sampling_jit_debug_audit.json must contain probes")
    if not all(isinstance(probe, dict) for probe in probes):
        raise FlashInferSamplingJitOutcomeError("sampling_jit_debug_audit probes must be objects")

    by_name = {str(probe.get("name") or ""): probe for probe in probes}
    missing = sorted(REQUIRED_SAMPLING_PROBABILITY_PROBES - set(by_name))
    if missing:
        raise FlashInferSamplingJitOutcomeError(
            "missing sampling probability probes: " + ", ".join(missing)
        )
    if CONTROL_PROBE not in by_name:
        raise FlashInferSamplingJitOutcomeError(f"missing control probe: {CONTROL_PROBE}")

    issue_signatures = audit.get("issue_signatures") or []
    if not isinstance(issue_signatures, list):
        raise FlashInferSamplingJitOutcomeError("issue_signatures must be a list")
    probability_probes = [by_name[name] for name in sorted(REQUIRED_SAMPLING_PROBABILITY_PROBES)]
    failed_probability = [
        str(probe.get("name"))
        for probe in probability_probes
        if probe.get("status") != "passed" or probe.get("return_code") != 0
    ]
    passed_probability = [
        str(probe.get("name"))
        for probe in probability_probes
        if probe.get("status") == "passed" and probe.get("return_code") == 0
    ]
    control = by_name[CONTROL_PROBE]

    if expected == "blocked":
        if audit.get("sampling_jit_blocked") is not True and audit.get("status") != "failed":
            raise FlashInferSamplingJitOutcomeError(
                "expected blocked sampling JIT, but audit is not blocked/failed"
            )
        if not failed_probability:
            raise FlashInferSamplingJitOutcomeError(
                "expected at least one failed sampling probability probe"
            )
        if FLAGHEADS_SIGNATURE not in issue_signatures and not any(
            probe.get("issue_signature") == FLAGHEADS_SIGNATURE for probe in probability_probes
        ):
            raise FlashInferSamplingJitOutcomeError(
                f"expected issue signature {FLAGHEADS_SIGNATURE!r}"
            )
    else:
        if audit.get("sampling_jit_blocked") is not False:
            raise FlashInferSamplingJitOutcomeError("expected repaired sampling JIT")
        if audit.get("status") != "passed":
            raise FlashInferSamplingJitOutcomeError("expected audit status='passed'")
        if issue_signatures:
            raise FlashInferSamplingJitOutcomeError(
                f"expected no issue signatures, got {issue_signatures!r}"
            )
        if failed_probability:
            raise FlashInferSamplingJitOutcomeError(
                "expected all sampling probability probes to pass; failed: "
                + ", ".join(failed_probability)
            )
        if control.get("status") != "passed" or control.get("return_code") != 0:
            raise FlashInferSamplingJitOutcomeError("expected top_k control probe to pass")
        if require_repair_patch and not (root / "repair_patch.diff").is_file():
            raise FlashInferSamplingJitOutcomeError("missing repair_patch.diff")
        if require_repair_patch and not (root / "repair_patch_audit.json").is_file():
            raise FlashInferSamplingJitOutcomeError("missing repair_patch_audit.json")

    return {
        "expected": expected,
        "sampling_jit_blocked": audit.get("sampling_jit_blocked"),
        "status": audit.get("status"),
        "failed_probability_probe_count": len(failed_probability),
        "passed_probability_probe_count": len(passed_probability),
        "control_probe_status": control.get("status"),
        "issue_signatures": issue_signatures,
        "require_repair_patch": require_repair_patch,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify FlashInfer sampling JIT debug/repair outcome artifacts."
    )
    parser.add_argument("root", nargs="?", default=".", help="artifact directory")
    parser.add_argument("--expected", choices=["blocked", "repaired"], required=True)
    parser.add_argument("--require-repair-patch", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify_flashinfer_sampling_jit_outcome(
            args.root,
            expected=args.expected,
            require_repair_patch=args.require_repair_patch,
        )
    except FlashInferSamplingJitOutcomeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
    print("FLASHINFER_SAMPLING_JIT_OUTCOME_VERIFIER_PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FlashInferSamplingJitOutcomeError(
            f"{path.name} is not valid JSON: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
