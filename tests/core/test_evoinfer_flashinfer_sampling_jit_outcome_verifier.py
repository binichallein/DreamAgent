from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoinfer_mcp.evoinfer.flashinfer_sampling_jit_outcome_verifier import (
    FLAGHEADS_SIGNATURE,
    FlashInferSamplingJitOutcomeError,
    verify_flashinfer_sampling_jit_outcome,
)


def test_flashinfer_sampling_jit_outcome_accepts_blocked_audit(tmp_path: Path) -> None:
    _write_audit(tmp_path, blocked=True)

    result = verify_flashinfer_sampling_jit_outcome(tmp_path, expected="blocked")

    assert result["sampling_jit_blocked"] is True
    assert result["failed_probability_probe_count"] == 4


def test_flashinfer_sampling_jit_outcome_accepts_repaired_audit_with_patch(
    tmp_path: Path,
) -> None:
    _write_audit(tmp_path, blocked=False)
    (tmp_path / "repair_patch.diff").write_text("--- a\n+++ b\n", encoding="utf-8")
    (tmp_path / "repair_patch_audit.json").write_text("{}", encoding="utf-8")

    result = verify_flashinfer_sampling_jit_outcome(
        tmp_path,
        expected="repaired",
        require_repair_patch=True,
    )

    assert result["sampling_jit_blocked"] is False
    assert result["passed_probability_probe_count"] == 4


def test_flashinfer_sampling_jit_outcome_rejects_repaired_audit_with_failed_probe(
    tmp_path: Path,
) -> None:
    _write_audit(tmp_path, blocked=True)

    with pytest.raises(FlashInferSamplingJitOutcomeError, match="expected repaired"):
        verify_flashinfer_sampling_jit_outcome(tmp_path, expected="repaired")


def test_flashinfer_sampling_jit_outcome_rejects_missing_required_probe(
    tmp_path: Path,
) -> None:
    _write_audit(tmp_path, blocked=False, omit="flashinfer.top_p_sampling_from_probs")

    with pytest.raises(FlashInferSamplingJitOutcomeError, match="missing sampling"):
        verify_flashinfer_sampling_jit_outcome(tmp_path, expected="repaired")


def _write_audit(tmp_path: Path, *, blocked: bool, omit: str | None = None) -> None:
    names = [
        "flashinfer.softmax",
        "flashinfer.sampling_from_probs",
        "flashinfer.top_k_sampling_from_probs",
        "flashinfer.top_p_sampling_from_probs",
        "flashinfer.top_k_control",
    ]
    probes = []
    for name in names:
        if name == omit:
            continue
        should_fail = blocked and name != "flashinfer.top_k_control"
        probes.append(
            {
                "name": name,
                "status": "failed" if should_fail else "passed",
                "return_code": 1 if should_fail else 0,
                "issue_signature": FLAGHEADS_SIGNATURE if should_fail else None,
            }
        )
    payload = {
        "status": "failed" if blocked else "passed",
        "sampling_jit_blocked": blocked,
        "issue_signatures": [FLAGHEADS_SIGNATURE] if blocked else [],
        "probes": probes,
    }
    (tmp_path / "sampling_jit_debug_audit.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
