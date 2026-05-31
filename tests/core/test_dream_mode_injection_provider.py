from __future__ import annotations

from kimi_cli.soul.dynamic_injections.dream_mode import _full_reminder, _sparse_reminder


def test_dream_full_reminder_guides_rigorous_inference_optimization() -> None:
    reminder = _full_reminder()

    assert "nsys" in reminder
    assert "ncu" in reminder
    assert "baseline" in reminder
    assert "Triton" in reminder
    assert "CUDA custom kernels" in reminder
    assert "CuTe" in reminder
    assert "specific operator" in reminder
    assert "Do not fake benchmarks" in reminder
    assert "Validate output correctness" in reminder
    assert "before/after metrics" in reminder


def test_dream_sparse_reminder_prefers_evidence_backed_memories() -> None:
    reminder = _sparse_reminder()

    assert "profiler/bottleneck" in reminder
    assert "baseline" in reminder
    assert "before/after benchmark" in reminder
    assert "correctness check" in reminder
    assert "honest success or failure" in reminder
