from __future__ import annotations

import json
from pathlib import Path

from evoinfer_mcp.hooks.dream_checkpoint import run_dream_checkpoint_hook


def test_dream_checkpoint_hook_injects_context_every_n_tool_events(tmp_path: Path) -> None:
    share_dir = tmp_path / "share"
    memory_file = share_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "version": 1,
                "memories": [
                    {
                        "id": "opt_hook_cuda_softmax",
                        "category": "optimization",
                        "title": "CUDA softmax shared reduction",
                        "summary": "Use block shared-memory reduction for row-wise softmax.",
                        "environment": "RTX 3090",
                        "model_type": "operator-kernel",
                        "model_arch": "row-wise softmax",
                        "inference_backend": "cuda",
                        "success": True,
                        "detail_description": "Benchmark and correctness artifacts supported it.",
                        "artifacts": ["benchmark_raw.json"],
                        "correctness_artifacts": ["correctness_raw.json"],
                        "tags": ["cuda", "softmax"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_file = tmp_path / "hook_state.json"
    context_file = tmp_path / "dream_context.md"
    event = {
        "hook_event_name": "PostToolUse",
        "cwd": str(tmp_path),
        "tool_name": "Bash",
        "tool_input": {"command": "python bench_softmax.py --backend cuda"},
        "tool_response": {"output": "latency_ms=0.19 correctness passed"},
    }

    first = run_dream_checkpoint_hook(
        event,
        client="codex",
        every_steps=2,
        session_dir=tmp_path,
        share_dir=share_dir,
        state_file=state_file,
        context_file=context_file,
    )

    assert first is None
    assert json.loads(state_file.read_text(encoding="utf-8"))["tool_checkpoint_count"] == 1

    second = run_dream_checkpoint_hook(
        event,
        client="codex",
        every_steps=2,
        session_dir=tmp_path,
        share_dir=share_dir,
        state_file=state_file,
        context_file=context_file,
    )

    assert second is not None
    assert second["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "opt_hook_cuda_softmax" in second["hookSpecificOutput"]["additionalContext"]
    assert "opt_hook_cuda_softmax" in context_file.read_text(encoding="utf-8")


def test_dream_checkpoint_hook_uses_claude_post_tool_batch_event_name(tmp_path: Path) -> None:
    share_dir = tmp_path / "share"
    (share_dir / "dream").mkdir(parents=True)
    (share_dir / "dream" / "memories.json").write_text(
        json.dumps({"version": 1, "memories": []}),
        encoding="utf-8",
    )

    output = run_dream_checkpoint_hook(
        {"hook_event_name": "PostToolBatch", "cwd": str(tmp_path), "tool_calls": []},
        client="claude",
        every_steps=1,
        session_dir=tmp_path,
        share_dir=share_dir,
        state_file=tmp_path / "state.json",
        context_file=tmp_path / "dream_context.md",
    )

    assert output is not None
    assert output["hookSpecificOutput"]["hookEventName"] == "PostToolBatch"
    assert "Dream checkpoint" in output["hookSpecificOutput"]["additionalContext"]
