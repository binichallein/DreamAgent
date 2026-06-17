from __future__ import annotations

import json
import shlex
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from evoinfer_mcp.cli import cli
from evoinfer_mcp.dream.memory import DreamMemorySearchInput, search_dream_memories


def test_evoinfer_mcp_config_outputs_codex_compatible_stdio_config(
    tmp_path: Path,
) -> None:
    share_dir = tmp_path / "share"

    result = CliRunner().invoke(
        cli,
        ["mcp-config", "--client", "codex", "--share-dir", str(share_dir)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    server = payload["mcpServers"]["evoinfer-dream"]
    assert server["command"] == "python"
    assert server["args"] == ["-m", "evoinfer_mcp.dream.mcp_server"]
    assert server["env"]["EVOINFER_SHARE_DIR"] == str(share_dir)


def test_evoinfer_mcp_config_outputs_claude_compatible_stdio_config(
    tmp_path: Path,
) -> None:
    share_dir = tmp_path / "share"

    result = CliRunner().invoke(
        cli,
        ["mcp-config", "--client", "claude", "--share-dir", str(share_dir)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mcpServers"]["evoinfer-dream"]["args"] == [
        "-m",
        "evoinfer_mcp.dream.mcp_server",
    ]
    assert payload["client"] == "claude"


def test_evoinfer_mcp_config_defaults_to_shared_seeded_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVOINFER_SHARE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    result = CliRunner().invoke(
        cli,
        ["mcp-config", "--client", "codex"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    share_dir = tmp_path / "home" / ".evoinfer" / "dream-share"
    server = payload["mcpServers"]["evoinfer-dream"]
    assert server["env"]["EVOINFER_SHARE_DIR"] == str(share_dir.resolve())
    memories = json.loads((share_dir / "dream" / "memories.json").read_text(encoding="utf-8"))
    assert len(memories["memories"]) == 7


def test_evoinfer_mcp_config_can_enable_cpu_embedding_backend(
    tmp_path: Path,
) -> None:
    share_dir = tmp_path / "share"

    result = CliRunner().invoke(
        cli,
        [
            "mcp-config",
            "--client",
            "codex",
            "--share-dir",
            str(share_dir),
            "--enable-embedding",
            "--embedding-model",
            "BAAI/bge-small-zh-v1.5",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    env = payload["mcpServers"]["evoinfer-dream"]["env"]
    assert env["EVOINFER_SHARE_DIR"] == str(share_dir)
    assert env["EVOINFER_EMBEDDING_BACKEND"] == "local"
    assert env["EVOINFER_EMBEDDING_MODEL"] == "BAAI/bge-small-zh-v1.5"
    assert env["EVOINFER_EMBEDDING_DEVICE"] == "cpu"


def test_evoinfer_force_session_creates_isolated_mandatory_dream_bundle(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "dream-session"
    share_dir = tmp_path / "share"
    workdir = tmp_path / "work"

    result = CliRunner().invoke(
        cli,
        [
            "force-session",
            "--json",
            "--session-dir",
            str(session_dir),
            "--share-dir",
            str(share_dir),
            "--workdir",
            str(workdir),
            "--command",
            "/usr/bin/python3",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "mandatory-session"
    assert payload["session_dir"] == str(session_dir.resolve())
    assert payload["share_dir"] == str(share_dir.resolve())
    assert payload["workdir"] == str(workdir.resolve())
    assert payload["mcp_config_path"] == str((session_dir / "mcp.json").resolve())
    assert payload["call_log_path"] == str((session_dir / "mcp_calls.jsonl").resolve())
    assert payload["instruction_paths"] == [
        str((workdir / "AGENTS.md").resolve()),
        str((workdir / "CLAUDE.md").resolve()),
    ]
    assert "claude" in payload["commands"]
    assert "codex" in payload["commands"]
    assert "kimi" in payload["commands"]

    config = json.loads((session_dir / "mcp.json").read_text(encoding="utf-8"))
    server = config["mcpServers"]["evoinfer-dream"]
    assert server == {
        "type": "stdio",
        "command": "/usr/bin/python3",
        "args": ["-m", "evoinfer_mcp.dream.mcp_server"],
        "env": {
            "EVOINFER_SHARE_DIR": str(share_dir.resolve()),
            "EVOINFER_MCP_CALL_LOG": str((session_dir / "mcp_calls.jsonl").resolve()),
            "EVOINFER_DREAM_SESSION_ID": session_dir.name,
            "EVOINFER_DREAM_MANDATORY": "1",
        },
    }
    protocol = (workdir / "AGENTS.md").read_text(encoding="utf-8")
    assert "MANDATORY EvoInfer Dream session protocol" in protocol
    assert "Before doing task-local exploration" in protocol
    assert "dream_get_agent_protocol" in protocol
    assert "dream_search_memories" in protocol
    assert "dream_extract_and_write_memories" in protocol
    assert (workdir / "CLAUDE.md").read_text(encoding="utf-8") == protocol


def test_evoinfer_root_cli_creates_codex_hooked_session_bundle(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "codex-session"
    share_dir = tmp_path / "share"
    workdir = tmp_path / "work"

    result = CliRunner().invoke(
        cli,
        [
            "--client",
            "codex",
            "--hook-every-steps",
            "10",
            "--session-dir",
            str(session_dir),
            "--share-dir",
            str(share_dir),
            "--workdir",
            str(workdir),
            "--command",
            "/usr/bin/python3",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "hooked-session"
    assert payload["client"] == "codex"
    assert payload["hook_every_steps"] == 10
    assert payload["hook_config_path"] == str((workdir / ".codex" / "hooks.json").resolve())
    assert payload["hook_state_path"] == str((session_dir / "hook_state.json").resolve())
    assert payload["dream_context_path"] == str((session_dir / "dream_context.md").resolve())
    assert payload["launch_command"][0] == "codex"
    assert "exec" not in payload["launch_command"]
    assert "--dangerously-bypass-hook-trust" in payload["launch_command"]
    assert "kimi" not in payload["commands"]

    hooks = json.loads((workdir / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert set(hooks["hooks"]) == {"SessionStart", "PostToolUse", "Stop"}
    post_tool = hooks["hooks"]["PostToolUse"][0]["hooks"][0]
    assert post_tool["type"] == "command"
    assert "evoinfer_mcp.hooks.dream_checkpoint" in post_tool["command"]
    assert "--every-steps 10" in post_tool["command"]


def test_evoinfer_root_cli_creates_claude_hooked_session_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVOINFER_SHARE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    session_dir = tmp_path / "claude-session"
    workdir = tmp_path / "work"

    result = CliRunner().invoke(
        cli,
        [
            "--client",
            "claude",
            "--hook-every-steps",
            "7",
            "--session-dir",
            str(session_dir),
            "--workdir",
            str(workdir),
            "--command",
            "/usr/bin/python3",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["client"] == "claude"
    assert payload["hook_every_steps"] == 7
    assert payload["share_dir"] == str((tmp_path / "home" / ".evoinfer" / "dream-share").resolve())
    assert payload["seed_memory_merge"]["imported_count"] == 7
    assert payload["hook_config_path"] == str((workdir / ".claude" / "settings.local.json").resolve())
    assert payload["launch_command"][0] == "claude"
    assert "--settings" in payload["launch_command"]

    settings = json.loads(
        (workdir / ".claude" / "settings.local.json").read_text(encoding="utf-8")
    )
    assert set(settings["hooks"]) == {"SessionStart", "PostToolBatch", "Stop"}
    post_batch = settings["hooks"]["PostToolBatch"][0]["hooks"][0]
    assert post_batch["type"] == "command"
    assert post_batch["command"] == "/usr/bin/python3"
    assert post_batch["args"][:2] == ["-m", "evoinfer_mcp.hooks.dream_checkpoint"]
    assert "--every-steps" in post_batch["args"]
    assert "7" in post_batch["args"]


def test_evoinfer_root_cli_prompts_for_client_and_step_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVOINFER_SHARE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    session_dir = tmp_path / "interactive-session"

    result = CliRunner().invoke(
        cli,
        [
            "--session-dir",
            str(session_dir),
            "--dry-run",
            "--json",
        ],
        input="codex\n6\n",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["client"] == "codex"
    assert payload["hook_every_steps"] == 6
    assert payload["share_dir"] == str((tmp_path / "home" / ".evoinfer" / "dream-share").resolve())
    assert Path(payload["hook_config_path"]).is_file()


def test_evoinfer_mcp_config_outputs_codex_toml_template(tmp_path: Path) -> None:
    share_dir = tmp_path / "share"

    result = CliRunner().invoke(
        cli,
        [
            "mcp-config",
            "--client",
            "codex",
            "--format",
            "codex-toml",
            "--share-dir",
            str(share_dir),
            "--command",
            "/usr/bin/python3",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = tomllib.loads(result.output)
    server = payload["mcp_servers"]["evoinfer-dream"]
    assert server["command"] == "/usr/bin/python3"
    assert server["args"] == ["-m", "evoinfer_mcp.dream.mcp_server"]
    assert server["env"]["EVOINFER_SHARE_DIR"] == str(share_dir)


def test_evoinfer_mcp_config_outputs_claude_add_json_command(tmp_path: Path) -> None:
    share_dir = tmp_path / "share"

    result = CliRunner().invoke(
        cli,
        [
            "mcp-config",
            "--client",
            "claude",
            "--format",
            "claude-add-json",
            "--scope",
            "user",
            "--share-dir",
            str(share_dir),
            "--command",
            "/usr/bin/python3",
        ],
    )

    assert result.exit_code == 0, result.output
    parts = shlex.split(result.output.strip())
    assert parts[:4] == ["claude", "mcp", "add-json", "evoinfer-dream"]
    config = json.loads(parts[4])
    assert config == {
        "type": "stdio",
        "command": "/usr/bin/python3",
        "args": ["-m", "evoinfer_mcp.dream.mcp_server"],
        "env": {"EVOINFER_SHARE_DIR": str(share_dir)},
    }
    assert parts[5:] == ["--scope", "user"]


def test_evoinfer_doctor_json_checks_open_box_mcp_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    share_dir = tmp_path / "share"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(share_dir))

    result = CliRunner().invoke(cli, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["mcp_server_import"]["ok"] is True
    assert checks["share_dir_writable"]["ok"] is True
    assert checks["tool_surface"]["ok"] is True
    assert checks["mcp_stdio_launch"]["ok"] is True
    assert "dream_get_agent_protocol" in payload["tool_names"]
    assert "dream_search_memories" in payload["tool_names"]
    assert "dream_stage_memory_candidate" in payload["tool_names"]
    assert "dream_extract_and_write_memories" in payload["tool_names"]
    assert "dream_promote_memory" in payload["tool_names"]


def test_evoinfer_schema_outputs_versioned_dream_memory_schema() -> None:
    result = CliRunner().invoke(cli, ["schema", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert "DreamMemory" in payload["schemas"]
    assert "OptimizationMemoryInput" in payload["schemas"]
    assert "EnvironmentDebugMemoryInput" in payload["schemas"]
    assert "profiler_artifacts" in json.dumps(payload["schemas"]["DreamMemory"])


def test_evoinfer_readme_documents_open_box_mcp_usage() -> None:
    readme = Path("docs/evoinfer/README.md")
    chinese_readme = Path("README.zh-CN.md")

    assert readme.is_file()
    assert chinese_readme.is_file()
    text = readme.read_text(encoding="utf-8")
    chinese_text = chinese_readme.read_text(encoding="utf-8")
    assert "uv tool install --force --editable ." in text
    assert "uv run" not in text
    assert "uv run" not in chinese_text
    assert "src/evoinfer_mcp/dream/seed_memories.json" in text
    assert "evoinfer memory-seed --json" in text
    assert "dream_extract_and_write_memories" in text
    assert "evoinfer doctor --json" in text
    assert "evoinfer mcp-config --client codex" in text
    assert "evoinfer force-session" in text
    assert "mandatory Dream session" in text
    assert "evoinfer --client codex --hook-every-steps 10" in text
    assert "Kimi CLI is not wired into hook mode" in text
    assert "claude mcp add-json" in text
    assert "--enable-embedding" in text
    assert "EvoInfer Dream 是一个面向推理优化 agent 的开盒 MCP 记忆管理器" in chinese_text
    assert "本地共享记忆库" in chinese_text
    assert "evoinfer --client codex --hook-every-steps 10" in chinese_text


def test_evoinfer_lifecycle_smoke_runs_mcp_stdio_memory_flow(
    tmp_path: Path,
) -> None:
    share_dir = tmp_path / "share"
    workdir = tmp_path / "campaign"

    result = CliRunner().invoke(
        cli,
        [
            "lifecycle-smoke",
            "--json",
            "--share-dir",
            str(share_dir),
            "--workdir",
            str(workdir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["workdir"] == str(workdir)
    assert payload["share_dir"] == str(share_dir)
    assert payload["phases"] == [
        "protocol",
        "search",
        "stuck_search",
        "stage",
        "extract",
        "write",
        "promote",
        "feedback",
        "list",
        "protocol_verify",
    ]
    assert payload["prior_memory"]["id"] == "opt_lifecycle_prior"
    assert payload["prior_memory"]["chosen"] == 2
    assert payload["prior_memory"]["useful_when_chosen"] == 1
    assert payload["promoted_memory"]["id"] == "opt_lifecycle_candidate"
    assert payload["promoted_memory"]["status"] == "promoted"
    assert (workdir / "dream_write_candidates.json").exists()
    campaign_result_path = Path(payload["campaign_result_path"])
    assert campaign_result_path.exists()
    assert campaign_result_path.parent == workdir
    protocol_verification = payload["protocol_verification"]
    assert protocol_verification["passed"] is True
    assert protocol_verification["retrieval_event_count"] == 2
    assert protocol_verification["completion_candidate_count"] == 1


def test_evoinfer_memory_export_and_import_cli_roundtrip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    share_dir = tmp_path / "share"
    memory_file = share_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "version": 1,
                "memories": [
                    {
                        "id": "opt_exported",
                        "category": "optimization",
                        "title": "Exported CUDA optimization memory",
                        "summary": "Artifact-backed memory for CLI export.",
                        "environment": "RTX 3090",
                        "model_type": "operator-kernel",
                        "inference_backend": "cuda",
                        "success": True,
                        "detail_description": "Verified by benchmark artifacts.",
                        "artifacts": ["runs/baseline.json"],
                        "correctness_artifacts": ["runs/correctness.json"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(share_dir))

    export_result = CliRunner().invoke(
        cli,
        ["memory-export", "--json"],
    )

    assert export_result.exit_code == 0, export_result.output
    exported = json.loads(export_result.output)
    assert [memory["id"] for memory in exported["memories"]] == ["opt_exported"]

    import_path = tmp_path / "import.json"
    import_path.write_text(
        json.dumps(
            {
                "version": 1,
                "memories": [
                    {
                        "id": "env_imported",
                        "category": "environment_debug",
                        "title": "Imported FlashInfer environment fix",
                        "summary": "CLI-imported environment debug memory.",
                        "environment": "RTX 3090",
                        "debug_type": "install",
                        "component": "flashinfer",
                        "issue_signature": "flashinfer import failed",
                        "symptoms": "ImportError during benchmark setup.",
                        "root_cause": "Wheel did not match local CUDA runtime.",
                        "solution": "Install the CUDA-matched FlashInfer wheel.",
                        "verification": "FlashInfer import and smoke benchmark passed.",
                        "success": True,
                        "diagnostic_artifacts": ["logs/import-error.txt"],
                        "verification_artifacts": ["logs/import-ok.txt"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dry_run = CliRunner().invoke(
        cli,
        ["memory-import", str(import_path), "--json"],
    )

    assert dry_run.exit_code == 0, dry_run.output
    dry_payload = json.loads(dry_run.output)
    assert dry_payload == {
        "dry_run": True,
        "imported_count": 1,
        "memory_ids": ["env_imported"],
    }
    assert [memory["id"] for memory in json.loads(memory_file.read_text())["memories"]] == [
        "opt_exported"
    ]

    apply_result = CliRunner().invoke(
        cli,
        ["memory-import", str(import_path), "--apply", "--json"],
    )

    assert apply_result.exit_code == 0, apply_result.output
    apply_payload = json.loads(apply_result.output)
    assert apply_payload == {
        "dry_run": False,
        "imported_count": 1,
        "memory_ids": ["env_imported"],
    }
    assert [memory["id"] for memory in json.loads(memory_file.read_text())["memories"]] == [
        "env_imported"
    ]


def test_evoinfer_memory_seed_merges_packaged_memories_without_overwriting(
    tmp_path: Path,
) -> None:
    share_dir = tmp_path / "share"
    memory_file = share_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text(
        json.dumps(
            {
                "version": 1,
                "memories": [
                    {
                        "id": "opt_seed_cuda_softmax_shared_memory_v1",
                        "category": "optimization",
                        "title": "Local edited Softmax memory",
                        "summary": "The user has already edited this seed memory locally.",
                        "environment": "local",
                        "model_type": "operator-kernel",
                        "inference_backend": "cuda",
                        "success": True,
                        "detail_description": "Keep local edits when seeding.",
                        "artifacts": ["local/benchmark.json"],
                        "correctness_artifacts": ["local/correctness.json"],
                        "chosen": 9,
                        "useful_when_chosen": 6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["memory-seed", "--share-dir", str(share_dir), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["seed_count"] == 7
    assert payload["imported_count"] == 6
    assert "opt_seed_cuda_softmax_shared_memory_v1" not in payload["memory_ids"]
    persisted = json.loads(memory_file.read_text(encoding="utf-8"))["memories"]
    by_id = {memory["id"]: memory for memory in persisted}
    assert len(by_id) == 7
    assert by_id["opt_seed_cuda_softmax_shared_memory_v1"]["title"] == "Local edited Softmax memory"
    assert by_id["opt_seed_cuda_softmax_shared_memory_v1"]["chosen"] == 9

    second = CliRunner().invoke(
        cli,
        ["memory-seed", "--share-dir", str(share_dir), "--json"],
    )

    assert second.exit_code == 0, second.output
    assert json.loads(second.output)["imported_count"] == 0


def test_evoinfer_seed_memories_are_searchable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    share_dir = tmp_path / "share"
    monkeypatch.setenv("EVOINFER_SHARE_DIR", str(share_dir))

    result = CliRunner().invoke(
        cli,
        ["memory-seed", "--share-dir", str(share_dir), "--json"],
    )

    assert result.exit_code == 0, result.output
    response = search_dream_memories(
        DreamMemorySearchInput(
            query="optimize cuda row-wise softmax shared memory seq=2048",
            category="optimization",
            tags=["cuda", "softmax"],
            top_k=3,
            record_choice=False,
        )
    )
    assert response.results
    assert response.results[0].memory.id == "opt_seed_cuda_softmax_shared_memory_v1"


def test_evoinfer_verify_protocol_cli_validates_campaign_artifacts(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "with_memory"
    workdir.mkdir()
    (workdir / "benchmark_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "correctness_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "opt_protocol_candidate",
                    "category": "candidate_optimization",
                    "title": "Artifact-backed protocol candidate",
                    "artifact_refs": ["benchmark_raw.json", "correctness_raw.json"],
                }
            ]
        ),
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "name": "protocol-cli",
                "runs": [
                    {
                        "arm_name": "with_memory",
                        "dream_enabled": True,
                        "session_id": "with",
                        "work_dir": str(workdir),
                        "status": "finished",
                        "started_at": 1,
                        "ended_at": 2,
                        "duration_seconds": 1,
                        "verification_status": "passed",
                        "dream_retrieval_events": [
                            {
                                "trigger": "campaign_start",
                                "query": "cuda rmsnorm",
                                "categories": ["optimization"],
                                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                                "result_count": 1,
                                "step_count": 0,
                            },
                            {
                                "trigger": "stuck",
                                "query": "cuda rmsnorm regression",
                                "categories": ["optimization"],
                                "memory_ids": ["opt_cuda_rmsnorm_block_parallel_v1"],
                                "result_count": 1,
                                "step_count": 12,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "verify-protocol",
            str(campaign_path),
            "--require-stuck-retrieval",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert payload["retrieval_event_count"] == 2
    assert payload["completion_candidate_count"] == 1


def test_evoinfer_verify_protocol_cli_accepts_relocated_artifact_root(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "bundle" / "fla-suite-work"
    local_workdir = artifact_root / "rep01" / "with_memory"
    local_workdir.mkdir(parents=True)
    (local_workdir / "benchmark_raw.json").write_text("{}", encoding="utf-8")
    (local_workdir / "correctness_raw.json").write_text("{}", encoding="utf-8")
    (local_workdir / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "opt_relocated_candidate",
                    "category": "candidate_optimization",
                    "title": "Relocated artifact candidate",
                    "artifact_refs": ["benchmark_raw.json", "correctness_raw.json"],
                }
            ]
        ),
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "name": "relocated-protocol-cli",
                "runs": [
                    {
                        "arm_name": "with_memory",
                        "dream_enabled": True,
                        "session_id": "with",
                        "work_dir": "/remote/evoinfer/fla-suite/rep01/with_memory",
                        "status": "finished",
                        "started_at": 1,
                        "ended_at": 2,
                        "duration_seconds": 1,
                        "verification_status": "passed",
                        "dream_retrieval_events": [
                            {
                                "trigger": "campaign_start",
                                "query": "fla route policy",
                                "categories": ["optimization"],
                                "memory_ids": ["opt_fla_route_policy"],
                                "result_count": 1,
                                "step_count": 0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "verify-protocol",
            str(campaign_path),
            "--artifact-root",
            str(artifact_root),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert payload["resolved_work_dirs"] == [str(local_workdir)]


def test_evoinfer_verify_protocol_cli_rejects_unsafe_route_transfer(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "with_memory"
    workdir.mkdir()
    (workdir / "benchmark_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "correctness_raw.json").write_text("{}", encoding="utf-8")
    (workdir / "dream_write_candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "opt_protocol_candidate",
                    "category": "candidate_optimization",
                    "title": "Artifact-backed protocol candidate",
                    "artifact_refs": ["benchmark_raw.json", "correctness_raw.json"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (workdir / "route_decision.json").write_text(
        json.dumps(
            {
                "selection_policy": "memory_route_policy",
                "selected_dtypes": ["float32"],
                "audit_dtypes": [],
                "avoid_dtypes": ["float16"],
                "selected_memory_ids": ["opt_fla_float16_negative"],
                "skip_evidence": {},
            }
        ),
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "name": "unsafe-transfer-cli",
                "runs": [
                    {
                        "arm_name": "with_memory",
                        "dream_enabled": True,
                        "session_id": "with",
                        "work_dir": str(workdir),
                        "status": "finished",
                        "started_at": 1,
                        "ended_at": 2,
                        "duration_seconds": 1,
                        "verification_status": "passed",
                        "dream_retrieval_events": [
                            {
                                "trigger": "campaign_start",
                                "query": "fla float16 dtype boundary",
                                "categories": ["optimization"],
                                "memory_ids": ["opt_fla_float16_negative"],
                                "result_count": 1,
                                "step_count": 0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "verify-protocol",
            str(campaign_path),
            "--require-transfer-safety",
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is False
    assert "missing skip evidence" in payload["error"]


def test_evoinfer_verify_protocol_suite_cli_enforces_strict_pass_rate(
    tmp_path: Path,
) -> None:
    campaign_paths: list[Path] = []
    for index, events in enumerate(
        [
            [
                {
                    "trigger": "campaign_start",
                    "query": "fla route policy",
                    "categories": ["optimization"],
                    "memory_ids": ["opt_fla_route_policy"],
                    "result_count": 1,
                    "step_count": 0,
                }
            ],
            [
                {
                    "trigger": "campaign_start",
                    "query": "fla route policy",
                    "categories": ["optimization"],
                    "memory_ids": ["opt_fla_route_policy"],
                    "result_count": 1,
                    "step_count": 0,
                },
                {
                    "trigger": "stuck",
                    "query": "fla route policy after failed dtype branch",
                    "categories": ["optimization"],
                    "memory_ids": ["opt_fla_route_policy"],
                    "result_count": 1,
                    "step_count": 12,
                },
            ],
        ],
        start=1,
    ):
        workdir = tmp_path / f"rep{index:02d}" / "with_memory"
        workdir.mkdir(parents=True)
        (workdir / "benchmark_raw.json").write_text("{}", encoding="utf-8")
        (workdir / "correctness_raw.json").write_text("{}", encoding="utf-8")
        (workdir / "dream_write_candidates.json").write_text(
            json.dumps(
                [
                    {
                        "category": "candidate_optimization",
                        "title": "FLA route policy candidate",
                        "artifact_refs": ["benchmark_raw.json", "correctness_raw.json"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        campaign_path = tmp_path / f"campaign-{index}.json"
        campaign_path.write_text(
            json.dumps(
                {
                    "name": f"fla-route-rep{index}",
                    "prompt": "Run FLA route policy campaign.",
                    "work_dir": str(workdir.parent),
                    "started_at": index,
                    "ended_at": index + 1,
                    "duration_seconds": 1,
                    "memory_before": {},
                    "memory_after": {},
                    "runs": [
                        {
                            "arm_name": "with_memory",
                            "dream_enabled": True,
                            "session_id": f"with-{index}",
                            "work_dir": str(workdir),
                            "status": "finished",
                            "started_at": index,
                            "ended_at": index + 1,
                            "duration_seconds": 1,
                            "verification_status": "passed",
                            "dream_retrieval_count": len(events),
                            "dream_retrieval_events": events,
                            "dream_retrieved_memory_ids": ["opt_fla_route_policy"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        campaign_paths.append(campaign_path)

    failed = CliRunner().invoke(
        cli,
        [
            "verify-protocol-suite",
            *[str(path) for path in campaign_paths],
            "--min-protocol-pass-rate",
            "1.0",
            "--min-strict-protocol-pass-rate",
            "0.75",
            "--json",
        ],
    )

    assert failed.exit_code == 1, failed.output
    failed_payload = json.loads(failed.output)
    assert failed_payload["ok"] is False
    assert failed_payload["checked_count"] == 2
    assert failed_payload["protocol_pass_rate"] == 1.0
    assert failed_payload["strict_protocol_pass_rate"] == 0.5
    assert "strict_protocol_pass_rate" in failed_payload["failures"][0]

    passed = CliRunner().invoke(
        cli,
        [
            "verify-protocol-suite",
            *[str(path) for path in campaign_paths],
            "--min-protocol-pass-rate",
            "1.0",
            "--min-strict-protocol-pass-rate",
            "0.5",
            "--json",
        ],
    )

    assert passed.exit_code == 0, passed.output
    passed_payload = json.loads(passed.output)
    assert passed_payload["ok"] is True
    assert passed_payload["strict_protocol_pass_count"] == 1
