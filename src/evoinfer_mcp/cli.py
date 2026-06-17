from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

import typer

cli = typer.Typer(help="Manage EvoInfer Dream MCP tooling.")

EvoInferMCPClient = Literal["codex", "claude", "generic"]
EvoInferAgentClient = Literal["codex", "claude"]
EvoInferMCPConfigFormat = Literal[
    "json",
    "codex-toml",
    "claude-json",
    "claude-add-json",
]
EvoInferClaudeScope = Literal["local", "project", "user"]


def build_evoinfer_mcp_config(
    *,
    client: EvoInferMCPClient = "generic",
    share_dir: Path | None = None,
    command: str = "python",
    enable_embedding: bool = False,
    embedding_model: str | None = None,
    call_log_path: Path | None = None,
    session_id: str | None = None,
    mandatory: bool = False,
) -> dict[str, object]:
    """Build a stdio MCP config for the EvoInfer Dream memory manager."""

    server: dict[str, object] = {
        "command": command,
        "args": ["-m", "evoinfer_mcp.dream.mcp_server"],
    }
    env = _build_evoinfer_mcp_env(
        share_dir=share_dir,
        enable_embedding=enable_embedding,
        embedding_model=embedding_model,
        call_log_path=call_log_path,
        session_id=session_id,
        mandatory=mandatory,
    )
    if env:
        server["env"] = env
    return {
        "client": client,
        "mcpServers": {
            "evoinfer-dream": server,
        },
    }


def build_evoinfer_mcp_server_config(
    *,
    share_dir: Path | None = None,
    command: str = "python",
    include_type: bool = False,
    enable_embedding: bool = False,
    embedding_model: str | None = None,
    call_log_path: Path | None = None,
    session_id: str | None = None,
    mandatory: bool = False,
) -> dict[str, object]:
    """Build one stdio MCP server config entry."""

    server: dict[str, object] = {
        "command": command,
        "args": ["-m", "evoinfer_mcp.dream.mcp_server"],
    }
    if include_type:
        server = {"type": "stdio", **server}
    env = _build_evoinfer_mcp_env(
        share_dir=share_dir,
        enable_embedding=enable_embedding,
        embedding_model=embedding_model,
        call_log_path=call_log_path,
        session_id=session_id,
        mandatory=mandatory,
    )
    if env:
        server["env"] = env
    return server


def render_evoinfer_mcp_config_template(
    *,
    client: EvoInferMCPClient,
    config_format: EvoInferMCPConfigFormat,
    share_dir: Path | None,
    command: str,
    scope: EvoInferClaudeScope,
    enable_embedding: bool = False,
    embedding_model: str | None = None,
) -> str:
    """Render a client-specific EvoInfer Dream MCP setup template."""

    if config_format == "json":
        return json.dumps(
            build_evoinfer_mcp_config(
                client=client,
                share_dir=share_dir,
                command=command,
                enable_embedding=enable_embedding,
                embedding_model=embedding_model,
            ),
            ensure_ascii=False,
            indent=2,
        )

    if config_format == "claude-json":
        return json.dumps(
            {
                "mcpServers": {
                    "evoinfer-dream": build_evoinfer_mcp_server_config(
                        share_dir=share_dir,
                        command=command,
                        include_type=True,
                        enable_embedding=enable_embedding,
                        embedding_model=embedding_model,
                    )
                }
            },
            ensure_ascii=False,
            indent=2,
        )

    if config_format == "claude-add-json":
        server_json = json.dumps(
            build_evoinfer_mcp_server_config(
                share_dir=share_dir,
                command=command,
                include_type=True,
                enable_embedding=enable_embedding,
                embedding_model=embedding_model,
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return " ".join(
            [
                "claude",
                "mcp",
                "add-json",
                "evoinfer-dream",
                shlex.quote(server_json),
                "--scope",
                scope,
            ]
        )

    if config_format == "codex-toml":
        return _render_codex_mcp_toml(
            share_dir=share_dir,
            command=command,
            enable_embedding=enable_embedding,
            embedding_model=embedding_model,
        )

    raise ValueError(f"unsupported EvoInfer MCP config format: {config_format}")


def _render_codex_mcp_toml(
    *,
    share_dir: Path | None,
    command: str,
    enable_embedding: bool = False,
    embedding_model: str | None = None,
    call_log_path: Path | None = None,
    session_id: str | None = None,
    mandatory: bool = False,
) -> str:
    lines = [
        "[mcp_servers.evoinfer-dream]",
        f"command = {_toml_string(command)}",
        'args = ["-m", "evoinfer_mcp.dream.mcp_server"]',
    ]
    env = _build_evoinfer_mcp_env(
        share_dir=share_dir,
        enable_embedding=enable_embedding,
        embedding_model=embedding_model,
        call_log_path=call_log_path,
        session_id=session_id,
        mandatory=mandatory,
    )
    if env:
        lines.extend(
            [
                "",
                "[mcp_servers.evoinfer-dream.env]",
            ]
        )
        for key, value in sorted(env.items()):
            lines.append(f"{key} = {_toml_string(value)}")
    return "\n".join(lines) + "\n"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _build_evoinfer_mcp_env(
    *,
    share_dir: Path | None,
    enable_embedding: bool,
    embedding_model: str | None,
    call_log_path: Path | None = None,
    session_id: str | None = None,
    mandatory: bool = False,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if share_dir is not None:
        env["EVOINFER_SHARE_DIR"] = str(share_dir)
    if call_log_path is not None:
        env["EVOINFER_MCP_CALL_LOG"] = str(call_log_path)
    if session_id:
        env["EVOINFER_DREAM_SESSION_ID"] = session_id
    if mandatory:
        env["EVOINFER_DREAM_MANDATORY"] = "1"
    if enable_embedding:
        from evoinfer_mcp.dream.embedding import (
            DEFAULT_EMBEDDING_MODEL,
            embedding_env_for_local_cpu,
        )

        env.update(
            embedding_env_for_local_cpu(
                model=embedding_model or DEFAULT_EMBEDDING_MODEL,
            )
        )
    return env


@cli.callback(invoke_without_command=True)
def evoinfer_main(
    ctx: typer.Context,
    client: Annotated[
        EvoInferAgentClient | None,
        typer.Option(
            "--client",
            help="Agent client to launch when no subcommand is supplied.",
        ),
    ] = None,
    hook_every_steps: Annotated[
        int,
        typer.Option(
            "--hook-every-steps",
            min=1,
            help="Run an EvoInfer Dream checkpoint after this many tool checkpoints.",
        ),
    ] = 10,
    session_dir: Annotated[
        Path | None,
        typer.Option(
            "--session-dir",
            help="Directory for the hooked EvoInfer session bundle.",
        ),
    ] = None,
    share_dir: Annotated[
        Path | None,
        typer.Option(
            "--share-dir",
            help="Durable Dream memory store. Defaults to SESSION_DIR/share.",
        ),
    ] = None,
    workdir: Annotated[
        Path | None,
        typer.Option(
            "--workdir",
            help="Agent workdir. Defaults to SESSION_DIR/work.",
        ),
    ] = None,
    command: Annotated[
        str,
        typer.Option(
            "--command",
            help="Python executable used by hooks and the MCP server.",
        ),
    ] = sys.executable,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="Optional initial prompt for the launched agent."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Create the session bundle but do not launch the agent."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON for the generated session."),
    ] = False,
) -> None:
    """Launch a Claude/Codex session with EvoInfer Dream hooks."""

    if ctx.invoked_subcommand is not None:
        return

    selected_client = client
    selected_steps = hook_every_steps
    if selected_client is None:
        selected_client = typer.prompt(
            "Client",
            default="codex",
            type=click_choice(["codex", "claude"]),
        )
        selected_steps = typer.prompt(
            "Dream hook every tool checkpoints",
            default=hook_every_steps,
            type=int,
        )

    if session_dir is None:
        session_dir = Path.cwd() / ".evoinfer" / selected_client

    payload = build_evoinfer_hooked_session_bundle(
        client=selected_client,
        hook_every_steps=selected_steps,
        session_dir=session_dir,
        share_dir=share_dir,
        workdir=workdir,
        command=command,
        prompt=prompt,
    )

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"EvoInfer hooked session: {payload['client']}")
        typer.echo(f"Session dir: {payload['session_dir']}")
        typer.echo(f"Workdir: {payload['workdir']}")
        typer.echo(f"Hook every tool checkpoints: {payload['hook_every_steps']}")
        typer.echo(f"Hook config: {payload['hook_config_path']}")
        typer.echo(f"Dream context: {payload['dream_context_path']}")
        typer.echo("Launch command:")
        typer.echo(" ".join(shlex.quote(str(part)) for part in payload["launch_command"]))

    if dry_run:
        return

    completed = subprocess.run(
        [str(part) for part in payload["launch_command"]],
        cwd=str(payload["workdir"]),
        check=False,
    )
    raise typer.Exit(code=completed.returncode)


def click_choice(values: list[str]):
    import click

    return click.Choice(values, case_sensitive=False)


@cli.command("mcp-config")
def evoinfer_mcp_config(
    client: Annotated[
        EvoInferMCPClient,
        typer.Option(
            "--client",
            help=(
                "Target MCP client template. Codex and Claude currently use compatible stdio JSON."
            ),
        ),
    ] = "generic",
    share_dir: Annotated[
        Path | None,
        typer.Option(
            "--share-dir",
            help="Optional EVOINFER_SHARE_DIR to pin the EvoInfer Dream memory store.",
        ),
    ] = None,
    command: Annotated[
        str,
        typer.Option(
            "--command",
            help="Python executable used by the MCP client to start the server.",
        ),
    ] = "python",
    config_format: Annotated[
        EvoInferMCPConfigFormat,
        typer.Option(
            "--format",
            help=("Template format: json, codex-toml, claude-json, or claude-add-json."),
        ),
    ] = "json",
    scope: Annotated[
        EvoInferClaudeScope,
        typer.Option(
            "--scope",
            help="Claude Code scope used only by --format claude-add-json.",
        ),
    ] = "project",
    enable_embedding: Annotated[
        bool,
        typer.Option(
            "--enable-embedding",
            help="Enable optional local CPU embeddings inside the EvoInfer Dream MCP server.",
        ),
    ] = False,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="SentenceTransformer model name used when --enable-embedding is set.",
        ),
    ] = None,
) -> None:
    """Print a JSON MCP config for EvoInfer Dream."""

    typer.echo(
        render_evoinfer_mcp_config_template(
            client=client,
            config_format=config_format,
            share_dir=share_dir,
            command=command,
            scope=scope,
            enable_embedding=enable_embedding,
            embedding_model=embedding_model,
        )
    )


MANDATORY_DREAM_SESSION_PROTOCOL = """# MANDATORY EvoInfer Dream session protocol

This is a dedicated EvoInfer Dream session. Dream is mandatory in this session,
not an optional helper.

Before doing task-local exploration:

1. Call `dream_get_agent_protocol` for the current task type and workdir.
2. Call `dream_search_memories` using the hardware, backend, dtype, workload,
   model/operator, and optimization or debug goal.
3. Treat retrieved successful memories as evidence-backed hypotheses. Treat
   negative memories as transfer constraints.

During the task:

4. If stuck, changing route, or choosing between CUDA, Triton, PyTorch,
   FlashInfer, FLA, another backend, or an environment fix, call
   `dream_search_memories` again before continuing.
5. Do real benchmark, correctness, profiler, source, and environment checks.
   Do not claim speedup or correctness without artifacts.

Before the final report:

6. If benchmark, correctness, profiler, verifier, source, or environment
   artifacts exist in the workdir, call `dream_extract_and_write_memories`.
7. Record feedback with `dream_record_feedback` when retrieved memories were
   useful or not useful.
8. Mention retrieved and written Dream memory IDs and the evidence artifacts in
   the final report.
"""


@cli.command("force-session")
def evoinfer_force_session(
    session_dir: Annotated[
        Path,
        typer.Option(
            "--session-dir",
            help="Directory that stores this mandatory Dream session bundle.",
        ),
    ],
    share_dir: Annotated[
        Path | None,
        typer.Option(
            "--share-dir",
            help="Durable Dream memory store for this session. Defaults to SESSION_DIR/share.",
        ),
    ] = None,
    workdir: Annotated[
        Path | None,
        typer.Option(
            "--workdir",
            help="Agent workdir for this session. Defaults to SESSION_DIR/work.",
        ),
    ] = None,
    command: Annotated[
        str,
        typer.Option(
            "--command",
            help="Python executable used by the MCP client to start EvoInfer Dream.",
        ),
    ] = "python",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Create a per-session bundle that makes Dream mandatory for one agent run."""

    payload = build_evoinfer_force_session_bundle(
        session_dir=session_dir,
        share_dir=share_dir,
        workdir=workdir,
        command=command,
    )
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    typer.echo("EvoInfer mandatory Dream session")
    typer.echo(f"Session dir: {payload['session_dir']}")
    typer.echo(f"Workdir: {payload['workdir']}")
    typer.echo(f"MCP config: {payload['mcp_config_path']}")
    typer.echo(f"Call log: {payload['call_log_path']}")
    typer.echo("Commands:")
    commands = payload["commands"]
    if isinstance(commands, dict):
        for name, command_text in commands.items():
            typer.echo(f"- {name}: {command_text}")


def build_evoinfer_force_session_bundle(
    *,
    session_dir: Path,
    share_dir: Path | None,
    workdir: Path | None,
    command: str,
) -> dict[str, object]:
    session_dir = session_dir.expanduser().resolve()
    share_dir = (share_dir or session_dir / "share").expanduser().resolve()
    workdir = (workdir or session_dir / "work").expanduser().resolve()
    call_log_path = session_dir / "mcp_calls.jsonl"
    mcp_config_path = session_dir / "mcp.json"
    session_id = session_dir.name

    session_dir.mkdir(parents=True, exist_ok=True)
    share_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    server = build_evoinfer_mcp_server_config(
        share_dir=share_dir,
        command=command,
        include_type=True,
        call_log_path=call_log_path,
        session_id=session_id,
        mandatory=True,
    )
    mcp_config = {"mcpServers": {"evoinfer-dream": server}}
    mcp_config_path.write_text(
        json.dumps(mcp_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    instruction_paths = [workdir / "AGENTS.md", workdir / "CLAUDE.md"]
    for instruction_path in instruction_paths:
        instruction_path.write_text(MANDATORY_DREAM_SESSION_PROTOCOL, encoding="utf-8")

    commands = {
        "claude": " ".join(
            [
                "claude",
                "-p",
                shlex.quote("<task prompt>"),
                "--mcp-config",
                shlex.quote(str(mcp_config_path)),
                "--strict-mcp-config",
                "--permission-mode",
                "bypassPermissions",
            ]
        ),
        "codex": " ".join(
            [
                "codex",
                "exec",
                "--cd",
                shlex.quote(str(workdir)),
                "--skip-git-repo-check",
                "-s",
                "danger-full-access",
                "-c",
                shlex.quote(
                    f"mcp_servers.evoinfer-dream.command={_toml_string(command)}"
                ),
                "-c",
                shlex.quote('mcp_servers.evoinfer-dream.args=["-m","evoinfer_mcp.dream.mcp_server"]'),
                "-c",
                shlex.quote(
                    "mcp_servers.evoinfer-dream.env.EVOINFER_SHARE_DIR="
                    f"{_toml_string(str(share_dir))}"
                ),
                "-c",
                shlex.quote(
                    "mcp_servers.evoinfer-dream.env.EVOINFER_MCP_CALL_LOG="
                    f"{_toml_string(str(call_log_path))}"
                ),
                "-c",
                shlex.quote(
                    "mcp_servers.evoinfer-dream.env.EVOINFER_DREAM_SESSION_ID="
                    f"{_toml_string(session_id)}"
                ),
                "-c",
                'mcp_servers.evoinfer-dream.env.EVOINFER_DREAM_MANDATORY="1"',
                shlex.quote("<task prompt>"),
            ]
        ),
        "kimi": " ".join(
            [
                "kimi",
                "--work-dir",
                shlex.quote(str(workdir)),
                "--mcp-config-file",
                shlex.quote(str(mcp_config_path)),
                "--prompt",
                shlex.quote("<task prompt>"),
            ]
        ),
    }

    return {
        "mode": "mandatory-session",
        "session_id": session_id,
        "session_dir": str(session_dir),
        "share_dir": str(share_dir),
        "workdir": str(workdir),
        "mcp_config_path": str(mcp_config_path),
        "call_log_path": str(call_log_path),
        "instruction_paths": [str(path) for path in instruction_paths],
        "commands": commands,
    }


def build_evoinfer_hooked_session_bundle(
    *,
    client: EvoInferAgentClient,
    hook_every_steps: int,
    session_dir: Path,
    share_dir: Path | None,
    workdir: Path | None,
    command: str,
    prompt: str | None = None,
) -> dict[str, object]:
    if hook_every_steps < 1:
        raise ValueError("hook_every_steps must be >= 1")

    base = build_evoinfer_force_session_bundle(
        session_dir=session_dir,
        share_dir=share_dir,
        workdir=workdir,
        command=command,
    )
    session_dir_path = Path(str(base["session_dir"]))
    share_dir_path = Path(str(base["share_dir"]))
    workdir_path = Path(str(base["workdir"]))
    hook_state_path = session_dir_path / "hook_state.json"
    dream_context_path = session_dir_path / "dream_context.md"
    hook_state_path.write_text('{"tool_checkpoint_count": 0}\n', encoding="utf-8")
    dream_context_path.write_text(
        "# EvoInfer Dream Context\n\nNo checkpoint has run yet.\n",
        encoding="utf-8",
    )

    if client == "claude":
        hook_config_path = _write_claude_hook_config(
            workdir=workdir_path,
            command=command,
            session_dir=session_dir_path,
            share_dir=share_dir_path,
            hook_state_path=hook_state_path,
            dream_context_path=dream_context_path,
            hook_every_steps=hook_every_steps,
        )
        launch_command = _build_claude_launch_command(
            mcp_config_path=Path(str(base["mcp_config_path"])),
            hook_config_path=hook_config_path,
            prompt=prompt,
        )
    else:
        hook_config_path = _write_codex_hook_config(
            workdir=workdir_path,
            command=command,
            session_dir=session_dir_path,
            share_dir=share_dir_path,
            hook_state_path=hook_state_path,
            dream_context_path=dream_context_path,
            hook_every_steps=hook_every_steps,
        )
        launch_command = _build_codex_launch_command(
            workdir=workdir_path,
            mcp_config_path=Path(str(base["mcp_config_path"])),
            prompt=prompt,
        )

    payload = {
        **base,
        "mode": "hooked-session",
        "client": client,
        "hook_every_steps": hook_every_steps,
        "hook_config_path": str(hook_config_path),
        "hook_state_path": str(hook_state_path),
        "dream_context_path": str(dream_context_path),
        "launch_command": launch_command,
        "commands": {client: " ".join(shlex.quote(part) for part in launch_command)},
    }
    return payload


def _hook_command_args(
    *,
    client: EvoInferAgentClient,
    session_dir: Path,
    share_dir: Path,
    hook_state_path: Path,
    dream_context_path: Path,
    hook_every_steps: int,
) -> list[str]:
    return [
        "-m",
        "evoinfer_mcp.hooks.dream_checkpoint",
        "--client",
        client,
        "--session-dir",
        str(session_dir),
        "--share-dir",
        str(share_dir),
        "--state-file",
        str(hook_state_path),
        "--context-file",
        str(dream_context_path),
        "--every-steps",
        str(hook_every_steps),
    ]


def _hook_shell_command(command: str, args: list[str]) -> str:
    return " ".join([shlex.quote(command), *[shlex.quote(arg) for arg in args]])


def _write_claude_hook_config(
    *,
    workdir: Path,
    command: str,
    session_dir: Path,
    share_dir: Path,
    hook_state_path: Path,
    dream_context_path: Path,
    hook_every_steps: int,
) -> Path:
    hook_dir = workdir / ".claude"
    hook_dir.mkdir(parents=True, exist_ok=True)
    args = _hook_command_args(
        client="claude",
        session_dir=session_dir,
        share_dir=share_dir,
        hook_state_path=hook_state_path,
        dream_context_path=dream_context_path,
        hook_every_steps=hook_every_steps,
    )
    hook = {
        "type": "command",
        "command": command,
        "args": args,
        "timeout": 60,
    }
    payload = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume",
                    "hooks": [hook],
                }
            ],
            "PostToolBatch": [
                {
                    "hooks": [hook],
                }
            ],
            "Stop": [
                {
                    "hooks": [hook],
                }
            ],
        }
    }
    path = hook_dir / "settings.local.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _write_codex_hook_config(
    *,
    workdir: Path,
    command: str,
    session_dir: Path,
    share_dir: Path,
    hook_state_path: Path,
    dream_context_path: Path,
    hook_every_steps: int,
) -> Path:
    hook_dir = workdir / ".codex"
    hook_dir.mkdir(parents=True, exist_ok=True)
    args = _hook_command_args(
        client="codex",
        session_dir=session_dir,
        share_dir=share_dir,
        hook_state_path=hook_state_path,
        dream_context_path=dream_context_path,
        hook_every_steps=hook_every_steps,
    )
    hook = {
        "type": "command",
        "command": _hook_shell_command(command, args),
        "timeout": 60,
        "statusMessage": "EvoInfer Dream checkpoint",
    }
    payload = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume",
                    "hooks": [hook],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "*",
                    "hooks": [hook],
                }
            ],
            "Stop": [
                {
                    "hooks": [hook],
                }
            ],
        }
    }
    path = hook_dir / "hooks.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _build_codex_launch_command(
    *,
    workdir: Path,
    mcp_config_path: Path,
    prompt: str | None,
) -> list[str]:
    server = json.loads(mcp_config_path.read_text(encoding="utf-8"))["mcpServers"][
        "evoinfer-dream"
    ]
    command = [
        "codex",
        "--cd",
        str(workdir),
        "-s",
        "danger-full-access",
        "--dangerously-bypass-hook-trust",
        "-c",
        f"mcp_servers.evoinfer-dream.command={_toml_string(server['command'])}",
        "-c",
        'mcp_servers.evoinfer-dream.args=["-m","evoinfer_mcp.dream.mcp_server"]',
    ]
    env = server.get("env", {})
    if isinstance(env, dict):
        for key, value in sorted(env.items()):
            command.extend(
                [
                    "-c",
                    f"mcp_servers.evoinfer-dream.env.{key}={_toml_string(str(value))}",
                ]
            )
    if prompt:
        command.append(prompt)
    return command


def _build_claude_launch_command(
    *,
    mcp_config_path: Path,
    hook_config_path: Path,
    prompt: str | None,
) -> list[str]:
    command = [
        "claude",
        "--mcp-config",
        str(mcp_config_path),
        "--strict-mcp-config",
        "--settings",
        str(hook_config_path),
        "--permission-mode",
        "bypassPermissions",
    ]
    if prompt:
        command.append(prompt)
    return command


@cli.command("doctor")
def evoinfer_doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Check whether the EvoInfer Dream MCP runtime is usable."""

    payload = run_evoinfer_doctor()
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if not payload["ok"]:
            raise typer.Exit(code=1)
        return

    typer.echo("EvoInfer Dream MCP doctor")
    for check in payload["checks"]:
        marker = "ok" if check["ok"] else "failed"
        detail = f": {check['detail']}" if check.get("detail") else ""
        typer.echo(f"- {check['name']}: {marker}{detail}")
    typer.echo(f"Tools: {', '.join(payload['tool_names'])}")
    if not payload["ok"]:
        raise typer.Exit(code=1)


@cli.command("schema")
def evoinfer_schema(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Print the versioned EvoInfer Dream memory JSON schemas."""

    payload = build_evoinfer_schema_payload()
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    typer.echo(text)


@cli.command("lifecycle-smoke")
def evoinfer_lifecycle_smoke(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
    share_dir: Annotated[
        Path | None,
        typer.Option(
            "--share-dir",
            help="EVOINFER_SHARE_DIR used by the spawned EvoInfer Dream MCP server.",
        ),
    ] = None,
    workdir: Annotated[
        Path | None,
        typer.Option(
            "--workdir",
            help="Workdir where smoke artifacts and dream_write_candidates.json are written.",
        ),
    ] = None,
) -> None:
    """Run an MCP stdio lifecycle smoke test for EvoInfer Dream."""

    payload = asyncio.run(run_evoinfer_lifecycle_smoke(share_dir=share_dir, workdir=workdir))
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if not payload["ok"]:
            raise typer.Exit(code=1)
        return

    marker = "ok" if payload["ok"] else "failed"
    typer.echo(f"EvoInfer Dream lifecycle smoke: {marker}")
    typer.echo(f"Share dir: {payload['share_dir']}")
    typer.echo(f"Workdir: {payload['workdir']}")
    typer.echo(f"Phases: {', '.join(payload['phases'])}")
    promoted = payload.get("promoted_memory") or {}
    if isinstance(promoted, dict) and promoted:
        typer.echo(f"Promoted memory: {promoted.get('id')} ({promoted.get('status')})")
    if not payload["ok"]:
        raise typer.Exit(code=1)


@cli.command("memory-export")
def evoinfer_memory_export(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to write the exported memory store."),
    ] = None,
) -> None:
    """Export the EvoInfer Dream memory store."""

    from evoinfer_mcp.dream.mcp_server import dream_export_memory_store_tool

    payload = json.loads(dream_export_memory_store_tool())
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if json_output or output is None:
        typer.echo(text)
    else:
        typer.echo(f"Exported {len(payload.get('memories', []))} Dream memories to {output}")


@cli.command("memory-import")
def evoinfer_memory_import(
    memory_store: Annotated[
        Path,
        typer.Argument(help="Memory store JSON exported by `evoinfer memory-export`."),
    ],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write the imported memory store. Default is dry-run validation.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Validate or import an EvoInfer Dream memory store."""

    from evoinfer_mcp.dream.mcp_server import dream_import_memory_store_tool

    payload = json.loads(
        dream_import_memory_store_tool(
            memory_store_json=memory_store.read_text(encoding="utf-8"),
            dry_run=not apply,
        )
    )
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    mode = "Imported" if apply else "Validated"
    typer.echo(
        f"{mode} {payload['imported_count']} Dream memories: " + ", ".join(payload["memory_ids"])
    )


@cli.command("verify-protocol")
def evoinfer_verify_protocol(
    campaign_result: Annotated[
        Path,
        typer.Argument(help="Campaign result JSON path to verify."),
    ],
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help=(
                "Local root for relocated campaign artifacts when campaign JSON "
                "contains remote work_dir paths."
            ),
        ),
    ] = None,
    require_stuck_retrieval: Annotated[
        bool,
        typer.Option(
            "--require-stuck-retrieval",
            help="Require a non-start Dream retrieval event such as stuck/periodic/branch-point.",
        ),
    ] = False,
    no_completion_candidates: Annotated[
        bool,
        typer.Option(
            "--no-completion-candidates",
            help="Do not require workdir/dream_write_candidates.json.",
        ),
    ] = False,
    no_artifact_valid_success: Annotated[
        bool,
        typer.Option(
            "--no-artifact-valid-success",
            help="Do not require a dream-enabled run with verification_status='passed'.",
        ),
    ] = False,
    require_transfer_safety: Annotated[
        bool,
        typer.Option(
            "--require-transfer-safety",
            help=(
                "Require route_decision.json avoided routes to cite retrieved "
                "skip_evidence memory IDs."
            ),
        ),
    ] = False,
    require_artifact_memory_write: Annotated[
        bool,
        typer.Option(
            "--require-artifact-memory-write",
            help=("Require artifact-driven memory auto-write evidence or explicit write blockers."),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Verify active Dream protocol evidence in a campaign result."""

    from evoinfer_mcp.evoinfer.dream_protocol_verifier import (
        DreamProtocolVerificationError,
        verify_dream_protocol_campaign_result,
    )

    try:
        payload = verify_dream_protocol_campaign_result(
            campaign_result,
            artifact_root=artifact_root,
            require_stuck_retrieval=require_stuck_retrieval,
            require_completion_candidates=not no_completion_candidates,
            require_artifact_valid_success=not no_artifact_valid_success,
            require_transfer_safety=require_transfer_safety,
            require_artifact_memory_write=require_artifact_memory_write,
        )
    except DreamProtocolVerificationError as exc:
        if json_output:
            typer.echo(
                json.dumps(
                    {"passed": False, "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            typer.echo(f"EvoInfer Dream protocol verifier: failed\n{exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    typer.echo("EvoInfer Dream protocol verifier: ok")
    typer.echo(f"Dream-enabled runs: {payload['dream_enabled_run_count']}")
    typer.echo(f"Retrieval events: {payload['retrieval_event_count']}")
    typer.echo(f"Completion candidates: {payload['completion_candidate_count']}")


@cli.command("verify-protocol-suite")
def evoinfer_verify_protocol_suite(
    campaign_results: Annotated[
        list[Path],
        typer.Argument(help="Campaign result JSON paths to aggregate and gate."),
    ],
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help=(
                "Local root for relocated campaign artifacts when campaign JSON "
                "contains remote work_dir paths."
            ),
        ),
    ] = None,
    min_protocol_pass_rate: Annotated[
        float,
        typer.Option(
            "--min-protocol-pass-rate",
            min=0.0,
            max=1.0,
            help="Minimum accepted task-start Dream protocol pass rate.",
        ),
    ] = 1.0,
    min_strict_protocol_pass_rate: Annotated[
        float,
        typer.Option(
            "--min-strict-protocol-pass-rate",
            min=0.0,
            max=1.0,
            help="Minimum accepted strict protocol pass rate requiring stuck/branch retrieval.",
        ),
    ] = 1.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Gate a set of campaign results by Dream protocol pass-rate thresholds."""

    payload = run_evoinfer_protocol_suite_gate(
        campaign_results,
        artifact_root=artifact_root,
        min_protocol_pass_rate=min_protocol_pass_rate,
        min_strict_protocol_pass_rate=min_strict_protocol_pass_rate,
    )
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        marker = "ok" if payload["ok"] else "failed"
        typer.echo(f"EvoInfer Dream protocol suite gate: {marker}")
        typer.echo(f"Checked runs: {payload['checked_count']}")
        typer.echo(f"Protocol pass rate: {payload['protocol_pass_rate']}")
        typer.echo(f"Strict protocol pass rate: {payload['strict_protocol_pass_rate']}")
        for failure in payload["failures"]:
            typer.echo(f"- {failure}")
    if not payload["ok"]:
        raise typer.Exit(code=1)


def run_evoinfer_doctor() -> dict[str, object]:
    checks: list[dict[str, object]] = []

    try:
        import evoinfer_mcp.dream.mcp_server as mcp_server

        checks.append({"name": "mcp_server_import", "ok": True, "detail": ""})
    except Exception as exc:
        checks.append(
            {
                "name": "mcp_server_import",
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return {"ok": False, "checks": checks, "tool_names": []}

    try:
        from evoinfer_mcp.share import get_share_dir

        share_dir = get_share_dir()
        share_dir.mkdir(parents=True, exist_ok=True)
        probe = share_dir / "evoinfer-dream-doctor.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append({"name": "share_dir_writable", "ok": True, "detail": str(share_dir)})
    except Exception as exc:
        checks.append(
            {
                "name": "share_dir_writable",
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    try:
        tool_names = _list_mcp_tool_names(mcp_server.mcp)
        required = {
            "dream_get_agent_protocol",
            "dream_search_memories",
            "dream_stage_memory_candidate",
            "dream_extract_memory_candidates",
            "dream_extract_and_write_memories",
            "dream_promote_memory",
            "dream_record_feedback",
        }
        missing = sorted(required - set(tool_names))
        checks.append(
            {
                "name": "tool_surface",
                "ok": not missing,
                "detail": (
                    f"missing={', '.join(missing)}" if missing else f"{len(tool_names)} tools"
                ),
            }
        )
    except Exception as exc:
        tool_names = []
        checks.append(
            {
                "name": "tool_surface",
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    try:
        from evoinfer_mcp.dream.embedding import describe_embedding_runtime

        embedding = describe_embedding_runtime(load_model=False)
        checks.append(
            {
                "name": "embedding_backend",
                "ok": bool(embedding["ok"]),
                "detail": str(embedding["detail"]),
            }
        )
    except Exception as exc:
        embedding = {"enabled": False, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        checks.append(
            {
                "name": "embedding_backend",
                "ok": False,
                "detail": str(embedding["detail"]),
            }
        )

    try:
        stdio_tool_names = asyncio.run(_list_mcp_stdio_tool_names(share_dir))
        missing = sorted(required - set(stdio_tool_names))
        checks.append(
            {
                "name": "mcp_stdio_launch",
                "ok": not missing,
                "detail": (
                    f"missing={', '.join(missing)}"
                    if missing
                    else f"{len(stdio_tool_names)} tools via stdio"
                ),
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "mcp_stdio_launch",
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    return {
        "ok": all(bool(check["ok"]) for check in checks),
        "checks": checks,
        "embedding": embedding,
        "tool_names": tool_names,
    }


def build_evoinfer_schema_payload() -> dict[str, object]:
    from evoinfer_mcp.dream.memory import (
        DreamMemory,
        DreamMemoryFeedbackInput,
        DreamMemorySearchInput,
        EnvironmentDebugMemoryInput,
        OptimizationMemoryInput,
    )

    return {
        "schema_version": 1,
        "schemas": {
            "DreamMemory": DreamMemory.model_json_schema(),
            "OptimizationMemoryInput": OptimizationMemoryInput.model_json_schema(),
            "EnvironmentDebugMemoryInput": EnvironmentDebugMemoryInput.model_json_schema(),
            "DreamMemorySearchInput": DreamMemorySearchInput.model_json_schema(),
            "DreamMemoryFeedbackInput": DreamMemoryFeedbackInput.model_json_schema(),
        },
    }


def run_evoinfer_protocol_suite_gate(
    campaign_results: list[Path],
    *,
    artifact_root: Path | None,
    min_protocol_pass_rate: float,
    min_strict_protocol_pass_rate: float,
) -> dict[str, object]:
    from evoinfer_mcp.evoinfer.protocol_suite_gate import (
        run_evoinfer_protocol_suite_gate as run_gate,
    )

    return run_gate(
        campaign_results,
        artifact_root=artifact_root,
        min_protocol_pass_rate=min_protocol_pass_rate,
        min_strict_protocol_pass_rate=min_strict_protocol_pass_rate,
    ).model_dump(mode="json")


async def run_evoinfer_lifecycle_smoke(
    *,
    share_dir: Path | None,
    workdir: Path | None,
) -> dict[str, object]:
    if share_dir is None:
        from evoinfer_mcp.share import get_share_dir

        share_dir = get_share_dir()
    if workdir is None:
        workdir = share_dir / "evoinfer" / "lifecycle-smoke"
    share_dir = share_dir.expanduser().resolve()
    workdir = workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    _seed_lifecycle_smoke_prior_memory(share_dir)
    _write_lifecycle_smoke_artifacts(workdir)

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    phases: list[str] = []
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(share_dir)},
    )

    with _real_stderr_for_stdio_subprocess():
        async with stdio_client(
            server,
            errlog=sys.__stderr__ or sys.stderr,
        ) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool(
                "dream_get_agent_protocol",
                arguments={"task_type": "optimization", "workdir": str(workdir)},
            )
            phases.append("protocol")

            search = await session.call_tool(
                "dream_search_memories",
                arguments={
                    "query": "FLA route policy dtype branch expansion",
                    "category": "optimization",
                    "tags": ["fla", "route_policy"],
                    "top_k": 1,
                    "record_choice": True,
                    "render_mode": "artifact_protocol",
                },
            )
            if "opt_lifecycle_prior" not in _mcp_tool_text(search):
                raise RuntimeError("lifecycle smoke prior memory was not retrieved")
            phases.append("search")

            stuck_search = await session.call_tool(
                "dream_search_memories",
                arguments={
                    "query": "FLA route policy stuck after broad dtype branch expansion",
                    "category": "optimization",
                    "tags": ["fla", "route_policy"],
                    "top_k": 1,
                    "record_choice": True,
                    "render_mode": "artifact_protocol",
                    "task_context": (
                        "Simulated branch point: local candidate expansion stalled, "
                        "so the agent searches Dream again before changing route."
                    ),
                },
            )
            if "opt_lifecycle_prior" not in _mcp_tool_text(stuck_search):
                raise RuntimeError("lifecycle smoke stuck search did not retrieve prior memory")
            phases.append("stuck_search")

            await session.call_tool(
                "dream_stage_memory_candidate",
                arguments={
                    "workdir": str(workdir),
                    "candidate_json": json.dumps(_lifecycle_smoke_candidate_payload()),
                },
            )
            phases.append("stage")

            extract = await session.call_tool(
                "dream_extract_memory_candidates",
                arguments={"workdir": str(workdir)},
            )
            candidates = json.loads(_mcp_tool_text(extract))["candidates"]
            promotion_input = _first_promotion_input(candidates)
            phases.append("extract")

            write = await session.call_tool(
                "dream_write_optimization_memory",
                arguments={"memory_json": json.dumps(promotion_input)},
            )
            written_memory = json.loads(_mcp_tool_text(write))["memory"]
            phases.append("write")

            promote = await session.call_tool(
                "dream_promote_memory",
                arguments={
                    "memory_id": written_memory["id"],
                    "reason": "CLI lifecycle smoke verifier artifacts passed.",
                    "evidence_artifacts": [
                        str(workdir / "baseline.json"),
                        str(workdir / "candidate.json"),
                        str(workdir / "correctness.json"),
                    ],
                    "evidence_level": "verified",
                },
            )
            promoted_memory = json.loads(_mcp_tool_text(promote))["memory"]
            phases.append("promote")

            await session.call_tool(
                "dream_record_feedback",
                arguments={
                    "memory_ids": ["opt_lifecycle_prior"],
                    "useful": True,
                    "reason": "CLI lifecycle smoke used prior memory and verifier artifacts passed.",
                    "evidence_artifacts": [
                        str(workdir / "baseline.json"),
                        str(workdir / "candidate.json"),
                        str(workdir / "correctness.json"),
                    ],
                    "source_session_id": "cli-lifecycle-smoke",
                },
            )
            phases.append("feedback")

            listed = await session.call_tool("dream_list_memories", arguments={})
            memories = {
                memory["id"]: memory
                for memory in json.loads(_mcp_tool_text(listed))["memories"]
            }
            phases.append("list")

    campaign_result_path = _write_lifecycle_smoke_campaign_result(workdir)
    from evoinfer_mcp.evoinfer.dream_protocol_verifier import (
        verify_dream_protocol_campaign_result,
    )

    protocol_verification = verify_dream_protocol_campaign_result(
        campaign_result_path,
        require_stuck_retrieval=True,
    )
    phases.append("protocol_verify")

    return {
        "ok": True,
        "share_dir": str(share_dir),
        "workdir": str(workdir),
        "phases": phases,
        "prior_memory": memories["opt_lifecycle_prior"],
        "promoted_memory": promoted_memory,
        "campaign_result_path": str(campaign_result_path),
        "protocol_verification": protocol_verification,
    }


def _seed_lifecycle_smoke_prior_memory(share_dir: Path) -> None:
    memory_file = share_dir / "dream" / "memories.json"
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    if memory_file.exists():
        try:
            payload = json.loads(memory_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"version": 1, "memories": []}
    else:
        payload = {"version": 1, "memories": []}
    raw_memories = payload.get("memories", payload)
    memories = [memory for memory in raw_memories if isinstance(memory, dict)]
    smoke_ids = {"opt_lifecycle_prior", "opt_lifecycle_candidate"}
    memories = [memory for memory in memories if memory.get("id") not in smoke_ids]
    memories.append(
        {
            "id": "opt_lifecycle_prior",
            "category": "optimization",
            "title": "Use route evidence before expanding dtype branches",
            "summary": "A prior FLA route memory reduced local candidate search.",
            "tags": ["fla", "route_policy"],
            "environment": "RTX 3090",
            "model_type": "operator-kernel",
            "inference_backend": "fla",
            "success": True,
            "artifacts": ["prior/baseline.json", "prior/candidate.json"],
            "correctness_artifacts": ["prior/correctness.json"],
            "chosen": 0,
            "useful_when_chosen": 0,
            "useful_rate": 0,
        }
    )
    memory_file.write_text(
        json.dumps({"version": 1, "memories": memories}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_lifecycle_smoke_artifacts(workdir: Path) -> None:
    (workdir / "baseline.json").write_text('{"latency_ms": 1.0}', encoding="utf-8")
    (workdir / "candidate.json").write_text('{"latency_ms": 0.5}', encoding="utf-8")
    (workdir / "correctness.json").write_text('{"max_abs_error": 0.0}', encoding="utf-8")


def _lifecycle_smoke_candidate_payload() -> dict[str, object]:
    return {
        "id": "opt_lifecycle_candidate",
        "category": "candidate_optimization",
        "title": "FLA route policy halves dtype branch search time",
        "summary": "Route memory selected a narrower candidate set.",
        "tags": ["fla", "route_policy", "dtype"],
        "environment": "RTX 3090",
        "model_type": "operator-kernel",
        "inference_backend": "fla",
        "metrics_before": {"wall_clock_s": 1.0},
        "metrics_after": {"wall_clock_s": 0.5},
        "success": True,
        "detail_description": (
            "The agent used retrieved route evidence to avoid broad dtype branch expansion, "
            "then verified correctness."
        ),
        "artifact_refs": ["baseline.json", "candidate.json", "correctness.json"],
        "artifacts": ["baseline.json", "candidate.json"],
        "correctness_artifacts": ["correctness.json"],
    }


def _write_lifecycle_smoke_campaign_result(workdir: Path) -> Path:
    campaign_result_path = workdir / "lifecycle_smoke_campaign.json"
    payload = {
        "name": "evoinfer-lifecycle-smoke",
        "prompt": (
            "Run a minimal EvoInfer Dream MCP lifecycle: retrieve prior memory, "
            "search again at a branch point, stage an artifact-backed candidate, "
            "promote it, and record feedback."
        ),
        "work_dir": str(workdir),
        "started_at": 1,
        "ended_at": 2,
        "duration_seconds": 1,
        "memory_before": {"memory_count": 1, "total_chosen": 0},
        "memory_after": {"memory_count": 2, "total_chosen": 2},
        "runs": [
            {
                "arm_name": "with_memory",
                "dream_enabled": True,
                "session_id": "cli-lifecycle-smoke",
                "work_dir": str(workdir),
                "status": "finished",
                "started_at": 1,
                "ended_at": 2,
                "duration_seconds": 1,
                "verification_status": "passed",
                "verification_output": (
                    "baseline_wall_clock_s=1.0 candidate_wall_clock_s=0.5 max_abs_error=0.0"
                ),
                "dream_retrieval_count": 2,
                "dream_retrieved_memory_ids": ["opt_lifecycle_prior"],
                "dream_retrieval_events": [
                    {
                        "trigger": "campaign_start",
                        "query": "FLA route policy dtype branch expansion",
                        "categories": ["optimization"],
                        "top_k_per_category": 1,
                        "memory_ids": ["opt_lifecycle_prior"],
                        "result_count": 1,
                        "step_count": 0,
                    },
                    {
                        "trigger": "stuck",
                        "query": ("FLA route policy stuck after broad dtype branch expansion"),
                        "categories": ["optimization"],
                        "top_k_per_category": 1,
                        "memory_ids": ["opt_lifecycle_prior"],
                        "result_count": 1,
                        "step_count": 12,
                    },
                ],
            }
        ],
    }
    campaign_result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return campaign_result_path


def _first_promotion_input(candidates: list[dict]) -> dict:
    for candidate in candidates:
        if candidate.get("promotion_ready") and isinstance(candidate.get("promotion_input"), dict):
            return candidate["promotion_input"]
    raise RuntimeError("no promotion-ready candidate returned by Dream extraction")


def _mcp_tool_text(result) -> str:
    return "\n".join(str(content.text) for content in result.content)


def _list_mcp_tool_names(mcp_server) -> list[str]:
    async def _list() -> list[str]:
        tools = await mcp_server.list_tools()
        iterable = tools.tools if hasattr(tools, "tools") else tools
        return sorted(tool.name for tool in iterable)

    return asyncio.run(_list())


async def _list_mcp_stdio_tool_names(share_dir: Path) -> list[str]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "evoinfer_mcp.dream.mcp_server"],
        env={"EVOINFER_SHARE_DIR": str(share_dir)},
    )
    with _real_stderr_for_stdio_subprocess():
        async with stdio_client(
            server,
            errlog=sys.__stderr__ or sys.stderr,
        ) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
    return sorted(tool.name for tool in tools.tools)


@contextmanager
def _real_stderr_for_stdio_subprocess():
    old = sys.stderr
    try:
        old.fileno()
        yield
        return
    except (AttributeError, OSError):
        pass
    replacement = sys.__stderr__ or old
    sys.stderr = replacement
    try:
        yield
    finally:
        sys.stderr = old
