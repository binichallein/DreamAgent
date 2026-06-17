"""Paired w/wo Dream-memory campaign runner for EvoInfer experiments."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import httpx
import websockets
from pydantic import BaseModel

from evoinfer_mcp.dream.memory import DreamMemoryCategory

DEFAULT_DREAM_RETRIEVAL_CATEGORIES: tuple[DreamMemoryCategory, ...] = (
    "optimization",
    "environment_debug",
)
DreamRetrievalRenderMode = Literal[
    "full",
    "route_policy",
    "route_policy_minimal",
    "route_policy_template",
    "route_policy_artifact",
    "route_policy_artifact_protocol",
]


class CampaignArm(BaseModel):
    """One experimental condition in a paired campaign."""

    name: str
    dream_enabled: bool
    initial_dream_retrieval: bool | None = None
    dream_retrieval_query: str | None = None
    dream_retrieval_tags: list[str] | None = None
    dream_retrieval_categories: tuple[DreamMemoryCategory, ...] | None = None
    dream_retrieval_memory_ids: list[str] | None = None
    dream_retrieval_top_k_per_category: int | None = None
    dream_retrieval_render_mode: DreamRetrievalRenderMode | None = None


class CampaignSpec(BaseModel):
    """A task specification that can be run with and without Dream memory."""

    name: str
    prompt: str
    work_dir: str
    create_dir: bool = False
    isolate_work_dirs: bool = True
    seed_dir: str | None = None
    reset_work_dirs: bool = False
    timer_interval_seconds: int | None = None
    observer_interval_seconds: int | None = None
    timeout_seconds: float = 1800.0
    verifier_command: str | None = None
    verifier_timeout_seconds: float = 120.0
    yolo_enabled: bool = False
    initial_dream_retrieval: bool = False
    dream_retrieval_query: str | None = None
    dream_retrieval_tags: list[str] = []
    dream_retrieval_categories: tuple[DreamMemoryCategory, ...] = (
        DEFAULT_DREAM_RETRIEVAL_CATEGORIES
    )
    dream_retrieval_memory_ids: list[str] = []
    dream_retrieval_top_k_per_category: int = 3
    dream_retrieval_render_mode: DreamRetrievalRenderMode = "full"
    active_dream_protocol: bool = False
    environment_command: str | None = None
    restore_dream_memory_after_arm: bool = False
    arms: tuple[CampaignArm, ...] = (
        CampaignArm(name="without_memory", dream_enabled=False),
        CampaignArm(name="with_memory", dream_enabled=True),
    )

    def work_dir_for_arm(self, arm: CampaignArm) -> str:
        if not self.isolate_work_dirs:
            return self.work_dir
        return str(Path(self.work_dir) / _slugify_arm_name(arm.name))


class DreamMemoryStats(BaseModel):
    """Aggregate Dream memory usage counters."""

    memory_count: int = 0
    total_chosen: int = 0
    total_useful_when_chosen: int = 0

    @classmethod
    def from_payload(cls, payload: Any) -> DreamMemoryStats:
        memories = payload.get("memories", []) if isinstance(payload, dict) else payload
        if not isinstance(memories, list):
            memories = []
        total_chosen = 0
        total_useful = 0
        for memory in memories:
            if not isinstance(memory, dict):
                continue
            total_chosen += _as_int(memory.get("chosen"))
            total_useful += _as_int(memory.get("useful_when_chosen"))
        return cls(
            memory_count=len([memory for memory in memories if isinstance(memory, dict)]),
            total_chosen=total_chosen,
            total_useful_when_chosen=total_useful,
        )


class DreamRetrievalEvent(BaseModel):
    """One structured Dream retrieval event observed during a session run."""

    trigger: str = ""
    query: str = ""
    categories: list[str] = []
    top_k_per_category: int = 0
    memory_ids: list[str] = []
    result_count: int = 0
    step_count: int | None = None


class DreamRetrievalContext(BaseModel):
    """Prompt-ready Dream retrieval context plus audit metadata."""

    text: str = ""
    event: DreamRetrievalEvent


class DreamCompletionExtraction(BaseModel):
    """Summary of completion-time Dream candidate extraction."""

    workdir: str
    candidate_count: int = 0
    artifact_ref_count: int = 0
    promotion_ready_count: int = 0
    extraction_sources: list[str] = []
    error: str | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> DreamCompletionExtraction:
        if not isinstance(payload, dict):
            return cls(
                workdir="",
                error="Dream candidate extraction response was not an object.",
            )
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            candidates = []

        artifact_ref_count = 0
        promotion_ready_count = 0
        extraction_sources: list[str] = []
        candidate_count = 0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_count += 1
            refs = candidate.get("artifact_refs")
            if isinstance(refs, list):
                artifact_ref_count += len(
                    [ref for ref in refs if isinstance(ref, str)]
                )
            if candidate.get("promotion_ready") is True:
                promotion_ready_count += 1
            source = candidate.get("extraction_source")
            if isinstance(source, str) and source:
                extraction_sources.append(source)

        error = payload.get("error")
        return cls(
            workdir=str(payload.get("workdir") or ""),
            candidate_count=candidate_count,
            artifact_ref_count=artifact_ref_count,
            promotion_ready_count=promotion_ready_count,
            extraction_sources=list(dict.fromkeys(extraction_sources)),
            error=error if isinstance(error, str) else None,
        )


class DreamAutoWriteResult(BaseModel):
    """Summary of artifact-driven Dream memory auto-write."""

    workdir: str
    written_count: int = 0
    written_memory_ids: list[str] = []
    rejected_count: int = 0
    blockers: list[str] = []
    error: str | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> DreamAutoWriteResult:
        if not isinstance(payload, dict):
            return cls(
                workdir="",
                error="Dream auto-write response was not an object.",
            )
        memory_ids = payload.get("written_memory_ids")
        blockers = payload.get("blockers")
        error = payload.get("error")
        return cls(
            workdir=str(payload.get("workdir") or ""),
            written_count=_as_int(payload.get("written_count")),
            written_memory_ids=[
                memory_id
                for memory_id in memory_ids
                if isinstance(memory_id, str) and memory_id
            ]
            if isinstance(memory_ids, list)
            else [],
            rejected_count=_as_int(payload.get("rejected_count")),
            blockers=[
                blocker for blocker in blockers if isinstance(blocker, str) and blocker
            ]
            if isinstance(blockers, list)
            else [],
            error=error if isinstance(error, str) else None,
        )


class ObserverCheckpointSample(BaseModel):
    """Controller-side non-invasive run sample."""

    timestamp: float
    elapsed_seconds: float
    context_tokens: int | None = None
    max_context_tokens: int | None = None
    timer_checkpoint_count: int = 0


class WireEventSizeSummary(BaseModel):
    """Aggregated wire-event size counters for context-overhead diagnosis."""

    message_kind: str
    count: int = 0
    total_json_bytes: int = 0
    max_json_bytes: int = 0
    total_payload_json_bytes: int = 0
    total_text_chars: int = 0
    total_think_chars: int = 0
    total_tool_argument_chars: int = 0
    total_tool_output_chars: int = 0


class SessionRunResult(BaseModel):
    """Observed metrics for one session arm."""

    arm_name: str = ""
    dream_enabled: bool = False
    session_id: str
    work_dir: str = ""
    status: str
    started_at: float
    ended_at: float
    duration_seconds: float
    assistant_text: str = ""
    input_prompt_chars: int = 0
    input_prompt_line_count: int = 0
    initial_dream_retrieval_chars: int = 0
    initial_dream_retrieval_render_mode: DreamRetrievalRenderMode | None = None
    controller_route_decision_path: str | None = None
    controller_route_decision_audit_path: str | None = None
    context_tokens: int | None = None
    max_context_tokens: int | None = None
    checkpoint_count: int = 0
    observer_checkpoint_count: int = 0
    observer_checkpoint_interval_seconds: int | None = None
    observer_checkpoint_samples: list[ObserverCheckpointSample] = []
    event_size_summaries: list[WireEventSizeSummary] = []
    tool_call_count: int = 0
    failed_tool_call_count: int = 0
    failed_tool_call_summaries: list[str] = []
    shell_call_count: int = 0
    shell_failure_count: int = 0
    benchmark_attempt_count: int = 0
    benchmark_failure_count: int = 0
    edit_command_count: int = 0
    candidate_metric_report_count: int = 0
    valid_speedup_count: int = 0
    failed_variant_count: int = 0
    first_valid_speedup_benchmark_index: int | None = None
    dream_chosen_delta: int = 0
    dream_useful_delta: int = 0
    dream_retrieval_count: int = 0
    dream_retrieved_memory_ids: list[str] = []
    dream_retrieval_events: list[DreamRetrievalEvent] = []
    dream_completion_candidate_count: int = 0
    dream_completion_artifact_ref_count: int = 0
    dream_completion_promotion_ready_count: int = 0
    dream_completion_extraction_sources: list[str] = []
    dream_completion_extraction_error: str | None = None
    dream_auto_write_count: int = 0
    dream_written_memory_ids: list[str] = []
    dream_auto_write_rejected_count: int = 0
    dream_auto_write_blockers: list[str] = []
    dream_auto_write_error: str | None = None
    error: str | None = None
    timed_out_process_cleanup_count: int = 0
    verification_command: str | None = None
    verification_status: str | None = None
    verification_exit_code: int | None = None
    verification_output: str = ""
    verification_error: str | None = None
    verification_artifact_path: str | None = None
    memory_snapshot_before_arm_path: str | None = None
    memory_snapshot_after_arm_path: str | None = None
    memory_restore_audit_path: str | None = None


class CampaignResult(BaseModel):
    """A paired campaign result suitable for JSON and markdown reporting."""

    name: str
    prompt: str
    work_dir: str
    started_at: float
    ended_at: float
    duration_seconds: float
    runs: list[SessionRunResult]
    memory_before: DreamMemoryStats
    memory_after: DreamMemoryStats
    environment_snapshot: dict[str, Any] = {}
    memory_snapshot_before: dict[str, Any] = {}
    memory_snapshot_after: dict[str, Any] = {}
    environment_json_path: str | None = None
    memory_snapshot_before_path: str | None = None
    memory_snapshot_after_path: str | None = None


class CampaignClient(Protocol):
    async def create_session(self, *, work_dir: str, create_dir: bool) -> str: ...

    async def delete_session(self, session_id: str) -> None: ...

    async def dream_memory_stats(self) -> DreamMemoryStats: ...

    async def dream_memory_snapshot(self) -> dict[str, Any]: ...

    async def restore_dream_memory_snapshot(self, snapshot: dict[str, Any]) -> None: ...

    async def record_dream_memory_feedback(
        self,
        *,
        memory_ids: list[str],
        useful: bool,
        reason: str | None,
        evidence_artifacts: list[str],
        source_session_id: str | None,
    ) -> None: ...

    async def retrieve_dream_memories(
        self,
        *,
        query: str,
        tags: list[str],
        categories: tuple[DreamMemoryCategory, ...],
        top_k_per_category: int,
        render_mode: DreamRetrievalRenderMode,
        memory_ids: list[str] | None = None,
    ) -> DreamRetrievalContext | None: ...

    async def extract_dream_memory_candidates(
        self,
        *,
        workdir: str,
        category_hint: DreamMemoryCategory | None = None,
    ) -> DreamCompletionExtraction: ...

    async def extract_and_write_dream_memories(
        self,
        *,
        workdir: str,
        category_hint: DreamMemoryCategory | None = None,
        dry_run: bool = False,
    ) -> DreamAutoWriteResult: ...

    async def run_prompt(
        self,
        *,
        session_id: str,
        prompt: str,
        dream_enabled: bool,
        yolo_enabled: bool,
        timer_interval_seconds: int | None,
        observer_interval_seconds: int | None,
        timeout_seconds: float,
    ) -> SessionRunResult: ...


class CampaignRunner:
    """Run a campaign over all configured arms using an injected client."""

    def __init__(self, client: CampaignClient) -> None:
        self._client = client

    async def run(self, spec: CampaignSpec, *, delete_sessions: bool = False) -> CampaignResult:
        started_at = time.time()
        environment_snapshot = _collect_environment_snapshot(spec.environment_command)
        memory_before = await self._client.dream_memory_stats()
        memory_snapshot_before = await self._client.dream_memory_snapshot()
        runs: list[SessionRunResult] = []

        for arm in spec.arms:
            arm_work_dir = spec.work_dir_for_arm(arm)
            self._prepare_arm_work_dir(spec, arm_work_dir)
            session_id = await self._client.create_session(
                work_dir=arm_work_dir,
                create_dir=spec.create_dir,
            )
            arm_memory_before = await self._client.dream_memory_stats()
            arm_memory_snapshot_before: dict[str, Any] | None = None
            arm_memory_snapshot_before_path: str | None = None
            arm_memory_restored = False
            if spec.restore_dream_memory_after_arm:
                arm_memory_snapshot_before = await self._client.dream_memory_snapshot()
            run: SessionRunResult | None = None
            try:
                prompt = spec.prompt
                initial_retrieval: DreamRetrievalContext | None = None
                initial_retrieval_chars = 0
                initial_retrieval_render_mode: DreamRetrievalRenderMode | None = None
                controller_route_decision_path: str | None = None
                controller_route_decision_audit_path: str | None = None
                if arm.dream_enabled and _arm_initial_dream_retrieval_enabled(spec, arm):
                    initial_retrieval_render_mode = _arm_dream_retrieval_render_mode(
                        spec,
                        arm,
                    )
                    initial_retrieval = await self._client.retrieve_dream_memories(
                        query=_arm_dream_retrieval_query(spec, arm),
                        tags=_arm_dream_retrieval_tags(spec, arm),
                        categories=_arm_dream_retrieval_categories(spec, arm),
                        top_k_per_category=_arm_dream_retrieval_top_k_per_category(
                            spec,
                            arm,
                        ),
                        render_mode=initial_retrieval_render_mode,
                        memory_ids=_arm_dream_retrieval_memory_ids(spec, arm),
                    )
                    if initial_retrieval and initial_retrieval.text:
                        initial_retrieval_chars = len(initial_retrieval.text)
                        if _is_route_policy_artifact_mode(initial_retrieval_render_mode):
                            controller_route_decision_path = (
                                _write_controller_route_decision_artifact(
                                    arm_work_dir,
                                    initial_retrieval.text,
                                )
                            )
                            controller_route_decision_audit_path = str(
                                Path(arm_work_dir)
                                / "controller_route_decision_audit.json"
                            )
                            prompt = _inject_controller_route_decision_context(
                                spec.prompt,
                                protocol=(
                                    initial_retrieval_render_mode
                                    == "route_policy_artifact_protocol"
                                ),
                            )
                        else:
                            prompt = _inject_initial_dream_retrieval_context(
                                spec.prompt,
                                initial_retrieval.text,
                            )
                run = await self._client.run_prompt(
                    session_id=session_id,
                    prompt=prompt,
                    dream_enabled=arm.dream_enabled,
                    yolo_enabled=spec.yolo_enabled,
                    timer_interval_seconds=spec.timer_interval_seconds,
                    observer_interval_seconds=spec.observer_interval_seconds,
                    timeout_seconds=spec.timeout_seconds,
                )
                run.arm_name = arm.name
                run.dream_enabled = arm.dream_enabled
                run.work_dir = arm_work_dir
                run.input_prompt_chars = len(prompt)
                run.input_prompt_line_count = _line_count(prompt)
                run.initial_dream_retrieval_chars = initial_retrieval_chars
                run.initial_dream_retrieval_render_mode = initial_retrieval_render_mode
                run.controller_route_decision_path = controller_route_decision_path
                run.controller_route_decision_audit_path = (
                    controller_route_decision_audit_path
                )
                run.memory_snapshot_before_arm_path = arm_memory_snapshot_before_path
                if initial_retrieval is not None:
                    _prepend_dream_retrieval_event(run, initial_retrieval.event)
                if spec.verifier_command:
                    await _run_verifier_for_arm(
                        run,
                        command=spec.verifier_command,
                        cwd=arm_work_dir,
                        timeout_seconds=spec.verifier_timeout_seconds,
                    )
                if arm.dream_enabled and spec.active_dream_protocol:
                    await self._extract_dream_completion_candidates(
                        run,
                        workdir=arm_work_dir,
                    )
                arm_memory_after = await self._client.dream_memory_stats()
                run.dream_chosen_delta = (
                    arm_memory_after.total_chosen - arm_memory_before.total_chosen
                )
                if arm_memory_snapshot_before is not None:
                    await self._restore_dream_memory_after_arm(
                        arm_work_dir=arm_work_dir,
                        snapshot_before=arm_memory_snapshot_before,
                        snapshot_before_path=arm_memory_snapshot_before_path,
                        run=run,
                    )
                    arm_memory_restored = True
                runs.append(run)
            finally:
                if arm_memory_snapshot_before is not None and not arm_memory_restored:
                    try:
                        await self._restore_dream_memory_after_arm(
                            arm_work_dir=arm_work_dir,
                            snapshot_before=arm_memory_snapshot_before,
                            snapshot_before_path=arm_memory_snapshot_before_path,
                            run=run,
                        )
                    except Exception as exc:
                        if run is None:
                            raise
                        cleanup_error = f"Dream memory restore failed: {exc}"
                        run.error = (
                            f"{run.error}\n{cleanup_error}" if run.error else cleanup_error
                        )
                if delete_sessions:
                    try:
                        await self._client.delete_session(session_id)
                    except Exception as exc:
                        if run is not None:
                            cleanup_error = f"Session cleanup failed: {exc}"
                            run.error = (
                                f"{run.error}\n{cleanup_error}"
                                if run.error
                                else cleanup_error
                            )
                if run is not None and run.status == "timeout":
                    run.timed_out_process_cleanup_count = _terminate_processes_in_work_dir(
                        arm_work_dir
                    )

        if not spec.restore_dream_memory_after_arm:
            await self._record_dream_feedback_for_runs(runs)
        memory_after = await self._client.dream_memory_stats()
        memory_snapshot_after = await self._client.dream_memory_snapshot()
        ended_at = time.time()
        return CampaignResult(
            name=spec.name,
            prompt=spec.prompt,
            work_dir=spec.work_dir,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=ended_at - started_at,
            runs=runs,
            memory_before=memory_before,
            memory_after=memory_after,
            environment_snapshot=environment_snapshot,
            memory_snapshot_before=memory_snapshot_before,
            memory_snapshot_after=memory_snapshot_after,
        )

    async def _restore_dream_memory_after_arm(
        self,
        *,
        arm_work_dir: str,
        snapshot_before: dict[str, Any],
        snapshot_before_path: str | None,
        run: SessionRunResult | None,
    ) -> None:
        if snapshot_before_path is None:
            snapshot_before_path = _write_arm_memory_artifact(
                arm_work_dir,
                "memory_snapshot_before_arm.json",
                snapshot_before,
            )
        snapshot_after = await self._client.dream_memory_snapshot()
        snapshot_after_path = _write_arm_memory_artifact(
            arm_work_dir,
            "memory_snapshot_after_arm.json",
            snapshot_after,
        )
        await self._client.restore_dream_memory_snapshot(snapshot_before)
        restored_snapshot = await self._client.dream_memory_snapshot()
        audit = {
            "restore_enabled": True,
            "reason": "A/B isolation",
            "restored_at": time.time(),
            "snapshot_before": snapshot_before_path,
            "snapshot_after_arm": snapshot_after_path,
            "before_count": _memory_snapshot_count(snapshot_before),
            "after_count": _memory_snapshot_count(snapshot_after),
            "restored_count": _memory_snapshot_count(restored_snapshot),
        }
        audit_path = _write_arm_memory_artifact(
            arm_work_dir,
            "memory_restore_audit.json",
            audit,
        )
        if run is not None:
            run.memory_snapshot_before_arm_path = snapshot_before_path
            run.memory_snapshot_after_arm_path = snapshot_after_path
            run.memory_restore_audit_path = audit_path

    @staticmethod
    def _prepare_arm_work_dir(spec: CampaignSpec, arm_work_dir: str) -> None:
        if spec.seed_dir is None:
            return
        seed_dir = Path(spec.seed_dir)
        if not seed_dir.is_dir():
            raise NotADirectoryError(f"Seed directory does not exist: {seed_dir}")

        target = Path(arm_work_dir)
        if target.exists():
            has_contents = any(target.iterdir()) if target.is_dir() else True
            if has_contents and not spec.reset_work_dirs:
                raise FileExistsError(
                    f"Refusing to mix seed project into non-empty workdir: {target}. "
                    "Use --reset-work-dirs for a fresh campaign run."
                )
            if spec.reset_work_dirs:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            elif target.is_dir():
                target.rmdir()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_dir, target)

    def build_result_from_runs(
        self,
        *,
        spec: CampaignSpec,
        runs: list[SessionRunResult],
    ) -> CampaignResult:
        started_at = min((run.started_at for run in runs), default=time.time())
        ended_at = max((run.ended_at for run in runs), default=started_at)
        return CampaignResult(
            name=spec.name,
            prompt=spec.prompt,
            work_dir=spec.work_dir,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=ended_at - started_at,
            runs=runs,
            memory_before=DreamMemoryStats(),
            memory_after=DreamMemoryStats(),
            environment_snapshot=_collect_environment_snapshot(spec.environment_command),
            memory_snapshot_before={"memories": []},
            memory_snapshot_after={"memories": []},
        )

    async def _record_dream_feedback_for_runs(self, runs: list[SessionRunResult]) -> None:
        comparison_runs = [run for run in runs if not run.dream_enabled]
        for run in runs:
            if not run.dream_enabled or not run.dream_retrieved_memory_ids:
                continue
            useful, decision_reason = _decide_dream_feedback_usefulness(
                run,
                comparison_runs=comparison_runs,
            )
            evidence_artifacts = _dream_feedback_evidence_artifacts(run)
            if useful and not evidence_artifacts:
                useful = False
                decision_reason = (
                    decision_reason
                    + " Feedback was not marked useful because no evidence artifacts "
                    "were present for the run."
                )
            feedback_before = await self._client.dream_memory_stats()
            await self._client.record_dream_memory_feedback(
                memory_ids=run.dream_retrieved_memory_ids,
                useful=useful,
                reason=_build_dream_feedback_reason(run, decision_reason=decision_reason),
                evidence_artifacts=evidence_artifacts,
                source_session_id=run.session_id,
            )
            feedback_after = await self._client.dream_memory_stats()
            run.dream_useful_delta = (
                feedback_after.total_useful_when_chosen
                - feedback_before.total_useful_when_chosen
            )

    async def _extract_dream_completion_candidates(
        self,
        run: SessionRunResult,
        *,
        workdir: str,
    ) -> None:
        try:
            extraction = await self._client.extract_dream_memory_candidates(
                workdir=workdir,
                category_hint=None,
            )
        except Exception as exc:
            extraction = DreamCompletionExtraction(workdir=workdir, error=str(exc))
        _apply_dream_completion_extraction(run, extraction)
        try:
            auto_write = await self._client.extract_and_write_dream_memories(
                workdir=workdir,
                category_hint=None,
                dry_run=False,
            )
        except Exception as exc:
            auto_write = DreamAutoWriteResult(workdir=workdir, error=str(exc))
        _apply_dream_auto_write_result(run, auto_write)


class SessionEventRecorder:
    """Collect useful metrics from GUI WebSocket messages."""

    def __init__(self, *, prompt_id: str) -> None:
        self.prompt_id = prompt_id
        self._text_parts: list[str] = []
        self.context_tokens: int | None = None
        self.max_context_tokens: int | None = None
        self.checkpoint_count = 0
        self.tool_call_count = 0
        self.failed_tool_call_count = 0
        self.failed_tool_call_summaries: list[str] = []
        self.shell_call_count = 0
        self.shell_failure_count = 0
        self.benchmark_attempt_count = 0
        self.benchmark_failure_count = 0
        self.edit_command_count = 0
        self.candidate_metric_report_count = 0
        self.valid_speedup_count = 0
        self.failed_variant_count = 0
        self.first_valid_speedup_benchmark_index: int | None = None
        self.dream_retrieval_events: list[DreamRetrievalEvent] = []
        self.finished_status: str | None = None
        self.error: str | None = None
        self._current_tool_call_id: str | None = None
        self._tool_arguments_by_id: dict[str, str] = {}
        self._tool_commands_by_id: dict[str, str] = {}
        self._tool_names_by_id: dict[str, str] = {}
        self._shell_tool_ids: set[str] = set()
        self._benchmark_tool_ids: set[str] = set()
        self._edit_tool_ids: set[str] = set()
        self._event_size_summaries: dict[str, WireEventSizeSummary] = {}

    @property
    def assistant_text(self) -> str:
        return "".join(self._text_parts)

    @property
    def dream_retrieval_count(self) -> int:
        return len(self.dream_retrieval_events)

    @property
    def dream_retrieved_memory_ids(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for event in self.dream_retrieval_events:
            for memory_id in event.memory_ids:
                if memory_id in seen:
                    continue
                seen.add(memory_id)
                ordered.append(memory_id)
        return ordered

    @property
    def inferred_finished_status(self) -> str | None:
        return "finished" if _looks_like_completed_campaign_text(self.assistant_text) else None

    @property
    def event_size_summaries(self) -> list[WireEventSizeSummary]:
        return [
            self._event_size_summaries[key]
            for key in sorted(self._event_size_summaries)
        ]

    def record(self, message: dict[str, Any]) -> None:
        self._record_event_size(message)

        if message.get("id") == self.prompt_id and isinstance(message.get("result"), dict):
            status = message["result"].get("status")
            if isinstance(status, str):
                self.finished_status = status

        error = message.get("error")
        if isinstance(error, dict):
            self.error = str(error.get("message") or error)

        if message.get("method") == "session_status":
            params = message.get("params")
            if isinstance(params, dict) and params.get("state") == "error":
                reason = params.get("reason")
                detail = params.get("detail")
                error_detail = detail or reason or "session entered error state"
                self.error = f"session_status error: {error_detail}"
            return

        if message.get("method") != "event":
            return

        params = message.get("params")
        if not isinstance(params, dict):
            return
        event_type = params.get("type")
        payload = params.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        if event_type == "ContentPart" and payload.get("type") == "text":
            text = payload.get("text")
            if isinstance(text, str):
                self._text_parts.append(text)
            return

        if event_type == "DreamMemoryRetrieval":
            self.dream_retrieval_events.append(_parse_dream_retrieval_event(payload))
            return

        if event_type == "ToolCall":
            self._record_tool_call(payload)
            return

        if event_type == "ToolCallPart":
            self._record_tool_call_part(payload)
            return

        if event_type == "ToolResult":
            self._record_tool_result(payload)
            return

        if event_type != "StatusUpdate":
            return

        context_tokens = _as_optional_int(payload.get("context_tokens"))
        if context_tokens is None:
            context_tokens = _as_optional_int(payload.get("contextTokens"))
        if context_tokens is not None:
            self.context_tokens = context_tokens

        max_context_tokens = _as_optional_int(payload.get("max_context_tokens"))
        if max_context_tokens is None:
            max_context_tokens = _as_optional_int(payload.get("maxContextTokens"))
        if max_context_tokens is not None:
            self.max_context_tokens = max_context_tokens

        timer = payload.get("checkpoint_timer")
        if timer is None:
            timer = payload.get("checkpointTimer")
        if isinstance(timer, dict):
            self.checkpoint_count = max(self.checkpoint_count, _as_int(timer.get("run_count")))

    def _record_event_size(self, message: dict[str, Any]) -> None:
        message_kind = _message_size_kind(message)
        summary = self._event_size_summaries.setdefault(
            message_kind,
            WireEventSizeSummary(message_kind=message_kind),
        )
        json_bytes = _stable_json_size_bytes(message)
        summary.count += 1
        summary.total_json_bytes += json_bytes
        summary.max_json_bytes = max(summary.max_json_bytes, json_bytes)

        params = message.get("params")
        if not isinstance(params, dict):
            return
        payload = params.get("payload")
        if not isinstance(payload, dict):
            return
        summary.total_payload_json_bytes += _stable_json_size_bytes(payload)

        event_type = params.get("type")
        if event_type == "ContentPart":
            text = payload.get("text")
            if isinstance(text, str):
                summary.total_text_chars += len(text)
            think = payload.get("think")
            if isinstance(think, str):
                summary.total_think_chars += len(think)
            return

        if event_type == "ToolCall":
            function = payload.get("function")
            if isinstance(function, dict):
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    summary.total_tool_argument_chars += len(arguments)
            return

        if event_type == "ToolCallPart":
            arguments_part = payload.get("arguments_part")
            if isinstance(arguments_part, str):
                summary.total_tool_argument_chars += len(arguments_part)
            return

        if event_type == "ToolResult":
            return_value = payload.get("return_value")
            if isinstance(return_value, dict):
                output = return_value.get("output")
                if isinstance(output, str):
                    summary.total_tool_output_chars += len(output)

    def _record_tool_call(self, payload: dict[str, Any]) -> None:
        self.tool_call_count += 1
        tool_call_id = payload.get("id")
        if not isinstance(tool_call_id, str):
            return
        self._current_tool_call_id = tool_call_id

        function = payload.get("function")
        if not isinstance(function, dict):
            return
        name = function.get("name")
        if isinstance(name, str):
            self._tool_names_by_id[tool_call_id] = name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            self._tool_arguments_by_id[tool_call_id] = arguments
        command = _extract_shell_command_from_tool_call(function)
        if command:
            self._tool_commands_by_id[tool_call_id] = command

        if name == "Shell":
            self.shell_call_count += 1
            self._shell_tool_ids.add(tool_call_id)
            self._record_shell_command_metrics(tool_call_id, command)

    def _record_tool_call_part(self, payload: dict[str, Any]) -> None:
        tool_call_id = payload.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            tool_call_id = self._current_tool_call_id
        if not isinstance(tool_call_id, str):
            return

        arguments_part = payload.get("arguments_part")
        if not isinstance(arguments_part, str):
            return

        arguments = self._tool_arguments_by_id.get(tool_call_id, "") + arguments_part
        self._tool_arguments_by_id[tool_call_id] = arguments
        command = _extract_shell_command_from_arguments(arguments)
        if command:
            self._tool_commands_by_id[tool_call_id] = command
            self._record_shell_command_metrics(tool_call_id, command)

    def _record_shell_command_metrics(self, tool_call_id: str, command: str) -> None:
        if tool_call_id not in self._shell_tool_ids or not command:
            return
        if _is_benchmark_command(command) and tool_call_id not in self._benchmark_tool_ids:
            self.benchmark_attempt_count += 1
            self._benchmark_tool_ids.add(tool_call_id)
        if _is_edit_command(command) and tool_call_id not in self._edit_tool_ids:
            self.edit_command_count += 1
            self._edit_tool_ids.add(tool_call_id)

    def _record_tool_result(self, payload: dict[str, Any]) -> None:
        tool_call_id = payload.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            return

        return_value = payload.get("return_value")
        if not isinstance(return_value, dict):
            return
        is_error = bool(return_value.get("is_error"))
        output = str(return_value.get("output") or "")

        if is_error:
            self.failed_tool_call_count += 1
            self.failed_tool_call_summaries.append(
                _failed_tool_call_summary(
                    name=self._tool_names_by_id.get(tool_call_id),
                    command=self._tool_commands_by_id.get(tool_call_id),
                    output=output,
                )
            )
            if tool_call_id in self._shell_tool_ids:
                self.shell_failure_count += 1

        if tool_call_id not in self._benchmark_tool_ids:
            return

        metrics = _extract_benchmark_result_metrics(output)
        if metrics:
            self.candidate_metric_report_count += 1
        valid_speedup = _is_valid_speedup_metrics(metrics)
        failed_variant = is_error or _is_failed_variant_metrics(metrics, output)
        if failed_variant:
            self.benchmark_failure_count += 1
            self.failed_variant_count += 1
        if valid_speedup:
            self.valid_speedup_count += 1
            if self.first_valid_speedup_benchmark_index is None:
                self.first_valid_speedup_benchmark_index = self.candidate_metric_report_count


class _ObserverCheckpointSampler:
    """Non-invasive controller-side checkpoint sampler for campaign runs."""

    def __init__(self, *, interval_seconds: int | None, started_at: float) -> None:
        self.interval_seconds = (
            interval_seconds if interval_seconds is not None and interval_seconds > 0 else None
        )
        self._started_at = started_at
        self._next_sample_at = (
            started_at + self.interval_seconds if self.interval_seconds is not None else None
        )
        self.samples: list[ObserverCheckpointSample] = []

    @property
    def enabled(self) -> bool:
        return self.interval_seconds is not None

    @property
    def count(self) -> int:
        return len(self.samples)

    def seconds_until_next(self, now: float) -> float:
        if self._next_sample_at is None:
            return 1.0
        return max(0.1, self._next_sample_at - now)

    def record_due(self, now: float, recorder: SessionEventRecorder) -> None:
        if self._next_sample_at is None or self.interval_seconds is None:
            return
        while self._next_sample_at <= now:
            self.samples.append(
                ObserverCheckpointSample(
                    timestamp=self._next_sample_at,
                    elapsed_seconds=self._next_sample_at - self._started_at,
                    context_tokens=recorder.context_tokens,
                    max_context_tokens=recorder.max_context_tokens,
                    timer_checkpoint_count=recorder.checkpoint_count,
                )
            )
            self._next_sample_at += self.interval_seconds


class EvoInferWebClient:
    """Thin client for running campaign arms against a live EvoInfer GUI backend."""

    def __init__(self, *, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._headers = {"Authorization": f"Bearer {token}"}

    async def create_session(self, *, work_dir: str, create_dir: bool) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/sessions/",
                headers=self._headers,
                json={"work_dir": work_dir, "create_dir": create_dir},
            )
            response.raise_for_status()
            session_id = response.json().get("session_id")
            if not isinstance(session_id, str):
                raise RuntimeError("Session creation response did not include session_id.")
            return session_id

    async def delete_session(self, session_id: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{self.base_url}/api/sessions/{session_id}",
                headers=self._headers,
            )
            response.raise_for_status()

    async def dream_memory_stats(self) -> DreamMemoryStats:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/api/dream/memories",
                headers=self._headers,
            )
            response.raise_for_status()
            return DreamMemoryStats.from_payload(response.json())

    async def dream_memory_snapshot(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/api/dream/memories",
                headers=self._headers,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"memories": payload}

    async def restore_dream_memory_snapshot(self, snapshot: dict[str, Any]) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(
                f"{self.base_url}/api/dream/memories/snapshot",
                headers=self._headers,
                json=snapshot,
            )
            response.raise_for_status()

    async def record_dream_memory_feedback(
        self,
        *,
        memory_ids: list[str],
        useful: bool,
        reason: str | None,
        evidence_artifacts: list[str],
        source_session_id: str | None,
    ) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/dream/memories/feedback",
                headers=self._headers,
                json={
                    "memory_ids": memory_ids,
                    "useful": useful,
                    "reason": reason,
                    "evidence_artifacts": evidence_artifacts,
                    "source_session_id": source_session_id,
                },
            )
            response.raise_for_status()

    async def retrieve_dream_memories(
        self,
        *,
        query: str,
        tags: list[str],
        categories: tuple[DreamMemoryCategory, ...],
        top_k_per_category: int,
        render_mode: DreamRetrievalRenderMode,
        memory_ids: list[str] | None = None,
    ) -> DreamRetrievalContext | None:
        query = query.strip()
        if not query:
            return None

        results: list[dict[str, Any]] = []
        pinned_memory_ids = list(dict.fromkeys(memory_ids or []))
        async with httpx.AsyncClient(timeout=30.0) as client:
            for category in categories:
                response = await client.post(
                    f"{self.base_url}/api/dream/memories/search",
                    headers=self._headers,
                    json={
                        "query": query,
                        "category": category,
                        "tags": tags,
                        "top_k": top_k_per_category,
                        "record_choice": True,
                        "memory_ids": pinned_memory_ids,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                category_results = payload.get("results")
                if isinstance(category_results, list):
                    results.extend(item for item in category_results if isinstance(item, dict))

        results.sort(key=lambda item: _search_result_score(item), reverse=True)
        memory_ids = _memory_ids_from_search_results(results)
        event = DreamRetrievalEvent(
            trigger="campaign_start",
            query=query,
            categories=list(categories),
            top_k_per_category=top_k_per_category,
            memory_ids=memory_ids,
            result_count=len(results),
            step_count=0,
        )
        return DreamRetrievalContext(
            text=_render_dream_search_results_for_prompt(
                results,
                render_mode=render_mode,
            ),
            event=event,
        )

    async def extract_dream_memory_candidates(
        self,
        *,
        workdir: str,
        category_hint: DreamMemoryCategory | None = None,
    ) -> DreamCompletionExtraction:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/dream/memories/extract-candidates",
                headers=self._headers,
                json={"workdir": workdir, "category_hint": category_hint},
            )
            response.raise_for_status()
            return DreamCompletionExtraction.from_payload(response.json())

    async def extract_and_write_dream_memories(
        self,
        *,
        workdir: str,
        category_hint: DreamMemoryCategory | None = None,
        dry_run: bool = False,
    ) -> DreamAutoWriteResult:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/dream/memories/extract-and-write",
                headers=self._headers,
                json={
                    "workdir": workdir,
                    "category_hint": category_hint,
                    "dry_run": dry_run,
                },
            )
            response.raise_for_status()
            return DreamAutoWriteResult.from_payload(response.json())

    async def run_prompt(
        self,
        *,
        session_id: str,
        prompt: str,
        dream_enabled: bool,
        yolo_enabled: bool,
        timer_interval_seconds: int | None,
        observer_interval_seconds: int | None,
        timeout_seconds: float,
    ) -> SessionRunResult:
        started_at = time.time()
        prompt_id = str(uuid4())
        recorder = SessionEventRecorder(prompt_id=prompt_id)
        observer = _ObserverCheckpointSampler(
            interval_seconds=observer_interval_seconds,
            started_at=started_at,
        )
        ws_url = self._websocket_url(session_id)

        async with websockets.connect(ws_url, ping_interval=None, max_size=20_000_000) as ws:
            await ws.send(json.dumps(build_initialize_message(str(uuid4()))))
            await self._wait_for_response(ws, timeout_seconds=timeout_seconds)

            if yolo_enabled:
                yolo_id = str(uuid4())
                await ws.send(json.dumps(build_set_yolo_mode_message(yolo_id, enabled=True)))
                await self._wait_for_response(
                    ws,
                    expected_id=yolo_id,
                    timeout_seconds=timeout_seconds,
                )

            dream_id = str(uuid4())
            await ws.send(json.dumps(build_set_dream_mode_message(dream_id, enabled=dream_enabled)))
            await self._wait_for_response(ws, expected_id=dream_id, timeout_seconds=timeout_seconds)

            timer_started = False
            prompt_started = False
            deadline = time.time()
            try:
                if timer_interval_seconds is not None:
                    timer_id = str(uuid4())
                    await ws.send(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "method": "prompt",
                                "id": timer_id,
                                "params": {"user_input": f"/timer start {timer_interval_seconds}"},
                            }
                        )
                    )
                    await self._wait_for_response(
                        ws,
                        expected_id=timer_id,
                        timeout_seconds=timeout_seconds,
                    )
                    timer_started = True

                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "prompt",
                            "id": prompt_id,
                            "params": {"user_input": prompt},
                        },
                        ensure_ascii=False,
                    )
                )

                prompt_started = True
                deadline = time.time() + timeout_seconds
                while time.time() < deadline:
                    now = time.time()
                    recv_timeout = max(0.1, deadline - now)
                    if observer.enabled:
                        recv_timeout = min(recv_timeout, observer.seconds_until_next(now))
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=recv_timeout,
                        )
                    except TimeoutError:
                        if observer.enabled:
                            observer.record_due(time.time(), recorder)
                        if time.time() >= deadline:
                            break
                        continue
                    message = _decode_json_object(raw)
                    recorder.record(message)
                    if observer.enabled:
                        observer.record_due(time.time(), recorder)
                    if (
                        recorder.error
                        or recorder.finished_status is not None
                        or recorder.inferred_finished_status is not None
                    ):
                        break
            finally:
                if observer.enabled:
                    observer.record_due(time.time(), recorder)
                if (
                    prompt_started
                    and recorder.error is None
                    and recorder.finished_status is None
                    and recorder.inferred_finished_status is None
                    and time.time() >= deadline
                ):
                    await self._cancel_active_prompt(ws, timeout_seconds=timeout_seconds)
                if timer_started:
                    await self._stop_checkpoint_timer(ws, timeout_seconds=timeout_seconds)

        ended_at = time.time()
        return SessionRunResult(
            session_id=session_id,
            status=recorder.finished_status
            or recorder.inferred_finished_status
            or ("error" if recorder.error else "timeout"),
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=ended_at - started_at,
            assistant_text=recorder.assistant_text,
            context_tokens=recorder.context_tokens,
            max_context_tokens=recorder.max_context_tokens,
            checkpoint_count=recorder.checkpoint_count,
            observer_checkpoint_count=observer.count,
            observer_checkpoint_interval_seconds=observer.interval_seconds,
            observer_checkpoint_samples=observer.samples,
            tool_call_count=recorder.tool_call_count,
            failed_tool_call_count=recorder.failed_tool_call_count,
            failed_tool_call_summaries=recorder.failed_tool_call_summaries,
            shell_call_count=recorder.shell_call_count,
            shell_failure_count=recorder.shell_failure_count,
            benchmark_attempt_count=recorder.benchmark_attempt_count,
            benchmark_failure_count=recorder.benchmark_failure_count,
            edit_command_count=recorder.edit_command_count,
            candidate_metric_report_count=recorder.candidate_metric_report_count,
            valid_speedup_count=recorder.valid_speedup_count,
            failed_variant_count=recorder.failed_variant_count,
            first_valid_speedup_benchmark_index=recorder.first_valid_speedup_benchmark_index,
            dream_retrieval_count=recorder.dream_retrieval_count,
            dream_retrieved_memory_ids=recorder.dream_retrieved_memory_ids,
            dream_retrieval_events=recorder.dream_retrieval_events,
            event_size_summaries=recorder.event_size_summaries,
            error=recorder.error,
        )

    async def _cancel_active_prompt(
        self,
        ws: websockets.ClientConnection,
        *,
        timeout_seconds: float,
    ) -> None:
        cancel_id = str(uuid4())
        try:
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "cancel",
                        "id": cancel_id,
                    }
                )
            )
            await self._wait_for_response(
                ws,
                expected_id=cancel_id,
                timeout_seconds=min(10.0, max(1.0, timeout_seconds)),
            )
        except Exception:
            # Best-effort cleanup. The result must remain a timeout even if cancellation
            # races with a finishing turn or the socket is already closed.
            return

    def _websocket_url(self, session_id: str) -> str:
        if self.base_url.startswith("https://"):
            ws_base = "wss://" + self.base_url[len("https://") :]
        elif self.base_url.startswith("http://"):
            ws_base = "ws://" + self.base_url[len("http://") :]
        else:
            raise ValueError("base_url must start with http:// or https://")
        return f"{ws_base}/api/sessions/{session_id}/stream?token={self.token}"

    async def _stop_checkpoint_timer(
        self,
        ws: websockets.ClientConnection,
        *,
        timeout_seconds: float,
    ) -> None:
        stop_id = str(uuid4())
        try:
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "prompt",
                        "id": stop_id,
                        "params": {"user_input": "/timer stop"},
                    }
                )
            )
            await self._wait_for_response(
                ws,
                expected_id=stop_id,
                timeout_seconds=min(10.0, max(1.0, timeout_seconds)),
            )
        except Exception:
            # Best-effort cleanup. Preserve the original run result if stopping the timer fails.
            return

    @staticmethod
    async def _wait_for_response(
        ws: websockets.ClientConnection,
        *,
        expected_id: str | None = None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, deadline - time.time()))
            message = _decode_json_object(raw)
            if message.get("error"):
                raise RuntimeError(str(message["error"]))
            if expected_id is None and message.get("id"):
                return message
            if expected_id is not None and message.get("id") == expected_id:
                return message
        raise TimeoutError("Timed out waiting for WebSocket response.")


def build_initialize_message(message_id: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": message_id,
        "params": {
            "protocol_version": "1.9",
            "client": {"name": "evoinfer-campaign", "version": "1"},
            "capabilities": {
                "supports_question": True,
                "supports_plan_mode": True,
                "supports_dream_mode": True,
            },
        },
    }


def build_set_dream_mode_message(message_id: str, *, enabled: bool) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "set_dream_mode",
        "id": message_id,
        "params": {"enabled": enabled},
    }


def build_set_yolo_mode_message(message_id: str, *, enabled: bool) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "set_yolo_mode",
        "id": message_id,
        "params": {"enabled": enabled},
    }


def render_campaign_markdown(result: CampaignResult) -> str:
    lines = [
        f"# EvoInfer Campaign: {result.name}",
        "",
        "## Prompt",
        "",
        result.prompt,
        "",
        "## Summary",
        "",
        (
            "| arm | dream | status | duration_s | context_tokens | checkpoints | "
            "dream_chosen_delta | dream_useful_delta | dream_retrievals | dream_memory_ids |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in result.runs:
        memory_ids = ", ".join(f"`{memory_id}`" for memory_id in run.dream_retrieved_memory_ids)
        lines.append(
            "| "
            f"{run.arm_name} | "
            f"{'on' if run.dream_enabled else 'off'} | "
            f"{run.status} | "
            f"{run.duration_seconds:.3f} | "
            f"{run.context_tokens if run.context_tokens is not None else ''} | "
            f"{run.checkpoint_count} | "
            f"{run.dream_chosen_delta} | "
            f"{run.dream_useful_delta} | "
            f"{run.dream_retrieval_count} | "
            f"{memory_ids} |"
        )
    lines.extend(["", "## Work Directories", ""])
    for run in result.runs:
        if run.work_dir:
            lines.append(f"- {run.arm_name}: `{run.work_dir}`")

    if any(
        run.input_prompt_chars
        or run.initial_dream_retrieval_chars
        or run.initial_dream_retrieval_render_mode
        or run.controller_route_decision_path
        for run in result.runs
    ):
        lines.extend(
            [
                "",
                "## Prompt Diagnostics",
                "",
                (
                    "| arm | prompt_chars | prompt_lines | retrieval_chars | "
                    "render_mode | route_decision | route_audit |"
                ),
                "| --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for run in result.runs:
            route_decision = (
                f"`{run.controller_route_decision_path}`"
                if run.controller_route_decision_path
                else ""
            )
            route_audit = (
                f"`{run.controller_route_decision_audit_path}`"
                if run.controller_route_decision_audit_path
                else ""
            )
            lines.append(
                "| "
                f"{run.arm_name} | "
                f"{run.input_prompt_chars} | "
                f"{run.input_prompt_line_count} | "
                f"{run.initial_dream_retrieval_chars} | "
                f"{run.initial_dream_retrieval_render_mode or ''} | "
                f"{route_decision} | "
                f"{route_audit} |"
            )

    if any(run.observer_checkpoint_count for run in result.runs):
        lines.extend(
            [
                "",
                "## Controller Observer Checkpoints",
                "",
                "| arm | observer_checkpoints | interval_s | last_context_tokens |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for run in result.runs:
            last_sample = (
                run.observer_checkpoint_samples[-1]
                if run.observer_checkpoint_samples
                else None
            )
            lines.append(
                "| "
                f"{run.arm_name} | "
                f"{run.observer_checkpoint_count} | "
                f"{run.observer_checkpoint_interval_seconds if run.observer_checkpoint_interval_seconds is not None else ''} | "
                f"{last_sample.context_tokens if last_sample and last_sample.context_tokens is not None else ''} |"
            )

    if any(run.event_size_summaries for run in result.runs):
        lines.extend(
            [
                "",
                "## Wire Event Size Diagnostics",
                "",
                (
                    "| arm | message_kind | count | json_bytes | max_json_bytes | "
                    "payload_json_bytes | text_chars | think_chars | tool_argument_chars | "
                    "tool_output_chars |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in result.runs:
            for summary in run.event_size_summaries:
                lines.append(
                    "| "
                    f"{run.arm_name} | "
                    f"{summary.message_kind} | "
                    f"{summary.count} | "
                    f"{summary.total_json_bytes} | "
                    f"{summary.max_json_bytes} | "
                    f"{summary.total_payload_json_bytes} | "
                    f"{summary.total_text_chars} | "
                    f"{summary.total_think_chars} | "
                    f"{summary.total_tool_argument_chars} | "
                    f"{summary.total_tool_output_chars} |"
                )

    if any(run.tool_call_count for run in result.runs):
        lines.extend(
            [
                "",
                "## Path Quality",
                "",
                (
                    "| arm | tool_calls | failed_tools | shell_calls | shell_failures | "
                    "benchmarks | benchmark_failures | edit_commands | metric_reports | "
                    "valid_speedups | failed_variants | first_valid_speedup_idx |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in result.runs:
            lines.append(
                "| "
                f"{run.arm_name} | "
                f"{run.tool_call_count} | "
                f"{run.failed_tool_call_count} | "
                f"{run.shell_call_count} | "
                f"{run.shell_failure_count} | "
                f"{run.benchmark_attempt_count} | "
                f"{run.benchmark_failure_count} | "
                f"{run.edit_command_count} | "
                f"{run.candidate_metric_report_count} | "
                f"{run.valid_speedup_count} | "
                f"{run.failed_variant_count} | "
                f"{run.first_valid_speedup_benchmark_index if run.first_valid_speedup_benchmark_index is not None else ''} |"
            )

    if any(run.verification_command for run in result.runs):
        lines.extend(["", "## Verification", ""])
        for run in result.runs:
            if not run.verification_command:
                continue
            lines.extend(
                [
                    f"### {run.arm_name}",
                    "",
                    f"- Command: `{run.verification_command}`",
                    f"- Status: {run.verification_status or ''}",
                    f"- Exit code: {run.verification_exit_code if run.verification_exit_code is not None else ''}",
                ]
            )
            if run.verification_error:
                lines.append(f"- Error: {run.verification_error}")
            if run.verification_output:
                lines.extend(
                    [
                        "",
                        "```text",
                        _truncate_markdown_block(run.verification_output),
                        "```",
                    ]
                )
    lines.extend(
        [
            "",
            "## Dream Memory Counters",
            "",
            f"- Before chosen: {result.memory_before.total_chosen}",
            f"- After chosen: {result.memory_after.total_chosen}",
            f"- Before useful_when_chosen: {result.memory_before.total_useful_when_chosen}",
            f"- After useful_when_chosen: {result.memory_after.total_useful_when_chosen}",
        ]
    )
    return "\n".join(lines) + "\n"


async def run_campaign_from_cli(args: argparse.Namespace) -> CampaignResult:
    spec = CampaignSpec(
        name=args.name,
        prompt=args.prompt,
        work_dir=args.work_dir,
        create_dir=args.create_dir,
        timer_interval_seconds=args.timer_interval_seconds,
        observer_interval_seconds=args.observer_interval_seconds,
        timeout_seconds=args.timeout_seconds,
        isolate_work_dirs=not args.shared_work_dir,
        seed_dir=args.seed_dir,
        reset_work_dirs=args.reset_work_dirs,
        verifier_command=args.verifier_command,
        verifier_timeout_seconds=args.verifier_timeout_seconds,
        yolo_enabled=args.yolo,
        initial_dream_retrieval=(
            args.initial_dream_retrieval and not args.disable_initial_dream_retrieval
        ),
        dream_retrieval_query=args.dream_retrieval_query,
        dream_retrieval_tags=args.dream_retrieval_tag,
        dream_retrieval_categories=normalize_dream_retrieval_categories(
            args.dream_retrieval_category
        ),
        dream_retrieval_memory_ids=args.dream_retrieval_memory_id,
        dream_retrieval_top_k_per_category=args.dream_retrieval_top_k_per_category,
        dream_retrieval_render_mode=args.dream_retrieval_render_mode,
        active_dream_protocol=args.active_dream_protocol,
        environment_command=args.environment_command,
        arms=_parse_campaign_arms(args.arm),
    )
    client = EvoInferWebClient(base_url=args.base_url, token=args.token)
    return await CampaignRunner(client).run(spec, delete_sessions=args.delete_sessions)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5494")
    parser.add_argument("--token", default="evoinfer-local")
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--create-dir", action="store_true")
    parser.add_argument("--seed-dir")
    parser.add_argument(
        "--reset-work-dirs",
        action="store_true",
        help="Delete each arm work directory before copying --seed-dir.",
    )
    parser.add_argument(
        "--shared-work-dir",
        action="store_true",
        help="Use the exact same work directory for all arms. Default is isolated per-arm dirs.",
    )
    parser.add_argument("--timer-interval-seconds", type=int)
    parser.add_argument(
        "--observer-interval-seconds",
        type=int,
        help=(
            "Controller-side non-invasive checkpoint sampling interval. "
            "Unlike --timer-interval-seconds, this does not send prompts to the agent."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--verifier-command",
        help="Command to run in each arm work directory after the agent finishes.",
    )
    parser.add_argument("--verifier-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--environment-command",
        help=(
            "Optional command that prints JSON describing the experiment runtime. "
            "It is captured as an audit artifact and does not affect run metrics."
        ),
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help=(
            "Enable session-local YOLO auto-approval for campaign runs. "
            "This does not change global agent client settings."
        ),
    )
    parser.add_argument(
        "--initial-dream-retrieval",
        action="store_true",
        help=(
            "Explicitly prefetch Dream memories in the campaign runner before sending "
            "the prompt. The Codex runner already performs auditable prompt-start "
            "retrieval when Dream mode is enabled."
        ),
    )
    parser.add_argument(
        "--disable-initial-dream-retrieval",
        action="store_true",
        help="Compatibility flag: keep campaign-runner prefetch disabled.",
    )
    parser.add_argument(
        "--dream-retrieval-query",
        help="Override the query used for campaign-start Dream retrieval. Default: prompt.",
    )
    parser.add_argument(
        "--dream-retrieval-tag",
        action="append",
        default=[],
        help="Structured Dream retrieval tag. Can be repeated.",
    )
    parser.add_argument(
        "--dream-retrieval-category",
        action="append",
        choices=["optimization", "environment_debug"],
        default=None,
        help="Dream memory category to retrieve. Can be repeated.",
    )
    parser.add_argument(
        "--dream-retrieval-memory-id",
        action="append",
        default=[],
        help=(
            "Pin campaign-start Dream retrieval to a specific memory ID. "
            "Can be repeated for retrieval-controlled A/B experiments."
        ),
    )
    parser.add_argument("--dream-retrieval-top-k-per-category", type=int, default=3)
    parser.add_argument(
        "--dream-retrieval-render-mode",
        choices=[
            "full",
            "route_policy",
            "route_policy_minimal",
            "route_policy_template",
            "route_policy_artifact",
            "route_policy_artifact_protocol",
        ],
        default="full",
        help=(
            "How campaign-start Dream retrieval is injected into the prompt. "
            "'full' preserves detailed memory context; 'route_policy' keeps compact "
            "route constraints and evidence IDs; 'route_policy_minimal' keeps only "
            "route-decision fields for low-token A/B prompts; 'route_policy_template' "
            "keeps a compact route_decision.json scaffold; 'route_policy_artifact' "
            "writes route_decision.json before the run and injects only a short notice; "
            "'route_policy_artifact_protocol' uses the same artifact but injects a "
            "stricter consumption protocol to reduce repair loops."
        ),
    )
    parser.add_argument(
        "--active-dream-protocol",
        action="store_true",
        help=(
            "Enable controller-side active Dream protocol hooks: task-start retrieval "
            "and completion-time candidate extraction from campaign artifacts."
        ),
    )
    parser.add_argument(
        "--arm",
        action="append",
        help=(
            "JSON campaign arm config. Repeat to define custom arms. "
            'Example: {"name":"with_library_learning_memory","dream_enabled":true,'
            '"initial_dream_retrieval":true,"dream_retrieval_top_k_per_category":1}'
        ),
    )
    parser.add_argument("--delete-sessions", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/evoinfer/campaign-runs"))
    args = parser.parse_args(argv)

    result = asyncio.run(run_campaign_from_cli(args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(result.started_at)}-{result.name}"
    json_path = args.out_dir / f"{stem}.json"
    markdown_path = args.out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_campaign_markdown(result), encoding="utf-8")
    print(json_path)
    print(markdown_path)


def _decode_json_object(raw: str | bytes) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object message.")
    return data


def _stable_json_size_bytes(value: Any) -> int:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raw = json.dumps(str(value), ensure_ascii=False)
    return len(raw.encode("utf-8"))


def _message_size_kind(message: dict[str, Any]) -> str:
    method = message.get("method")
    if method == "event":
        params = message.get("params")
        if isinstance(params, dict):
            event_type = params.get("type")
            if isinstance(event_type, str) and event_type:
                return f"event:{event_type}"
        return "event:unknown"
    if isinstance(method, str) and method:
        return method
    if "result" in message:
        return "response:result"
    if "error" in message:
        return "response:error"
    return "unknown"


def _as_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_int(value: Any) -> int:
    return _as_optional_int(value) or 0


def _collect_environment_snapshot(environment_command: str | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "cwd": os.getcwd(),
        "env": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "PATH": os.environ.get("PATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
    }
    if environment_command:
        completed = subprocess.run(
            environment_command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        command_record: dict[str, Any] = {
            "command": environment_command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        try:
            command_record["payload"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            command_record["payload"] = None
        snapshot["command"] = command_record
    return snapshot


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _parse_dream_retrieval_event(payload: dict[str, Any]) -> DreamRetrievalEvent:
    return DreamRetrievalEvent(
        trigger=str(payload.get("trigger") or ""),
        query=str(payload.get("query") or ""),
        categories=_as_str_list(payload.get("categories")),
        top_k_per_category=_as_int(payload.get("top_k_per_category")),
        memory_ids=_as_str_list(payload.get("memory_ids")),
        result_count=_as_int(payload.get("result_count")),
        step_count=_as_optional_int(payload.get("step_count")),
    )


def _extract_shell_command_from_tool_call(function: dict[str, Any]) -> str:
    return _extract_shell_command_from_arguments(function.get("arguments"))


def _extract_shell_command_from_arguments(arguments: Any) -> str:
    parsed: Any = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return ""
    if not isinstance(parsed, dict):
        return ""
    command = parsed.get("command")
    return command if isinstance(command, str) else ""


def _failed_tool_call_summary(
    *,
    name: str | None,
    command: str | None,
    output: str,
) -> str:
    tool_name = name or "Tool"
    command_text = _truncate_one_line(command or "", 96)
    first_output_line = next(
        (line.strip() for line in output.splitlines() if line.strip()),
        "",
    )
    output_text = _truncate_one_line(first_output_line, 96)
    if command_text and output_text:
        return f"{tool_name} failed: {command_text} | {output_text}"
    if command_text:
        return f"{tool_name} failed: {command_text}"
    if output_text:
        return f"{tool_name} failed: {output_text}"
    return f"{tool_name} failed"


def _line_count(text: str) -> int:
    return text.count("\n") + 1 if text else 0


def _is_benchmark_command(command: str) -> bool:
    normalized = command.lower()
    return any(
        marker in normalized
        for marker in (
            "benchmark",
            "bench.py",
            "_bench",
            "-bench",
            "/bench",
            "rmsnorm_bench",
            "./bench",
            "preexpansion_probe.py",
            "evoinfer_preexpansion_profile",
        )
    )


def _is_edit_command(command: str) -> bool:
    normalized = command.lower()
    if any(
        marker in normalized
        for marker in (
            "cat >",
            "tee ",
            "sed -i",
            "perl -pi",
            "apply_patch",
        )
    ):
        return True
    return bool(
        re.search(
            r"\bpython3?\b.*(?:patch|apply|update|edit|fix).*\.py",
            normalized,
        )
    )


def _extract_benchmark_result_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw_value in _METRIC_RE.findall(output):
        metrics[name.strip().lower()] = float(raw_value)
    return metrics


def _is_valid_speedup_metrics(metrics: dict[str, float]) -> bool:
    if not metrics:
        return False
    if _has_excessive_correctness_error(metrics):
        return False
    speedup = metrics.get("speedup") or metrics.get("speedup_x")
    if speedup is not None:
        return speedup > 1.0
    baseline_ms = metrics.get("baseline_ms")
    candidate_ms = metrics.get("candidate_ms")
    return (
        baseline_ms is not None
        and candidate_ms is not None
        and candidate_ms < baseline_ms
    )


def _is_failed_variant_metrics(metrics: dict[str, float], output: str) -> bool:
    if "correctness_failed" in output:
        return True
    if not metrics:
        return False
    if _has_excessive_correctness_error(metrics):
        return True
    speedup = metrics.get("speedup") or metrics.get("speedup_x")
    if speedup is not None:
        return speedup <= 1.0
    baseline_ms = metrics.get("baseline_ms")
    candidate_ms = metrics.get("candidate_ms")
    return (
        baseline_ms is not None
        and candidate_ms is not None
        and candidate_ms >= baseline_ms
    )


def _has_excessive_correctness_error(metrics: dict[str, float]) -> bool:
    metric_thresholds = {
        "max_abs_error": 1e-3,
        "max_row_sum_error": 1e-5,
    }
    return any(
        metrics.get(name) is not None and metrics[name] > threshold
        for name, threshold in metric_thresholds.items()
    )


def normalize_dream_retrieval_categories(
    categories: list[DreamMemoryCategory] | None,
) -> tuple[DreamMemoryCategory, ...]:
    values = categories or list(DEFAULT_DREAM_RETRIEVAL_CATEGORIES)
    seen: set[DreamMemoryCategory] = set()
    normalized: list[DreamMemoryCategory] = []
    for category in values:
        if category in seen:
            continue
        seen.add(category)
        normalized.append(category)
    return tuple(normalized) or DEFAULT_DREAM_RETRIEVAL_CATEGORIES


def _parse_campaign_arms(raw_arms: list[str] | None) -> tuple[CampaignArm, ...]:
    """Parse repeatable JSON arm arguments for campaign CLIs."""

    if not raw_arms:
        return (
            CampaignArm(name="without_memory", dream_enabled=False),
            CampaignArm(name="with_memory", dream_enabled=True),
        )

    arms: list[CampaignArm] = []
    for raw_arm in raw_arms:
        try:
            payload = json.loads(raw_arm)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid --arm JSON: {raw_arm}") from exc
        if not isinstance(payload, dict):
            raise ValueError("--arm must be a JSON object")
        if "dream_retrieval_categories" in payload:
            payload["dream_retrieval_categories"] = normalize_dream_retrieval_categories(
                payload["dream_retrieval_categories"]
            )
        arms.append(CampaignArm.model_validate(payload))
    return tuple(arms)


def _prepend_dream_retrieval_event(
    run: SessionRunResult,
    event: DreamRetrievalEvent,
) -> None:
    run.dream_retrieval_events = [event, *run.dream_retrieval_events]
    run.dream_retrieval_count = len(run.dream_retrieval_events)
    run.dream_retrieved_memory_ids = _unique_memory_ids_from_events(run.dream_retrieval_events)


def _apply_dream_completion_extraction(
    run: SessionRunResult,
    extraction: DreamCompletionExtraction,
) -> None:
    run.dream_completion_candidate_count = extraction.candidate_count
    run.dream_completion_artifact_ref_count = extraction.artifact_ref_count
    run.dream_completion_promotion_ready_count = extraction.promotion_ready_count
    run.dream_completion_extraction_sources = list(extraction.extraction_sources)
    run.dream_completion_extraction_error = extraction.error


def _apply_dream_auto_write_result(
    run: SessionRunResult,
    result: DreamAutoWriteResult,
) -> None:
    run.dream_auto_write_count = result.written_count
    run.dream_written_memory_ids = list(result.written_memory_ids)
    run.dream_auto_write_rejected_count = result.rejected_count
    run.dream_auto_write_blockers = list(result.blockers)
    run.dream_auto_write_error = result.error


def _unique_memory_ids_from_events(events: list[DreamRetrievalEvent]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for event in events:
        for memory_id in event.memory_ids:
            if memory_id in seen:
                continue
            seen.add(memory_id)
            ordered.append(memory_id)
    return ordered


def _arm_initial_dream_retrieval_enabled(spec: CampaignSpec, arm: CampaignArm) -> bool:
    if arm.initial_dream_retrieval is not None:
        return arm.initial_dream_retrieval
    return spec.initial_dream_retrieval or spec.active_dream_protocol


def _arm_dream_retrieval_query(spec: CampaignSpec, arm: CampaignArm) -> str:
    return arm.dream_retrieval_query or spec.dream_retrieval_query or spec.prompt


def _arm_dream_retrieval_tags(spec: CampaignSpec, arm: CampaignArm) -> list[str]:
    if arm.dream_retrieval_tags is not None:
        return arm.dream_retrieval_tags
    return spec.dream_retrieval_tags


def _arm_dream_retrieval_categories(
    spec: CampaignSpec,
    arm: CampaignArm,
) -> tuple[DreamMemoryCategory, ...]:
    if arm.dream_retrieval_categories is not None:
        return arm.dream_retrieval_categories
    return spec.dream_retrieval_categories


def _arm_dream_retrieval_memory_ids(
    spec: CampaignSpec,
    arm: CampaignArm,
) -> list[str]:
    if arm.dream_retrieval_memory_ids is not None:
        return arm.dream_retrieval_memory_ids
    return spec.dream_retrieval_memory_ids


def _arm_dream_retrieval_top_k_per_category(
    spec: CampaignSpec,
    arm: CampaignArm,
) -> int:
    if arm.dream_retrieval_top_k_per_category is not None:
        return arm.dream_retrieval_top_k_per_category
    return spec.dream_retrieval_top_k_per_category


def _arm_dream_retrieval_render_mode(
    spec: CampaignSpec,
    arm: CampaignArm,
) -> DreamRetrievalRenderMode:
    return arm.dream_retrieval_render_mode or spec.dream_retrieval_render_mode


def _inject_initial_dream_retrieval_context(prompt: str, retrieval_text: str) -> str:
    return (
        "Initial EvoInfer Dream memory retrieval was performed before this campaign arm.\n"
        "Use the retrieved memories only as prior experience; validate every decision with "
        "real commands, benchmarks, profiling evidence, and correctness checks.\n\n"
        f"{retrieval_text.strip()}\n\n"
        "Original campaign task:\n"
        f"{prompt}"
    )


def _inject_controller_route_decision_context(prompt: str, *, protocol: bool = False) -> str:
    if protocol:
        return (
            "Controller generated route_decision.json from EvoInfer Dream retrieval "
            "before this campaign arm.\n"
            "Consumption protocol:\n"
            "- Treat the original task's instruction to create route_decision.json as "
            "already satisfied; do not recreate or overwrite the file unless it is "
            "schema-invalid or contradicted by newly measured task evidence.\n"
            "- First inspect README.md, preexpansion_probe.py, and route_decision.json. "
            "Do not print or restate the full route JSON in chat unless debugging a "
            "schema failure.\n"
            "- Use route_decision.json as the route prior. Still run the real "
            "correctness, benchmark, artifact inspection, and verifier commands from "
            "the task; never claim success from memory alone.\n"
            "- Run campaign commands inside the environment specified by the original "
            "task in the same shell command. If that environment is activated by "
            "`source .../activate`, keep that activation before invoking python.\n"
            "- If a command fails because the environment was not activated or the "
            "python executable was wrong, fix only that command and rerun once before "
            "exploring unrelated environment changes.\n\n"
            "Original campaign task:\n"
            f"{prompt}"
        )
    return (
        "Controller generated route_decision.json from EvoInfer Dream retrieval "
        "before this campaign arm.\n"
        "Use that file as prior route evidence, but still validate every decision "
        "with real commands, benchmarks, and correctness checks. Edit it only if "
        "new task evidence disagrees.\n\n"
        "Original campaign task:\n"
        f"{prompt}"
    )


def _is_route_policy_artifact_mode(render_mode: DreamRetrievalRenderMode | None) -> bool:
    return render_mode in {"route_policy_artifact", "route_policy_artifact_protocol"}


def _write_controller_route_decision_artifact(arm_work_dir: str, text: str) -> str:
    try:
        route_decision = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("route_policy_artifact retrieval did not produce JSON") from exc
    if not isinstance(route_decision, dict):
        raise ValueError("route_policy_artifact retrieval must produce a JSON object")
    path = Path(arm_work_dir) / "route_decision.json"
    route_decision_json = (
        json.dumps(route_decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    path.write_text(route_decision_json, encoding="utf-8")
    route_decision_bytes = route_decision_json.encode("utf-8")
    audit_path = Path(arm_work_dir) / "controller_route_decision_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "artifact_path": path.name,
                "controller_generated": True,
                "controller_generated_at": time.time(),
                "route_decision_bytes": len(route_decision_bytes),
                "route_decision_sha256": hashlib.sha256(
                    route_decision_bytes
                ).hexdigest(),
                "source": "dream_retrieval",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _search_result_score(item: dict[str, Any]) -> float:
    score = item.get("score")
    return float(score) if isinstance(score, int | float) else 0.0


def _memory_ids_from_search_results(results: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    memory_ids: list[str] = []
    for item in results:
        memory = item.get("memory")
        if not isinstance(memory, dict):
            continue
        memory_id = memory.get("id")
        if not isinstance(memory_id, str) or memory_id in seen:
            continue
        seen.add(memory_id)
        memory_ids.append(memory_id)
    return memory_ids


def _render_dream_search_results_for_prompt(
    results: list[dict[str, Any]],
    *,
    render_mode: DreamRetrievalRenderMode = "full",
) -> str:
    if not results:
        return ""
    if render_mode == "route_policy":
        return _render_dream_route_policy_for_prompt(results)
    if render_mode == "route_policy_minimal":
        return _render_dream_minimal_route_policy_for_prompt(results)
    if render_mode == "route_policy_template":
        return _render_dream_route_decision_template_for_prompt(results)
    if _is_route_policy_artifact_mode(render_mode):
        return _render_dream_route_decision_artifact(results)
    lines = [
        "Dream memory retrieval results:",
        "",
        "These memories are retrieved for the current inference optimization/debug campaign.",
    ]
    for index, item in enumerate(results, start=1):
        memory = item.get("memory")
        if not isinstance(memory, dict):
            continue
        memory_id = str(memory.get("id") or "")
        category = str(memory.get("category") or "")
        title = str(memory.get("title") or "")
        score = _search_result_score(item)
        reasons = item.get("reasons")
        reasons_text = ", ".join(reason for reason in reasons if isinstance(reason, str)) if isinstance(reasons, list) else ""
        lines.extend(
            [
                "",
                f"{index}. [{category}] {title} ({memory_id})",
                f"   score={score:.3f}; reasons={reasons_text or 'semantic'}",
            ]
        )
        for label, key in [
            ("summary", "summary"),
            ("environment", "environment"),
            ("backend", "inference_backend"),
            ("applicability", "applicability"),
            ("issue", "issue_signature"),
            ("root_cause", "root_cause"),
            ("solution", "solution"),
            ("verification", "verification"),
        ]:
            value = memory.get(key)
            if value:
                lines.append(f"   {label}={_one_line(value)}")
        for label, key in [
            ("metrics_after", "metrics_after"),
            ("metrics_before", "metrics_before"),
        ]:
            value = memory.get(key)
            if isinstance(value, dict) and value:
                lines.append(
                    f"   {label}="
                    + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
        for label, key in [
            ("correctness", "correctness_invariants"),
            ("safe", "safe_transfer_notes"),
            ("unsafe", "unsafe_transfer_notes"),
        ]:
            value = memory.get(key)
            if isinstance(value, list) and value:
                lines.append(f"   {label}={_one_line(' | '.join(str(item) for item in value[:3]))}")
        tags = memory.get("tags")
        if isinstance(tags, list) and tags:
            lines.append("   tags=" + ", ".join(str(tag) for tag in tags if tag))
    return "\n".join(lines)


def _render_dream_route_policy_for_prompt(results: list[dict[str, Any]]) -> str:
    lines = [
        "Dream route policy retrieval results:",
        "",
        "Use these compact memories as route constraints only; verify every decision with correctness and benchmark artifacts.",
    ]
    for index, item in enumerate(results, start=1):
        memory = item.get("memory")
        if not isinstance(memory, dict):
            continue
        memory_id = str(memory.get("id") or "")
        category = str(memory.get("category") or "")
        title = str(memory.get("title") or "")
        score = _search_result_score(item)
        lines.extend(
            [
                "",
                f"{index}. [{category}] {title} ({memory_id})",
                f"   score={score:.3f}",
            ]
        )
        summary = memory.get("summary")
        if summary:
            lines.append(f"   evidence={_one_line(summary)}")
        for label, key in [
            ("correctness", "correctness_invariants"),
            ("unsafe", "unsafe_transfer_notes"),
            ("safe", "safe_transfer_notes"),
        ]:
            value = memory.get(key)
            if isinstance(value, list) and value:
                lines.append(
                    f"   {label}="
                    + _one_line(" | ".join(str(item) for item in value[:2]))
                )
        tags = memory.get("tags")
        if isinstance(tags, list) and tags:
            compact_tags = ", ".join(str(tag) for tag in tags[:8] if tag)
            if compact_tags:
                lines.append(f"   tags={compact_tags}")
    return "\n".join(lines)


_ROUTE_POLICY_TAG_PRIORITY = (
    "fp16_failure",
    "bf16_failure",
    "float16_failure",
    "bfloat16_failure",
    "dtype_boundary",
    "negative_optimization",
    "correctness_failure",
    "unsafe_transfer",
)


def _render_dream_minimal_route_policy_for_prompt(results: list[dict[str, Any]]) -> str:
    lines = [
        "Dream minimal route policy retrieval results:",
        "Use memory IDs only for route_decision.selected_memory_ids and skip_evidence; verify before benchmark.",
    ]
    for index, item in enumerate(results, start=1):
        memory = item.get("memory")
        if not isinstance(memory, dict):
            continue
        memory_id = str(memory.get("id") or "")
        title = _truncate_one_line(str(memory.get("title") or ""), 96)
        score = _search_result_score(item)
        tags = _route_policy_tags(memory)
        avoid_hint = _route_policy_avoid_hint(memory)
        fields = [
            f"{index}. id={memory_id}",
            f"score={score:.3f}",
        ]
        if avoid_hint:
            fields.append(f"avoid_hint={avoid_hint}")
        if tags:
            fields.append(f"tags={tags}")
        if title:
            fields.append(f"title={title}")
        lines.append(" ".join(fields))
    return "\n".join(lines)


def _render_dream_route_decision_template_for_prompt(results: list[dict[str, Any]]) -> str:
    selected_memory_ids: list[str] = []
    dtype_evidence: dict[str, list[str]] = {}
    evidence_titles: dict[str, str] = {}
    for item in results:
        memory = item.get("memory")
        if not isinstance(memory, dict):
            continue
        memory_id = str(memory.get("id") or "")
        if not memory_id:
            continue
        selected_memory_ids.append(memory_id)
        evidence_titles[memory_id] = _truncate_one_line(str(memory.get("title") or ""), 72)
        for dtype in _route_policy_evidence_dtypes(memory):
            dtype_evidence.setdefault(dtype, []).append(memory_id)

    avoid_dtypes = sorted(dtype_evidence)
    lines = [
        "Dream route decision template:",
        "Use this as a compact route_decision.json scaffold; edit it if the task evidence disagrees.",
        "Only skip a dtype if the retrieved IDs above justify the skip.",
        "{",
        '  "selection_policy": "memory_route_policy",',
        '  "selected_dtypes": [],',
        '  "audit_dtypes": [],',
        f'  "avoid_dtypes": {_json_inline(avoid_dtypes)},',
        f'  "selected_memory_ids": {_json_inline(selected_memory_ids)},',
        '  "skip_evidence": {',
    ]
    for index, dtype in enumerate(sorted(dtype_evidence)):
        suffix = "," if index < len(dtype_evidence) - 1 else ""
        lines.append(f'    "{dtype}": {_json_inline(dtype_evidence[dtype])}{suffix}')
    lines.extend(
        [
            "  },",
            '  "reason": "Use retrieved negative dtype-boundary memory IDs; audit instead of skip when unsure."',
            "}",
        ]
    )
    if selected_memory_ids:
        lines.append("Retrieved evidence IDs:")
        for memory_id in selected_memory_ids:
            title = evidence_titles.get(memory_id, "")
            lines.append(f"- {memory_id}" + (f": {title}" if title else ""))
    return "\n".join(lines)


def _render_dream_route_decision_artifact(results: list[dict[str, Any]]) -> str:
    selected_memory_ids: list[str] = []
    dtype_evidence: dict[str, list[str]] = {}
    for item in results:
        memory = item.get("memory")
        if not isinstance(memory, dict):
            continue
        memory_id = str(memory.get("id") or "")
        if not memory_id:
            continue
        selected_memory_ids.append(memory_id)
        for dtype in _route_policy_evidence_dtypes(memory):
            dtype_evidence.setdefault(dtype, []).append(memory_id)

    route_decision = {
        "selection_policy": "memory_route_policy",
        "selected_dtypes": ["float32"] if dtype_evidence else [],
        "audit_dtypes": [],
        "avoid_dtypes": sorted(dtype_evidence),
        "selected_memory_ids": selected_memory_ids,
        "skip_evidence": {
            dtype: dtype_evidence[dtype] for dtype in sorted(dtype_evidence)
        },
        "reason": (
            "Controller-side Dream retrieval generated this route from retrieved "
            "dtype-boundary memories."
        ),
    }
    return json.dumps(
        route_decision,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _route_policy_evidence_dtypes(memory: dict[str, Any]) -> list[str]:
    dtypes = _route_policy_avoid_hint(memory).split(",")
    return [dtype for dtype in dtypes if dtype]


def _json_inline(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ": "))


def _route_policy_tags(memory: dict[str, Any]) -> str:
    raw_tags = memory.get("tags")
    if not isinstance(raw_tags, list):
        return ""
    tag_set = {str(tag) for tag in raw_tags if tag}
    ordered = [tag for tag in _ROUTE_POLICY_TAG_PRIORITY if tag in tag_set]
    if len(ordered) < 4:
        for tag in sorted(tag_set):
            if tag in ordered:
                continue
            if tag in {"fla", "linear_attention", "recurrent_generation"}:
                continue
            ordered.append(tag)
            if len(ordered) >= 4:
                break
    return ",".join(ordered[:4])


def _route_policy_avoid_hint(memory: dict[str, Any]) -> str:
    tags = memory.get("tags")
    tag_text = " ".join(str(tag).lower() for tag in tags if tag) if isinstance(tags, list) else ""
    text = " ".join(
        str(memory.get(key) or "").lower()
        for key in ("id", "title", "summary")
    )
    evidence = f"{tag_text} {text}"
    hints: list[str] = []
    if "fp16" in evidence or "float16" in evidence:
        hints.append("float16")
    if "bf16" in evidence or "bfloat16" in evidence:
        hints.append("bfloat16")
    return ",".join(hints)


def _truncate_one_line(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _one_line(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict | list) else str(value)
    return " ".join(text.split())


_METRIC_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

_LOWER_IS_BETTER_EXACT_METRICS = {
    "candidate_ms",
    "latency_ms",
    "decode_ms",
    "prefill_ms",
    "time_ms",
}

_HIGHER_IS_BETTER_EXACT_METRICS = {
    "speedup",
    "speedup_x",
    "throughput",
    "tokens_per_second",
    "tok_s",
}


def _decide_dream_feedback_usefulness(
    run: SessionRunResult,
    *,
    comparison_runs: list[SessionRunResult],
) -> tuple[bool, str]:
    """Decide whether retrieved memory helped using verifier evidence and controls."""

    if run.verification_status != "passed":
        return False, f"Dream memory was not useful: verifier_status={run.verification_status}."

    run_metrics = _extract_numeric_metrics_for_feedback(run)
    passed_controls = [
        control for control in comparison_runs if control.verification_status == "passed"
    ]
    control_metrics = [
        _extract_numeric_metrics_for_feedback(control) for control in passed_controls
    ]
    metric_name = _select_comparison_metric(run_metrics, control_metrics)
    if metric_name is None:
        return (
            True,
            "Dream memory marked useful because verifier passed and no comparable "
            "non-memory target metric was available.",
        )

    run_value = run_metrics[metric_name]
    control_values = [metrics[metric_name] for metrics in control_metrics if metric_name in metrics]
    if not control_values:
        return (
            True,
            f"Dream memory marked useful because verifier passed and no non-memory "
            f"value existed for metric {metric_name}.",
        )

    lower_is_better = _metric_lower_is_better(metric_name)
    if lower_is_better:
        best_control = min(control_values)
        useful = run_value <= best_control * 1.02
        relation = "<="
    else:
        best_control = max(control_values)
        useful = run_value >= best_control * 0.98
        relation = ">="

    if useful:
        return (
            True,
            f"Dream memory was useful: {metric_name}={run_value:.6g} {relation} "
            f"control={best_control:.6g} within tolerance.",
        )
    return (
        False,
        f"Dream memory was not useful: {metric_name}={run_value:.6g} did not beat "
        f"or match control={best_control:.6g} within tolerance.",
    )


def _select_comparison_metric(
    run_metrics: dict[str, float],
    control_metrics: list[dict[str, float]],
) -> str | None:
    control_metric_names = set().union(*(metrics.keys() for metrics in control_metrics)) if control_metrics else set()
    common_names = set(run_metrics) & control_metric_names
    priority = [
        "candidate_ms",
        "latency_ms",
        "decode_ms",
        "prefill_ms",
        "time_ms",
        "speedup",
        "speedup_x",
        "tokens_per_second",
        "tok_s",
    ]
    for name in priority:
        if name in common_names:
            return name
    for name in sorted(common_names):
        if _is_feedback_target_metric(name):
            return name
    return None


def _extract_numeric_metrics_for_feedback(run: SessionRunResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for text in (run.assistant_text, run.verification_output):
        metrics.update(_extract_feedback_metrics_from_text(text))
    return metrics


def _extract_feedback_metrics_from_text(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, raw_value in _METRIC_RE.findall(text):
        normalized = name.strip().lower()
        if _is_feedback_target_metric(normalized):
            metrics[normalized] = float(raw_value)
    return metrics


def _is_feedback_target_metric(name: str) -> bool:
    if name in {"baseline_ms", "max_abs_error"}:
        return False
    return (
        name in _LOWER_IS_BETTER_EXACT_METRICS
        or name in _HIGHER_IS_BETTER_EXACT_METRICS
        or name.endswith("_ms")
        or name.endswith("_tokens_per_second")
        or name.endswith("_tok_s")
    )


def _metric_lower_is_better(name: str) -> bool:
    if name in _HIGHER_IS_BETTER_EXACT_METRICS:
        return False
    if name.endswith("_tokens_per_second") or name.endswith("_tok_s"):
        return False
    return True


def _build_dream_feedback_reason(
    run: SessionRunResult,
    *,
    decision_reason: str | None = None,
) -> str:
    parts = [
        f"Campaign arm {run.arm_name} verifier passed.",
    ]
    if decision_reason:
        parts.append(decision_reason)
    if run.verification_command:
        parts.append(f"command={run.verification_command}")
    output = run.verification_output.strip()
    if output:
        parts.append("output=" + output[:1000])
    return "\n".join(parts)


def _dream_feedback_evidence_artifacts(run: SessionRunResult) -> list[str]:
    artifacts: list[str] = []
    for value in (
        run.controller_route_decision_path,
        run.controller_route_decision_audit_path,
        run.verification_artifact_path,
    ):
        if value and Path(value).exists():
            artifacts.append(value)

    work_dir = Path(run.work_dir) if run.work_dir else None
    if work_dir is not None:
        for name in (
            "benchmark_raw.json",
            "correctness_raw.json",
            "dream_write_candidates.json",
        ):
            path = work_dir / name
            if path.is_file():
                artifacts.append(str(path))

    return list(dict.fromkeys(artifacts))


async def _run_verifier_for_arm(
    run: SessionRunResult,
    *,
    command: str,
    cwd: str,
    timeout_seconds: float,
) -> None:
    run.verification_command = command
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            output_bytes, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=max(0.1, timeout_seconds),
            )
        except TimeoutError:
            process.kill()
            output_bytes, _ = await process.communicate()
            run.timed_out_process_cleanup_count += _terminate_processes_in_work_dir(cwd)
            run.verification_status = "timeout"
            run.verification_exit_code = process.returncode
            run.verification_output = output_bytes.decode("utf-8", errors="replace")
            run.verification_error = f"Verifier timed out after {timeout_seconds:.3f}s."
            _write_verifier_result_artifact(run, cwd=cwd, timeout_seconds=timeout_seconds)
            return
        run.verification_exit_code = process.returncode
        run.verification_output = output_bytes.decode("utf-8", errors="replace")
        run.verification_status = "passed" if process.returncode == 0 else "failed"
    except Exception as exc:
        run.verification_status = "error"
        run.verification_error = str(exc)
    finally:
        if run.verification_artifact_path is None:
            _write_verifier_result_artifact(run, cwd=cwd, timeout_seconds=timeout_seconds)


def _write_verifier_result_artifact(
    run: SessionRunResult,
    *,
    cwd: str,
    timeout_seconds: float,
) -> None:
    run.verification_artifact_path = _write_arm_memory_artifact(
        cwd,
        "verifier_result.json",
        {
            "command": run.verification_command,
            "status": run.verification_status,
            "exit_code": run.verification_exit_code,
            "output": run.verification_output,
            "error": run.verification_error,
            "timeout_seconds": timeout_seconds,
        },
    )


def _terminate_processes_in_work_dir(work_dir: str) -> int:
    root = Path(work_dir).resolve()
    pids = sorted(_process_ids_in_work_dir(root))
    if not pids:
        return 0

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    deadline = time.time() + 1.0
    while time.time() < deadline:
        remaining = [pid for pid in pids if _pid_exists(pid)]
        if not remaining:
            return len(pids)
        time.sleep(0.05)

    for pid in pids:
        if not _pid_exists(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    return len(pids)


def _write_arm_memory_artifact(arm_work_dir: str, filename: str, payload: Any) -> str:
    audit_dir = Path(arm_work_dir) / ".evoinfer"
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def _memory_snapshot_count(snapshot: dict[str, Any]) -> int:
    memories = snapshot.get("memories")
    return len(memories) if isinstance(memories, list) else 0


def _process_ids_in_work_dir(root: Path) -> set[int]:
    current_pid = os.getpid()
    pids: set[int] = set()
    proc_root = Path("/proc")
    if not proc_root.exists():
        return pids
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        try:
            cwd = (entry / "cwd").resolve()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        try:
            cwd.relative_to(root)
        except ValueError:
            continue
        pids.add(pid)
    return pids


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _truncate_markdown_block(text: str, *, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text.rstrip()
    return text[:limit].rstrip() + "\n...[truncated]"


def _looks_like_completed_campaign_text(text: str) -> bool:
    normalized = text.lower()
    if "library_campaign_verifier_pass" in normalized:
        return True
    return "campaign completed" in normalized and "verifier passed" in normalized


def _slugify_arm_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip()).strip(".-")
    return slug or "arm"


if __name__ == "__main__":
    main()
