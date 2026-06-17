"""Dream memory retrieval helpers for agent context injection."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from evoinfer_mcp.dream.memory import (
    DreamMemory,
    DreamMemoryCategory,
    DreamMemorySearchInput,
    DreamMemorySearchResult,
    search_dream_memories,
)

DreamMemoryRetrievalDetailMode = Literal["full", "compact"]


_DREAM_MEMORY_APPLICATION_POLICY = [
    "Dream memory application policy:",
    "- First try the closest correctness-valid successful memory as an implementation recipe "
    "before broad search.",
    "- Validate it with the real benchmark, profiling or source evidence, and correctness checks; "
    "do not trust speed without correctness.",
    "- If the candidate is correct and materially faster than the baseline, stop broad exploration "
    "and report exact metrics.",
    "- Treat failed memories or memories with failed correctness as negative constraints, not recipes.",
]


@dataclass(frozen=True, slots=True)
class DreamMemoryRetrievalContext:
    """Structured Dream retrieval result for prompt injection and experiment audit."""

    text: str
    query: str
    categories: tuple[DreamMemoryCategory, ...]
    top_k_per_category: int
    memory_ids: tuple[str, ...]
    result_count: int


def build_dream_memory_retrieval_context(
    *,
    query: str,
    tags: Iterable[str] = (),
    top_k_per_category: int = 3,
    categories: Iterable[DreamMemoryCategory] = ("optimization", "environment_debug"),
    detail_mode: DreamMemoryRetrievalDetailMode = "full",
) -> str | None:
    """Build a compact context block with memories relevant to a stalled Dream task."""

    retrieval = build_dream_memory_retrieval(
        query=query,
        tags=tags,
        top_k_per_category=top_k_per_category,
        categories=categories,
        detail_mode=detail_mode,
    )
    return retrieval.text if retrieval is not None else None


def build_dream_memory_retrieval(
    *,
    query: str,
    tags: Iterable[str] = (),
    top_k_per_category: int = 3,
    categories: Iterable[DreamMemoryCategory] = ("optimization", "environment_debug"),
    detail_mode: DreamMemoryRetrievalDetailMode = "full",
) -> DreamMemoryRetrievalContext | None:
    """Build a compact context block and return structured retrieval metadata."""

    query = query.strip()
    if not query:
        return None

    category_tuple = tuple(categories)
    results: list[DreamMemorySearchResult] = []
    for category in category_tuple:
        response = search_dream_memories(
            DreamMemorySearchInput(
                query=query,
                category=category,
                tags=list(tags),
                top_k=top_k_per_category,
                record_choice=True,
            )
        )
        results.extend(response.results)

    if not results:
        return None

    results.sort(key=lambda item: item.score, reverse=True)
    lines = [
        "Dream memory retrieval results:",
        "",
        "Use these only as evidence-backed prior experience. Validate applicability with real "
        "environment checks, benchmarks, profiling, and correctness tests before applying.",
        "",
        *_DREAM_MEMORY_APPLICATION_POLICY,
    ]
    for index, result in enumerate(results, start=1):
        memory = result.memory
        lines.extend(
            [
                "",
                f"{index}. [{memory.category}] {memory.title} ({memory.id})",
                f"   score={result.score:.3f}; reasons={', '.join(result.reasons) or 'semantic'}",
                f"   recommended_action={result.recommended_action}; status={memory.status}; evidence_level={memory.evidence_level}",
            ]
        )
        summary = _memory_summary(memory)
        if summary:
            lines.append(f"   summary={summary}")
        if detail_mode == "compact":
            lines.extend(_memory_compact_detail_lines(memory))
        else:
            details = _memory_actionable_details(memory)
            if details:
                lines.append(f"   details={details}")
        tags_text = ", ".join(memory.tags)
        if tags_text:
            lines.append(f"   tags={tags_text}")
    memory_ids = tuple(result.memory.id for result in results)
    return DreamMemoryRetrievalContext(
        text="\n".join(lines),
        query=query,
        categories=category_tuple,
        top_k_per_category=top_k_per_category,
        memory_ids=memory_ids,
        result_count=len(results),
    )


def _memory_summary(memory: DreamMemory) -> str:
    parts = [
        memory.summary,
        f"environment={memory.environment}" if memory.environment else "",
        f"backend={memory.inference_backend}" if memory.inference_backend else "",
        f"success={memory.success}" if memory.success is not None else "",
        f"useful_rate={memory.useful_rate:.3f}" if memory.useful_rate else "",
        f"chosen={memory.chosen}" if memory.chosen else "",
    ]
    return "; ".join(part for part in parts if part)


def _memory_compact_detail_lines(memory: DreamMemory) -> list[str]:
    if memory.category == "optimization":
        lines: list[str] = []
        if memory.metrics_after:
            lines.append(f"   metrics_after={_compact_json(memory.metrics_after)}")
        if memory.metrics_before:
            lines.append(f"   metrics_before={_compact_json(memory.metrics_before)}")
        if memory.objective_metric:
            lines.append(f"   objective={memory.objective_metric}")
        if memory.applicability:
            lines.append(f"   applies={_truncate(memory.applicability, 220)}")
        if memory.caveats:
            lines.append(f"   caveats={_truncate(memory.caveats, 220)}")
        if memory.operation_semantics:
            lines.append(f"   semantics={_join_limited(memory.operation_semantics)}")
        if memory.correctness_invariants:
            lines.append(f"   correctness={_join_limited(memory.correctness_invariants)}")
        if memory.safe_transfer_notes:
            lines.append(f"   safe={_join_limited(memory.safe_transfer_notes)}")
        if memory.unsafe_transfer_notes:
            lines.append(f"   unsafe={_join_limited(memory.unsafe_transfer_notes)}")
        return lines

    lines = []
    if memory.issue_signature:
        lines.append(f"   issue={_truncate(memory.issue_signature, 180)}")
    if memory.root_cause:
        lines.append(f"   root_cause={_truncate(memory.root_cause, 220)}")
    if memory.solution:
        lines.append(f"   solution={_truncate(memory.solution, 260)}")
    if memory.verification:
        lines.append(f"   verification={_truncate(memory.verification, 220)}")
    if memory.resolved_by_command:
        lines.append(f"   resolved_by={_truncate(memory.resolved_by_command, 220)}")
    return lines


def _compact_json(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _join_limited(values: list[str], *, item_limit: int = 3, char_limit: int = 360) -> str:
    text = " | ".join(_truncate(value, 140) for value in values[:item_limit])
    if len(values) > item_limit:
        text += f" | ...(+{len(values) - item_limit})"
    return _truncate(text, char_limit)


def _truncate(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _memory_actionable_details(memory: DreamMemory) -> str:
    if memory.category == "optimization":
        details = {
            "bottleneck": memory.bottleneck_type,
            "objective_metric": memory.objective_metric,
            "metrics_before": memory.metrics_before or None,
            "metrics_after": memory.metrics_after or None,
            "benchmark_command": memory.benchmark_command,
            "promotion_decision": memory.promotion_decision,
            "applicability": memory.applicability,
            "caveats": memory.caveats,
            "operation_semantics": memory.operation_semantics or None,
            "correctness_invariants": memory.correctness_invariants or None,
            "safe_transfer_notes": memory.safe_transfer_notes or None,
            "unsafe_transfer_notes": memory.unsafe_transfer_notes or None,
            "description": memory.detail_description or None,
        }
    else:
        details = {
            "issue_signature": memory.issue_signature,
            "root_cause": memory.root_cause,
            "solution": memory.solution,
            "verification": memory.verification,
            "resolved_by_command": memory.resolved_by_command,
            "commands": memory.commands or None,
        }
    return json.dumps(
        {key: value for key, value in details.items() if value},
        ensure_ascii=False,
        sort_keys=True,
    )
