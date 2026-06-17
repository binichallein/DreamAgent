"""Optional embedding backend for EvoInfer Dream retrieval."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from evoinfer_mcp import logger

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_DEVICE = "cpu"

SentenceTransformer: Any | None = None


def _resolve_sentence_transformer() -> Any | None:
    try:
        return import_module("sentence_transformers").SentenceTransformer
    except Exception:  # pragma: no cover - depends on optional local install state
        return None


class DreamEmbeddingProvider(Protocol):
    """Scores query/document semantic similarity for Dream retrieval."""

    def score_texts(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one similarity score per text."""


@dataclass(frozen=True, slots=True)
class DreamEmbeddingSettings:
    """Runtime settings for optional Dream embeddings."""

    backend: str = "none"
    model: str = DEFAULT_EMBEDDING_MODEL
    device: str = DEFAULT_EMBEDDING_DEVICE

    @property
    def enabled(self) -> bool:
        return self.backend not in {"", "0", "false", "none", "off", "disabled"}


class LocalSentenceTransformerEmbeddingProvider:
    """CPU-first local SentenceTransformer embedding provider."""

    def __init__(self, *, model_name: str, device: str = DEFAULT_EMBEDDING_DEVICE) -> None:
        global SentenceTransformer
        if SentenceTransformer is None:
            SentenceTransformer = _resolve_sentence_transformer()
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; install the embedding extra "
                "or disable EVOINFER_EMBEDDING_BACKEND"
            )
        self.model_name = model_name
        self.device = device or DEFAULT_EMBEDDING_DEVICE
        self._model = SentenceTransformer(model_name, device=self.device)

    def score_texts(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        embeddings = self._model.encode(
            [query, *texts],
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        vectors = [_as_float_vector(vector) for vector in embeddings]
        query_vector = vectors[0]
        return [_cosine_similarity(query_vector, vector) for vector in vectors[1:]]


_PROVIDER_CACHE: tuple[DreamEmbeddingSettings, DreamEmbeddingProvider] | None = None
_WARNED_UNAVAILABLE = False


def embedding_settings_from_env(
    environ: Mapping[str, str] | None = None,
) -> DreamEmbeddingSettings:
    env = os.environ if environ is None else environ
    backend = env.get("EVOINFER_EMBEDDING_BACKEND", "none").strip().lower()
    model = env.get("EVOINFER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    device = env.get("EVOINFER_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE).strip()
    return DreamEmbeddingSettings(
        backend=backend or "none",
        model=model or DEFAULT_EMBEDDING_MODEL,
        device=device or DEFAULT_EMBEDDING_DEVICE,
    )


def embedding_env_for_local_cpu(model: str = DEFAULT_EMBEDDING_MODEL) -> dict[str, str]:
    """Return env vars that make the MCP server use a local CPU embedding model."""

    return {
        "EVOINFER_EMBEDDING_BACKEND": "local",
        "EVOINFER_EMBEDDING_MODEL": model,
        "EVOINFER_EMBEDDING_DEVICE": DEFAULT_EMBEDDING_DEVICE,
    }


def clear_embedding_provider_cache() -> None:
    """Clear the provider cache. Intended for tests and config reloads."""

    global _PROVIDER_CACHE
    _PROVIDER_CACHE = None


def get_embedding_provider() -> DreamEmbeddingProvider | None:
    """Return the configured embedding provider, or None when disabled."""

    global _PROVIDER_CACHE
    settings = embedding_settings_from_env()
    if not settings.enabled:
        return None
    if settings.backend != "local":
        raise RuntimeError(f"unsupported EvoInfer embedding backend: {settings.backend}")
    if _PROVIDER_CACHE is not None and _PROVIDER_CACHE[0] == settings:
        return _PROVIDER_CACHE[1]
    provider = LocalSentenceTransformerEmbeddingProvider(
        model_name=settings.model,
        device=settings.device,
    )
    _PROVIDER_CACHE = (settings, provider)
    return provider


def score_texts_with_optional_embedding(
    query: str,
    texts: Sequence[str],
) -> list[float] | None:
    """Score texts with the optional provider, falling back to existing retrieval."""

    global _WARNED_UNAVAILABLE
    try:
        provider = get_embedding_provider()
    except Exception as exc:
        if not _WARNED_UNAVAILABLE:
            logger.warning("EvoInfer embedding backend unavailable: {}", exc)
            _WARNED_UNAVAILABLE = True
        return None
    if provider is None:
        return None
    try:
        return provider.score_texts(query, texts)
    except Exception as exc:
        if not _WARNED_UNAVAILABLE:
            logger.warning("EvoInfer embedding scoring failed: {}", exc)
            _WARNED_UNAVAILABLE = True
        return None


def describe_embedding_runtime(*, load_model: bool = False) -> dict[str, object]:
    """Return doctor-friendly embedding backend status."""

    settings = embedding_settings_from_env()
    payload: dict[str, object] = {
        "enabled": settings.enabled,
        "backend": settings.backend,
        "model": settings.model if settings.enabled else None,
        "device": settings.device if settings.enabled else None,
        "ok": True,
        "detail": "disabled",
    }
    if not settings.enabled:
        return payload
    if settings.backend != "local":
        payload.update(
            {
                "ok": False,
                "detail": f"unsupported backend: {settings.backend}",
            }
        )
        return payload
    global SentenceTransformer
    if SentenceTransformer is None:
        SentenceTransformer = _resolve_sentence_transformer()
    if SentenceTransformer is None:
        payload.update(
            {
                "ok": False,
                "detail": "sentence-transformers is not installed",
            }
        )
        return payload
    if load_model:
        try:
            get_embedding_provider()
        except Exception as exc:
            payload.update({"ok": False, "detail": f"{type(exc).__name__}: {exc}"})
            return payload
    payload["detail"] = f"local model={settings.model} device={settings.device}"
    return payload


def _as_float_vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))
