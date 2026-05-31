"""Speech transcription API routes for the web UI."""

from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/speech", tags=["speech"])

ENV_STT_MODEL = "KIMI_WEB_STT_MODEL"
ENV_STT_DEVICE = "KIMI_WEB_STT_DEVICE"
ENV_STT_COMPUTE_TYPE = "KIMI_WEB_STT_COMPUTE_TYPE"
ENV_STT_LANGUAGE = "KIMI_WEB_STT_LANGUAGE"
ENV_STT_MAX_UPLOAD_MB = "KIMI_WEB_STT_MAX_UPLOAD_MB"

DEFAULT_STT_MODEL = "medium"
DEFAULT_STT_DEVICE = "auto"
DEFAULT_STT_COMPUTE_TYPE = "float16"
DEFAULT_MAX_UPLOAD_MB = 50
READ_CHUNK_SIZE = 1024 * 1024
MAX_AUDIO_UPLOAD_SIZE = (
    int(os.environ.get(ENV_STT_MAX_UPLOAD_MB, str(DEFAULT_MAX_UPLOAD_MB))) * 1024 * 1024
)
ALLOWED_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


class SpeechTranscriptionResponse(BaseModel):
    """Speech transcription response."""

    text: str = Field(description="Transcribed text.")
    language: str | None = Field(default=None, description="Detected or configured language.")
    duration_ms: int = Field(description="End-to-end transcription duration in milliseconds.")


class _WhisperSegment(Protocol):
    text: str


class _WhisperInfo(Protocol):
    language: str | None


class _WhisperModel(Protocol):
    def transcribe(
        self,
        audio: str,
        *,
        beam_size: int,
        vad_filter: bool,
        language: str | None = None,
    ) -> tuple[Iterable[_WhisperSegment], _WhisperInfo]: ...


class _WhisperModelFactory(Protocol):
    def __call__(
        self,
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
    ) -> _WhisperModel: ...


class _FasterWhisperModule(Protocol):
    WhisperModel: _WhisperModelFactory


_MODEL_LOCK = Lock()
_model_cache: _WhisperModel | None = None


def _safe_audio_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_AUDIO_SUFFIXES:
        return suffix
    return ".webm"


async def _save_upload_to_temp_file(file: UploadFile) -> Path:
    suffix = _safe_audio_suffix(file.filename)
    total = 0
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await file.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_AUDIO_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            "Audio upload too large "
                            f"(max {MAX_AUDIO_UPLOAD_SIZE // 1024 // 1024}MB)."
                        ),
                    )
                tmp.write(chunk)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    if total == 0:
        assert tmp_path is not None
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio upload is empty.",
        )
    assert tmp_path is not None
    return tmp_path


def _get_whisper_model() -> _WhisperModel:
    global _model_cache

    with _MODEL_LOCK:
        if _model_cache is not None:
            return _model_cache

        try:
            faster_whisper = cast(
                _FasterWhisperModule,
                importlib.import_module("faster_whisper"),
            )
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install the web-stt extra before using voice input."
            ) from exc

        model_name = os.environ.get(ENV_STT_MODEL, DEFAULT_STT_MODEL)
        device = os.environ.get(ENV_STT_DEVICE, DEFAULT_STT_DEVICE)
        compute_type = os.environ.get(ENV_STT_COMPUTE_TYPE, DEFAULT_STT_COMPUTE_TYPE)
        _model_cache = faster_whisper.WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        return _model_cache


def _transcribe_audio_file(path: Path) -> tuple[str, str | None]:
    model = _get_whisper_model()
    configured_language = os.environ.get(ENV_STT_LANGUAGE) or None
    segments, info = model.transcribe(
        str(path),
        beam_size=5,
        vad_filter=True,
        language=configured_language,
    )
    text = "".join(segment.text for segment in segments).strip()
    return text, info.language


@router.post("/transcribe", summary="Transcribe uploaded speech")
async def transcribe_speech(file: UploadFile) -> SpeechTranscriptionResponse:
    """Transcribe a short browser-recorded audio upload with faster-whisper."""
    started_at = time.perf_counter()
    audio_path = await _save_upload_to_temp_file(file)
    try:
        try:
            text, language = await asyncio.to_thread(_transcribe_audio_file, audio_path)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
    finally:
        audio_path.unlink(missing_ok=True)

    return SpeechTranscriptionResponse(
        text=text,
        language=language,
        duration_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
    )
