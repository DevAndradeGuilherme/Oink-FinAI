import asyncio
import logging
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from oink_finai.schemas.audio import MAX_TRANSCRIPT_CHARACTERS, AudioTranscription
from oink_finai.services.audio_transcriber import AudioTranscriber, ValidatedAudio
from oink_finai.services.transcription_errors import TranscriptionError, TranscriptionErrorCode

_PRIVATE_OPERATION: ContextVar[bool] = ContextVar("openai_audio_private_operation", default=False)
_SDK_LOGGERS = (
    "openai",
    "openai._base_client",
    "openai._response",
    "openai._legacy_response",
    "openai.audio.transcriptions",
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)
_AUDIO_FILES = {
    "audio/ogg": ("audio.ogg", "audio/ogg"),
    "audio/opus": ("audio.ogg", "audio/ogg"),
    "audio/mpeg": ("audio.mp3", "audio/mpeg"),
    "audio/mp4": ("audio.m4a", "audio/mp4"),
    "audio/wav": ("audio.wav", "audio/wav"),
    "audio/flac": ("audio.flac", "audio/flac"),
    "audio/webm": ("audio.webm", "audio/webm"),
}


class _PrivateOperationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _PRIVATE_OPERATION.get()


_PRIVATE_FILTER = _PrivateOperationFilter()


@contextmanager
def _private_operation() -> Iterator[None]:
    # Logger filters run before any handler sees SDK bodies, headers or transport URLs.
    # Context-local suppression leaves concurrent Gemini/Evolution operations unchanged.
    for name in _SDK_LOGGERS:
        logging.getLogger(name).addFilter(_PRIVATE_FILTER)
    token = _PRIVATE_OPERATION.set(True)
    try:
        yield
    finally:
        _PRIVATE_OPERATION.reset(token)


def _positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


class OpenAIAudioTranscriber(AudioTranscriber):
    """One in-memory transcription per durable attempt, using the official async SDK.

    Injected SDK clients remain caller-owned. Their HTTP transport must disable retries;
    redirect-enabled clients are rejected and SDK retries are always overridden here.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "gpt-4o-mini-transcribe",
        timeout_seconds: float = 90.0,
        language: str = "pt",
        max_audio_bytes: int = 10 * 1024 * 1024,
        max_duration_seconds: int = 300,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or any(ord(character) < 33 or ord(character) > 126 for character in api_key)
            or not isinstance(model, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model)
            or not _positive_number(timeout_seconds)
            or not isinstance(language, str)
            or not re.fullmatch(r"[a-z]{2}", language)
            or not isinstance(max_audio_bytes, int)
            or not _positive_number(max_audio_bytes)
            or not isinstance(max_duration_seconds, int)
            or not _positive_number(max_duration_seconds)
        ):
            raise TranscriptionError(TranscriptionErrorCode.CONFIGURATION, transient=False)
        self._model = model
        self._language = language
        self._timeout_seconds = timeout_seconds
        self._max_audio_bytes = max_audio_bytes
        self._max_duration_seconds = max_duration_seconds
        self._owns_client = client is None
        self._closed = False
        failure = None
        with _private_operation():
            try:
                # The SDK has no public HTTP-client accessor. Inspect without mutating the
                # borrowed client so a redirect cannot create another request in an attempt.
                if client is not None and client._client.follow_redirects:
                    raise TranscriptionError(TranscriptionErrorCode.CONFIGURATION, transient=False)
                self._client = (
                    AsyncOpenAI(
                        api_key=api_key,
                        base_url="https://api.openai.com/v1",
                        max_retries=0,
                        timeout=timeout_seconds,
                        http_client=httpx.AsyncClient(
                            timeout=timeout_seconds, follow_redirects=False
                        ),
                    )
                    if client is None
                    else client.with_options(max_retries=0, timeout=timeout_seconds)
                )
            except Exception:
                failure = TranscriptionError(TranscriptionErrorCode.CONFIGURATION, transient=False)
        if failure is not None:
            # Raise outside the except block: no raw SDK exception retained as context/cause.
            raise failure

    async def aclose(self) -> None:
        if self._closed:
            return
        failure = None
        with _private_operation():
            try:
                if self._owns_client:
                    async with asyncio.timeout(self._timeout_seconds):
                        await self._client.close()
                self._closed = True
            except Exception:
                failure = TranscriptionError(TranscriptionErrorCode.UNAVAILABLE, transient=True)
        if failure is not None:
            raise failure

    async def transcribe(self, audio: ValidatedAudio) -> AudioTranscription:
        failure = None
        with _private_operation():
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    if self._closed:
                        raise TranscriptionError(
                            TranscriptionErrorCode.CONFIGURATION, transient=False
                        )
                    filename, mime_type = self._validate_audio(audio)
                    response = await self._client.audio.transcriptions.create(
                        model=self._model,
                        file=(filename, audio.content, mime_type),
                        language=self._language,
                        response_format="json",
                    )
                    text = getattr(response, "text", None)
                    if not isinstance(text, str) or not text.strip():
                        raise TranscriptionError(
                            TranscriptionErrorCode.INVALID_RESPONSE, transient=False
                        )
                    if len(text.strip()) > MAX_TRANSCRIPT_CHARACTERS:
                        raise TranscriptionError(TranscriptionErrorCode.TOO_LONG, transient=False)
                    # The input language is a hint, not language detection returned by the API.
                    return AudioTranscription(
                        transcript=text.strip(), has_speech=True, detected_language=None
                    )
            except TranscriptionError as error:
                failure = error
            except (TimeoutError, APITimeoutError, httpx.TimeoutException):
                failure = TranscriptionError(TranscriptionErrorCode.TIMEOUT, transient=True)
            except APIStatusError as error:
                failure = self._status_error(error.status_code)
            except (APIConnectionError, httpx.TransportError, ConnectionError):
                failure = TranscriptionError(TranscriptionErrorCode.UNAVAILABLE, transient=True)
            except Exception:
                failure = TranscriptionError(
                    TranscriptionErrorCode.INVALID_RESPONSE, transient=False
                )
        raise failure

    def _validate_audio(self, audio: ValidatedAudio) -> tuple[str, str]:
        if not isinstance(audio.content, bytes) or not audio.content:
            raise TranscriptionError(TranscriptionErrorCode.INVALID_RESPONSE, transient=False)
        if len(audio.content) > self._max_audio_bytes:
            raise TranscriptionError(TranscriptionErrorCode.TOO_LONG, transient=False)
        mime_type = audio.mime_type.partition(";")[0].strip().lower()
        if mime_type not in _AUDIO_FILES:
            raise TranscriptionError(TranscriptionErrorCode.INVALID_RESPONSE, transient=False)
        if mime_type in {"audio/ogg", "audio/opus"} and not audio.content.startswith(b"OggS"):
            raise TranscriptionError(TranscriptionErrorCode.INVALID_RESPONSE, transient=False)
        duration = audio.declared_duration_seconds
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 0
            or duration > self._max_duration_seconds
        ):
            raise TranscriptionError(TranscriptionErrorCode.TOO_LONG, transient=False)
        return _AUDIO_FILES[mime_type]

    @staticmethod
    def _status_error(status: int) -> TranscriptionError:
        if status in {401, 403}:
            return TranscriptionError(TranscriptionErrorCode.AUTHENTICATION, transient=False)
        if status == 404:
            return TranscriptionError(TranscriptionErrorCode.MODEL_UNAVAILABLE, transient=False)
        if status == 429:
            return TranscriptionError(TranscriptionErrorCode.QUOTA_EXCEEDED, transient=True)
        if status in {408, 504}:
            return TranscriptionError(TranscriptionErrorCode.TIMEOUT, transient=True)
        if 500 <= status <= 599:
            return TranscriptionError(TranscriptionErrorCode.UNAVAILABLE, transient=True)
        return TranscriptionError(TranscriptionErrorCode.INVALID_RESPONSE, transient=False)
