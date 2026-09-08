import asyncio
import math
import re

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from oink_finai.schemas.audio import MAX_TRANSCRIPT_CHARACTERS, AudioTranscription
from oink_finai.services.audio_transcriber import AudioTranscriber, ValidatedAudio
from oink_finai.services.openai_privacy import openai_private_operation
from oink_finai.services.transcription_errors import TranscriptionError, TranscriptionErrorCode

_AUDIO_FILES = {
    "audio/ogg": ("audio.ogg", "audio/ogg"),
    "audio/opus": ("audio.ogg", "audio/ogg"),
    "audio/mpeg": ("audio.mp3", "audio/mpeg"),
    "audio/mp4": ("audio.m4a", "audio/mp4"),
    "audio/wav": ("audio.wav", "audio/wav"),
    "audio/flac": ("audio.flac", "audio/flac"),
    "audio/webm": ("audio.webm", "audio/webm"),
}


def _build_ogg_crc_table() -> tuple[int, ...]:
    table = []
    for value in range(256):
        checksum = value << 24
        for _ in range(8):
            checksum = (checksum << 1) ^ (0x04C11DB7 if checksum & 0x80000000 else 0)
        table.append(checksum & 0xFFFFFFFF)
    return tuple(table)


_OGG_CRC_TABLE = _build_ogg_crc_table()


def _ogg_crc(content: bytes | bytearray) -> int:
    checksum = 0
    for value in content:
        checksum = ((checksum << 8) & 0xFFFFFFFF) ^ _OGG_CRC_TABLE[(checksum >> 24) ^ value]
    return checksum


def _opus_packet_is_structurally_valid(packet: bytes) -> bool:
    if not packet:
        return False
    frame_code = packet[0] & 0x03
    if frame_code == 3:
        if len(packet) < 2:
            return False
        frame_count = packet[1] & 0x3F
    else:
        frame_count = 1 if frame_code == 0 else 2
    if not 0 < frame_count <= 48:
        return False
    configuration = packet[0] >> 3
    if configuration < 12:
        frame_duration_ms = (10, 20, 40, 60)[configuration % 4]
    elif configuration < 16:
        frame_duration_ms = (10, 20)[configuration % 2]
    else:
        frame_duration_ms = (2.5, 5, 10, 20)[configuration % 4]
    return frame_count * frame_duration_ms <= 120


def _validate_opus_headers(packets: list[bytes]) -> bool:
    if len(packets) < 3:
        return False
    head = packets[0]
    if not head.startswith(b"OpusHead") or len(head) < 19:
        return False
    if not 0 < head[8] < 16:
        return False
    channels = head[9]
    mapping_family = head[18]
    if channels == 0 or (mapping_family == 0 and len(head) != 19):
        return False
    if mapping_family != 0 and len(head) < 21 + channels:
        return False

    tags = packets[1]
    if not tags.startswith(b"OpusTags") or len(tags) < 16:
        return False
    offset = 8
    vendor_length = int.from_bytes(tags[offset : offset + 4], "little")
    offset += 4
    if offset + vendor_length + 4 > len(tags):
        return False
    offset += vendor_length
    comment_count = int.from_bytes(tags[offset : offset + 4], "little")
    offset += 4
    for _ in range(comment_count):
        if offset + 4 > len(tags):
            return False
        comment_length = int.from_bytes(tags[offset : offset + 4], "little")
        offset += 4
        if offset + comment_length > len(tags):
            return False
        offset += comment_length
    return all(_opus_packet_is_structurally_valid(packet) for packet in packets[2:])


def _valid_ogg_opus_stream(content: bytes) -> bool:
    offset = 0
    stream_serial = None
    expected_sequence = None
    pending_packet = bytearray()
    packets: list[bytes] = []
    page_index = 0
    saw_eos = False

    while offset < len(content):
        if saw_eos or offset + 27 > len(content) or content[offset : offset + 4] != b"OggS":
            return False
        if content[offset + 4] != 0:
            return False
        flags = content[offset + 5]
        serial = int.from_bytes(content[offset + 14 : offset + 18], "little")
        sequence = int.from_bytes(content[offset + 18 : offset + 22], "little")
        segment_count = content[offset + 26]
        table_start = offset + 27
        table_end = table_start + segment_count
        if table_end > len(content):
            return False
        lacing_values = content[table_start:table_end]
        payload_end = table_end + sum(lacing_values)
        if payload_end > len(content):
            return False

        page = bytearray(content[offset:payload_end])
        stored_checksum = int.from_bytes(page[22:26], "little")
        page[22:26] = b"\0\0\0\0"
        if _ogg_crc(page) != stored_checksum:
            return False
        if page_index == 0:
            if not flags & 0x02 or sequence != 0:
                return False
            stream_serial = serial
            expected_sequence = sequence
        elif flags & 0x02:
            return False
        if serial != stream_serial or sequence != expected_sequence:
            return False
        if bool(flags & 0x01) != bool(pending_packet):
            return False
        expected_sequence += 1

        payload_offset = table_end
        for length in lacing_values:
            pending_packet.extend(content[payload_offset : payload_offset + length])
            payload_offset += length
            if length < 255:
                packets.append(bytes(pending_packet))
                pending_packet.clear()

        saw_eos = bool(flags & 0x04)
        page_index += 1
        offset = payload_end

    return (
        offset == len(content)
        and not pending_packet
        and page_index > 0
        and _validate_opus_headers(packets)
    )


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
        model: str = "gpt-transcribe",
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
        with openai_private_operation():
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
        with openai_private_operation():
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
        with openai_private_operation():
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
        if mime_type in {"audio/ogg", "audio/opus"} and not _valid_ogg_opus_stream(audio.content):
            raise TranscriptionError(TranscriptionErrorCode.INVALID_RESPONSE, transient=False)
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
