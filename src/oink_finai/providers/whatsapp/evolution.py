import base64
import binascii
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from oink_finai.domain.enums import MediaType, MessageType
from oink_finai.providers.whatsapp.base import IncomingMessage, MediaMetadata, WhatsAppProvider
from oink_finai.providers.whatsapp.errors import MediaError, MediaErrorCode

ALLOWED_MIME_TYPES = frozenset(
    {"audio/ogg", "audio/opus", "audio/mpeg", "audio/mp4", "audio/aac", "audio/wav"}
)
WRAPPERS = (
    "ephemeralMessage",
    "documentWithCaptionMessage",
    "viewOnceMessage",
    "viewOnceMessageV2",
)


@dataclass(frozen=True, slots=True, repr=False)
class EvolutionMediaReference:
    message: dict[str, object]

    def __repr__(self) -> str:
        return "EvolutionMediaReference(<redacted>)"


class EvolutionWhatsAppProvider(WhatsAppProvider):
    def __init__(
        self,
        *,
        base_url: str | None,
        instance: str | None,
        api_key: str | None,
        max_bytes: int = 10 * 1024 * 1024,
        max_duration_seconds: int = 300,
        timeout_seconds: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._instance = instance
        self._api_key = api_key
        self._max_bytes = max_bytes
        self._max_duration_seconds = max_duration_seconds
        self._timeout = httpx.Timeout(timeout_seconds)
        self._client = client

    async def parse_webhook(self, payload: dict[str, object]) -> IncomingMessage:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("invalid webhook data")
        key = data.get("key")
        message = data.get("message")
        if not isinstance(key, dict) or not isinstance(message, dict):
            raise ValueError("invalid webhook message")

        message_id = key.get("id")
        remote_jid = key.get("remoteJid")
        if not isinstance(message_id, str) or not isinstance(remote_jid, str):
            raise ValueError("invalid webhook key")
        from_me = key.get("fromMe")
        if not isinstance(from_me, bool):
            raise ValueError("invalid fromMe")

        timestamp = data.get("messageTimestamp")
        if not isinstance(timestamp, int | float) or isinstance(timestamp, bool):
            raise ValueError("invalid message timestamp")
        sent_at = datetime.fromtimestamp(timestamp, tz=UTC)
        sender = remote_jid.split("@", 1)[0]
        instance = payload.get("instance")
        if not isinstance(instance, str):
            raise ValueError("invalid instance")

        unwrapped = self._unwrap_message(message)
        audio = unwrapped.get("audioMessage")
        if isinstance(audio, dict):
            mime = audio.get("mimetype")
            seconds = audio.get("seconds")
            ptt = audio.get("ptt", False)
            if not isinstance(mime, str) or not isinstance(ptt, bool):
                raise ValueError("invalid audio metadata")
            if seconds is not None and (
                not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 0
            ):
                raise ValueError("invalid audio duration")
            # The 2.3.7 endpoint resolves a stored message when only its key is supplied.
            # This avoids retaining the webhook's mediaKey, directPath, URL, or raw payload.
            reference_message = {
                "key": {"id": message_id, "remoteJid": remote_jid, "fromMe": from_me}
            }
            media = MediaMetadata(
                media_type=MediaType.AUDIO,
                declared_mime_type=self._safe_mime(mime),
                declared_duration_seconds=seconds,
                is_voice_note=ptt,
                reference=EvolutionMediaReference(reference_message),
            )
            return IncomingMessage(
                provider="evolution",
                instance_id=instance,
                external_message_id=message_id,
                sender_phone=sender,
                sent_at=sent_at,
                kind=MessageType.AUDIO,
                from_bot=from_me,
                media=media,
            )

        text = unwrapped.get("conversation")
        if not isinstance(text, str):
            extended = unwrapped.get("extendedTextMessage")
            text = extended.get("text") if isinstance(extended, dict) else None
        if isinstance(text, str):
            return IncomingMessage(
                provider="evolution",
                instance_id=instance,
                external_message_id=message_id,
                sender_phone=sender,
                sent_at=sent_at,
                kind=MessageType.TEXT,
                from_bot=from_me,
                content=text,
            )
        raise ValueError("unsupported message type")

    async def send_text(self, phone_number: str, text: str) -> None:
        raise NotImplementedError("sending is outside audio pipeline phase 1")

    async def download_media(self, media: MediaMetadata) -> bytes:
        if not self._base_url or not self._instance or not self._api_key:
            raise MediaError(MediaErrorCode.CONFIGURATION, transient=False)
        if not isinstance(media.reference, EvolutionMediaReference):
            raise MediaError(MediaErrorCode.CONFIGURATION, transient=False)
        mime = self._mime_for_comparison(media.declared_mime_type)
        if mime not in ALLOWED_MIME_TYPES:
            raise MediaError(MediaErrorCode.UNSUPPORTED_TYPE, transient=False)
        if (
            media.declared_duration_seconds is not None
            and media.declared_duration_seconds > self._max_duration_seconds
        ):
            raise MediaError(MediaErrorCode.TOO_LONG, transient=False)

        url = f"{self._base_url}/chat/getBase64FromMediaMessage/{quote(self._instance, safe='')}"
        try:
            if self._client is None:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        headers={"apikey": self._api_key},
                        json={"message": media.reference.message, "convertToMp4": False},
                        timeout=self._timeout,
                    )
            else:
                response = await self._client.post(
                    url,
                    headers={"apikey": self._api_key},
                    json={"message": media.reference.message, "convertToMp4": False},
                    timeout=self._timeout,
                )
        except httpx.TimeoutException:
            raise MediaError(MediaErrorCode.TIMEOUT, transient=True) from None
        except httpx.HTTPError:
            raise MediaError(MediaErrorCode.UNAVAILABLE, transient=True) from None

        if response.status_code in {401, 403}:
            raise MediaError(MediaErrorCode.AUTHENTICATION, transient=False)
        if response.status_code == 404:
            raise MediaError(MediaErrorCode.NOT_FOUND, transient=False)
        if response.status_code == 429 or response.status_code >= 500:
            raise MediaError(MediaErrorCode.UNAVAILABLE, transient=True)
        if response.status_code >= 400:
            raise MediaError(MediaErrorCode.UNAVAILABLE, transient=False)

        try:
            body = response.json()
        except ValueError:
            raise MediaError(MediaErrorCode.UNAVAILABLE, transient=False) from None
        encoded = body.get("base64") if isinstance(body, dict) else None
        if not isinstance(encoded, str):
            raise MediaError(MediaErrorCode.INVALID_BASE64, transient=False)
        max_encoded = ((self._max_bytes + 2) // 3) * 4
        if len(encoded) > max_encoded:
            raise MediaError(MediaErrorCode.TOO_LARGE, transient=False)
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise MediaError(MediaErrorCode.INVALID_BASE64, transient=False) from None
        if len(content) > self._max_bytes:
            raise MediaError(MediaErrorCode.TOO_LARGE, transient=False)
        response_mime = body.get("mimetype")
        if isinstance(response_mime, str) and self._mime_for_comparison(response_mime) != mime:
            raise MediaError(MediaErrorCode.CONTENT_MISMATCH, transient=False)
        if not self._signature_matches(mime, content):
            raise MediaError(MediaErrorCode.CONTENT_MISMATCH, transient=False)
        return content

    @staticmethod
    def _unwrap_message(message: dict[str, Any]) -> dict[str, Any]:
        current = message
        for _ in range(len(WRAPPERS)):
            wrapper = next((name for name in WRAPPERS if isinstance(current.get(name), dict)), None)
            if wrapper is None:
                break
            wrapped = current[wrapper].get("message")
            if not isinstance(wrapped, dict):
                raise ValueError("invalid message wrapper")
            current = wrapped
        return current

    @staticmethod
    def _safe_mime(value: str) -> str:
        if (
            not value
            or len(value) > 200
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("invalid MIME type")
        return value.strip()

    @staticmethod
    def _mime_for_comparison(value: str) -> str:
        return value.partition(";")[0].strip().lower()

    @staticmethod
    def _signature_matches(mime: str, content: bytes) -> bool:
        if mime == "audio/ogg":
            return content.startswith(b"OggS")
        if mime == "audio/mpeg":
            return content.startswith(b"ID3") or (
                len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0
            )
        if mime == "audio/mp4":
            return len(content) >= 12 and content[4:8] == b"ftyp"
        if mime == "audio/wav":
            return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE"
        return True  # AAC has multiple valid transports; MIME/size validation still applies.
