import asyncio
import base64
import binascii
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from oink_finai.providers.whatsapp.base import (
    InteractiveMessage,
    InteractiveMessageUnsupportedError,
    WhatsAppProvider,
)
from oink_finai.providers.whatsapp.media_errors import MediaError, MediaErrorCode
from oink_finai.schemas.whatsapp import InboundMedia, InboundWhatsAppMessage

ALLOWED_AUDIO_MIME_TYPES = frozenset(
    {"audio/ogg", "audio/opus", "audio/mpeg", "audio/mp4", "audio/aac", "audio/wav"}
)
MEDIA_MESSAGE_WRAPPERS = (
    "ephemeralMessage",
    "documentWithCaptionMessage",
    "viewOnceMessage",
    "viewOnceMessageV2",
)


@dataclass(frozen=True, slots=True, repr=False)
class EvolutionMediaReference:
    external_message_id: str
    remote_jid: str
    from_me: bool

    def __repr__(self) -> str:
        return "EvolutionMediaReference(<redacted>)"

    def request_message(self) -> dict[str, object]:
        return {
            "key": {
                "id": self.external_message_id,
                "remoteJid": self.remote_jid,
                "fromMe": self.from_me,
            }
        }


class EvolutionProviderError(RuntimeError):
    """Raised when Evolution cannot send a message."""

    def __init__(self, message: str, *, outcome_unknown: bool = False) -> None:
        super().__init__(message)
        self.outcome_unknown = outcome_unknown


class EvolutionWebhookInstanceError(ValueError):
    """Raised when a webhook does not identify the configured Evolution instance."""


class EvolutionWhatsAppProvider(WhatsAppProvider):
    """Evolution client with retries limited to failures known to precede transmission.

    ``max_retries`` controls only retries while acquiring a pooled connection or establishing
    a connection. A sent or potentially sent ``sendText`` request is never retried.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        instance: str,
        *,
        timeout_seconds: float = 10.0,
        media_timeout_seconds: float = 15.0,
        media_max_bytes: int = 10 * 1024 * 1024,
        media_max_duration_seconds: int = 300,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        parsed_base_url = httpx.URL(normalized_base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.host:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not instance.strip():
            raise ValueError("instance must not be empty")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if not math.isfinite(media_timeout_seconds) or media_timeout_seconds <= 0:
            raise ValueError("media_timeout_seconds must be a positive finite number")
        if (
            isinstance(media_max_bytes, bool)
            or not isinstance(media_max_bytes, int)
            or media_max_bytes <= 0
        ):
            raise ValueError("media_max_bytes must be a positive integer")
        if (
            isinstance(media_max_duration_seconds, bool)
            or not isinstance(media_max_duration_seconds, int)
            or media_max_duration_seconds <= 0
        ):
            raise ValueError("media_max_duration_seconds must be a positive integer")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")

        self._base_url = normalized_base_url
        self._api_key = api_key
        self._instance = instance
        self._timeout_seconds = timeout_seconds
        self._media_timeout_seconds = media_timeout_seconds
        self._media_max_bytes = media_max_bytes
        self._media_max_duration_seconds = media_max_duration_seconds
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def __aenter__(self) -> "EvolutionWhatsAppProvider":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def send_text(self, phone_number: str, text: str) -> str | None:
        return await self._send_message(
            "sendText",
            {"number": phone_number, "text": text},
            interactive=False,
        )

    async def send_interactive(self, phone_number: str, message: InteractiveMessage) -> str | None:
        payload: dict[str, object] = {
            "number": phone_number,
            "title": message.title,
            "buttons": [
                {"type": "reply", "displayText": action.label, "id": action.id}
                for action in message.actions
            ],
        }
        if message.body:
            payload["description"] = message.body
        return await self._send_message("sendButtons", payload, interactive=True)

    async def _send_message(
        self, endpoint: str, payload: dict[str, object], *, interactive: bool
    ) -> str | None:
        url = f"{self._base_url}/message/{endpoint}/{quote(self._instance, safe='')}"
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    url,
                    json=payload,
                    headers={"apikey": self._api_key},
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                try:
                    body = response.json()
                except ValueError:
                    return None
                key = body.get("key") if isinstance(body, dict) else None
                return key.get("id") if isinstance(key, dict) else None
            except httpx.HTTPStatusError as exc:
                if interactive and exc.response.status_code in {400, 404, 405, 422}:
                    raise InteractiveMessageUnsupportedError from None
                raise EvolutionProviderError(
                    "Evolution send result is unknown after an HTTP response",
                    outcome_unknown=True,
                ) from None
            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
                pass
            except httpx.HTTPError:
                raise EvolutionProviderError(
                    "Evolution send result is unknown after a transport failure",
                    outcome_unknown=True,
                ) from None

            if attempt == self._max_retries:
                raise EvolutionProviderError(
                    "Evolution connection failed after limited retries"
                ) from None
            await asyncio.sleep(0.1 * (2**attempt))

    async def parse_webhook(self, payload: dict[str, object]) -> InboundWhatsAppMessage | None:
        if payload.get("event") != "messages.upsert":
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        instance_id = self._extract_instance(payload, data)
        if instance_id is None:
            raise EvolutionWebhookInstanceError("Evolution webhook instance is missing")
        if instance_id != self._instance:
            raise EvolutionWebhookInstanceError("Evolution webhook instance is not allowed")

        key = data.get("key")
        message = data.get("message")
        if not isinstance(key, dict) or not isinstance(message, dict):
            return None

        external_id = key.get("id")
        remote_jid = key.get("remoteJid")
        if not all(isinstance(value, str) and value for value in (external_id, remote_jid)):
            return None

        from_me = key.get("fromMe")
        if not isinstance(from_me, bool):
            return None
        remote_jid_alt = self._first_string(
            key, data, names=("remoteJidAlt", "remoteJidAlternative")
        )
        participant_jid = self._first_string(key, data, names=("participant", "participantJid"))
        participant_jid_alt = self._first_string(
            key, data, names=("participantAlt", "participantJidAlt")
        )
        content = self._unwrap_message(message)
        message_type, text = self._extract_content(content, data.get("messageType"))
        interaction_id = self._extract_interaction_id(content)
        media = self._extract_audio_media(content, key)
        timestamp = self._parse_timestamp(data.get("messageTimestamp"), payload.get("date_time"))
        phone_number = self._phone_from_jids(remote_jid_alt, remote_jid)

        return InboundWhatsAppMessage(
            external_message_id=external_id,
            instance_id=instance_id,
            remote_jid=remote_jid,
            remote_jid_alt=remote_jid_alt,
            participant_jid=participant_jid,
            participant_jid_alt=participant_jid_alt,
            phone_number=phone_number,
            from_me=from_me,
            message_type=message_type,
            text_content=text,
            interaction_id=interaction_id,
            media=media,
            timestamp=timestamp,
        )

    async def download_media(self, media: InboundMedia) -> bytes:
        reference = media.reference
        if media.media_type != "audio" or not isinstance(reference, EvolutionMediaReference):
            raise MediaError(MediaErrorCode.CONFIGURATION, transient=False)
        mime_type = self._mime_for_comparison(media.declared_mime_type)
        if mime_type not in ALLOWED_AUDIO_MIME_TYPES:
            raise MediaError(MediaErrorCode.UNSUPPORTED_TYPE, transient=False)
        if (
            media.declared_duration_seconds is not None
            and media.declared_duration_seconds > self._media_max_duration_seconds
        ):
            raise MediaError(MediaErrorCode.TOO_LONG, transient=False)

        url = f"{self._base_url}/chat/getBase64FromMediaMessage/{quote(self._instance, safe='')}"
        try:
            response = await self._client.post(
                url,
                json={"message": reference.request_message(), "convertToMp4": False},
                headers={"apikey": self._api_key},
                timeout=self._media_timeout_seconds,
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
        maximum_encoded_length = ((self._media_max_bytes + 2) // 3) * 4
        if len(encoded) > maximum_encoded_length:
            raise MediaError(MediaErrorCode.TOO_LARGE, transient=False)
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise MediaError(MediaErrorCode.INVALID_BASE64, transient=False) from None
        if len(content) > self._media_max_bytes:
            raise MediaError(MediaErrorCode.TOO_LARGE, transient=False)

        response_mime = body.get("mimetype")
        if isinstance(response_mime, str) and self._mime_for_comparison(response_mime) != mime_type:
            raise MediaError(MediaErrorCode.CONTENT_MISMATCH, transient=False)
        if not self._signature_matches(mime_type, content):
            raise MediaError(MediaErrorCode.CONTENT_MISMATCH, transient=False)
        return content

    @staticmethod
    def _unwrap_message(message: dict[str, Any]) -> dict[str, Any]:
        content = message
        for _ in range(len(MEDIA_MESSAGE_WRAPPERS)):
            wrapper = next(
                (name for name in MEDIA_MESSAGE_WRAPPERS if isinstance(content.get(name), dict)),
                None,
            )
            if wrapper is None:
                return content
            nested = content[wrapper].get("message")
            if not isinstance(nested, dict):
                return {}
            content = nested
        return content

    @staticmethod
    def _extract_audio_media(message: dict[str, Any], key: dict[str, Any]) -> InboundMedia | None:
        audio = message.get("audioMessage")
        if not isinstance(audio, dict):
            return None
        mime_type = audio.get("mimetype")
        duration = audio.get("seconds")
        is_voice_note = audio.get("ptt", False)
        external_id = key.get("id")
        remote_jid = key.get("remoteJid")
        from_me = key.get("fromMe")
        if (
            not isinstance(mime_type, str)
            or not EvolutionWhatsAppProvider._is_safe_mime(mime_type)
            or (
                duration is not None
                and (not isinstance(duration, int) or isinstance(duration, bool) or duration < 0)
            )
            or not isinstance(is_voice_note, bool)
            or not isinstance(external_id, str)
            or not isinstance(remote_jid, str)
            or not isinstance(from_me, bool)
        ):
            return None
        return InboundMedia(
            media_type="audio",
            declared_mime_type=mime_type.strip(),
            declared_duration_seconds=duration,
            is_voice_note=is_voice_note,
            reference=EvolutionMediaReference(external_id, remote_jid, from_me),
        )

    @staticmethod
    def _is_safe_mime(value: str) -> bool:
        return (
            bool(value.strip())
            and len(value) <= 200
            and not any(ord(character) < 32 or ord(character) == 127 for character in value)
        )

    @staticmethod
    def _mime_for_comparison(value: str) -> str:
        return value.partition(";")[0].strip().lower()

    @staticmethod
    def _signature_matches(mime_type: str, content: bytes) -> bool:
        if mime_type == "audio/ogg":
            return content.startswith(b"OggS")
        if mime_type == "audio/opus":
            return content.startswith((b"OggS", b"OpusHead"))
        if mime_type == "audio/mpeg":
            return content.startswith(b"ID3") or (
                len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0
            )
        if mime_type == "audio/mp4":
            return len(content) >= 12 and content[4:8] == b"ftyp"
        if mime_type == "audio/wav":
            return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE"
        if mime_type == "audio/aac":
            return content.startswith(b"ADIF") or (
                len(content) >= 2 and content[0] == 0xFF and content[1] & 0xF6 == 0xF0
            )
        return True

    @staticmethod
    def _first_string(*containers: dict[str, Any], names: tuple[str, ...]) -> str | None:
        for container in containers:
            for name in names:
                value = container.get(name)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _phone_from_jids(*jids: str | None) -> str:
        for jid in jids:
            if jid and jid.endswith("@s.whatsapp.net"):
                return jid.partition("@")[0]
        return next(jid.partition("@")[0] for jid in jids if jid)

    @staticmethod
    def _extract_instance(payload: dict[str, object], data: dict[str, Any]) -> str | None:
        envelope_instance = payload.get("instance")
        if isinstance(envelope_instance, str) and envelope_instance:
            return envelope_instance
        data_instance = data.get("instanceId")
        if isinstance(data_instance, str) and data_instance:
            return data_instance
        return None

    @staticmethod
    def _extract_content(message: dict[str, Any], reported_type: object) -> tuple[str, str | None]:
        conversation = message.get("conversation")
        if isinstance(conversation, str):
            return "conversation", conversation
        extended = message.get("extendedTextMessage")
        if isinstance(extended, dict) and isinstance(extended.get("text"), str):
            return "extendedTextMessage", extended["text"]
        if isinstance(reported_type, str) and reported_type:
            return reported_type, None
        return next(iter(message), "unknown"), None

    @staticmethod
    def _extract_interaction_id(message: dict[str, Any]) -> str | None:
        interactive = message.get("interactiveResponseMessage")
        if isinstance(interactive, dict):
            native_flow = interactive.get("nativeFlowResponseMessage")
            if isinstance(native_flow, dict):
                params_json = native_flow.get("paramsJson")
                if isinstance(params_json, str):
                    try:
                        params = json.loads(params_json)
                    except (TypeError, ValueError):
                        return None
                    action_id = params.get("id") if isinstance(params, dict) else None
                    return action_id if isinstance(action_id, str) and action_id else None
        buttons = message.get("buttonsResponseMessage")
        if isinstance(buttons, dict):
            action_id = buttons.get("selectedButtonId")
            return action_id if isinstance(action_id, str) and action_id else None
        template = message.get("templateButtonReplyMessage")
        if isinstance(template, dict):
            action_id = template.get("selectedId")
            return action_id if isinstance(action_id, str) and action_id else None
        return None

    @staticmethod
    def _parse_timestamp(raw: object, fallback: object) -> datetime:
        timestamp = EvolutionWhatsAppProvider._safe_unix_timestamp(raw)
        if timestamp is not None:
            return timestamp
        if isinstance(fallback, str):
            try:
                parsed_fallback = datetime.fromisoformat(fallback.replace("Z", "+00:00"))
                if parsed_fallback.tzinfo is None:
                    return parsed_fallback.replace(tzinfo=UTC)
                return parsed_fallback
            except ValueError:
                pass
        return datetime.now(UTC)

    @staticmethod
    def _safe_unix_timestamp(raw: object) -> datetime | None:
        if isinstance(raw, bool):
            return None
        value: int | float
        if isinstance(raw, (int, float)):
            value = raw
        elif isinstance(raw, str) and raw.lstrip("+-").isdigit():
            try:
                value = int(raw)
            except ValueError:
                return None
        else:
            return None

        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError, TypeError):
            return None
