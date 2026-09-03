from copy import deepcopy


def audio_webhook(
    *,
    mime: str = "audio/ogg; codecs=opus",
    seconds: int = 12,
    ptt: bool = True,
    wrapper: str | None = None,
    jid: str = "5511999999999@s.whatsapp.net",
    from_me: bool = False,
) -> dict[str, object]:
    message: dict[str, object] = {
        "audioMessage": {
            "url": "https://untrusted.invalid/encrypted",
            "mimetype": mime,
            "fileSha256": {"0": 1},
            "fileLength": {"low": 8},
            "seconds": seconds,
            "ptt": ptt,
            "mediaKey": {"0": 99},
            "directPath": "/opaque/path",
        }
    }
    if wrapper:
        message = {wrapper: {"message": message}}
    return {
        "event": "messages.upsert",
        "instance": "test-instance",
        "data": {
            "key": {"remoteJid": jid, "fromMe": from_me, "id": "MSG-001"},
            "message": message,
            "messageType": "audioMessage",
            "messageTimestamp": 1_700_000_000,
        },
        "destination": "https://app.invalid/webhooks/evolution",
        "date_time": "2023-11-14T22:13:20.000Z",
        "sender": "redacted",
        "server_url": "https://evolution.invalid",
        "apikey": None,
    }


def text_webhook() -> dict[str, object]:
    payload = deepcopy(audio_webhook())
    payload["data"]["message"] = {"conversation": "cafe 10"}
    payload["data"]["messageType"] = "conversation"
    return payload
