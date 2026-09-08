import math

import httpx
from openai import AsyncOpenAI

from oink_finai.services.openai_privacy import openai_private_operation


def create_openai_client(*, api_key: str | None, timeout_seconds: float) -> AsyncOpenAI:
    """Create the shared runtime client with retries and redirects disabled."""
    if (
        not isinstance(api_key, str)
        or not api_key.strip()
        or any(ord(character) < 33 or ord(character) > 126 for character in api_key)
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise RuntimeError("OpenAI provider configuration is invalid")
    client = None
    with openai_private_operation():
        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.openai.com/v1",
                max_retries=0,
                timeout=timeout_seconds,
                http_client=httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False),
            )
        except Exception:
            pass
    if client is None:
        raise RuntimeError("OpenAI provider configuration is invalid")
    return client
