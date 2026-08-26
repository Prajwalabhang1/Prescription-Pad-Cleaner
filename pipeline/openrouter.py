"""OpenRouter vision-chat adapter used by the reconstruction pipeline."""

from __future__ import annotations

import base64
from typing import Any

import requests

from config import (
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_MODEL,
    get_openrouter_api_key,
)


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterRequestError(RuntimeError):
    """An OpenRouter request could not produce a usable model response."""


def _response_message(response: requests.Response) -> str:
    try:
        payload: Any = response.json()
    except ValueError:
        return response.text.strip() or response.reason
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or response.reason)
    return str(error or response.reason)


def _send_openrouter_request(
    api_key: str,
    model: str,
    image_bytes: bytes,
    image_mime_type: str,
    system_instruction: str,
    user_instruction: str,
    max_tokens: int,
    response_format: dict[str, str] | None = None,
) -> str:
    image_data = base64.b64encode(image_bytes).decode("ascii")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime_type};base64,{image_data}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if response_format:
        payload["response_format"] = response_format
    try:
        response = requests.post(
            OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Prescription Pad Cleaner",
            },
            json=payload,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise OpenRouterRequestError(f"OpenRouter request failed: {exc}") from exc
    if not response.ok:
        raise OpenRouterRequestError(
            f"OpenRouter request failed ({response.status_code}): {_response_message(response)}"
        )
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise OpenRouterRequestError("OpenRouter returned no completion content.") from exc
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise OpenRouterRequestError("OpenRouter returned an empty completion.")
    return content.strip()


def generate_openrouter_content(
    image_bytes: bytes,
    image_mime_type: str,
    system_instruction: str,
    user_instruction: str,
    max_tokens: int,
    response_format: dict[str, str] | None = None,
    model: str | None = None,
) -> str:
    """Send one image-and-text request through OpenRouter and return text."""
    api_key = get_openrouter_api_key()
    if not api_key:
        raise OpenRouterRequestError("OPENROUTER_API_KEY is not configured.")

    primary_model = model or OPENROUTER_MODEL
    try:
        return _send_openrouter_request(
            api_key,
            primary_model,
            image_bytes,
            image_mime_type,
            system_instruction,
            user_instruction,
            max_tokens,
            response_format,
        )
    except OpenRouterRequestError as exc:
        fallback_model = OPENROUTER_FALLBACK_MODEL
        if fallback_model and fallback_model != primary_model and not model:
            try:
                return _send_openrouter_request(
                    api_key,
                    fallback_model,
                    image_bytes,
                    image_mime_type,
                    system_instruction,
                    user_instruction,
                    max_tokens,
                    response_format,
                )
            except Exception:
                pass
        raise exc

