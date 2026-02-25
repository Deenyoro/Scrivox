"""Unified LLM client: handles both OpenAI-compatible and Anthropic Messages APIs."""

import json
import time
from typing import Optional

import requests


ANTHROPIC_API_VERSION = "2023-06-01"


def is_anthropic_api(api_base: str) -> bool:
    """Detect if the API base URL points to Anthropic's Messages API."""
    return "anthropic.com" in (api_base or "")


def _convert_openai_to_anthropic_messages(messages):
    """Convert OpenAI-format messages to Anthropic Messages API format.

    Handles:
    - Simple string content -> string content
    - Image content blocks (image_url -> Anthropic image source)
    - System messages -> extracted as top-level system parameter
    """
    system_text = None
    converted = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Extract system messages
        if role == "system":
            system_text = content if isinstance(content, str) else str(content)
            continue

        # Simple string content
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        # Content blocks (list of dicts) — need to convert image format
        if isinstance(content, list):
            new_blocks = []
            for block in content:
                block_type = block.get("type", "")

                if block_type == "text":
                    new_blocks.append({"type": "text", "text": block["text"]})

                elif block_type == "image_url":
                    # Convert OpenAI image_url to Anthropic image source
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        # Parse data URI: data:image/jpeg;base64,<data>
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        new_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        })
                    else:
                        # URL-based image (Anthropic supports this too)
                        new_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            },
                        })

                else:
                    # Pass through unknown block types
                    new_blocks.append(block)

            converted.append({"role": role, "content": new_blocks})

    return converted, system_text


def _parse_anthropic_response(resp):
    """Extract text from an Anthropic Messages API response."""
    try:
        data = resp.json()
        # Anthropic returns: {"content": [{"type": "text", "text": "..."}], ...}
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"].strip()
        return "[Anthropic: no text in response]"
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"[API parse error: {e}]"


def _parse_openai_response(resp):
    """Extract text from an OpenAI-compatible API response."""
    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        return f"[API parse error: {e}]"


def chat_completion(
    messages: list,
    model: str,
    api_key: str,
    api_base: str,
    max_tokens: int = 2000,
    temperature: Optional[float] = None,
    max_retries: int = 3,
    timeout: int = 120,
) -> Optional[str]:
    """Send a chat completion request, auto-detecting API format.

    Args:
        messages: OpenAI-format messages list.
        model: Model ID (e.g. "google/gemini-2.5-flash" or "claude-sonnet-4-20250514").
        api_key: API key for authentication.
        api_base: Full API endpoint URL.
        max_tokens: Max tokens in response.
        temperature: Sampling temperature (None = API default).
        max_retries: Number of retry attempts on transient errors.
        timeout: Request timeout in seconds.

    Returns:
        Response text string, or None on complete failure.
        Strings starting with "[" indicate errors.
    """
    use_anthropic = is_anthropic_api(api_base)

    if use_anthropic:
        return _anthropic_completion(
            messages, model, api_key, api_base,
            max_tokens, temperature, max_retries, timeout,
        )
    else:
        return _openai_completion(
            messages, model, api_key, api_base,
            max_tokens, temperature, max_retries, timeout,
        )


def _anthropic_completion(messages, model, api_key, api_base, max_tokens,
                          temperature, max_retries, timeout):
    """Make an Anthropic Messages API call."""
    converted_messages, system_text = _convert_openai_to_anthropic_messages(messages)

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": converted_messages,
    }
    if system_text:
        payload["system"] = system_text
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                api_base, headers=headers, json=payload, timeout=timeout,
            )
            if resp.status_code == 200:
                text = _parse_anthropic_response(resp)
                return text
            elif resp.status_code >= 500 or resp.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return f"[API error {resp.status_code} after {max_retries} retries]"
            else:
                body = ""
                try:
                    body = resp.text[:200]
                except Exception:
                    pass
                return f"[API error {resp.status_code}: {body}]"
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.SSLError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[{type(e).__name__} after {max_retries} retries]"

    return None


def _openai_completion(messages, model, api_key, api_base, max_tokens,
                       temperature, max_retries, timeout):
    """Make an OpenAI-compatible chat completion call."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                api_base, headers=headers, json=payload, timeout=timeout,
            )
            if resp.status_code == 200:
                text = _parse_openai_response(resp)
                return text
            elif resp.status_code >= 500 or resp.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return f"[API error {resp.status_code} after {max_retries} retries]"
            else:
                body = ""
                try:
                    body = resp.text[:200]
                except Exception:
                    pass
                return f"[API error {resp.status_code}: {body}]"
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.SSLError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[{type(e).__name__} after {max_retries} retries]"

    return None
