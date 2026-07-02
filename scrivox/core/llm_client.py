"""Unified LLM client: handles both OpenAI-compatible and Anthropic Messages APIs."""

import json
import time
from typing import Optional

import requests


ANTHROPIC_API_VERSION = "2023-06-01"

# All error sentinels produced by this module (and the vision module) start
# with one of these prefixes. Callers must use is_error_response() rather
# than checking startswith("[") — legitimate content can start with "[".
_ERROR_PREFIXES = (
    "[API error",
    "[Vision error",
)


def is_error_response(text) -> bool:
    """True if text is an error sentinel (or empty) rather than real content."""
    if not text:
        return True
    return text.startswith(_ERROR_PREFIXES)


def is_anthropic_api(api_base: str) -> bool:
    """Detect if the API base URL points to Anthropic's Messages API."""
    return "anthropic.com" in (api_base or "")


def _convert_openai_to_anthropic_messages(messages):
    """Convert OpenAI-format messages to Anthropic Messages API format.

    Handles:
    - Simple string content -> string content
    - Image content blocks (image_url -> Anthropic image source)
    - System messages -> extracted (and concatenated) as top-level system param
    """
    system_parts = []
    converted = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Extract system messages (concatenate if there are several)
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
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
                    new_blocks.append({"type": "text", "text": block.get("text", "")})

                elif block_type == "image_url":
                    # Convert OpenAI image_url to Anthropic image source
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        # Parse data URI: data:image/jpeg;base64,<data>
                        header, data = url.split(",", 1)
                        media_parts = header.split(":", 1)
                        media_type = media_parts[1].split(";")[0] if len(media_parts) > 1 and media_parts[1] else "image/jpeg"
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

    system_text = "\n\n".join(p for p in system_parts if p) or None
    return converted, system_text


def _parse_anthropic_response(resp):
    """Extract text from an Anthropic Messages API response.

    Returns None if the response has no usable text so the caller can retry.
    """
    try:
        data = resp.json()
        # Anthropic returns: {"content": [{"type": "text", "text": "..."}], ...}
        texts = [
            block["text"] for block in data.get("content", [])
            if block.get("type") == "text" and block.get("text")
        ]
        joined = "\n".join(texts).strip()
        return joined or None
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None


def _parse_openai_response(resp):
    """Extract text from an OpenAI-compatible API response.

    Returns None if the response has no usable text content (e.g. the model
    spent the whole token budget on reasoning, or the provider returned a
    200-with-error body) so the caller can retry.
    """
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Some OpenAI-compatible backends return content as a list of blocks
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if content is None or not content.strip():
            return None
        return content.strip()
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        return None


def _retry_delay(resp, attempt):
    """Backoff delay in seconds: honor Retry-After on 429, exponential otherwise."""
    if resp is not None and resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return min(60.0, max(1.0, float(retry_after)))
            except ValueError:
                pass
        # Rate limits rarely clear in 1-2s — back off harder than 5xx
        return min(60.0, 5.0 * (2 ** attempt))
    return float(2 ** attempt)


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
        model: Model ID (e.g. "google/gemini-2.5-flash" or "claude-sonnet-4-6").
        api_key: API key for authentication.
        api_base: Full API endpoint URL.
        max_tokens: Max tokens in response.
        temperature: Sampling temperature (None = API default).
        max_retries: Number of retry attempts on transient errors.
        timeout: Request timeout in seconds.

    Returns:
        Response text string, or None on complete failure.
        Use is_error_response() to detect error sentinels.
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


def _post_with_retry(api_base, headers, payload, parse_fn, max_retries, timeout):
    """Shared request/retry driver for both API formats.

    Retries on 429/5xx (honoring Retry-After), transient network errors, and
    200s whose content parses to nothing. Returns the parsed text or an
    "[API error ...]" sentinel (see is_error_response).
    """
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                api_base, headers=headers, json=payload, timeout=timeout,
            )
            if resp.status_code == 200:
                text = parse_fn(resp)
                if text is None:
                    # Empty/unparseable content (reasoning consumed max_tokens,
                    # or a 200-with-error body) — retry
                    if attempt < max_retries - 1:
                        time.sleep(_retry_delay(None, attempt))
                        continue
                    return f"[API error: empty response after {max_retries} retries]"
                return text
            elif resp.status_code >= 500 or resp.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(_retry_delay(resp, attempt))
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
                requests.exceptions.SSLError,
                requests.exceptions.ChunkedEncodingError) as e:
            if attempt < max_retries - 1:
                time.sleep(_retry_delay(None, attempt))
            else:
                return f"[API error: {type(e).__name__} after {max_retries} retries]"

    return None


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

    return _post_with_retry(api_base, headers, payload,
                            _parse_anthropic_response, max_retries, timeout)


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

    return _post_with_retry(api_base, headers, payload,
                            _parse_openai_response, max_retries, timeout)
