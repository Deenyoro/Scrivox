"""LLM-based transcript translation via unified LLM client."""

import copy
import time

from .llm_client import chat_completion


def _parse_numbered_lines(response_text, expected_count):
    """Parse numbered-line response back into a list of strings.

    Accepts formats like:
        1: translated text
        2: translated text
    Or just plain lines (one per input) as fallback.
    """
    lines = response_text.strip().split("\n")
    result = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Try "N: text" format
        if ":" in line:
            parts = line.split(":", 1)
            try:
                num = int(parts[0].strip())
                text = parts[1].strip()
                result[num] = text
                continue
            except ValueError:
                pass
        # Try "N. text" format
        if "." in line:
            parts = line.split(".", 1)
            try:
                num = int(parts[0].strip())
                text = parts[1].strip()
                result[num] = text
                continue
            except ValueError:
                pass

    # If we got the right count via numbered parsing, use it
    if len(result) >= expected_count:
        return [result.get(i + 1, "") for i in range(expected_count)]

    # Fallback: strip numbering prefixes and use line-by-line
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading "N: " or "N. "
        for sep in [":", "."]:
            if sep in line:
                prefix, rest = line.split(sep, 1)
                if prefix.strip().isdigit():
                    line = rest.strip()
                    break
        cleaned.append(line)

    # Pad or truncate to expected count
    while len(cleaned) < expected_count:
        cleaned.append("")
    return cleaned[:expected_count]


def translate_segments(segments, target_language, api_key, translation_model,
                       source_language=None, api_base=None, batch_size=25,
                       on_progress=print):
    """Translate transcript segments to a target language via LLM API.

    Args:
        segments: List of segment dicts with 'text', 'start', 'end', etc.
        target_language: Target language name (e.g. "Arabic", "French")
        api_key: LLM API key
        translation_model: Model ID (e.g. "google/gemini-2.5-flash")
        source_language: Optional source language name for context
        api_base: Optional API base URL (defaults to OpenRouter)
        batch_size: Number of segments per API call
        on_progress: Progress callback

    Returns:
        List of translated segment dicts (deep copies with translated text).
    """
    if not segments:
        return []

    on_progress(f"Translating {len(segments)} segments to {target_language} "
                f"with {translation_model}...")
    t0 = time.time()

    from .constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
    url = api_base or LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]

    translated = []
    total_batches = (len(segments) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(segments))
        batch = segments[start:end]

        on_progress(f"  Translating batch {batch_idx + 1}/{total_batches} "
                    f"({len(batch)} segments)...")

        # Build numbered lines
        numbered_lines = []
        for i, seg in enumerate(batch):
            numbered_lines.append(f"{i + 1}: {seg['text']}")
        numbered_text = "\n".join(numbered_lines)

        source_hint = f" from {source_language}" if source_language else ""
        prompt = (
            f"Translate the following numbered lines{source_hint} to {target_language}. "
            f"Return ONLY the translated lines in the exact same numbered format. "
            f"Preserve the numbering exactly (1:, 2:, etc). "
            f"Do not add explanations, notes, or extra text. "
            f"Keep each translation on a single line.\n\n"
            f"{numbered_text}"
        )

        messages = [{"role": "user", "content": prompt}]
        response_text = chat_completion(
            messages=messages,
            model=translation_model,
            api_key=api_key,
            api_base=url,
            max_tokens=4096,
            temperature=0.3,
            max_retries=3,
            timeout=120,
        )

        batch_translations = None
        if response_text and not response_text.startswith("["):
            batch_translations = _parse_numbered_lines(response_text, len(batch))
        elif response_text:
            on_progress(f"  Warning: {response_text}")

        # Build translated segments for this batch
        for i, seg in enumerate(batch):
            new_seg = copy.deepcopy(seg)
            if batch_translations and i < len(batch_translations) and batch_translations[i]:
                new_seg["text"] = batch_translations[i]
            # else: keep original text as fallback
            # Clear word-level timestamps (they don't apply to translated text)
            new_seg.pop("words", None)
            translated.append(new_seg)

    elapsed = time.time() - t0
    on_progress(f"Translation complete in {elapsed:.1f}s")
    return translated
