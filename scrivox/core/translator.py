"""LLM-based transcript translation via unified LLM client."""

import copy
import time

from .llm_client import chat_completion, is_error_response

# Header strings used by the formatter that should be translated
# when "translate all content" is enabled.
TRANSLATABLE_HEADERS = [
    "Meeting Transcript",
    "Full Transcript",
    "FULL TRANSCRIPT",
    "File",
    "Duration",
    "Model",
    "Language",
    "Speakers",
    "SCREEN",
]


def _parse_numbered_lines(response_text, expected_count):
    """Parse numbered-line response back into a list of strings.

    Accepts "N: text" / "N. text" where N must be the next sequential item
    number — a translation that legitimately starts with "12." (a date) or
    "1:30" (a time) cannot hijack the numbering. Lines without a valid prefix
    are treated as continuations of the previous item, not dropped.

    Returns None when the response cannot be aligned to expected_count.
    Callers must then keep the original untranslated strings — misaligned
    translations attached to the wrong timestamps are worse than no
    translation.
    """
    lines = response_text.strip().split("\n")
    items = {}

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        matched = False
        for sep in (":", "."):
            prefix, found, rest = line.partition(sep)
            if not found:
                continue
            try:
                num = int(prefix.strip())
            except ValueError:
                continue
            # Require "N: text" shape (space or end after the separator) —
            # a bare "1:30" (time) or "12.," is content, not numbering
            if rest and not rest[0].isspace():
                continue
            if num == len(items) + 1 and num <= expected_count:
                items[num] = rest.strip()
                matched = True
            break
        if not matched and items and len(items) < expected_count:
            # Continuation of the previous item (model wrapped a line).
            # Once all items are parsed, unnumbered lines are trailing
            # commentary ("Note: ...") and must not be glued to the last item.
            items[len(items)] = (items[len(items)] + " " + line).strip()

    if len(items) == expected_count:
        return [items[i + 1] for i in range(expected_count)]

    # Fallback: plain lines, one per input — only when the count matches
    # exactly. Strip a numbered prefix only if it matches the line's position.
    plain = [l.strip() for l in lines if l.strip()]
    if len(plain) != expected_count:
        return None
    cleaned = []
    for idx, line in enumerate(plain):
        for sep in (":", "."):
            prefix, found, rest = line.partition(sep)
            if found and (not rest or rest[0].isspace()):
                try:
                    if int(prefix.strip()) == idx + 1:
                        line = rest.strip()
                        break
                except ValueError:
                    pass
        cleaned.append(line)
    return cleaned


def translate_segments(segments, target_language, api_key, translation_model,
                       source_language=None, api_base=None, batch_size=25,
                       on_progress=print, cancel_event=None):
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
        cancel_event: Optional threading.Event to check for cancellation

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
        # Check for cancellation between batches
        if cancel_event and cancel_event.is_set():
            on_progress("Translation cancelled.")
            break

        start = batch_idx * batch_size
        end = min(start + batch_size, len(segments))
        batch = segments[start:end]

        on_progress(f"  Translating batch {batch_idx + 1}/{total_batches} "
                    f"({len(batch)} segments)...")

        # Build numbered lines — flatten any embedded newlines or the
        # one-line-per-item protocol breaks
        numbered_lines = []
        for i, seg in enumerate(batch):
            numbered_lines.append(f"{i + 1}: {' '.join(seg['text'].split())}")
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

        # Scale max_tokens to batch content — translations can expand 2-3x
        # ~1 token per 4 chars, allow 3x expansion for verbose target languages
        input_chars = sum(len(seg["text"]) for seg in batch)
        estimated_tokens = max(4096, input_chars * 3 // 4)
        max_tokens = min(estimated_tokens, 16384)

        messages = [{"role": "user", "content": prompt}]
        response_text = chat_completion(
            messages=messages,
            model=translation_model,
            api_key=api_key,
            api_base=url,
            max_tokens=max_tokens,
            temperature=0.3,
            max_retries=3,
            timeout=120,
        )

        batch_translations = None
        if not is_error_response(response_text):
            batch_translations = _parse_numbered_lines(response_text, len(batch))
            if batch_translations is None:
                on_progress("  Warning: Could not align translated lines — keeping originals for this batch")
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


def translate_text(text, target_language, api_key, translation_model,
                   source_language=None, api_base=None, on_progress=print):
    """Translate a block of text (e.g. meeting summary) preserving markdown structure.

    Returns translated text, or original text on failure.
    """
    if not text:
        return text

    from .constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
    url = api_base or LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]

    source_hint = f" from {source_language}" if source_language else ""
    prompt = (
        f"Translate the following text{source_hint} to {target_language}. "
        f"Preserve all markdown formatting, headers, bullet points, and structure exactly. "
        f"Do not add explanations or notes. Translate ONLY the text content.\n\n"
        f"{text}"
    )

    input_chars = len(text)
    max_tokens = min(max(4096, input_chars * 3 // 4), 16384)

    messages = [{"role": "user", "content": prompt}]
    result = chat_completion(
        messages=messages,
        model=translation_model,
        api_key=api_key,
        api_base=url,
        max_tokens=max_tokens,
        temperature=0.3,
        max_retries=3,
        timeout=120,
    )

    if not is_error_response(result):
        return result

    on_progress(f"  Warning: Text translation failed: {result}")
    return text  # fallback to original


def translate_strings(strings, target_language, api_key, translation_model,
                      source_language=None, api_base=None, batch_size=25,
                      on_progress=print):
    """Translate a list of short strings. Returns list of translated strings.

    Falls back to originals on failure.
    """
    if not strings:
        return strings

    from .constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
    url = api_base or LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]

    results = []
    total_batches = (len(strings) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(strings))
        batch = strings[start:end]

        # Flatten multi-line strings (e.g. long vision descriptions) — the
        # numbered protocol requires one line per item
        numbered_lines = [f"{i + 1}: {' '.join(s.split())}" for i, s in enumerate(batch)]
        numbered_text = "\n".join(numbered_lines)

        source_hint = f" from {source_language}" if source_language else ""
        prompt = (
            f"Translate the following numbered lines{source_hint} to {target_language}. "
            f"Return ONLY the translated lines in the exact same numbered format. "
            f"Preserve the numbering exactly (1:, 2:, etc). "
            f"Do not add explanations, notes, or extra text.\n\n"
            f"{numbered_text}"
        )

        messages = [{"role": "user", "content": prompt}]
        result = chat_completion(
            messages=messages,
            model=translation_model,
            api_key=api_key,
            api_base=url,
            max_tokens=4096,
            temperature=0.3,
            max_retries=3,
            timeout=60,
        )

        if not is_error_response(result):
            batch_results = _parse_numbered_lines(result, len(batch))
            if batch_results is None:
                on_progress("  Warning: Could not align translated strings — keeping originals for this batch")
                batch_results = list(batch)
            results.extend(batch_results)
        else:
            on_progress(f"  Warning: String translation failed: {result}")
            results.extend(batch)  # fallback to originals

    return results
