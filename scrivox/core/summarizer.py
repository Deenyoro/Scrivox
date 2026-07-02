"""Meeting summary generation via LLM API."""

import time

from .formatter import format_timestamp_human
from .llm_client import chat_completion, is_error_response


def _truncate_middle(text, max_chars, marker):
    """Keep the head and tail of text, dropping the middle past max_chars."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n{marker}\n\n" + text[-half:]


def generate_meeting_summary(segments, api_key, summary_model, diarized=False,
                             visual_context=None, api_base=None, on_progress=print):
    """Generate a meeting summary with key points, decisions, and action items."""
    on_progress(f"Generating meeting summary with {summary_model}...")
    t0 = time.time()

    transcript_lines = []
    for seg in segments:
        ts = format_timestamp_human(seg["start"])
        speaker = f"[{seg['speaker']}] " if diarized and seg.get("speaker") else ""
        transcript_lines.append(f"[{ts}] {speaker}{seg['text']}")

    transcript_text = "\n".join(transcript_lines)

    context_section = ""
    if visual_context:
        context_lines = []
        for vc in visual_context:
            ts = format_timestamp_human(vc["timestamp"])
            context_lines.append(f"[{ts}] {vc['description']}")
        context_section = "\n\nVisual context from the video:\n" + "\n".join(context_lines)

    transcript_text = _truncate_middle(
        transcript_text, 80000, "[... middle portion omitted for length ...]")
    # Vision descriptions can be very long (8k-token budget per frame) — cap
    # the context section too, or vision-heavy runs overflow the model context
    # and the summary fails entirely
    context_section = _truncate_middle(
        context_section, 60000, "[... middle visual context omitted for length ...]")

    prompt = f"""Analyze this meeting transcript and provide a structured summary.

TRANSCRIPT:
{transcript_text}
{context_section}

Provide your response in this exact format:

## Meeting Summary
A 2-4 sentence overview of what the meeting was about.

## Key Discussion Points
- Bullet points of the main topics discussed

## Decisions Made
- Bullet points of any decisions that were reached (or "None identified" if no clear decisions)

## Action Items
- [ ] Specific action items with the responsible person if identifiable (or "None identified" if no clear action items)

## Notable Quotes
- Any particularly important or memorable statements (include speaker if known)

Be concise and factual. Only include information actually present in the transcript."""

    from .constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
    url = api_base or LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]

    messages = [{"role": "user", "content": prompt}]
    result = chat_completion(
        messages=messages,
        model=summary_model,
        api_key=api_key,
        api_base=url,
        max_tokens=2000,
        max_retries=3,
        timeout=120,
    )

    if is_error_response(result):
        on_progress(f"Warning: Summary generation failed: {result}")
        return None

    elapsed = time.time() - t0
    on_progress(f"Summary generated in {elapsed:.1f}s")
    return result
