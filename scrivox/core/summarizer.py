"""Meeting summary generation via OpenRouter LLM."""

import json
import sys
import time

import requests

from .formatter import format_timestamp_human


def _safe_parse_api_response(resp):
    """Safely extract text from an OpenRouter API response."""
    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        return f"[API parse error: {e}]"


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

    max_chars = 80000
    if len(transcript_text) > max_chars:
        half = max_chars // 2
        transcript_text = (
            transcript_text[:half]
            + "\n\n[... middle portion omitted for length ...]\n\n"
            + transcript_text[-half:]
        )

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

    payload = {
        "model": summary_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    from .constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
    url = api_base or LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]

    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers, json=payload, timeout=120,
            )
            if resp.status_code == 200:
                summary = _safe_parse_api_response(resp)
                if summary.startswith("[API parse error"):
                    on_progress(f"Warning: {summary}")
                    return None
                elapsed = time.time() - t0
                on_progress(f"Summary generated in {elapsed:.1f}s")
                return summary
            elif resp.status_code >= 500 or resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            else:
                on_progress(f"Warning: Summary API error {resp.status_code}: {resp.text[:200]}")
                return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                on_progress(f"Warning: Summary failed after retries: {type(e).__name__}")
                return None

    on_progress("Warning: Summary generation failed after max retries")
    return None
