"""Audio transcription using faster-whisper with post-processing."""

import os
import re
import time

import torch


# ── Post-processing ────────────────────────────────────────


def _detect_repeated_phrase(text, min_phrase_words=2, min_repeats=2):
    """Detect and collapse repeated phrases (classic Whisper hallucination pattern).
    Loops until no more repeats found.
    """
    changed = True
    while changed:
        changed = False
        words = text.split()
        if len(words) < min_phrase_words * min_repeats:
            break
        for phrase_len in range(len(words) // min_repeats, min_phrase_words - 1, -1):
            found = False
            for start in range(len(words) - phrase_len * min_repeats + 1):
                phrase = words[start:start + phrase_len]
                repeats = 1
                pos = start + phrase_len
                while pos + phrase_len <= len(words) and words[pos:pos + phrase_len] == phrase:
                    repeats += 1
                    pos += phrase_len
                if repeats >= min_repeats:
                    before = words[:start]
                    after = words[pos:]
                    text = " ".join(before + phrase + after)
                    changed = True
                    found = True
                    break
            if found:
                break
    return text


def _detect_gapped_repeat(text, min_phrase_words=5, max_gap=5):
    """Detect when a phrase repeats non-consecutively with a few words gap between."""
    changed = True
    while changed:
        changed = False
        words = text.split()
        if len(words) < min_phrase_words * 2:
            break
        for phrase_len in range(len(words) // 2, min_phrase_words - 1, -1):
            found = False
            for start1 in range(len(words) - phrase_len):
                phrase = words[start1:start1 + phrase_len]
                end1 = start1 + phrase_len
                search_limit = min(end1 + max_gap + 1, len(words) - phrase_len + 1)
                for start2 in range(end1 + 1, search_limit):
                    if words[start2:start2 + phrase_len] == phrase:
                        words = words[:start2] + words[start2 + phrase_len:]
                        text = " ".join(words)
                        changed = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return text


def _is_non_speech(text):
    """Check if text is likely non-speech (noise, music, foreign character hallucination)."""
    stripped = text.strip()
    if not stripped:
        return True
    latin_chars = sum(1 for c in stripped if c.isascii() or c in "'\u2019\u2014\u2013")
    if len(stripped) > 0 and latin_chars / len(stripped) < 0.5:
        return True
    return False


def _filter_low_confidence_segments(segments, min_avg_prob=0.50):
    """Remove segments where average word probability is very low."""
    filtered = []
    for seg in segments:
        if seg.get("words"):
            probs = [w["probability"] for w in seg["words"] if "probability" in w]
            if probs and (sum(probs) / len(probs)) < min_avg_prob:
                continue
        filtered.append(seg)
    return filtered


def _normalize_punctuation(text):
    """Fix common punctuation spacing issues from Whisper output."""
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def _capitalize_text(text):
    """Capitalize sentence starts, the pronoun I, and common acronyms."""
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
    text = re.sub(r"\bi\b", "I", text)
    for acronym in ("ai", "hr", "api", "pto", "sso", "saas", "usa", "uk", "eu"):
        text = re.sub(r'\b' + acronym + r'\b', acronym.upper(), text, flags=re.IGNORECASE)
    return text


def clean_transcription(segments):
    """Post-process transcript segments to fix common Whisper issues."""
    cleaned = []
    for seg in segments:
        text = seg["text"]

        if _is_non_speech(text):
            continue

        text = _detect_repeated_phrase(text)
        text = _detect_gapped_repeat(text)
        text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u2018\u2019\u201C\u201D\u2014\u2013]+', '', text)
        text = _normalize_punctuation(text)
        text = _capitalize_text(text)

        if not text:
            continue

        seg = dict(seg)
        seg["text"] = text
        cleaned.append(seg)

    cleaned = _filter_low_confidence_segments(cleaned)

    # Remove consecutive duplicate segments
    deduped = []
    for seg in cleaned:
        if deduped and seg["text"].lower().strip() == deduped[-1]["text"].lower().strip():
            continue
        deduped.append(seg)

    return deduped


def _get_whisper_cache_dir():
    """Get cache directory for Whisper models.

    Checks for a 'models/whisper' directory next to the exe (for bundled/custom models).
    Users can place their own CTranslate2-converted Whisper models here.
    Falls back to the default faster-whisper cache location.
    """
    import sys
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    custom_dir = os.path.join(base, "models", "whisper")
    if os.path.isdir(custom_dir):
        return custom_dir
    return None


def transcribe_audio(audio_path, model_name="large-v3", language=None, on_progress=print):
    """Transcribe audio using faster-whisper on CUDA.

    Whisper models are downloaded automatically on first use.
    Users can also place custom models in the 'models/whisper/' directory.
    """
    from faster_whisper import WhisperModel

    # Check for custom model directory
    cache_dir = _get_whisper_cache_dir()
    if cache_dir:
        # Check if the model exists as a subdirectory
        custom_model_path = os.path.join(cache_dir, model_name)
        if os.path.isdir(custom_model_path):
            on_progress(f"Loading custom model from {custom_model_path}...")
            model_name = custom_model_path

    on_progress(f"Loading faster-whisper '{model_name}' on CUDA (float16)...")
    model = WhisperModel(model_name, device="cuda", compute_type="float16")

    on_progress(f"Transcribing: {audio_path}")
    t0 = time.time()

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language or None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
    )

    result_segments = []
    for seg in segments:
        entry = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [],
        }
        if seg.words:
            for w in seg.words:
                entry["words"].append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                })
        result_segments.append(entry)

    elapsed = time.time() - t0
    on_progress(f"Transcription done in {elapsed:.1f}s (language: {info.language}, prob: {info.language_probability:.2f})")

    del model
    torch.cuda.empty_cache()

    return result_segments, info
