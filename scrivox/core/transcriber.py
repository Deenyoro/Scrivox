"""Audio transcription using faster-whisper with post-processing."""

import os
import re
import time
import unicodedata

import torch


# ── Language helpers ──────────────────────────────────────

# Languages that use Latin script
_LATIN_LANGUAGES = frozenset({
    "en", "fr", "de", "es", "pt", "it", "nl", "pl", "cs", "sk",
    "ro", "hu", "sv", "da", "no", "fi", "et", "lv", "lt", "hr",
    "sl", "sq", "tr", "az", "id", "ms", "vi", "tl", "sw", "ca",
    "gl", "eu", "cy", "ga", "mt", "la", "eo",
})


def _is_latin_language(lang):
    """Return True if the language code is known to use Latin script."""
    if not lang:
        return False
    return lang.lower().split("-")[0] in _LATIN_LANGUAGES


def _detect_segment_language(text, primary_language):
    """Detect the dominant script/language of a text segment.

    Uses Unicode character categories to identify the script, then maps to a
    language code.  Only overrides *primary_language* when a non-primary script
    is clearly dominant (>70 % of alphabetic characters).

    Returns a language code string.
    """
    if not text or not text.strip():
        return primary_language or "en"

    script_counts = {}
    alpha_total = 0

    for ch in text:
        if not ch.isalpha():
            continue
        alpha_total += 1
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue

        if name.startswith("HANGUL"):
            script_counts["hangul"] = script_counts.get("hangul", 0) + 1
        elif name.startswith("CJK") or name.startswith("KANGXI"):
            script_counts["cjk"] = script_counts.get("cjk", 0) + 1
        elif name.startswith("HIRAGANA") or name.startswith("KATAKANA"):
            script_counts["kana"] = script_counts.get("kana", 0) + 1
        elif name.startswith("CYRILLIC"):
            script_counts["cyrillic"] = script_counts.get("cyrillic", 0) + 1
        elif name.startswith("ARABIC"):
            script_counts["arabic"] = script_counts.get("arabic", 0) + 1
        elif name.startswith("DEVANAGARI"):
            script_counts["devanagari"] = script_counts.get("devanagari", 0) + 1
        elif name.startswith("THAI"):
            script_counts["thai"] = script_counts.get("thai", 0) + 1
        elif ch.isascii() or "LATIN" in name:
            script_counts["latin"] = script_counts.get("latin", 0) + 1

    if alpha_total == 0:
        return primary_language or "en"

    # Find the dominant script
    dominant_script = max(script_counts, key=script_counts.get) if script_counts else "latin"
    dominant_ratio = script_counts.get(dominant_script, 0) / alpha_total

    # Only override when clearly dominant
    if dominant_ratio < 0.70:
        return primary_language or "en"

    # Map script -> language code
    script_to_lang = {
        "hangul": "ko",
        "cyrillic": "ru",
        "arabic": "ar",
        "devanagari": "hi",
        "thai": "th",
    }

    if dominant_script == "latin":
        return primary_language if primary_language and _is_latin_language(primary_language) else "en"
    elif dominant_script in script_to_lang:
        return script_to_lang[dominant_script]
    elif dominant_script == "kana":
        return "ja"
    elif dominant_script == "cjk":
        # CJK chars are shared; defer to primary if it's zh/ja/ko
        if primary_language in ("zh", "ja", "ko"):
            return primary_language
        return "zh"

    return primary_language or "en"


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


def _is_non_speech(text, language=None):
    """Check if text is likely non-speech (noise, music, hallucination).

    Language-aware: for non-Latin languages, only filters text with zero
    alphabetic characters (pure symbols/punctuation).  For Latin languages,
    keeps the existing Latin-ratio heuristic.
    """
    stripped = text.strip()
    if not stripped:
        return True

    if not _is_latin_language(language):
        # Non-Latin or unknown language: only filter if zero alphabetic chars
        return not any(c.isalpha() for c in stripped)

    # Latin-script language: original heuristic
    latin_chars = sum(1 for c in stripped if c.isascii() or c in "'\u2019\u2014\u2013")
    if latin_chars / len(stripped) < 0.5:
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
    for acronym in ("ai", "hr", "api", "pto", "sso", "saas", "usa", "uk", "eu",
                     "ceo", "cto", "cfo", "coo", "vp", "qa", "kpi", "roi", "it",
                     "pr", "ml", "sql", "url", "pdf", "faq", "gdpr", "eta", "asap", "rsvp"):
        text = re.sub(r'\b' + acronym + r'\b', acronym.upper(), text, flags=re.IGNORECASE)
    return text


def clean_transcription(segments, language=None):
    """Post-process transcript segments to fix common Whisper issues.

    Args:
        segments: List of segment dicts with "text", "start", "end", etc.
        language: Primary language code (e.g. "ko", "en").  Used to avoid
                  destroying non-Latin text.
    """
    cleaned = []
    for seg in segments:
        text = seg["text"]
        seg_lang = seg.get("language", language)

        if _is_non_speech(text, language=seg_lang):
            continue

        text = _detect_repeated_phrase(text)
        text = _detect_gapped_repeat(text)

        # Only strip non-Latin characters for Latin-script languages
        if _is_latin_language(seg_lang):
            text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u2018\u2019\u201C\u201D\u2014\u2013\u20AC\u00A3\u00A5\u20A9\u00B0\u00B1\u00A7\u00A9\u00AE\u2122\u00B7\u2022%]+', '', text)

        text = _normalize_punctuation(text)

        # Skip capitalization for scripts that have no concept of letter case
        if _is_latin_language(seg_lang):
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
        seg_text = seg.text.strip()
        entry = {
            "start": seg.start,
            "end": seg.end,
            "text": seg_text,
            "language": _detect_segment_language(seg_text, language),
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
