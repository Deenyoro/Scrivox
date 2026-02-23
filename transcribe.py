"""
Whisper GPU Transcription with Speaker Diarization + Visual Context + Meeting Summary.

Full pipeline:
  1. faster-whisper transcription on GPU (float16)
  2. pyannote speaker diarization on GPU
  3. ffmpeg keyframe extraction from video
  4. Vision LLM describes keyframes via OpenRouter
  5. LLM-generated meeting summary with action items
  6. Combined output with speaker labels + visual context

Usage:
    python transcribe.py input.mp3
    python transcribe.py input.mp4 --diarize
    python transcribe.py input.mp4 --diarize --vision
    python transcribe.py input.mp4 --diarize --vision --summarize
    python transcribe.py input.mp4 --all                              (diarize + vision + summarize)
    python transcribe.py input.mp4 --diarize --vision --format json -o out.json
    python transcribe.py input.mp4 --diarize --speaker-names "Alice,Bob,Charlie"
    python transcribe.py input.mp4 --all --format md -o meeting.md

Requires .env file with:
    HF_TOKEN=...              (for diarization)
    OPENROUTER_API_KEY=...    (for vision / summary)
"""

import argparse
import base64
import concurrent.futures
import contextlib
import glob
import json
import os
import re
import requests
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import torch

# PyTorch 2.6+ defaults to weights_only=True which breaks pyannote model loading.
# This context manager temporarily forces weights_only=False only where needed.
_original_torch_load = torch.load

@contextlib.contextmanager
def _allow_unsafe_torch_load():
    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = patched
    try:
        yield
    finally:
        torch.load = _original_torch_load

DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
DEFAULT_SUMMARY_MODEL = "google/gemini-2.5-flash"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def format_timestamp(seconds, fmt="srt"):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    if fmt == "vtt":
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_human(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def check_ffmpeg():
    """Verify ffmpeg and ffprobe are available."""
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except FileNotFoundError:
            print(f"Error: '{tool}' not found. Install ffmpeg and ensure it's in your PATH.", file=sys.stderr)
            sys.exit(1)
        except subprocess.TimeoutExpired:
            pass  # slow but exists


def has_video_stream(file_path):
    """Use ffprobe to check if file actually contains a video stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() == "video"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS


def get_media_duration(file_path):
    """Get duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# ── Post-processing ────────────────────────────────────────

def _detect_repeated_phrase(text, min_phrase_words=2, min_repeats=2):
    """Detect and collapse repeated phrases (classic Whisper hallucination pattern).
    Loops until no more repeats found.
    e.g. 'oh my god i'm like oh my god i'm like oh my god' -> 'oh my god i'm like'
    e.g. 'iso system iso system' -> 'iso system'
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
    """Detect when a phrase repeats non-consecutively with a few words gap between.
    Common Whisper hallucination where it re-hears the same audio segment.
    e.g. 'they have an iso system that's just so easy to do they have an iso system that's just so'
    """
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
                # Look for same phrase starting 1..max_gap words after first ends
                search_limit = min(end1 + max_gap + 1, len(words) - phrase_len + 1)
                for start2 in range(end1 + 1, search_limit):
                    if words[start2:start2 + phrase_len] == phrase:
                        # Remove the second occurrence of the phrase
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
    # Count characters that are Latin letters, digits, or common punctuation
    latin_chars = sum(1 for c in stripped if c.isascii() or c in "''-—")
    if len(stripped) > 0 and latin_chars / len(stripped) < 0.5:
        return True
    return False


def _filter_low_confidence_segments(segments, min_avg_prob=0.50):
    """Remove segments where average word probability is very low (likely hallucination)."""
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
    # Remove space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Ensure space after punctuation (unless end of string or followed by quote)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def _capitalize_text(text):
    """Capitalize sentence starts and the pronoun I for readability."""
    if not text:
        return text
    # Capitalize first character of segment
    text = text[0].upper() + text[1:]
    # Capitalize after sentence-ending punctuation followed by space
    text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
    # Capitalize the pronoun "i" (standalone and contractions like i'm, i'll, i've, i'd)
    text = re.sub(r"\bi\b", "I", text)
    return text


def clean_transcription(segments):
    """Post-process transcript segments to fix common Whisper issues."""
    cleaned = []
    for seg in segments:
        text = seg["text"]

        # Skip non-speech segments
        if _is_non_speech(text):
            continue

        # Collapse consecutive repeated phrases
        text = _detect_repeated_phrase(text)

        # Collapse gapped repeated phrases (same phrase with a few filler words between)
        text = _detect_gapped_repeat(text)

        # Strip stray non-Latin characters within otherwise English text
        text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u2018\u2019\u201C\u201D\u2014\u2013]+', '', text)

        # Fix punctuation spacing
        text = _normalize_punctuation(text)

        # Capitalize sentence starts and pronoun I
        text = _capitalize_text(text)

        if not text:
            continue

        seg = dict(seg)  # copy so we don't mutate original
        seg["text"] = text
        cleaned.append(seg)

    # Filter low-confidence segments
    cleaned = _filter_low_confidence_segments(cleaned)

    # Remove consecutive duplicate segments (same text back-to-back)
    deduped = []
    for seg in cleaned:
        if deduped and seg["text"].lower().strip() == deduped[-1]["text"].lower().strip():
            continue
        deduped.append(seg)

    return deduped


# ── Transcription ──────────────────────────────────────────

def transcribe_audio(audio_path, model_name="large-v3", language=None):
    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper '{model_name}' on CUDA (float16)...")
    model = WhisperModel(model_name, device="cuda", compute_type="float16")

    print(f"Transcribing: {audio_path}")
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
    print(f"Transcription done in {elapsed:.1f}s (language: {info.language}, prob: {info.language_probability:.2f})")

    # Free GPU memory for diarization
    del model
    torch.cuda.empty_cache()

    return result_segments, info


# ── Diarization ────────────────────────────────────────────

def extract_wav(input_path):
    """Extract audio to WAV for diarization. Returns path to temp WAV file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    print("Extracting audio to WAV for diarization...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise
    return wav_path


def diarize_audio(audio_path, hf_token, num_speakers=None, min_speakers=None, max_speakers=None):
    from pyannote.audio import Pipeline

    # Pyannote needs WAV
    wav_path = None
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in (".wav", ".wave"):
        wav_path = extract_wav(audio_path)
        diarize_input = wav_path
    else:
        diarize_input = audio_path

    try:
        print("Loading pyannote speaker-diarization-3.1 on CUDA...")
        with _allow_unsafe_torch_load():
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        pipeline.to(torch.device("cuda"))

        print("Running diarization... (this may take several minutes)")
        t0 = time.time()

        # Progress indicator in background
        stop_progress = threading.Event()
        def progress_ticker():
            while not stop_progress.is_set():
                elapsed = time.time() - t0
                print(f"\r  Diarizing... {elapsed:.0f}s elapsed", end="", flush=True)
                stop_progress.wait(5)
            print()

        ticker = threading.Thread(target=progress_ticker, daemon=True)
        ticker.start()

        try:
            diarization = pipeline(
                diarize_input,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        finally:
            stop_progress.set()
            ticker.join(timeout=5)

        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        elapsed = time.time() - t0
        speakers = set(s["speaker"] for s in speaker_segments)
        print(f"Diarization done in {elapsed:.1f}s ({len(speakers)} speakers detected)")

        # Free GPU memory
        del pipeline
        torch.cuda.empty_cache()

        return speaker_segments

    finally:
        # Always clean up temp WAV, even on error
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def assign_speakers(transcript_segments, speaker_segments, speaker_names=None):
    EPSILON = 1e-6
    for seg in transcript_segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for spk in speaker_segments:
            overlap_start = max(seg["start"], spk["start"])
            overlap_end = min(seg["end"], spk["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap + EPSILON:
                best_overlap = overlap
                best_speaker = spk["speaker"]

        # If no overlap found, fall back to nearest speaker in time
        if best_speaker == "UNKNOWN" and speaker_segments:
            min_gap = float('inf')
            for spk in speaker_segments:
                gap = max(0, max(seg["start"], spk["start"]) - min(seg["end"], spk["end"]))
                if gap < min_gap:
                    min_gap = gap
                    best_speaker = spk["speaker"]

        seg["speaker"] = best_speaker

    # Build speaker label map
    speaker_map = {}
    counter = 1
    for seg in transcript_segments:
        if seg["speaker"] not in speaker_map and seg["speaker"] != "UNKNOWN":
            if speaker_names and counter <= len(speaker_names):
                speaker_map[seg["speaker"]] = speaker_names[counter - 1]
            else:
                speaker_map[seg["speaker"]] = f"SPEAKER_{counter:02d}"
            counter += 1
        seg["speaker"] = speaker_map.get(seg["speaker"], "UNKNOWN")

    return transcript_segments


def rename_speakers(segments, speaker_names):
    """Re-map existing SPEAKER_XX labels to custom names (for cached data)."""
    # Collect unique speaker labels in order of first appearance
    seen = []
    for seg in segments:
        spk = seg.get("speaker", "")
        if spk and spk != "UNKNOWN" and spk not in seen:
            seen.append(spk)

    # Build rename map
    rename_map = {}
    for i, old_name in enumerate(seen):
        if i < len(speaker_names):
            rename_map[old_name] = speaker_names[i]

    if not rename_map:
        return segments

    for seg in segments:
        spk = seg.get("speaker", "")
        if spk in rename_map:
            seg["speaker"] = rename_map[spk]

    return segments


def _safe_parse_api_response(resp):
    """Safely extract text from an OpenRouter API response."""
    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        return f"[API parse error: {e}]"


# ── Vision / Keyframe Analysis ─────────────────────────────

def extract_keyframes(video_path, interval_secs=60, max_frames=30):
    """Extract keyframes from video at regular intervals."""
    tmpdir = tempfile.mkdtemp(prefix="whisper_frames_")

    duration = get_media_duration(video_path)
    if duration is None:
        print("Warning: Could not determine video duration, using default interval")
        duration = interval_secs * max_frames

    # Adjust interval if too many frames would be extracted
    if duration / interval_secs > max_frames:
        old_interval = interval_secs
        interval_secs = int(duration / max_frames)
        print(f"  Adjusted keyframe interval from {old_interval}s to {interval_secs}s (capped at {max_frames} frames)")

    print(f"Extracting keyframes every {interval_secs}s from {duration:.0f}s video...")

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps=1/{int(interval_secs)},scale=1280:-2",
         "-q:v", "3",
         os.path.join(tmpdir, "frame_%04d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        timeout=600,
    )

    frames = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
    keyframes = []
    for i, path in enumerate(frames):
        timestamp = i * interval_secs
        keyframes.append({"path": path, "timestamp": timestamp})

    print(f"Extracted {len(keyframes)} keyframes")
    return keyframes, tmpdir


def describe_keyframe(image_path, timestamp, api_key, vision_model, max_retries=3):
    """Send a keyframe to vision LLM and get a description, with retries."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    ts_str = format_timestamp_human(timestamp)
    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"This is a screenshot from a video at timestamp {ts_str}. "
                            "Briefly describe what's visible on screen — any text, UI elements, "
                            "people, slides, applications, or content shown. "
                            "Be concise (2-3 sentences max). Focus on what would provide useful "
                            "context alongside a transcript of the conversation."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 200,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=60,
            )
            if resp.status_code == 200:
                return _safe_parse_api_response(resp)
            elif resp.status_code >= 500 or resp.status_code == 429:
                wait = 2 ** attempt
                print(f"    (rate limited or server error {resp.status_code}, retrying in {wait}s...)")
                time.sleep(wait)
                continue
            else:
                body = ""
                try:
                    body = resp.text[:100]
                except Exception:
                    pass
                return f"[Vision API error: {resp.status_code} - {body}]"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.SSLError) as e:
            if attempt < max_retries - 1:
                print(f"    (retry {attempt+1}/{max_retries} after {type(e).__name__})")
                time.sleep(2 ** attempt)
            else:
                return f"[Vision error: {type(e).__name__}]"
    return "[Vision error: max retries exceeded]"


def analyze_keyframes(keyframes, api_key, vision_model, max_workers=4):
    """Describe all keyframes using vision LLM with concurrent requests."""
    print(f"Analyzing {len(keyframes)} keyframes with vision LLM ({vision_model})...")
    t0 = time.time()

    descriptions = [None] * len(keyframes)

    def process_frame(idx, kf):
        ts_str = format_timestamp_human(kf["timestamp"])
        print(f"  Frame {idx+1}/{len(keyframes)} @ {ts_str}...")
        desc = describe_keyframe(kf["path"], kf["timestamp"], api_key, vision_model)
        return idx, {"timestamp": kf["timestamp"], "description": desc}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_frame, i, kf): i for i, kf in enumerate(keyframes)}
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, result = future.result()
                descriptions[idx] = result
            except Exception as e:
                frame_idx = futures[future]
                ts_str = format_timestamp_human(keyframes[frame_idx]["timestamp"])
                print(f"  Frame {frame_idx+1} @ {ts_str} failed: {e}", file=sys.stderr)
                descriptions[frame_idx] = {
                    "timestamp": keyframes[frame_idx]["timestamp"],
                    "description": f"[Frame analysis failed: {type(e).__name__}]",
                }

    # Filter out any None entries (shouldn't happen but be safe)
    descriptions = [d for d in descriptions if d is not None]

    elapsed = time.time() - t0
    print(f"Vision analysis done in {elapsed:.1f}s")
    return descriptions


# ── Meeting Summary ────────────────────────────────────────

def generate_meeting_summary(segments, api_key, summary_model, diarized=False, visual_context=None):
    """Generate a meeting summary with key points, decisions, and action items."""
    print(f"Generating meeting summary with {summary_model}...")
    t0 = time.time()

    # Build transcript text for the LLM
    transcript_lines = []
    for seg in segments:
        ts = format_timestamp_human(seg["start"])
        speaker = f"[{seg['speaker']}] " if diarized and seg.get("speaker") else ""
        transcript_lines.append(f"[{ts}] {speaker}{seg['text']}")

    transcript_text = "\n".join(transcript_lines)

    # Add visual context if available
    context_section = ""
    if visual_context:
        context_lines = []
        for vc in visual_context:
            ts = format_timestamp_human(vc["timestamp"])
            context_lines.append(f"[{ts}] {vc['description']}")
        context_section = "\n\nVisual context from the video:\n" + "\n".join(context_lines)

    # Truncate if extremely long (keep first and last portions)
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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=120,
            )
            if resp.status_code == 200:
                summary = _safe_parse_api_response(resp)
                if summary.startswith("[API parse error"):
                    print(f"Warning: {summary}", file=sys.stderr)
                    return None
                elapsed = time.time() - t0
                print(f"Summary generated in {elapsed:.1f}s")
                return summary
            elif resp.status_code >= 500 or resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"Warning: Summary API error {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
                return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"Warning: Summary failed after retries: {type(e).__name__}", file=sys.stderr)
                return None

    print("Warning: Summary generation failed after max retries", file=sys.stderr)
    return None


# ── Output Formatting ──────────────────────────────────────

def format_output(segments, fmt="txt", diarized=False, visual_context=None, summary=None, metadata=None):
    lines = []

    if fmt == "txt":
        # Meeting summary at the top
        if summary:
            lines.append(summary)
            lines.append("\n" + "=" * 60)
            lines.append("  FULL TRANSCRIPT")
            lines.append("=" * 60 + "\n")

        # Insert visual context at appropriate timestamps
        vis_idx = 0
        vis = visual_context or []
        current_speaker = None

        for seg in segments:
            # Insert any visual context that falls before this segment
            while vis_idx < len(vis) and vis[vis_idx]["timestamp"] <= seg["start"]:
                ts = format_timestamp_human(vis[vis_idx]["timestamp"])
                lines.append(f"\n--- [{ts}] SCREEN: {vis[vis_idx]['description']} ---\n")
                vis_idx += 1

            ts = format_timestamp_human(seg["start"])
            if diarized and seg.get("speaker") != current_speaker:
                current_speaker = seg.get("speaker", "")
                lines.append(f"\n[{ts}] [{current_speaker}]")
            elif not diarized:
                lines.append(f"[{ts}] {seg['text']}")
                continue
            lines.append(seg["text"])

        # Any remaining visual context
        while vis_idx < len(vis):
            ts = format_timestamp_human(vis[vis_idx]["timestamp"])
            lines.append(f"\n--- [{ts}] SCREEN: {vis[vis_idx]['description']} ---\n")
            vis_idx += 1

        return "\n".join(lines).strip()

    elif fmt == "md":
        # Markdown format — ideal for meeting minutes
        if metadata:
            duration = metadata.get("duration_seconds")
            duration_str = format_timestamp_human(duration) if duration else "unknown"
            lines.append(f"# Meeting Transcript")
            lines.append("")
            lines.append(f"- **File:** {metadata.get('input_file', 'unknown')}")
            lines.append(f"- **Duration:** {duration_str}")
            lines.append(f"- **Model:** {metadata.get('model', 'unknown')}")
            lines.append(f"- **Language:** {metadata.get('language', 'unknown')}")
            if diarized:
                # Count unique speakers
                speakers = set(seg.get("speaker", "") for seg in segments)
                speakers.discard("")
                speakers.discard("UNKNOWN")
                lines.append(f"- **Speakers:** {', '.join(sorted(speakers)) if speakers else 'unknown'}")
            lines.append("")

        if summary:
            lines.append("---")
            lines.append("")
            lines.append(summary)
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("## Full Transcript")
            lines.append("")

        # Build transcript
        vis_idx = 0
        vis = visual_context or []
        current_speaker = None

        for seg in segments:
            while vis_idx < len(vis) and vis[vis_idx]["timestamp"] <= seg["start"]:
                ts = format_timestamp_human(vis[vis_idx]["timestamp"])
                lines.append(f"\n> *[{ts}] {vis[vis_idx]['description']}*\n")
                vis_idx += 1

            ts = format_timestamp_human(seg["start"])
            if diarized:
                if seg.get("speaker") != current_speaker:
                    current_speaker = seg.get("speaker", "")
                    lines.append(f"\n**[{ts}] {current_speaker}:**")
                lines.append(f"{seg['text']}")
            else:
                lines.append(f"`{ts}` {seg['text']}")

        while vis_idx < len(vis):
            ts = format_timestamp_human(vis[vis_idx]["timestamp"])
            lines.append(f"\n> *[{ts}] {vis[vis_idx]['description']}*\n")
            vis_idx += 1

        return "\n".join(lines).strip()

    elif fmt == "srt":
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp(seg["start"], "srt")
            end_ts = format_timestamp(seg["end"], "srt")
            speaker_prefix = f"[{seg['speaker']}] " if diarized and seg.get("speaker") else ""
            lines.append(f"{i}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"{speaker_prefix}{seg['text']}")
            lines.append("")
        return "\n".join(lines)

    elif fmt == "vtt":
        lines.append("WEBVTT\n")
        for seg in segments:
            start_ts = format_timestamp(seg["start"], "vtt")
            end_ts = format_timestamp(seg["end"], "vtt")
            speaker_prefix = f"<v {seg['speaker']}>" if diarized and seg.get("speaker") else ""
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"{speaker_prefix}{seg['text']}")
            lines.append("")
        return "\n".join(lines)

    elif fmt == "json":
        output = {}
        if metadata:
            output["metadata"] = metadata
        output["segments"] = segments
        if visual_context:
            output["visual_context"] = visual_context
        if summary:
            output["summary"] = summary
        return json.dumps(output, indent=2, ensure_ascii=False)

    elif fmt == "tsv":
        header = "start\tend"
        if diarized:
            header += "\tspeaker"
        header += "\ttext"
        lines.append(header)
        for seg in segments:
            row = f"{seg['start']:.3f}\t{seg['end']:.3f}"
            if diarized:
                row += f"\t{seg.get('speaker', '')}"
            row += f"\t{seg['text']}"
            lines.append(row)
        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format: {fmt}")


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whisper GPU Transcription + Diarization + Vision + Meeting Summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python transcribe.py meeting.mp3
  python transcribe.py meeting.mp4 --diarize
  python transcribe.py meeting.mp4 --all
  python transcribe.py meeting.mp4 --diarize --vision --summarize
  python transcribe.py meeting.mp4 --diarize --speaker-names "Alice,Bob"
  python transcribe.py meeting.mp4 --all --format md -o minutes.md
  python transcribe.py meeting.mp4 --diarize --vision -f json -o report.json
        """,
    )
    parser.add_argument("input", help="Audio or video file to transcribe")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model: tiny, base, small, medium, large-v3 (default: large-v3)")
    parser.add_argument("--language", default=None,
                        help="Language code e.g. 'en', 'fr', 'ja' (default: auto-detect)")
    parser.add_argument("--all", action="store_true",
                        help="Enable all features: diarize + vision + summarize")

    # Diarization options
    diarize_group = parser.add_argument_group("Speaker Diarization")
    diarize_group.add_argument("--diarize", action="store_true",
                               help="Enable speaker diarization (requires HF_TOKEN)")
    diarize_group.add_argument("--num-speakers", type=int, default=None,
                               help="Exact number of speakers (if known)")
    diarize_group.add_argument("--min-speakers", type=int, default=None,
                               help="Minimum number of speakers expected")
    diarize_group.add_argument("--max-speakers", type=int, default=None,
                               help="Maximum number of speakers expected")
    diarize_group.add_argument("--speaker-names", default=None,
                               help="Comma-separated speaker names, e.g. 'Alice,Bob,Charlie'")

    # Vision options
    vision_group = parser.add_argument_group("Vision Analysis")
    vision_group.add_argument("--vision", action="store_true",
                              help="Extract keyframes and describe with vision LLM (video files only)")
    vision_group.add_argument("--vision-interval", type=int, default=60,
                              help="Seconds between keyframe captures (default: 60)")
    vision_group.add_argument("--vision-model", default=DEFAULT_VISION_MODEL,
                              help=f"Vision LLM model (default: {DEFAULT_VISION_MODEL})")
    vision_group.add_argument("--vision-workers", type=int, default=4,
                              help="Concurrent vision API requests (default: 4)")

    # Summary options
    summary_group = parser.add_argument_group("Meeting Summary")
    summary_group.add_argument("--summarize", action="store_true",
                               help="Generate meeting summary with key points and action items")
    summary_group.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL,
                               help=f"LLM model for summary (default: {DEFAULT_SUMMARY_MODEL})")

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", default=None,
                              help="Output file path (default: print to console)")
    output_group.add_argument("--format", "-f", default="txt",
                              choices=["txt", "md", "srt", "vtt", "json", "tsv"],
                              help="Output format (default: txt)")

    # Cache / credential options
    parser.add_argument("--clear-cache", action="store_true",
                        help="Force re-transcription, ignoring cached results")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (overrides .env / cached login)")
    parser.add_argument("--openrouter-key", default=None,
                        help="OpenRouter API key (overrides .env)")
    args = parser.parse_args()

    # ── Expand --all ──
    if args.all:
        args.diarize = True
        args.vision = True
        args.summarize = True

    # ── Validate input ──
    if not os.path.isfile(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ext = os.path.splitext(args.input)[1].lower()
    if ext not in VIDEO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
        print(f"Warning: Unrecognized file extension '{ext}'. Attempting transcription anyway.", file=sys.stderr)

    # Validate speaker count args
    for arg_name, arg_val in [("--num-speakers", args.num_speakers),
                               ("--min-speakers", args.min_speakers),
                               ("--max-speakers", args.max_speakers)]:
        if arg_val is not None and arg_val < 1:
            print(f"Error: {arg_name} must be at least 1", file=sys.stderr)
            sys.exit(1)
    if args.min_speakers is not None and args.max_speakers is not None:
        if args.min_speakers > args.max_speakers:
            print("Error: --min-speakers cannot be greater than --max-speakers", file=sys.stderr)
            sys.exit(1)
    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        print("Warning: --num-speakers overrides --min-speakers/--max-speakers", file=sys.stderr)

    # Validate vision interval
    if args.vision_interval < 1:
        print("Error: --vision-interval must be at least 1 second", file=sys.stderr)
        sys.exit(1)

    # ── Check system tools ──
    check_ffmpeg()

    # ── Check CUDA ──
    if not torch.cuda.is_available():
        print("Error: CUDA GPU not available. This tool requires an NVIDIA GPU.", file=sys.stderr)
        sys.exit(1)

    # ── Resolve tokens ──
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    openrouter_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")

    if args.diarize and not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            pass
    if args.diarize and not hf_token:
        print("Error: Diarization requires HF_TOKEN in .env, --hf-token, or `huggingface-cli login`", file=sys.stderr)
        sys.exit(1)

    if (args.vision or args.summarize) and not openrouter_key:
        print("Error: Vision/Summary requires OPENROUTER_API_KEY in .env or --openrouter-key", file=sys.stderr)
        sys.exit(1)

    # ── Check video for vision ──
    is_video = has_video_stream(args.input)
    if args.vision and not is_video:
        print("Warning: --vision requires a video file with a video stream, skipping keyframe extraction", file=sys.stderr)
        args.vision = False

    # Parse speaker names
    speaker_names = None
    if args.speaker_names:
        speaker_names = [name.strip() for name in args.speaker_names.split(",") if name.strip()]

    # ── Banner ──
    print("=" * 60)
    print("  WHISPER GPU TRANSCRIPTION")
    if args.diarize:
        print("  + SPEAKER DIARIZATION")
    if args.vision:
        print("  + VISUAL CONTEXT (keyframe analysis)")
    if args.summarize:
        print("  + MEETING SUMMARY")
    print("=" * 60)
    print(f"  Input:    {args.input}")
    print(f"  Model:    {args.model}")
    print(f"  Language: {args.language or 'auto-detect'}")
    print(f"  Diarize:  {args.diarize}", end="")
    if args.diarize and speaker_names:
        print(f" (speakers: {', '.join(speaker_names)})", end="")
    print()
    print(f"  Vision:   {args.vision}", end="")
    if args.vision:
        print(f" (every {args.vision_interval}s, model: {args.vision_model})", end="")
    print()
    if args.summarize:
        print(f"  Summary:  True (model: {args.summary_model})")
    print(f"  Format:   {args.format}")
    print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print()

    total_t0 = time.time()

    # ── Cache ──
    cache_path = args.input + ".whisper_cache.json"
    if args.clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)
        print("Cleared cached results.")

    # Step 1: Transcribe (or load from cache)
    cache_hit = False
    if os.path.exists(cache_path) and not args.clear_cache:
        print("Loading cached transcription + diarization...")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            segments = cache["segments"]
            cached_model = cache.get("model")
            cached_language = cache.get("language")
            cached_diarized = cache.get("diarized", False)
            cached_diarize_params = cache.get("diarize_params", {})

            # Current diarization params for comparison
            current_diarize_params = {
                "num_speakers": args.num_speakers,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
            }

            # Invalidate cache if transcription settings changed
            if cached_model and cached_model != args.model:
                print(f"  Cache used model '{cached_model}', now using '{args.model}' — re-transcribing.")
            elif cached_language and args.language and cached_language != args.language:
                print(f"  Cache used language '{cached_language}', now using '{args.language}' — re-transcribing.")
            elif args.diarize and not cached_diarized:
                print(f"  Cache was not diarized, but --diarize requested — re-transcribing.")
            elif args.diarize and cached_diarized and cached_diarize_params != current_diarize_params:
                print(f"  Diarization parameters changed — re-transcribing.")
            else:
                print(f"Loaded {len(segments)} segments from cache")
                cache_hit = True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Cache file corrupt ({e}), re-transcribing.", file=sys.stderr)

    if not cache_hit:
        segments, info = transcribe_audio(args.input, args.model, args.language)

        # Post-process: remove hallucinations, non-speech, low-confidence
        before_count = len(segments)
        segments = clean_transcription(segments)
        removed = before_count - len(segments)
        if removed > 0:
            print(f"Post-processing: removed {removed} hallucinated/non-speech segments")

        # Step 2: Diarize
        if args.diarize:
            print()
            speaker_segments = diarize_audio(
                args.input, hf_token,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            segments = assign_speakers(segments, speaker_segments, speaker_names)

        # Save cache with metadata
        cache_data = {
            "segments": segments,
            "model": args.model,
            "language": args.language,
            "diarized": args.diarize,
            "diarize_params": {
                "num_speakers": args.num_speakers,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
            },
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
            print(f"Cached transcription to {cache_path}")
        except OSError as e:
            print(f"Warning: Could not save cache: {e}", file=sys.stderr)

    # Apply speaker names to cached data (if loading from cache with new names)
    if cache_hit and speaker_names:
        segments = rename_speakers(segments, speaker_names)

    # Step 3: Vision
    visual_context = None
    tmpdir = None
    try:
        if args.vision:
            print()
            keyframes, tmpdir = extract_keyframes(args.input, interval_secs=args.vision_interval)
            if keyframes:
                visual_context = analyze_keyframes(keyframes, openrouter_key, args.vision_model, args.vision_workers)

        # Step 4: Meeting Summary
        summary = None
        if args.summarize:
            print()
            summary = generate_meeting_summary(
                segments, openrouter_key, args.summary_model,
                diarized=args.diarize, visual_context=visual_context,
            )
    finally:
        # Always clean up temp keyframes, even on error
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Step 5: Build metadata
    duration = get_media_duration(args.input)
    metadata = {
        "input_file": os.path.basename(args.input),
        "model": args.model,
        "language": args.language or "auto-detect",
        "diarized": args.diarize,
        "vision": args.vision,
        "summarized": args.summarize,
        "segments_count": len(segments),
        "duration_seconds": duration,
        "gpu": torch.cuda.get_device_name(0),
    }

    # Step 6: Format output
    output_text = format_output(
        segments, args.format,
        diarized=args.diarize,
        visual_context=visual_context,
        summary=summary,
        metadata=metadata,
    )

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Segments: {len(segments)}")
    if visual_context:
        print(f"Keyframes analyzed: {len(visual_context)}")
    if summary:
        print("Meeting summary: generated")

    # ── Auto-generate output filename if no -o ──
    if not args.output:
        base = os.path.splitext(args.input)[0]
        if args.format != "txt":
            ext_map = {"md": ".md", "srt": ".srt", "vtt": ".vtt", "json": ".json", "tsv": ".tsv"}
            args.output = base + ext_map[args.format]
        elif args.diarize or args.vision or args.summarize:
            # Auto-save full pipeline results alongside input file
            args.output = base + "_transcript.txt"
        if args.output:
            print(f"Auto-saving to: {args.output}")

    # Output
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Saved to: {args.output}\n")
        except OSError as e:
            print(f"Error: Could not write output file: {e}", file=sys.stderr)
            print("\n" + output_text)
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("  TRANSCRIPT")
        print("=" * 60)
        print(output_text)
        print()


if __name__ == "__main__":
    main()
