"""Vision / keyframe analysis using LLM vision models."""

import base64
import concurrent.futures
import glob
import os
import subprocess
import tempfile
import time

from .formatter import format_timestamp_human
from .llm_client import chat_completion
from .media import get_media_duration


def _dhash(image_path, hash_size=8):
    """64-bit difference hash of an image. Returns int. Resilient to minor compression/noise."""
    from PIL import Image
    with Image.open(image_path) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = list(small.getdata())
    diff = 0
    width = hash_size + 1
    for row in range(hash_size):
        base = row * width
        for col in range(hash_size):
            left = pixels[base + col]
            right = pixels[base + col + 1]
            diff = (diff << 1) | (1 if left > right else 0)
    return diff


def _hamming(a, b):
    return bin(a ^ b).count("1")


def extract_keyframes(video_path, interval_secs=60, max_frames=0, change_threshold=0, on_progress=print):
    """Extract keyframes from video at regular intervals, optionally deduping unchanged frames.

    Args:
        interval_secs: Sampling interval in seconds. Float values < 1.0 are supported.
        max_frames: Maximum frames to extract. 0 = unlimited (use interval as-is).
        change_threshold: If > 0, dedupe frames whose dhash differs from the last kept
            frame by at most this many bits (out of 64). Typical: 5. 0 disables dedup.
    """
    tmpdir = tempfile.mkdtemp(prefix="whisper_frames_")

    duration = get_media_duration(video_path)
    if duration is None:
        on_progress("Warning: Could not determine video duration, using default interval")
        duration = interval_secs * (max_frames or 30)

    if max_frames > 0 and duration / interval_secs > max_frames:
        old_interval = interval_secs
        interval_secs = duration / max_frames
        on_progress(f"  Adjusted keyframe interval from {old_interval}s to {interval_secs:.2f}s (capped at {max_frames} frames)")

    interval_secs = float(interval_secs)
    fps_expr = f"{1.0 / interval_secs:.6f}"
    on_progress(f"Extracting keyframes every {interval_secs}s from {duration:.0f}s video (fps={fps_expr})...")

    from .media import _subprocess_flags
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", f"fps={fps_expr},scale=1280:-2",
             "-q:v", "3",
             os.path.join(tmpdir, "frame_%05d.jpg")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            timeout=7200,
            **_subprocess_flags(),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Caller never receives tmpdir on failure — clean it up here or the
        # partially-extracted frames leak in %TEMP%
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    frames = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
    all_keyframes = []
    for i, path in enumerate(frames):
        timestamp = i * interval_secs
        all_keyframes.append({"path": path, "timestamp": timestamp})

    if change_threshold <= 0 or len(all_keyframes) <= 1:
        on_progress(f"Extracted {len(all_keyframes)} keyframes")
        return all_keyframes, tmpdir

    on_progress(f"Deduping {len(all_keyframes)} frames (change_threshold={change_threshold} bits)...")
    kept = []
    last_hash = None
    dropped = 0
    for kf in all_keyframes:
        try:
            h = _dhash(kf["path"])
        except Exception as e:
            on_progress(f"  Hash failed for {os.path.basename(kf['path'])}: {e} — keeping frame")
            kept.append(kf)
            last_hash = None
            continue
        if last_hash is None or _hamming(h, last_hash) > change_threshold:
            kept.append(kf)
            last_hash = h
        else:
            dropped += 1
            try:
                os.remove(kf["path"])
            except OSError:
                pass
    on_progress(f"Kept {len(kept)} unique frames (dropped {dropped} near-duplicates)")
    return kept, tmpdir


def describe_keyframe(image_path, timestamp, api_key, vision_model, api_base=None, max_retries=3,
                      on_progress=print, cancel_event=None):
    """Send a keyframe to vision LLM and get a description, with retries."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    ts_str = format_timestamp_human(timestamp)

    # Build messages in OpenAI format — llm_client converts for Anthropic
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"This is a screenshot from a video at timestamp {ts_str}. "
                        "Describe everything visible on screen in detail: the application or "
                        "website shown, window titles, on-screen text (transcribe key headings, "
                        "labels, names, and numbers verbatim), tables or data values, UI elements, "
                        "people, slides, and any content being presented or edited. "
                        "Write a thorough description that captures all context someone reading "
                        "the meeting transcript would need to understand what was on screen."
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
    ]

    from .constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
    url = api_base or LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]

    result = chat_completion(
        messages=messages,
        model=vision_model,
        api_key=api_key,
        api_base=url,
        max_tokens=8000,
        max_retries=max_retries,
        # Detailed descriptions can take a while to stream at 8000 tokens
        timeout=300,
        cancel_event=cancel_event,
    )

    return result or "[Vision error: no response]"


def analyze_keyframes(keyframes, api_key, vision_model, max_workers=4, api_base=None,
                      on_progress=print, cancel_event=None):
    """Describe all keyframes using vision LLM with concurrent requests.

    Frames whose analysis failed are dropped (not embedded as error strings in
    the transcript). Honors cancel_event between frames.
    """
    on_progress(f"Analyzing {len(keyframes)} keyframes with vision LLM ({vision_model})...")
    t0 = time.time()

    descriptions = [None] * len(keyframes)

    def process_frame(idx, kf):
        if cancel_event and cancel_event.is_set():
            return idx, None
        ts_str = format_timestamp_human(kf["timestamp"])
        on_progress(f"  Frame {idx+1}/{len(keyframes)} @ {ts_str}...")
        desc = describe_keyframe(kf["path"], kf["timestamp"], api_key, vision_model,
                                 api_base=api_base, on_progress=on_progress,
                                 cancel_event=cancel_event)
        return idx, {"timestamp": kf["timestamp"], "description": desc}

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_frame, i, kf): i for i, kf in enumerate(keyframes)}
        for future in concurrent.futures.as_completed(futures):
            if cancel_event and cancel_event.is_set():
                # Drop queued frames immediately — in-flight requests bail out
                # via the cancel_event passed to chat_completion
                executor.shutdown(wait=False, cancel_futures=True)
                break
            try:
                idx, result = future.result()
                descriptions[idx] = result
            except Exception as e:
                failed += 1
                frame_idx = futures[future]
                ts_str = format_timestamp_human(keyframes[frame_idx]["timestamp"])
                on_progress(f"  Frame {frame_idx+1} @ {ts_str} failed: {e}")

    if cancel_event and cancel_event.is_set():
        on_progress("Vision analysis cancelled.")

    # Drop failed frames — an "[API error 429 ...]" string must never appear
    # in the transcript as a screen description
    from .llm_client import is_error_response
    kept = []
    for d in descriptions:
        if d is None:
            continue
        if is_error_response(d["description"]):
            failed += 1
            continue
        kept.append(d)
    if failed:
        on_progress(f"  Warning: {failed} frame(s) could not be described and were skipped")

    elapsed = time.time() - t0
    on_progress(f"Vision analysis done in {elapsed:.1f}s ({len(kept)} descriptions)")
    return kept
