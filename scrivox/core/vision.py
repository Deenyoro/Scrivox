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


def extract_keyframes(video_path, interval_secs=60, max_frames=30, on_progress=print):
    """Extract keyframes from video at regular intervals."""
    tmpdir = tempfile.mkdtemp(prefix="whisper_frames_")

    duration = get_media_duration(video_path)
    if duration is None:
        on_progress("Warning: Could not determine video duration, using default interval")
        duration = interval_secs * max_frames

    if duration / interval_secs > max_frames:
        old_interval = interval_secs
        interval_secs = int(duration / max_frames)
        on_progress(f"  Adjusted keyframe interval from {old_interval}s to {interval_secs}s (capped at {max_frames} frames)")

    on_progress(f"Extracting keyframes every {interval_secs}s from {duration:.0f}s video...")

    from .media import _subprocess_flags
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps=1/{int(interval_secs)},scale=1280:-2",
         "-q:v", "3",
         os.path.join(tmpdir, "frame_%04d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        timeout=600,
        **_subprocess_flags(),
    )

    frames = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
    keyframes = []
    for i, path in enumerate(frames):
        timestamp = i * interval_secs
        keyframes.append({"path": path, "timestamp": timestamp})

    on_progress(f"Extracted {len(keyframes)} keyframes")
    return keyframes, tmpdir


def describe_keyframe(image_path, timestamp, api_key, vision_model, api_base=None, max_retries=3, on_progress=print):
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
                        "Briefly describe what's visible on screen \u2014 any text, UI elements, "
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
    ]

    from .constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
    url = api_base or LLM_PROVIDERS[DEFAULT_LLM_PROVIDER]

    result = chat_completion(
        messages=messages,
        model=vision_model,
        api_key=api_key,
        api_base=url,
        max_tokens=200,
        max_retries=max_retries,
        timeout=60,
    )

    return result or "[Vision error: no response]"


def analyze_keyframes(keyframes, api_key, vision_model, max_workers=4, api_base=None, on_progress=print):
    """Describe all keyframes using vision LLM with concurrent requests."""
    on_progress(f"Analyzing {len(keyframes)} keyframes with vision LLM ({vision_model})...")
    t0 = time.time()

    descriptions = [None] * len(keyframes)

    def process_frame(idx, kf):
        ts_str = format_timestamp_human(kf["timestamp"])
        on_progress(f"  Frame {idx+1}/{len(keyframes)} @ {ts_str}...")
        desc = describe_keyframe(kf["path"], kf["timestamp"], api_key, vision_model, api_base=api_base, on_progress=on_progress)
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
                on_progress(f"  Frame {frame_idx+1} @ {ts_str} failed: {e}")
                descriptions[frame_idx] = {
                    "timestamp": keyframes[frame_idx]["timestamp"],
                    "description": f"[Frame analysis failed: {type(e).__name__}]",
                }

    descriptions = [d for d in descriptions if d is not None]

    elapsed = time.time() - t0
    on_progress(f"Vision analysis done in {elapsed:.1f}s")
    return descriptions
