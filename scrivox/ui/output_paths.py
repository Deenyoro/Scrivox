"""GUI-side decisions that don't need Tk: where output goes, which files are
accepted, and how pipeline errors are explained. Kept free of tkinter so the
rules can be unit-tested headlessly.
"""

import os

from ..core.constants import AUDIO_EXTENSIONS, OUTPUT_FORMATS, VIDEO_EXTENSIONS

MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# Human descriptions for the "Save as" format picker (values stay the raw
# format ids that config files and the CLI use)
FORMAT_DESCRIPTIONS = {
    "txt": "Plain text with timestamps",
    "md": "Markdown document (good for notes and summaries)",
    "srt": "Subtitles for video players (SubRip)",
    "vtt": "Subtitles for the web (WebVTT)",
    "json": "Structured data with word timings",
    "tsv": "Spreadsheet-friendly table",
}

# Approximate first-run download sizes of the stock Whisper models
MODEL_INFO = {
    "tiny": ("Fastest, lowest accuracy", 75e6),
    "base": ("Very fast, basic accuracy", 145e6),
    "small": ("Fast, good accuracy", 484e6),
    "medium": ("Balanced speed and accuracy", 1.5e9),
    "large-v3": ("Best accuracy, slower", 3.1e9),
    "large-v3-turbo": ("Near-best accuracy, much faster", 1.6e9),
    "distil-large-v3.5": ("Fast, English-focused", 1.5e9),
}


def format_size(num_bytes):
    """1536000 -> '1.5 MB', 3.1e9 -> '3.1 GB'."""
    num = float(num_bytes)
    for unit in ("bytes", "KB", "MB", "GB"):
        if num < 1000 or unit == "GB":
            if unit == "bytes":
                return f"{int(num)} bytes"
            return f"{num:.1f} {unit}" if num < 100 else f"{num:.0f} {unit}"
        num /= 1000.0
    return f"{num:.1f} GB"


def describe_model(name):
    """One-line hint shown under the model picker."""
    info = MODEL_INFO.get(name)
    if not info:
        return "Custom model name or folder"
    desc, size = info
    return f"{desc} · {format_size(size)} download on first use"


def is_media_file(path):
    return os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS


def output_extension(fmt):
    fmt = fmt if fmt in OUTPUT_FORMATS else "txt"
    return "_transcript.txt" if fmt == "txt" else f".{fmt}"


def unique_path(candidate, taken=()):
    """`candidate`, or `name (2).ext`, `name (3).ext`... - whichever neither
    exists on disk nor is already claimed by another job in this run."""
    taken_norm = {os.path.normcase(os.path.abspath(p)) for p in taken}

    def free(p):
        return (not os.path.exists(p)
                and os.path.normcase(os.path.abspath(p)) not in taken_norm)

    if free(candidate):
        return candidate
    base, ext = os.path.splitext(candidate)
    # "talk_transcript.txt" -> "talk_transcript (2).txt"
    n = 2
    while True:
        alt = f"{base} ({n}){ext}"
        if free(alt):
            return alt
        n += 1


def default_output_path(input_path, fmt, audio_track=0, out_dir=None, taken=()):
    """Where the GUI saves a job when the user did not pick a file name.

    Next to the input (or in `out_dir`), named after the input:
    `talk.srt`, `talk_transcript.txt`, `talk_track2.srt`. Never overwrites
    an existing file: a numbered name is used instead.
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    folder = out_dir or os.path.dirname(os.path.abspath(input_path))
    suffix = f"_track{audio_track}" if audio_track else ""
    candidate = os.path.join(folder, f"{stem}{suffix}{output_extension(fmt)}")
    return unique_path(candidate, taken)


def plan_output_paths(jobs, fmt, explicit_output="", out_dir=""):
    """Decide one output path per job (list of (file_path, audio_track)).

    - explicit_output with a single job: used as-is (the user chose it).
    - explicit_output with several jobs: `<chosen>_<input stem>.<ext>`, so
      jobs don't overwrite one another.
    - otherwise: default_output_path() in out_dir or next to each input.
    """
    paths = []
    taken = []
    default_ext = f".{fmt}" if fmt in OUTPUT_FORMATS else ".txt"
    for file_path, audio_track in jobs:
        if explicit_output and len(jobs) == 1:
            path = explicit_output
        elif explicit_output:
            out_base, out_ext = os.path.splitext(explicit_output)
            job_stem = os.path.splitext(os.path.basename(file_path))[0]
            suffix = f"_track{audio_track}" if audio_track else ""
            path = unique_path(f"{out_base}_{job_stem}{suffix}{out_ext or default_ext}", taken)
        else:
            path = default_output_path(file_path, fmt, audio_track,
                                       out_dir=out_dir or None, taken=taken)
        taken.append(path)
        paths.append(path)
    return paths


# ── Error explanations ──

FIX_FFMPEG = "ffmpeg"
FIX_GPU = "gpu"
FIX_KEYS = "keys"


def explain_error(message):
    """Turn a pipeline error into (headline, fix_topic or None).

    The raw text stays in the log; the headline is what a non-technical user
    reads first, and the fix topic picks which help to offer.
    """
    raw = (message or "").strip()
    low = raw.lower()
    if "ffmpeg" in low or "ffprobe" in low:
        return "ffmpeg isn't installed, so Scrivox can't read audio or video.", FIX_FFMPEG
    if "cuda" in low and ("not available" in low or "gpu" in low):
        return "No NVIDIA graphics card was found. Scrivox needs one to transcribe.", FIX_GPU
    if "hf_token" in low or "hugging" in low:
        return "Identifying speakers needs a Hugging Face token.", FIX_KEYS
    if "api key" in low:
        return "This extra needs an AI service key.", FIX_KEYS
    if low.startswith("file not found"):
        return raw.replace("File not found:", "This file no longer exists:"), None
    if "out of memory" in low:
        return ("The graphics card ran out of memory. Try a smaller model "
                "such as large-v3-turbo or medium."), None
    first_line = raw.splitlines()[0] if raw else "Something went wrong"
    if len(first_line) > 160:
        first_line = first_line[:157] + "..."
    return first_line, None
