"""Media utilities: ffmpeg checks, video detection, duration, WAV extraction."""

import os
import subprocess
import sys
import tempfile

from .constants import VIDEO_EXTENSIONS


def check_ffmpeg(on_progress=print):
    """Verify ffmpeg and ffprobe are available."""
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except FileNotFoundError:
            on_progress(f"Error: '{tool}' not found. Install ffmpeg and ensure it's in your PATH.")
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


def extract_wav(input_path, on_progress=print):
    """Extract audio to WAV for diarization. Returns path to temp WAV file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    on_progress("Extracting audio to WAV for diarization...")
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
