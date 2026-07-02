"""Media utilities: ffmpeg checks, video detection, duration, WAV extraction, audio tracks."""

import json as _json
import os
import subprocess
import sys
import tempfile

from .constants import VIDEO_EXTENSIONS


def _subprocess_flags():
    """Return kwargs to hide console windows on Windows."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def check_ffmpeg(on_progress=print):
    """Verify ffmpeg and ffprobe are available.

    Raises PipelineError if not found (instead of sys.exit).
    """
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                **_subprocess_flags(),
            )
        except FileNotFoundError:
            from .pipeline import PipelineError
            raise PipelineError(
                f"'{tool}' not found. Install ffmpeg and ensure it's in your PATH."
            )
        except subprocess.TimeoutExpired:
            pass  # slow but exists


def has_video_stream(file_path):
    """Use ffprobe to check if file contains a real video stream.

    Embedded cover art (attached_pic, e.g. album art in MP3/M4A) does not count —
    otherwise vision analysis would burn API calls describing a static image.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=codec_type",
             "-show_entries", "stream_disposition=attached_pic",
             "-of", "json", file_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
            **_subprocess_flags(),
        )
        streams = _json.loads(result.stdout or "{}").get("streams", [])
        return any(
            not s.get("disposition", {}).get("attached_pic", 0)
            for s in streams
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, _json.JSONDecodeError):
        return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS


def get_media_duration(file_path):
    """Get duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
            **_subprocess_flags(),
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def list_audio_tracks(file_path):
    """Return list of audio track dicts via ffprobe.

    Each dict: {index, codec, language, channels, sample_rate, title, is_default}
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index,codec_name,channels,sample_rate",
             "-show_entries", "stream_tags=language,title",
             "-show_entries", "stream_disposition=default",
             "-of", "json", file_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15,
            **_subprocess_flags(),
        )
        data = _json.loads(result.stdout)
    except Exception:
        return []

    streams = data.get("streams", [])
    tracks = []
    for i, stream in enumerate(streams):
        tags = stream.get("tags", {})
        disposition = stream.get("disposition", {})
        tracks.append({
            "index": i,
            "codec": stream.get("codec_name", "unknown"),
            "language": tags.get("language", ""),
            "channels": stream.get("channels", 0),
            "sample_rate": stream.get("sample_rate", ""),
            "title": tags.get("title", ""),
            "is_default": bool(disposition.get("default", 0)),
        })
    return tracks


def extract_wav(input_path, track_index=0, on_progress=print):
    """Extract audio to WAV for diarization. Returns path to temp WAV file.

    Args:
        input_path: Path to the media file.
        track_index: Audio stream index to extract (default 0 = first audio track).
        on_progress: Progress callback.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    on_progress(f"Extracting audio to WAV (track {track_index})...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-map", f"0:a:{track_index}",
             "-ac", "1", "-ar", "16000", "-vn", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            encoding="utf-8", errors="replace", check=True,
            timeout=600,
            **_subprocess_flags(),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        from .pipeline import PipelineError
        if isinstance(e, subprocess.TimeoutExpired):
            raise PipelineError(f"ffmpeg timed out extracting audio from {input_path}")
        stderr_tail = "\n".join((e.stderr or "").strip().splitlines()[-5:])
        raise PipelineError(
            f"ffmpeg failed to extract audio track {track_index} from {input_path}:\n{stderr_tail}"
        )
    return wav_path
