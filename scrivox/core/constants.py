"""Shared constants for the Scrivox transcription pipeline."""

DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
DEFAULT_SUMMARY_MODEL = "google/gemini-2.5-flash"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
OUTPUT_FORMATS = ["txt", "md", "srt", "vtt", "json", "tsv"]
