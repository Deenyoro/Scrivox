"""Shared constants for the Scrivox transcription pipeline."""

DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
DEFAULT_SUMMARY_MODEL = "google/gemini-2.5-flash"

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
DEFAULT_SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
OUTPUT_FORMATS = ["txt", "md", "srt", "vtt", "json", "tsv"]

# LLM API providers (all use OpenAI-compatible /v1/chat/completions format)
LLM_PROVIDERS = {
    "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
    "OpenAI": "https://api.openai.com/v1/chat/completions",
    "Ollama (local)": "http://localhost:11434/v1/chat/completions",
}
DEFAULT_LLM_PROVIDER = "OpenRouter"
