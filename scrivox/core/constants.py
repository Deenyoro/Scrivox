"""Shared constants for the Scrivox transcription pipeline."""

from collections import OrderedDict

DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
DEFAULT_SUMMARY_MODEL = "google/gemini-2.5-flash"
DEFAULT_TRANSLATION_MODEL = "google/gemini-2.5-flash"

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
DEFAULT_SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
OUTPUT_FORMATS = ["txt", "md", "srt", "vtt", "json", "tsv"]

# LLM API providers
# OpenAI-compatible providers use /v1/chat/completions format.
# Anthropic uses its own Messages API format (handled by llm_client).
LLM_PROVIDERS = {
    "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
    "OpenAI": "https://api.openai.com/v1/chat/completions",
    "Anthropic": "https://api.anthropic.com/v1/messages",
    "Ollama (local)": "http://localhost:11434/v1/chat/completions",
}
DEFAULT_LLM_PROVIDER = "OpenRouter"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Whisper-supported languages: display name -> ISO 639-1 code
# Ordered alphabetically by display name for UI dropdowns
WHISPER_LANGUAGES = OrderedDict([
    ("Afrikaans", "af"),
    ("Arabic", "ar"),
    ("Armenian", "hy"),
    ("Azerbaijani", "az"),
    ("Belarusian", "be"),
    ("Bosnian", "bs"),
    ("Bulgarian", "bg"),
    ("Catalan", "ca"),
    ("Chinese", "zh"),
    ("Croatian", "hr"),
    ("Czech", "cs"),
    ("Danish", "da"),
    ("Dutch", "nl"),
    ("English", "en"),
    ("Estonian", "et"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("Galician", "gl"),
    ("German", "de"),
    ("Greek", "el"),
    ("Hebrew", "he"),
    ("Hindi", "hi"),
    ("Hungarian", "hu"),
    ("Icelandic", "is"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Kannada", "kn"),
    ("Kazakh", "kk"),
    ("Korean", "ko"),
    ("Latvian", "lv"),
    ("Lithuanian", "lt"),
    ("Macedonian", "mk"),
    ("Malay", "ms"),
    ("Marathi", "mr"),
    ("Maori", "mi"),
    ("Nepali", "ne"),
    ("Norwegian", "no"),
    ("Persian", "fa"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Romanian", "ro"),
    ("Russian", "ru"),
    ("Serbian", "sr"),
    ("Slovak", "sk"),
    ("Slovenian", "sl"),
    ("Spanish", "es"),
    ("Swahili", "sw"),
    ("Swedish", "sv"),
    ("Tagalog", "tl"),
    ("Tamil", "ta"),
    ("Thai", "th"),
    ("Turkish", "tr"),
    ("Ukrainian", "uk"),
    ("Urdu", "ur"),
    ("Vietnamese", "vi"),
    ("Welsh", "cy"),
])

# Reverse lookup: code -> display name
LANGUAGE_CODE_TO_NAME = {code: name for name, code in WHISPER_LANGUAGES.items()}

# Build translation target list: Whisper languages + regional variants
# inserted after their parent language for clean dropdown ordering
def _build_translation_languages():
    """Build ordered translation target list with regional variants."""
    # Variants to insert after their parent language
    _extras = {
        "zh": [("Chinese (Simplified)", "zh-CN"), ("Chinese (Traditional)", "zh-TW")],
        "pt": [("Portuguese (Brazil)", "pt-BR")],
    }
    items = []
    for name, code in WHISPER_LANGUAGES.items():
        items.append((name, code))
        if code in _extras:
            items.extend(_extras[code])
    return OrderedDict(items)

TRANSLATION_LANGUAGES = _build_translation_languages()
TRANSLATION_CODE_TO_NAME = {code: name for name, code in TRANSLATION_LANGUAGES.items()}
