"""ConfigManager: JSON-based config persistence next to exe or in %APPDATA%."""

import json
import os
import sys


def _get_config_dir():
    """Determine config directory: next to exe (frozen) or next to project root (dev)."""
    if getattr(sys, "frozen", False):
        # PyInstaller exe — config lives next to the exe
        exe_dir = os.path.dirname(sys.executable)
        # Test if writable
        test_path = os.path.join(exe_dir, ".scrivox_write_test")
        try:
            with open(test_path, "w") as f:
                f.write("test")
            os.remove(test_path)
            return exe_dir
        except OSError:
            pass
    else:
        # Dev mode — config lives next to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return project_root

    # Fallback: %APPDATA%\Scrivox\
    appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
    config_dir = os.path.join(appdata, "Scrivox")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


CONFIG_FILENAME = "scrivox_config.json"

_DEFAULT_CONFIG = {
    "credentials": {
        "hf_token": "",
        "openrouter_key": "",
    },
    "last_settings": {
        "model": "large-v3",
        "language": "",
        "diarize": False,
        "vision": False,
        "summarize": False,
        "output_format": "txt",
        "num_speakers": None,
        "min_speakers": None,
        "max_speakers": None,
        "speaker_names": "",
        "vision_interval": 60,
        "vision_model": "google/gemini-2.5-flash",
        "vision_workers": 4,
        "summary_model": "google/gemini-2.5-flash",
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "use_system_cuda": False,
        "subtitle_max_chars": 84,
        "subtitle_max_duration": 4.0,
        "subtitle_max_gap": 0.8,
        "confidence_threshold": 0.50,
    },
    "api": {
        "provider": "OpenRouter",
        "custom_base": "",
    },
    "ui": {
        "geometry": "",
        "last_input_dir": "",
        "last_output_dir": "",
        "preferred_language": "",
        "recent_files": [],
    },
}


class ConfigManager:
    """Read/write JSON config file with section-based access."""

    def __init__(self):
        self._dir = _get_config_dir()
        self._path = os.path.join(self._dir, CONFIG_FILENAME)
        self._data = {}
        self._load()

    @property
    def path(self):
        return self._path

    def _load(self):
        """Load config from disk, merging with defaults."""
        self._data = json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                # Merge saved values into defaults (preserving new default keys)
                for section in _DEFAULT_CONFIG:
                    if section in saved and isinstance(saved[section], dict):
                        self._data[section].update(saved[section])
            except (json.JSONDecodeError, OSError):
                pass  # corrupt file, use defaults

    def save(self):
        """Persist config to disk."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass  # silently fail if can't write

    def get(self, section, key, default=None):
        """Get a config value."""
        return self._data.get(section, {}).get(key, default)

    def set(self, section, key, value):
        """Set a config value (does not auto-save)."""
        if section not in self._data:
            self._data[section] = {}
        self._data[section][key] = value

    def get_section(self, section):
        """Get an entire config section as a dict."""
        return dict(self._data.get(section, {}))

    def get_credentials(self):
        """Return (hf_token, openrouter_key) from config."""
        creds = self._data.get("credentials", {})
        return creds.get("hf_token", ""), creds.get("openrouter_key", "")

    def set_credentials(self, hf_token=None, openrouter_key=None):
        """Update credential values."""
        if hf_token is not None:
            self.set("credentials", "hf_token", hf_token)
        if openrouter_key is not None:
            self.set("credentials", "openrouter_key", openrouter_key)

    def save_last_settings(self, **kwargs):
        """Save last-used pipeline settings."""
        for key, value in kwargs.items():
            self.set("last_settings", key, value)
        self.save()

    def get_last_settings(self):
        """Get last-used pipeline settings."""
        return self.get_section("last_settings")
