"""Model, language, feature toggles, and speaker name settings."""

import tkinter as tk
from tkinter import ttk

from ...core.constants import WHISPER_MODELS, DEFAULT_VISION_MODEL, DEFAULT_SUMMARY_MODEL
from ..theme import COLORS, FONTS


class SettingsFrame(ttk.Frame):
    """Model/language combos, feature checkboxes, and sub-settings."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # ── Variables ──
        self.model_var = tk.StringVar(value="large-v3")
        self.language_var = tk.StringVar(value="")
        self.diarize_var = tk.BooleanVar(value=False)
        self.vision_var = tk.BooleanVar(value=False)
        self.summarize_var = tk.BooleanVar(value=False)

        # Diarization sub-settings
        self.num_speakers_var = tk.StringVar(value="")
        self.min_speakers_var = tk.StringVar(value="")
        self.max_speakers_var = tk.StringVar(value="")
        self.speaker_names_var = tk.StringVar(value="")

        # Vision sub-settings
        self.vision_interval_var = tk.StringVar(value="60")
        self.vision_model_var = tk.StringVar(value=DEFAULT_VISION_MODEL)
        self.vision_workers_var = tk.StringVar(value="4")

        # Summary sub-settings
        self.summary_model_var = tk.StringVar(value=DEFAULT_SUMMARY_MODEL)

        self._build()

    def _build(self):
        # ── MODEL & LANGUAGE ──
        model_frame = ttk.LabelFrame(self, text="MODEL & LANGUAGE")
        model_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        row1 = ttk.Frame(model_frame)
        row1.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row1, text="Model:").pack(side=tk.LEFT)
        model_combo = ttk.Combobox(row1, textvariable=self.model_var,
                                    values=WHISPER_MODELS, state="readonly", width=16)
        model_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        row2 = ttk.Frame(model_frame)
        row2.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(row2, text="Language:").pack(side=tk.LEFT)
        lang_entry = ttk.Entry(row2, textvariable=self.language_var, width=16)
        lang_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))
        # Tooltip-style hint
        ttk.Label(model_frame, text="Primary language, blank for auto-detect (e.g. ko, ja, en)",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 6), anchor=tk.W)

        # ── FEATURES ──
        features_frame = ttk.LabelFrame(self, text="FEATURES")
        features_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        ttk.Checkbutton(features_frame, text="Diarize (speaker labels)",
                        variable=self.diarize_var,
                        command=self._toggle_diarize).pack(padx=8, pady=(8, 2), anchor=tk.W)
        ttk.Checkbutton(features_frame, text="Vision (keyframe analysis)",
                        variable=self.vision_var,
                        command=self._toggle_vision).pack(padx=8, pady=2, anchor=tk.W)
        ttk.Checkbutton(features_frame, text="Summarize (meeting summary)",
                        variable=self.summarize_var,
                        command=self._toggle_summary).pack(padx=8, pady=(2, 8), anchor=tk.W)

        # ── DIARIZATION SUB-SETTINGS ──
        self._diarize_frame = ttk.LabelFrame(self, text="DIARIZATION")
        # Initially hidden; shown when diarize is checked

        row = ttk.Frame(self._diarize_frame)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row, text="Speakers:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.num_speakers_var, width=6).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Label(row, text="exact", style="Dim.TLabel").pack(side=tk.RIGHT)

        row = ttk.Frame(self._diarize_frame)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text="Range:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.max_speakers_var, width=4).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Label(row, text="max", style="Dim.TLabel").pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Entry(row, textvariable=self.min_speakers_var, width=4).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Label(row, text="min", style="Dim.TLabel").pack(side=tk.RIGHT)

        row = ttk.Frame(self._diarize_frame)
        row.pack(fill=tk.X, padx=8, pady=(2, 8))
        ttk.Label(row, text="Names:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.speaker_names_var).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))
        ttk.Label(self._diarize_frame, text="Comma-separated: Alice,Bob,Charlie",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 6), anchor=tk.W)

        # ── VISION SUB-SETTINGS ──
        self._vision_frame = ttk.LabelFrame(self, text="VISION")

        row = ttk.Frame(self._vision_frame)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row, text="Interval (s):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.vision_interval_var, width=6).pack(side=tk.RIGHT)

        row = ttk.Frame(self._vision_frame)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text="Model:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.vision_model_var).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        row = ttk.Frame(self._vision_frame)
        row.pack(fill=tk.X, padx=8, pady=(2, 8))
        ttk.Label(row, text="Workers:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.vision_workers_var, width=6).pack(side=tk.RIGHT)

        # ── SUMMARY SUB-SETTINGS ──
        self._summary_frame = ttk.LabelFrame(self, text="SUMMARY")

        row = ttk.Frame(self._summary_frame)
        row.pack(fill=tk.X, padx=8, pady=(8, 8))
        ttk.Label(row, text="Model:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.summary_model_var).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

    def _toggle_diarize(self):
        if self.diarize_var.get():
            self._diarize_frame.pack(fill=tk.X, padx=4, pady=(0, 6),
                                      after=self._find_features_frame())
        else:
            self._diarize_frame.pack_forget()

    def _toggle_vision(self):
        if self.vision_var.get():
            # Insert after diarize frame if visible, else after features
            after = self._diarize_frame if self.diarize_var.get() else self._find_features_frame()
            self._vision_frame.pack(fill=tk.X, padx=4, pady=(0, 6), after=after)
        else:
            self._vision_frame.pack_forget()

    def _toggle_summary(self):
        if self.summarize_var.get():
            after = self._vision_frame if self.vision_var.get() else (
                self._diarize_frame if self.diarize_var.get() else self._find_features_frame()
            )
            self._summary_frame.pack(fill=tk.X, padx=4, pady=(0, 6), after=after)
        else:
            self._summary_frame.pack_forget()

    def _find_features_frame(self):
        """Find the FEATURES labelframe widget."""
        for child in self.winfo_children():
            if isinstance(child, ttk.LabelFrame) and child.cget("text") == "FEATURES":
                return child
        return self

    def get_speaker_names(self):
        """Parse and return speaker names list, or None."""
        raw = self.speaker_names_var.get().strip()
        if not raw:
            return None
        return [n.strip() for n in raw.split(",") if n.strip()]

    def get_int_or_none(self, var):
        """Parse a StringVar as int or return None."""
        val = var.get().strip()
        if not val:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    def load_settings(self, settings):
        """Load settings dict into widget variables."""
        self.model_var.set(settings.get("model", "large-v3"))
        self.language_var.set(settings.get("language", ""))
        self.diarize_var.set(settings.get("diarize", False))
        self.vision_var.set(settings.get("vision", False))
        self.summarize_var.set(settings.get("summarize", False))
        self.speaker_names_var.set(settings.get("speaker_names", ""))
        self.vision_interval_var.set(str(settings.get("vision_interval", 60)))
        self.vision_model_var.set(settings.get("vision_model", DEFAULT_VISION_MODEL))
        self.vision_workers_var.set(str(settings.get("vision_workers", 4)))
        self.summary_model_var.set(settings.get("summary_model", DEFAULT_SUMMARY_MODEL))

        num = settings.get("num_speakers")
        self.num_speakers_var.set(str(num) if num else "")
        mins = settings.get("min_speakers")
        self.min_speakers_var.set(str(mins) if mins else "")
        maxs = settings.get("max_speakers")
        self.max_speakers_var.set(str(maxs) if maxs else "")

        # Show/hide sub-frames
        self._toggle_diarize()
        self._toggle_vision()
        self._toggle_summary()

    def get_settings_dict(self):
        """Return current settings as a dict for config persistence."""
        return {
            "model": self.model_var.get(),
            "language": self.language_var.get(),
            "diarize": self.diarize_var.get(),
            "vision": self.vision_var.get(),
            "summarize": self.summarize_var.get(),
            "num_speakers": self.get_int_or_none(self.num_speakers_var),
            "min_speakers": self.get_int_or_none(self.min_speakers_var),
            "max_speakers": self.get_int_or_none(self.max_speakers_var),
            "speaker_names": self.speaker_names_var.get(),
            "vision_interval": int(self.vision_interval_var.get() or 60),
            "vision_model": self.vision_model_var.get(),
            "vision_workers": int(self.vision_workers_var.get() or 4),
            "summary_model": self.summary_model_var.get(),
        }
