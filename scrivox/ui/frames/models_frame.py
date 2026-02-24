"""Advanced settings: model config, subtitle tuning, and post-processing thresholds."""

import tkinter as tk
from tkinter import ttk

from ...core.constants import DEFAULT_DIARIZATION_MODEL
from ..theme import COLORS, FONTS
from .settings_frame import ToolTip


class ModelsFrame(ttk.LabelFrame):
    """Collapsible advanced settings for models, subtitles, and post-processing."""

    def __init__(self, parent, show_diarization=True, **kwargs):
        super().__init__(parent, text="ADVANCED", **kwargs)

        self._show_diarization = show_diarization
        self.diarization_model_var = tk.StringVar(value=DEFAULT_DIARIZATION_MODEL)
        self._expanded = tk.BooleanVar(value=False)

        # Subtitle tuning
        self.subtitle_max_chars_var = tk.StringVar(value="84")
        self.subtitle_max_duration_var = tk.StringVar(value="4.0")
        self.subtitle_max_gap_var = tk.StringVar(value="0.8")

        # Post-processing
        self.confidence_threshold_var = tk.StringVar(value="0.50")

        self._build()

    def _build(self):
        # Toggle button
        self._toggle_btn = ttk.Checkbutton(
            self, text="Show advanced settings",
            variable=self._expanded, command=self._toggle,
        )
        self._toggle_btn.pack(padx=8, pady=(8, 4), anchor=tk.W)

        # Content frame (hidden by default)
        self._content = ttk.Frame(self)

        # ── Diarization model (only in Regular/Full builds) ──
        if self._show_diarization:
            ttk.Label(self._content, text="Models", style="Header.TLabel").pack(
                padx=8, pady=(4, 2), anchor=tk.W)

            row = ttk.Frame(self._content)
            row.pack(fill=tk.X, padx=8, pady=(0, 2))
            ttk.Label(row, text="Diarization:").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=self.diarization_model_var).pack(
                side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

            ttk.Label(self._content, text="HuggingFace model ID or local path",
                      style="Dim.TLabel").pack(padx=8, pady=(0, 6), anchor=tk.W)

            ttk.Separator(self._content, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=2)

        # ── Subtitle tuning ──
        ttk.Label(self._content, text="Subtitle Merging", style="Header.TLabel").pack(
            padx=8, pady=(4, 2), anchor=tk.W)

        row = ttk.Frame(self._content)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text="Max chars/cue:").pack(side=tk.LEFT)
        e = ttk.Entry(row, textvariable=self.subtitle_max_chars_var, width=6)
        e.pack(side=tk.RIGHT)
        ToolTip(e, "Maximum characters per subtitle cue\n(two lines of ~42 chars = 84)")

        row = ttk.Frame(self._content)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text="Max duration (s):").pack(side=tk.LEFT)
        e = ttk.Entry(row, textvariable=self.subtitle_max_duration_var, width=6)
        e.pack(side=tk.RIGHT)
        ToolTip(e, "Max seconds a single subtitle stays on screen\nLower = more frequent cue changes")

        row = ttk.Frame(self._content)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text="Max gap (s):").pack(side=tk.LEFT)
        e = ttk.Entry(row, textvariable=self.subtitle_max_gap_var, width=6)
        e.pack(side=tk.RIGHT)
        ToolTip(e, "Max silence gap to merge across\nLower = more subtitle breaks at pauses")

        ttk.Label(self._content, text="Controls how Whisper segments merge into subtitle cues",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 6), anchor=tk.W)

        # ── Post-processing ──
        ttk.Separator(self._content, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(self._content, text="Post-Processing", style="Header.TLabel").pack(
            padx=8, pady=(4, 2), anchor=tk.W)

        row = ttk.Frame(self._content)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text="Confidence threshold:").pack(side=tk.LEFT)
        e = ttk.Entry(row, textvariable=self.confidence_threshold_var, width=6)
        e.pack(side=tk.RIGHT)
        ToolTip(e, "Min avg word probability to keep a segment\n0.50 = default, lower keeps more, higher filters more")

        ttk.Label(self._content, text="Segments below this confidence are removed as hallucinations",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 8), anchor=tk.W)

    def _toggle(self):
        if self._expanded.get():
            self._content.pack(fill=tk.X, after=self._toggle_btn)
        else:
            self._content.pack_forget()

    def get_diarization_model(self):
        return self.diarization_model_var.get().strip() or DEFAULT_DIARIZATION_MODEL

    def load_settings(self, settings):
        """Load settings from config dict."""
        self.diarization_model_var.set(
            settings.get("diarization_model", DEFAULT_DIARIZATION_MODEL))
        self.subtitle_max_chars_var.set(str(settings.get("subtitle_max_chars", 84)))
        self.subtitle_max_duration_var.set(str(settings.get("subtitle_max_duration", 4.0)))
        self.subtitle_max_gap_var.set(str(settings.get("subtitle_max_gap", 0.8)))
        self.confidence_threshold_var.set(str(settings.get("confidence_threshold", 0.50)))

    def get_settings_dict(self):
        """Return settings as a dict for config persistence."""
        return {
            "diarization_model": self.diarization_model_var.get().strip() or DEFAULT_DIARIZATION_MODEL,
            "subtitle_max_chars": self._parse_int(self.subtitle_max_chars_var.get(), 84),
            "subtitle_max_duration": self._parse_float(self.subtitle_max_duration_var.get(), 4.0),
            "subtitle_max_gap": self._parse_float(self.subtitle_max_gap_var.get(), 0.8),
            "confidence_threshold": self._parse_float(self.confidence_threshold_var.get(), 0.50),
        }

    def _parse_int(self, val, default):
        try:
            return int(val.strip())
        except (ValueError, AttributeError):
            return default

    def _parse_float(self, val, default):
        try:
            return float(val.strip())
        except (ValueError, AttributeError):
            return default
