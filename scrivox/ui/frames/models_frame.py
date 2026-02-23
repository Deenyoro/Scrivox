"""Advanced model configuration frame (collapsed by default)."""

import tkinter as tk
from tkinter import ttk

from ...core.constants import DEFAULT_DIARIZATION_MODEL
from ..theme import COLORS, FONTS


class ModelsFrame(ttk.LabelFrame):
    """Collapsible advanced model settings for diarization model selection."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="MODELS (Advanced)", **kwargs)

        self.diarization_model_var = tk.StringVar(value=DEFAULT_DIARIZATION_MODEL)
        self._expanded = tk.BooleanVar(value=False)

        self._build()

    def _build(self):
        # Toggle button
        self._toggle_btn = ttk.Checkbutton(
            self, text="Show advanced model settings",
            variable=self._expanded, command=self._toggle,
        )
        self._toggle_btn.pack(padx=8, pady=(8, 4), anchor=tk.W)

        # Content frame (hidden by default)
        self._content = ttk.Frame(self)

        # Diarization model
        row = ttk.Frame(self._content)
        row.pack(fill=tk.X, padx=8, pady=(4, 2))
        ttk.Label(row, text="Diarization:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.diarization_model_var).pack(
            side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        ttk.Label(self._content, text="HuggingFace model ID or local path",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 8), anchor=tk.W)

    def _toggle(self):
        if self._expanded.get():
            self._content.pack(fill=tk.X, after=self._toggle_btn)
        else:
            self._content.pack_forget()

    def get_diarization_model(self):
        return self.diarization_model_var.get().strip() or DEFAULT_DIARIZATION_MODEL

    def load_settings(self, settings):
        """Load model settings from config dict."""
        self.diarization_model_var.set(
            settings.get("diarization_model", DEFAULT_DIARIZATION_MODEL))

    def get_settings_dict(self):
        """Return model settings as a dict for config persistence."""
        return {
            "diarization_model": self.diarization_model_var.get().strip() or DEFAULT_DIARIZATION_MODEL,
        }
