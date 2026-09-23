"""Advanced settings: speaker model, subtitle tuning, accuracy filter and
hardware. Shown on the "Advanced" tab of the Settings dialog."""

import tkinter as tk
from tkinter import ttk

from ...core.constants import DEFAULT_DIARIZATION_MODEL
from ..theme import SP_L, SP_S, SP_XS, px
from ..widgets import ToolTip, WrappingLabel


def _row(parent, label, var, tip, from_, to, increment=1.0, width=7):
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, pady=(0, SP_XS))
    ttk.Label(row, text=label).pack(side=tk.LEFT)
    spin = ttk.Spinbox(row, textvariable=var, from_=from_, to=to, increment=increment,
                       width=width)
    spin.pack(side=tk.RIGHT)
    ToolTip(spin, tip)
    return spin


class ModelsFrame(ttk.Frame):
    """Advanced models, subtitle and post-processing settings."""

    def __init__(self, parent, show_diarization=True, **kwargs):
        kwargs.setdefault("padding", (px(16), px(12)))
        super().__init__(parent, **kwargs)

        self._show_diarization = show_diarization
        self.diarization_model_var = tk.StringVar(value=DEFAULT_DIARIZATION_MODEL)
        # Always expanded now that it lives in its own tab (kept for callers)
        self._expanded = tk.BooleanVar(value=True)

        # Subtitle tuning
        self.subtitle_max_chars_var = tk.StringVar(value="84")
        self.subtitle_max_duration_var = tk.StringVar(value="4.0")
        self.subtitle_max_gap_var = tk.StringVar(value="0.8")
        self.subtitle_min_chars_var = tk.StringVar(value="15")

        # Post-processing
        self.confidence_threshold_var = tk.StringVar(value="0.50")
        self.use_system_cuda_var = tk.BooleanVar(value=False)

        self._build()

    def _build(self):
        self._content = self

        # ── Diarization model (only in Regular/Full builds) ──
        if self._show_diarization:
            ttk.Label(self, text="Speaker model", style="Header.TLabel").pack(
                anchor=tk.W, pady=(0, SP_XS))
            row = ttk.Frame(self)
            row.pack(fill=tk.X, pady=(0, SP_XS))
            ttk.Label(row, text="Model").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=self.diarization_model_var).pack(
                side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
            WrappingLabel(self, text="Hugging Face model ID or a local folder. "
                                     "Leave as is unless you know you need another.",
                          style="Dim.TLabel", justify=tk.LEFT).pack(fill=tk.X, pady=(0, SP_L))

        # ── Subtitle tuning ──
        ttk.Label(self, text="Subtitles (SRT/VTT)", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, SP_XS))
        self._max_chars = _row(self, "Max characters per subtitle", self.subtitle_max_chars_var,
                               "Two lines of about 42 characters = 84", 10, 400, 1)
        self._max_dur = _row(self, "Max seconds on screen", self.subtitle_max_duration_var,
                             "Lower = subtitles change more often", 0.5, 30, 0.5)
        self._max_gap = _row(self, "Merge across pauses up to (s)", self.subtitle_max_gap_var,
                             "Lower = more subtitle breaks at pauses", 0, 10, 0.1)
        self._min_chars = _row(self, "Min characters when splitting", self.subtitle_min_chars_var,
                               "Prevents tiny one-word subtitles", 0, 200, 1)
        self._sub_error = WrappingLabel(self, text="", style="SmallError.TLabel",
                                        justify=tk.LEFT)
        self._sub_error.pack(fill=tk.X, pady=(0, SP_L))

        # ── Post-processing ──
        ttk.Label(self, text="Accuracy filter", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, SP_XS))
        self._confidence = _row(self, "Drop lines below confidence", self.confidence_threshold_var,
                                "0.50 is the default. Lower keeps more text,\n"
                                "higher removes more likely mis-hearings.", 0.0, 1.0, 0.05)
        WrappingLabel(self, text="Removes low-confidence lines, which are usually "
                                 "background noise mistaken for speech.",
                      style="Dim.TLabel", justify=tk.LEFT).pack(fill=tk.X, pady=(0, SP_L))

        # ── Hardware ──
        ttk.Label(self, text="Hardware", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, SP_XS))
        cb_cuda = ttk.Checkbutton(self, text="Use the CUDA installed on this PC",
                                  variable=self.use_system_cuda_var)
        cb_cuda.pack(anchor=tk.W, pady=(0, SP_XS))
        ToolTip(cb_cuda, "Use the NVIDIA CUDA Toolkit installed on your\n"
                         "system instead of the bundled CUDA libraries.")
        WrappingLabel(self, text="Only change this if you have a matching CUDA Toolkit "
                                 "installed. Takes effect after restarting Scrivox.",
                      style="Dim.TLabel", justify=tk.LEFT).pack(fill=tk.X)

        for var in (self.subtitle_max_chars_var, self.subtitle_max_duration_var,
                    self.subtitle_max_gap_var, self.subtitle_min_chars_var,
                    self.confidence_threshold_var):
            var.trace_add("write", lambda *a: self._validate())

    def _toggle(self):
        """Kept for compatibility: the settings are always visible now."""

    def problems(self):
        """Invalid values as (message, widget)."""
        errors = []

        def num(var, cast):
            try:
                return cast(var.get().strip())
            except (ValueError, AttributeError):
                return None

        max_chars = num(self.subtitle_max_chars_var, int)
        min_chars = num(self.subtitle_min_chars_var, int)
        max_dur = num(self.subtitle_max_duration_var, float)
        max_gap = num(self.subtitle_max_gap_var, float)
        conf = num(self.confidence_threshold_var, float)
        if max_chars is None or max_chars <= 0:
            errors.append(("Max characters per subtitle must be a whole number above 0",
                           self._max_chars))
        if min_chars is None or min_chars < 0:
            errors.append(("Min characters when splitting must be 0 or more", self._min_chars))
        elif max_chars and max_chars > 0 and min_chars > max_chars:
            errors.append(("Min characters can't be more than max characters", self._min_chars))
        if max_dur is None or max_dur <= 0:
            errors.append(("Max seconds on screen must be above 0", self._max_dur))
        if max_gap is None or max_gap < 0:
            errors.append(("Pause merging must be 0 or more seconds", self._max_gap))
        if conf is None or not 0.0 <= conf <= 1.0:
            errors.append(("Confidence must be between 0.0 and 1.0", self._confidence))
        return errors

    def _validate(self):
        errors = self.problems()
        bad = {id(w) for _, w in errors}
        for w in (self._max_chars, self._max_dur, self._max_gap, self._min_chars,
                  self._confidence):
            w.state(["invalid"] if id(w) in bad else ["!invalid"])
        self._sub_error.configure(text=errors[0][0] if errors else "")

    def get_diarization_model(self):
        return self.diarization_model_var.get().strip() or DEFAULT_DIARIZATION_MODEL

    def load_settings(self, settings):
        """Load settings from config dict."""
        self.diarization_model_var.set(
            settings.get("diarization_model", DEFAULT_DIARIZATION_MODEL))
        self.subtitle_max_chars_var.set(str(settings.get("subtitle_max_chars", 84)))
        self.subtitle_max_duration_var.set(str(settings.get("subtitle_max_duration", 4.0)))
        self.subtitle_max_gap_var.set(str(settings.get("subtitle_max_gap", 0.8)))
        self.subtitle_min_chars_var.set(str(settings.get("subtitle_min_chars", 15)))
        self.confidence_threshold_var.set(str(settings.get("confidence_threshold", 0.50)))
        self.use_system_cuda_var.set(settings.get("use_system_cuda", False))

    def get_settings_dict(self):
        """Return settings as a dict for config persistence."""
        return {
            "diarization_model": self.diarization_model_var.get().strip() or DEFAULT_DIARIZATION_MODEL,
            "subtitle_max_chars": self._parse_int(self.subtitle_max_chars_var.get(), 84),
            "subtitle_max_duration": self._parse_float(self.subtitle_max_duration_var.get(), 4.0),
            "subtitle_max_gap": self._parse_float(self.subtitle_max_gap_var.get(), 0.8),
            "subtitle_min_chars": self._parse_int(self.subtitle_min_chars_var.get(), 15),
            "confidence_threshold": self._parse_float(self.confidence_threshold_var.get(), 0.50),
            "use_system_cuda": self.use_system_cuda_var.get(),
        }

    def all_vars(self):
        return [self.diarization_model_var, self.subtitle_max_chars_var,
                self.subtitle_max_duration_var, self.subtitle_max_gap_var,
                self.subtitle_min_chars_var, self.confidence_threshold_var,
                self.use_system_cuda_var]

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
