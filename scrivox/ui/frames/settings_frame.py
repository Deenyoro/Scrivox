"""Model, language, feature toggles, speaker config, and sub-settings with validation."""

import tkinter as tk
from tkinter import ttk

from ...core.constants import WHISPER_MODELS, DEFAULT_VISION_MODEL, DEFAULT_SUMMARY_MODEL
from ...core.features import has_diarization, has_advanced_features
from ..theme import COLORS, FONTS


class ToolTip:
    """Simple tooltip that appears on hover."""

    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self._tipwindow = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self._tipwindow:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tipwindow = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self._text, justify=tk.LEFT,
                         bg=COLORS["bg_secondary"], fg=COLORS["fg"],
                         font=FONTS["small"], relief=tk.SOLID, borderwidth=1,
                         padx=6, pady=4)
        label.pack()

    def _hide(self, event=None):
        if self._tipwindow:
            self._tipwindow.destroy()
            self._tipwindow = None


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

        # Hardware
        self.use_system_cuda_var = tk.BooleanVar(value=False)

        # Diarization sub-settings
        self.num_speakers_var = tk.StringVar(value="")
        self.min_speakers_var = tk.StringVar(value="")
        self.max_speakers_var = tk.StringVar(value="")
        self.speaker_names_var = tk.StringVar(value="")
        self._speaker_mode_var = tk.StringVar(value="range")  # "exact" or "range"

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
                                    values=WHISPER_MODELS, state="normal", width=16)
        model_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))
        ToolTip(model_combo, "tiny/base: fast, lower quality\n"
                             "small/medium: balanced\n"
                             "large-v3: best quality, slower\n"
                             "Or enter a custom model name/path")

        row2 = ttk.Frame(model_frame)
        row2.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(row2, text="Language:").pack(side=tk.LEFT)
        lang_entry = ttk.Entry(row2, textvariable=self.language_var, width=16)
        lang_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))
        ToolTip(lang_entry, "Leave blank for auto-detect\n"
                            "Examples: en, ko, ja, de, fr")

        ttk.Label(model_frame, text="Presets or custom model name/path",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 2), anchor=tk.W)
        ttk.Label(model_frame, text="Primary language, blank for auto-detect (e.g. ko, ja, en)",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 6), anchor=tk.W)

        # ── HARDWARE ──
        hw_frame = ttk.LabelFrame(self, text="HARDWARE")
        hw_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        cb_cuda = ttk.Checkbutton(hw_frame, text="Use system CUDA instead of bundled",
                                   variable=self.use_system_cuda_var)
        cb_cuda.pack(padx=8, pady=(8, 2), anchor=tk.W)
        ToolTip(cb_cuda, "Use the NVIDIA CUDA Toolkit installed on your\n"
                         "system instead of the bundled CUDA libraries.\n"
                         "Enable this if you have a compatible CUDA\n"
                         "installation and want to use it for better\n"
                         "compatibility or newer driver features.")
        ttk.Label(hw_frame, text="Uses NVIDIA CUDA Toolkit from your system. Requires restart.",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 6), anchor=tk.W)

        # ── FEATURES ──
        features_frame = ttk.LabelFrame(self, text="FEATURES")
        features_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        if has_diarization():
            cb = ttk.Checkbutton(features_frame, text="Diarize (speaker labels)",
                                 variable=self.diarize_var,
                                 command=self._toggle_diarize)
            cb.pack(padx=8, pady=(8, 2), anchor=tk.W)
            ToolTip(cb, "Identify and label different speakers\nRequires HuggingFace token")

        if has_advanced_features():
            cb_vision = ttk.Checkbutton(features_frame, text="Vision (keyframe analysis)",
                                        variable=self.vision_var,
                                        command=self._toggle_vision)
            cb_vision.pack(padx=8, pady=2, anchor=tk.W)
            ToolTip(cb_vision, "Extract and analyze keyframes from video\nRequires LLM API key")

            cb_summary = ttk.Checkbutton(features_frame, text="Summarize (meeting summary)",
                                         variable=self.summarize_var,
                                         command=self._toggle_summary)
            cb_summary.pack(padx=8, pady=(2, 8), anchor=tk.W)
            ToolTip(cb_summary, "Generate meeting summary with key points\nRequires LLM API key")

        if not has_diarization():
            ttk.Label(features_frame,
                      text="Upgrade to Regular or Full build for diarization,\nvision, and summary features",
                      style="Dim.TLabel").pack(padx=8, pady=(8, 8), anchor=tk.W)

        # ── DIARIZATION SUB-SETTINGS ──
        self._diarize_frame = ttk.LabelFrame(self, text="DIARIZATION")
        # Initially hidden; shown when diarize is checked

        # Speaker mode radio buttons
        mode_frame = ttk.Frame(self._diarize_frame)
        mode_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Radiobutton(mode_frame, text="Range", variable=self._speaker_mode_var,
                        value="range", command=self._update_speaker_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Exact count", variable=self._speaker_mode_var,
                        value="exact", command=self._update_speaker_mode).pack(side=tk.LEFT, padx=(12, 0))

        # Range row
        self._range_frame = ttk.Frame(self._diarize_frame)
        self._range_frame.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(self._range_frame, text="Range:").pack(side=tk.LEFT)
        self._min_entry = ttk.Entry(self._range_frame, textvariable=self.min_speakers_var, width=4)
        self._min_entry.pack(side=tk.LEFT, padx=(8, 4))
        ttk.Label(self._range_frame, text="to", style="Dim.TLabel").pack(side=tk.LEFT)
        self._max_entry = ttk.Entry(self._range_frame, textvariable=self.max_speakers_var, width=4)
        self._max_entry.pack(side=tk.LEFT, padx=(4, 0))

        # Exact count row
        self._exact_frame = ttk.Frame(self._diarize_frame)
        ttk.Label(self._exact_frame, text="Speakers:").pack(side=tk.LEFT)
        self._num_entry = ttk.Entry(self._exact_frame, textvariable=self.num_speakers_var, width=6)
        self._num_entry.pack(side=tk.LEFT, padx=(8, 0))

        # Validation label
        self._diarize_validation = ttk.Label(self._diarize_frame, text="", style="Error.TLabel")
        self._diarize_validation.pack(padx=8, pady=(0, 2), anchor=tk.W)

        # Speaker names
        row = ttk.Frame(self._diarize_frame)
        row.pack(fill=tk.X, padx=8, pady=(2, 8))
        ttk.Label(row, text="Names:").pack(side=tk.LEFT)
        names_entry = ttk.Entry(row, textvariable=self.speaker_names_var)
        names_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))
        ToolTip(names_entry, "Comma-separated speaker names\ne.g. Alice,Bob,Charlie")
        ttk.Label(self._diarize_frame, text="Comma-separated: Alice,Bob,Charlie",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 6), anchor=tk.W)

        # Add validation traces
        self.min_speakers_var.trace_add("write", self._validate_speakers)
        self.max_speakers_var.trace_add("write", self._validate_speakers)
        self.num_speakers_var.trace_add("write", self._validate_speakers)

        # ── VISION SUB-SETTINGS ──
        self._vision_frame = ttk.LabelFrame(self, text="VISION")

        row = ttk.Frame(self._vision_frame)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row, text="Interval (s):").pack(side=tk.LEFT)
        interval_entry = ttk.Entry(row, textvariable=self.vision_interval_var, width=6)
        interval_entry.pack(side=tk.RIGHT)
        ToolTip(interval_entry, "Seconds between keyframe captures\nLower = more detail, higher API cost")

        row = ttk.Frame(self._vision_frame)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text="Model:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.vision_model_var).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        row = ttk.Frame(self._vision_frame)
        row.pack(fill=tk.X, padx=8, pady=(2, 4))
        ttk.Label(row, text="Workers:").pack(side=tk.LEFT)
        workers_entry = ttk.Entry(row, textvariable=self.vision_workers_var, width=6)
        workers_entry.pack(side=tk.RIGHT)
        ToolTip(workers_entry, "Number of concurrent API requests\nHigher = faster, more API load")

        # Vision validation
        self._vision_validation = ttk.Label(self._vision_frame, text="", style="Error.TLabel")
        self._vision_validation.pack(padx=8, pady=(0, 6), anchor=tk.W)

        self.vision_interval_var.trace_add("write", self._validate_vision)
        self.vision_workers_var.trace_add("write", self._validate_vision)

        # ── SUMMARY SUB-SETTINGS ──
        self._summary_frame = ttk.LabelFrame(self, text="SUMMARY")

        row = ttk.Frame(self._summary_frame)
        row.pack(fill=tk.X, padx=8, pady=(8, 8))
        ttk.Label(row, text="Model:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.summary_model_var).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        # Initialize speaker mode display
        self._update_speaker_mode()

    def _update_speaker_mode(self):
        """Show/hide exact vs range speaker controls based on radio selection."""
        if self._speaker_mode_var.get() == "exact":
            self._range_frame.pack_forget()
            self._exact_frame.pack(fill=tk.X, padx=8, pady=2,
                                    after=self._diarize_frame.winfo_children()[0])
            self.min_speakers_var.set("")
            self.max_speakers_var.set("")
        else:
            self._exact_frame.pack_forget()
            self._range_frame.pack(fill=tk.X, padx=8, pady=2,
                                    after=self._diarize_frame.winfo_children()[0])
            self.num_speakers_var.set("")

    def _validate_speakers(self, *args):
        """Real-time validation of speaker count fields."""
        errors = []
        num = self._parse_int(self.num_speakers_var.get())
        mins = self._parse_int(self.min_speakers_var.get())
        maxs = self._parse_int(self.max_speakers_var.get())

        if num is not None and num < 1:
            errors.append("Exact speakers must be >= 1")
        if mins is not None and mins < 1:
            errors.append("Min speakers must be >= 1")
        if maxs is not None and maxs < 1:
            errors.append("Max speakers must be >= 1")
        if mins is not None and maxs is not None and mins > maxs:
            errors.append("Min speakers cannot exceed max")

        self._diarize_validation.configure(text=errors[0] if errors else "")

    def _validate_vision(self, *args):
        """Real-time validation of vision fields."""
        errors = []
        interval = self._parse_int(self.vision_interval_var.get())
        workers = self._parse_int(self.vision_workers_var.get())

        if interval is not None and interval < 1:
            errors.append("Interval must be >= 1")
        if workers is not None and workers < 1:
            errors.append("Workers must be >= 1")

        self._vision_validation.configure(text=errors[0] if errors else "")

    def _parse_int(self, val):
        """Parse string as int, return None if empty or invalid."""
        val = val.strip()
        if not val:
            return None
        try:
            return int(val)
        except ValueError:
            return None

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
        self.use_system_cuda_var.set(settings.get("use_system_cuda", False))

        # Only load advanced feature states if they're available
        if has_diarization():
            self.diarize_var.set(settings.get("diarize", False))
        if has_advanced_features():
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

        # Set speaker mode based on loaded values
        if num:
            self._speaker_mode_var.set("exact")
        else:
            self._speaker_mode_var.set("range")
        self._update_speaker_mode()

        # Show/hide sub-frames
        self._toggle_diarize()
        self._toggle_vision()
        self._toggle_summary()

    def get_settings_dict(self):
        """Return current settings as a dict for config persistence."""
        return {
            "model": self.model_var.get(),
            "language": self.language_var.get(),
            "use_system_cuda": self.use_system_cuda_var.get(),
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
