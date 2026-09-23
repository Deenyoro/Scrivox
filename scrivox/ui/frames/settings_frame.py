"""Step 2 "Options": model, language, and the optional extras (speakers,
on-screen content, summary, translation), each with its options shown right
under its checkbox."""

import tkinter as tk
from tkinter import ttk

from ...core.constants import (
    DEFAULT_TRANSLATION_MODEL, DEFAULT_VISION_MODEL, DEFAULT_SUMMARY_MODEL,
    LLM_MODEL_PRESETS, VISION_MODEL_PRESETS,
    TRANSLATION_LANGUAGES, WHISPER_LANGUAGES, WHISPER_MODELS,
)
from ...core.features import has_diarization, has_advanced_features
from ..output_paths import describe_model, model_label
from ..theme import SP_S, SP_XS, px
# ToolTip is also imported from here by older modules
from ..widgets import AutocompleteCombobox, LinkLabel, ToolTip, WrappingLabel

AUTO_DETECT = "Auto-detect"

# Display values for the language dropdown: ["Auto-detect", "Afrikaans (af)", ...]
_LANGUAGE_DISPLAY_VALUES = [AUTO_DETECT] + [f"{name} ({code})" for name, code in WHISPER_LANGUAGES.items()]


def _extract_language_code(display_str):
    """Extract language code from display string like 'Arabic (ar)' -> 'ar'.

    Also accepts raw codes like 'ar' or 'en' directly. Blank and
    "Auto-detect" both mean auto-detection ("").
    """
    display_str = display_str.strip()
    if not display_str or display_str.lower() == AUTO_DETECT.lower():
        return ""
    # Try to extract from "Name (code)" format
    if "(" in display_str and display_str.endswith(")"):
        code = display_str.rsplit("(", 1)[1].rstrip(")")
        return code.strip()
    # Raw code (e.g. "en", "ar")
    return display_str


def _field_row(parent, label, pady=(0, SP_XS)):
    """A label + control row. Returns the frame to put the control in."""
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, pady=pady)
    ttk.Label(row, text=label).pack(side=tk.LEFT)
    return row


class SettingsFrame(ttk.Frame):
    """Model/language combos, extras checkboxes, and their inline options."""

    def __init__(self, parent, on_setup_keys=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_setup_keys = on_setup_keys or (lambda tab=None: None)

        # ── Variables (names and meanings unchanged; persisted in config) ──
        self.model_var = tk.StringVar(value="large-v3")
        self.language_var = tk.StringVar(value=AUTO_DETECT)
        self.diarize_var = tk.BooleanVar(value=False)
        self.vision_var = tk.BooleanVar(value=False)
        self.summarize_var = tk.BooleanVar(value=False)

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
        self.vision_change_threshold_var = tk.StringVar(value="0")

        # Summary sub-settings
        self.summary_model_var = tk.StringVar(value=DEFAULT_SUMMARY_MODEL)

        # Translation sub-settings
        self.translate_var = tk.BooleanVar(value=False)
        self.translate_all_var = tk.BooleanVar(value=False)
        self.translate_to_var = tk.StringVar(value="")
        self.translation_model_var = tk.StringVar(value=DEFAULT_TRANSLATION_MODEL)

        self._key_links = {}
        self._missing_keys = set()
        self._loading = False
        self._extras_open = False
        self._build()

    # ── Layout ──

    def _build(self):
        # Transcription basics
        row = _field_row(self, "Model")
        # The list compares models (speed/accuracy/download size); the field
        # and the saved setting keep the plain model name
        labels = {model_label(m): m for m in WHISPER_MODELS}
        self._model_combo = AutocompleteCombobox(row, textvariable=self.model_var,
                                                 values=list(labels), display_map=labels,
                                                 state="normal", width=16,
                                                 style="Wide.TCombobox")
        self._model_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        ToolTip(self._model_combo, "Bigger models are more accurate but slower.\n"
                                   "large-v3-turbo is a good everyday choice.\n"
                                   "You can also type a custom model name or folder.")
        self._model_hint = WrappingLabel(self, text="", style="Dim.TLabel", justify=tk.LEFT)
        self._model_hint.pack(fill=tk.X, pady=(0, SP_S))
        self.model_var.trace_add("write", lambda *a: self._update_model_hint())
        self._update_model_hint()

        row = _field_row(self, "Language", pady=(0, SP_S))
        self._lang_combo = AutocompleteCombobox(row, textvariable=self.language_var,
                                                values=_LANGUAGE_DISPLAY_VALUES, state="normal",
                                                width=20)
        self._lang_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        ToolTip(self._lang_combo, "Auto-detect works for most recordings.\n"
                                  "Pick the language (or type a code like 'en')\n"
                                  "if detection guesses wrong.")
        self._lang_combo.bind("<FocusOut>", self._normalize_language, add="+")

        # ── Extras: one summary row until opened, so steps 1-3 fit ──
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(SP_XS, px(2)))
        head = ttk.Frame(self)
        head.pack(fill=tk.X)
        self._extras_btn = ttk.Button(head, text="", style="Disclosure.TButton", width=0,
                                      command=self.toggle_extras)
        self._extras_btn.pack(side=tk.LEFT)
        self._extras_summary = ttk.Label(head, text="", style="Dim.TLabel", cursor="hand2")
        self._extras_summary.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SP_XS, 0))
        self._extras_summary.bind("<Button-1>", lambda e: self.toggle_extras())
        ToolTip(self._extras_btn, "Speaker names, on-screen content, summary and translation")

        self._extras = ttk.Frame(self, padding=(0, SP_XS, 0, 0))

        self._diarize_cb = self._vision_cb = self._summary_cb = self._translate_cb = None
        if has_diarization():
            self._diarize_cb = ttk.Checkbutton(self._extras, text="Identify speakers",
                                               variable=self.diarize_var,
                                               command=self._toggle_diarize)
            self._diarize_cb.pack(anchor=tk.W, pady=px(2))
            ToolTip(self._diarize_cb, "Label who said what (Speaker 1, Speaker 2, ...).\n"
                                      "Uses a free Hugging Face token.")
        self._diarize_frame = self._build_diarize_options()

        if has_advanced_features():
            self._vision_cb = ttk.Checkbutton(self._extras,
                                              text="Describe on-screen content (video)",
                                              variable=self.vision_var,
                                              command=self._toggle_vision)
            self._vision_cb.pack(anchor=tk.W, pady=px(2))
            ToolTip(self._vision_cb, "Capture frames from the video and describe slides,\n"
                                     "screens and scenes with an AI service.")
        self._vision_frame = self._build_vision_options()

        if has_advanced_features():
            self._summary_cb = ttk.Checkbutton(self._extras, text="Summarize",
                                               variable=self.summarize_var,
                                               command=self._toggle_summary)
            self._summary_cb.pack(anchor=tk.W, pady=px(2))
            ToolTip(self._summary_cb, "Add a meeting summary with key points and\n"
                                      "action items, written by an AI service.")
        self._summary_frame = self._build_summary_options()

        if has_advanced_features():
            self._translate_cb = ttk.Checkbutton(self._extras, text="Translate",
                                                 variable=self.translate_var,
                                                 command=self._toggle_translate)
            self._translate_cb.pack(anchor=tk.W, pady=px(2))
            ToolTip(self._translate_cb, "Also save a translated copy in one or more\n"
                                        "languages, using an AI service.")
        self._translate_frame = self._build_translate_options()

        if not has_diarization():
            WrappingLabel(self._extras,
                          text="Speaker labels, summaries, translation and on-screen "
                               "descriptions are in the Regular and Full downloads.",
                          style="Dim.TLabel", justify=tk.LEFT).pack(fill=tk.X, pady=(0, SP_XS))

        # Initialize speaker mode display
        self._update_speaker_mode()
        for var in (self.diarize_var, self.vision_var, self.summarize_var, self.translate_var):
            var.trace_add("write", lambda *a: self._update_extras_summary())
        self._update_extras_summary()

    def _sub_frame(self):
        """Indented options block that appears under its checkbox."""
        return ttk.Frame(self._extras, padding=(px(26), 0, 0, px(6)))

    def _key_link(self, parent, key, text, tab):
        row = ttk.Frame(parent)
        ttk.Label(row, text=text, style="Warning.TLabel").pack(side=tk.LEFT)
        LinkLabel(row, text="Set up…", command=lambda: self._on_setup_keys(tab)).pack(
            side=tk.LEFT, padx=(SP_XS, 0))
        self._key_links[key] = row
        return row

    def _build_diarize_options(self):
        frame = self._sub_frame()
        self._key_link(frame, "hf", "Needs a Hugging Face token.", "keys")

        mode = ttk.Frame(frame)
        mode.pack(fill=tk.X, pady=(0, SP_XS))
        self._mode_frame = mode
        ttk.Label(mode, text="Speakers").pack(side=tk.LEFT, padx=(0, SP_S))
        ttk.Radiobutton(mode, text="Detect", variable=self._speaker_mode_var,
                        value="range", command=self._update_speaker_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode, text="Exactly", variable=self._speaker_mode_var,
                        value="exact", command=self._update_speaker_mode).pack(
            side=tk.LEFT, padx=(SP_S, 0))
        self._num_entry = ttk.Spinbox(mode, textvariable=self.num_speakers_var,
                                      from_=1, to=50, width=4)

        self._range_frame = ttk.Frame(frame)
        ttk.Label(self._range_frame, text="Between", style="Dim.TLabel").pack(side=tk.LEFT)
        self._min_entry = ttk.Spinbox(self._range_frame, textvariable=self.min_speakers_var,
                                      from_=1, to=50, width=4)
        self._min_entry.pack(side=tk.LEFT, padx=(SP_XS, SP_XS))
        ttk.Label(self._range_frame, text="and", style="Dim.TLabel").pack(side=tk.LEFT)
        self._max_entry = ttk.Spinbox(self._range_frame, textvariable=self.max_speakers_var,
                                      from_=1, to=50, width=4)
        self._max_entry.pack(side=tk.LEFT, padx=(SP_XS, SP_XS))
        ttk.Label(self._range_frame, text="(optional)", style="Dim.TLabel").pack(side=tk.LEFT)
        self._exact_frame = ttk.Frame(frame)  # kept for layout compatibility

        self._diarize_validation = WrappingLabel(frame, text="", style="SmallError.TLabel",
                                                 justify=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(SP_XS, 0))
        self._names_row = row
        ttk.Label(row, text="Names").pack(side=tk.LEFT)
        names_entry = ttk.Entry(row, textvariable=self.speaker_names_var)
        names_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        ToolTip(names_entry, "Optional. Replace Speaker 1, Speaker 2... with names,\n"
                             "in order of first appearance: Alice, Bob, Charlie")

        for var in (self.min_speakers_var, self.max_speakers_var, self.num_speakers_var):
            var.trace_add("write", self._validate_speakers)
        return frame

    def _build_vision_options(self):
        frame = self._sub_frame()
        self._key_link(frame, "llm_vision", "Needs an AI service key.", "keys")

        row = _field_row(frame, "Capture a frame every")
        ttk.Label(row, text="seconds", style="Dim.TLabel").pack(side=tk.RIGHT)
        interval_entry = ttk.Spinbox(row, textvariable=self.vision_interval_var,
                                     from_=0.5, to=3600, increment=5, width=6)
        interval_entry.pack(side=tk.RIGHT, padx=(SP_S, SP_XS))
        self._interval_entry = interval_entry
        ToolTip(interval_entry, "Lower = more detail and higher AI cost.\n"
                                "Fractions are allowed (e.g. 0.5).")

        row = _field_row(frame, "AI model")
        AutocompleteCombobox(row, textvariable=self.vision_model_var,
                             values=VISION_MODEL_PRESETS, state="normal").pack(
            side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))

        row = _field_row(frame, "Parallel requests")
        workers_entry = ttk.Spinbox(row, textvariable=self.vision_workers_var,
                                    from_=1, to=32, width=6)
        workers_entry.pack(side=tk.RIGHT)
        self._workers_entry = workers_entry
        ToolTip(workers_entry, "How many frames are described at once.\n"
                               "Higher is faster but may hit the service's rate limit.")

        row = _field_row(frame, "Skip near-duplicate frames")
        change_entry = ttk.Spinbox(row, textvariable=self.vision_change_threshold_var,
                                   from_=0, to=64, width=6)
        change_entry.pack(side=tk.RIGHT)
        self._change_entry = change_entry
        ToolTip(change_entry, "0 = off. 2 skips only near-identical frames,\n"
                              "5 skips more aggressively (range 0-64).")

        self._vision_validation = WrappingLabel(frame, text="", style="SmallError.TLabel",
                                                justify=tk.LEFT)

        for var in (self.vision_interval_var, self.vision_workers_var,
                    self.vision_change_threshold_var):
            var.trace_add("write", self._validate_vision)
        return frame

    def _build_summary_options(self):
        frame = self._sub_frame()
        self._key_link(frame, "llm_summary", "Needs an AI service key.", "keys")
        row = _field_row(frame, "AI model", pady=0)
        AutocompleteCombobox(row, textvariable=self.summary_model_var,
                             values=LLM_MODEL_PRESETS, state="normal").pack(
            side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        return frame

    def _build_translate_options(self):
        frame = self._sub_frame()
        self._key_link(frame, "llm_translate", "Needs an AI service key.", "keys")

        row = _field_row(frame, "Translate to")
        translate_langs = [f"{name} ({code})" for name, code in TRANSLATION_LANGUAGES.items()]
        self._translate_to_combo = AutocompleteCombobox(
            row, textvariable=self.translate_to_var,
            values=translate_langs, state="normal", width=20, multi_value=True)
        self._translate_to_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        ToolTip(self._translate_to_combo, "One or more languages, separated by commas:\n"
                                          "Arabic (ar), French (fr)")

        cb_all = ttk.Checkbutton(frame, text="Also translate summary and headings",
                                 variable=self.translate_all_var)
        cb_all.pack(anchor=tk.W, pady=(0, SP_XS))
        ToolTip(cb_all, "Translate the summary, on-screen descriptions and\n"
                        "document headings too, not just the transcript")

        row = _field_row(frame, "AI model")
        AutocompleteCombobox(row, textvariable=self.translation_model_var,
                             values=LLM_MODEL_PRESETS, state="normal").pack(
            side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        WrappingLabel(frame, text="Saves one extra file per language, e.g. talk.fr.srt",
                      style="Dim.TLabel", justify=tk.LEFT).pack(fill=tk.X)
        return frame

    # ── Behaviour ──

    def _update_model_hint(self):
        self._model_hint.configure(text=describe_model(self.model_var.get().strip()))

    def _normalize_language(self, event=None):
        """Blank means auto-detect: show it as such."""
        if not self.language_var.get().strip():
            self.language_var.set(AUTO_DETECT)

    def _update_speaker_mode(self):
        """Show the exact-count box or the optional range, per the radio."""
        if self._speaker_mode_var.get() == "exact":
            self._range_frame.pack_forget()
            self._num_entry.pack(side=tk.LEFT, padx=(SP_S, 0))
            self.min_speakers_var.set("")
            self.max_speakers_var.set("")
        else:
            self._num_entry.pack_forget()
            self._range_frame.pack(fill=tk.X, pady=(0, SP_XS), after=self._mode_frame)
            self.num_speakers_var.set("")

    def speaker_errors(self):
        errors = []
        num = self._parse_int(self.num_speakers_var.get())
        mins = self._parse_int(self.min_speakers_var.get())
        maxs = self._parse_int(self.max_speakers_var.get())
        for label, raw, val in (("Speaker count", self.num_speakers_var.get(), num),
                                ("Minimum speakers", self.min_speakers_var.get(), mins),
                                ("Maximum speakers", self.max_speakers_var.get(), maxs)):
            if raw.strip() and val is None:
                errors.append((f"{label} must be a whole number", label))
            elif val is not None and val < 1:
                errors.append((f"{label} must be at least 1", label))
        if mins is not None and maxs is not None and mins > maxs:
            errors.append(("Minimum speakers can't be more than the maximum", "Minimum speakers"))
        return errors

    def _validate_speakers(self, *args):
        """Real-time validation of speaker count fields."""
        errors = self.speaker_errors()
        self._diarize_validation.configure(text=errors[0][0] if errors else "")
        if errors:
            self._diarize_validation.pack(fill=tk.X, after=self._range_frame
                                          if self._range_frame.winfo_manager() else self._mode_frame)
        else:
            self._diarize_validation.pack_forget()
        for entry, label in ((self._num_entry, "Speaker count"),
                             (self._min_entry, "Minimum speakers"),
                             (self._max_entry, "Maximum speakers")):
            bad = any(lbl == label for _, lbl in errors)
            entry.state(["invalid"] if bad else ["!invalid"])

    def vision_errors(self):
        errors = []
        raw_interval = self.vision_interval_var.get()
        interval = self._parse_float(raw_interval)
        workers = self._parse_int(self.vision_workers_var.get())
        threshold = self._parse_int(self.vision_change_threshold_var.get())
        if (raw_interval.strip() and interval is None) or (interval is not None and interval <= 0):
            errors.append(("Frame interval must be a number above 0", self._interval_entry))
        if (self.vision_workers_var.get().strip() and workers is None) or (
                workers is not None and workers < 1):
            errors.append(("Parallel requests must be at least 1", self._workers_entry))
        if (self.vision_change_threshold_var.get().strip() and threshold is None) or (
                threshold is not None and not 0 <= threshold <= 64):
            errors.append(("Skip near-duplicates must be between 0 and 64", self._change_entry))
        return errors

    def _validate_vision(self, *args):
        """Real-time validation of vision fields."""
        errors = self.vision_errors()
        self._vision_validation.configure(text=errors[0][0] if errors else "")
        if errors and not self._vision_validation.winfo_manager():
            self._vision_validation.pack(fill=tk.X)
        elif not errors:
            self._vision_validation.pack_forget()
        bad = {id(w) for _, w in errors}
        for entry in (self._interval_entry, self._workers_entry, self._change_entry):
            entry.state(["invalid"] if id(entry) in bad else ["!invalid"])

    def set_key_hints(self, missing):
        """Show "Needs a key - Set up..." under extras whose key is missing.

        `missing` is a set drawn from {"hf", "llm"}.
        """
        self._missing_keys = set(missing)
        self._update_extras_summary()
        for key, row in self._key_links.items():
            need = ("hf" in missing) if key == "hf" else ("llm" in missing)
            if need and not row.winfo_manager():
                row.pack(anchor=tk.W, pady=(0, SP_XS), before=row.master.winfo_children()[1]
                         if len(row.master.winfo_children()) > 1 else None)
            elif not need and row.winfo_manager():
                row.pack_forget()

    def problems(self, missing_keys=()):
        """Everything that would stop a run, as (message, widget_to_focus)."""
        out = []
        if self.diarize_var.get():
            if "hf" in missing_keys:
                out.append(("Identifying speakers needs a Hugging Face token", "keys"))
            for msg, label in self.speaker_errors():
                widget = {"Speaker count": self._num_entry, "Minimum speakers": self._min_entry,
                          "Maximum speakers": self._max_entry}[label]
                out.append((msg, widget))
        uses_llm = self.vision_var.get() or self.summarize_var.get() or self.translate_var.get()
        if uses_llm and "llm" in missing_keys:
            out.append(("The selected extras need an AI service key", "keys"))
        if self.vision_var.get():
            out.extend(self.vision_errors())
        if self.translate_var.get() and not self.get_translate_to_codes():
            out.append(("Choose a language to translate to", self._translate_to_combo))
        return out

    def _parse_int(self, val):
        """Parse string as int, return None if empty or invalid."""
        val = val.strip()
        if not val:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    def _parse_float(self, val):
        """Parse string as float, return None if empty or invalid."""
        val = val.strip()
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def _show_sub(self, frame, checkbox, visible):
        if visible and checkbox is not None:
            frame.pack(fill=tk.X, after=checkbox)
            # Switching an extra on (not just restoring saved settings)
            # shows its options
            if not self._loading:
                self.set_extras_open(True)
        else:
            frame.pack_forget()

    # ── Extras disclosure ──

    @property
    def extras_open(self):
        return self._extras_open

    def toggle_extras(self):
        self.set_extras_open(not self._extras_open)

    def set_extras_open(self, is_open):
        is_open = bool(is_open)
        changed = is_open != self._extras_open
        self._extras_open = is_open
        if is_open and not self._extras.winfo_manager():
            self._extras.pack(fill=tk.X)
        elif not is_open and self._extras.winfo_manager():
            self._extras.pack_forget()
        self._update_extras_summary()
        if changed:
            self.event_generate("<<ExtrasToggled>>")

    def reveal(self, widget):
        """Open the extras section if `widget` lives inside it."""
        w = widget
        while w is not None:
            if w is self._extras:
                self.set_extras_open(True)
                return
            w = getattr(w, "master", None)

    def _update_extras_summary(self):
        if not hasattr(self, "_extras_btn"):
            return
        arrow = "\u25be" if self._extras_open else "\u25b8"
        self._extras_btn.configure(text=f"{arrow}  Extras")
        if not has_diarization():
            self._extras_summary.configure(text="not included in Lite", style="Dim.TLabel")
            return
        if self._extras_open:
            # Open: the ticked boxes speak for themselves
            self._extras_summary.configure(text="", style="Dim.TLabel")
            return
        on = []
        needs_key = False
        if self.diarize_var.get():
            on.append("speakers")
            needs_key |= "hf" in self._missing_keys
        llm_missing = "llm" in self._missing_keys
        for var, name in ((self.vision_var, "on-screen content"),
                          (self.summarize_var, "summary"), (self.translate_var, "translation")):
            if has_advanced_features() and var.get():
                on.append(name)
                needs_key |= llm_missing
        if on:
            text = ", ".join(on)
            text = text[0].upper() + text[1:]
            if needs_key:
                text += "  \u00b7  needs a key"
            style = "Warning.TLabel" if needs_key else "TLabel"
        else:
            text = ("speakers, summary, translation\u2026" if has_advanced_features()
                    else "identify speakers\u2026")
            style = "Dim.TLabel"
        self._extras_summary.configure(text=text, style=style)

    def _toggle_diarize(self):
        self._show_sub(self._diarize_frame, self._diarize_cb, self.diarize_var.get())
        self.event_generate("<<ExtrasChanged>>")

    def _toggle_vision(self):
        self._show_sub(self._vision_frame, self._vision_cb, self.vision_var.get())
        self.event_generate("<<ExtrasChanged>>")

    def _toggle_summary(self):
        self._show_sub(self._summary_frame, self._summary_cb, self.summarize_var.get())
        self.event_generate("<<ExtrasChanged>>")

    def _toggle_translate(self):
        self._show_sub(self._translate_frame, self._translate_cb, self.translate_var.get())
        self.event_generate("<<ExtrasChanged>>")

    def _find_features_frame(self):
        """Kept for compatibility: the extras container."""
        return self._extras

    def get_language_code(self):
        """Extract language code from the language combobox display value."""
        return _extract_language_code(self.language_var.get()) or None

    def get_translate_to_codes(self):
        """Extract language code(s) from the translate-to combobox display value.

        Supports comma-separated entries like 'Arabic (ar), French (fr)' or 'ar,fr'.
        Returns a comma-separated string of codes like 'ar,fr', or None if empty.
        """
        raw = self.translate_to_var.get().strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",")]
        codes = []
        for part in parts:
            code = _extract_language_code(part)
            if code:
                codes.append(code)
        return ",".join(codes) if codes else None

    def get_speaker_names(self):
        """Parse and return speaker names list, or None."""
        raw = self.speaker_names_var.get().strip()
        if not raw:
            return None
        return [n.strip() for n in raw.split(",") if n.strip()]

    @staticmethod
    def _safe_int(value, default):
        """Parse a string as int, returning default on failure."""
        try:
            return int(value) if value else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(value, default):
        """Parse a string as float, returning default on failure."""
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default

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

        # Convert raw language code to display format if needed
        lang_val = settings.get("language", "") or ""
        if lang_val and "(" not in lang_val and lang_val.lower() != AUTO_DETECT.lower():
            # Raw code like "en" -> "English (en)"
            from ...core.constants import LANGUAGE_CODE_TO_NAME
            name = LANGUAGE_CODE_TO_NAME.get(lang_val)
            if name:
                lang_val = f"{name} ({lang_val})"
        self.language_var.set(lang_val or AUTO_DETECT)

        # Only load advanced feature states if they're available
        if has_diarization():
            self.diarize_var.set(settings.get("diarize", False))
        if has_advanced_features():
            self.vision_var.set(settings.get("vision", False))
            self.summarize_var.set(settings.get("summarize", False))
            self.translate_var.set(settings.get("translate", False))
            self.translate_all_var.set(settings.get("translate_all", False))

        self.speaker_names_var.set(settings.get("speaker_names", ""))
        interval = settings.get("vision_interval", 60)
        if isinstance(interval, float) and interval == int(interval):
            interval = int(interval)
        self.vision_interval_var.set(str(interval))
        self.vision_model_var.set(settings.get("vision_model", DEFAULT_VISION_MODEL))
        self.vision_workers_var.set(str(settings.get("vision_workers", 4)))
        self.vision_change_threshold_var.set(str(settings.get("vision_change_threshold", 0)))
        self.summary_model_var.set(settings.get("summary_model", DEFAULT_SUMMARY_MODEL))

        # Translation settings — handle comma-separated codes like "ar,fr"
        translate_to_val = settings.get("translate_to", "")
        if translate_to_val and "(" not in translate_to_val:
            from ...core.constants import TRANSLATION_CODE_TO_NAME
            parts = [p.strip() for p in translate_to_val.split(",") if p.strip()]
            display_parts = []
            for code in parts:
                name = TRANSLATION_CODE_TO_NAME.get(code)
                display_parts.append(f"{name} ({code})" if name else code)
            translate_to_val = ", ".join(display_parts)
        self.translate_to_var.set(translate_to_val)
        self.translation_model_var.set(
            settings.get("translation_model", DEFAULT_TRANSLATION_MODEL))

        num = settings.get("num_speakers")
        mins = settings.get("min_speakers")
        maxs = settings.get("max_speakers")

        # Set speaker mode based on loaded values (switching mode clears the
        # other mode's fields, so do it before filling them in)
        self._speaker_mode_var.set("exact" if num else "range")
        self._update_speaker_mode()
        self.num_speakers_var.set(str(num) if num else "")
        self.min_speakers_var.set(str(mins) if mins else "")
        self.max_speakers_var.set(str(maxs) if maxs else "")

        # Show/hide sub-frames (without popping the extras section open)
        self._loading = True
        try:
            self._toggle_diarize()
            self._toggle_vision()
            self._toggle_summary()
            self._toggle_translate()
        finally:
            self._loading = False

    def get_settings_dict(self):
        """Return current settings as a dict for config persistence."""
        return {
            "model": self.model_var.get(),
            # "Auto-detect" is display-only; config keeps "" as before
            "language": "" if not _extract_language_code(self.language_var.get())
            else self.language_var.get(),
            "diarize": self.diarize_var.get(),
            "vision": self.vision_var.get(),
            "summarize": self.summarize_var.get(),
            "translate": self.translate_var.get(),
            "translate_all": self.translate_all_var.get(),
            "translate_to": self.translate_to_var.get(),
            "translation_model": self.translation_model_var.get(),
            "num_speakers": self.get_int_or_none(self.num_speakers_var),
            "min_speakers": self.get_int_or_none(self.min_speakers_var),
            "max_speakers": self.get_int_or_none(self.max_speakers_var),
            "speaker_names": self.speaker_names_var.get(),
            "vision_interval": self._safe_float(self.vision_interval_var.get(), 60.0),
            "vision_model": self.vision_model_var.get(),
            "vision_workers": self._safe_int(self.vision_workers_var.get(), 4),
            "vision_change_threshold": self._safe_int(self.vision_change_threshold_var.get(), 0),
            "summary_model": self.summary_model_var.get(),
        }

    def all_vars(self):
        """Every persisted variable (used to auto-save settings on change)."""
        return [self.model_var, self.language_var, self.diarize_var, self.vision_var,
                self.summarize_var, self.num_speakers_var, self.min_speakers_var,
                self.max_speakers_var, self.speaker_names_var, self.vision_interval_var,
                self.vision_model_var, self.vision_workers_var,
                self.vision_change_threshold_var, self.summary_model_var,
                self.translate_var, self.translate_all_var, self.translate_to_var,
                self.translation_model_var]
