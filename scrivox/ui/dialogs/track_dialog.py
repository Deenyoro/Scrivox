"""Audio track selection dialog for multi-track video files."""

import tkinter as tk
from tkinter import ttk

from .. import winnative
from ..theme import COLORS, SP_S, SP_XS, px


class TrackDialog(tk.Toplevel):
    """Modal dialog for selecting audio tracks from a multi-track video file.

    Returns list of selected track indices via self.result ([] = cancelled).
    """

    def __init__(self, parent, filename, tracks):
        """
        Args:
            parent: Parent window.
            filename: Display name of the file.
            tracks: List of track dicts from media.list_audio_tracks().
        """
        super().__init__(parent)
        self.withdraw()
        self.title(f"Audio Tracks: {filename}")
        self.transient(parent)
        self.resizable(False, False)

        self.result = []
        self._tracks = tracks
        self._check_vars = []

        self.configure(bg=COLORS["bg"])
        self._build(filename)
        self._center(parent)
        self.deiconify()
        winnative.use_dark_title_bar(self)
        self.grab_set()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        # "break" consumes the event so the app's global Escape-to-cancel
        # binding doesn't also fire and kill a running batch
        self.bind("<Escape>", self._on_escape)
        self.bind("<Return>", lambda e: self._on_ok())
        # Focus OK so Enter/Space confirm immediately
        self.after(10, self._ok_btn.focus_set)

    def _on_escape(self, event=None):
        self._on_cancel()
        return "break"

    def _build(self, filename):
        body = ttk.Frame(self, padding=(px(20), px(16), px(20), px(16)))
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="This video has several audio tracks",
                  style="CardTitle.TLabel").pack(anchor=tk.W)
        ttk.Label(body, text=f"Choose which ones to transcribe from {filename}. "
                             "Each track becomes its own transcript.",
                  style="Dim.TLabel", wraplength=int(420 * _scale(self)),
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(SP_XS, SP_S))

        tracks_frame = ttk.Frame(body)
        tracks_frame.pack(fill=tk.X, pady=(0, SP_S))
        for track in self._tracks:
            var = tk.BooleanVar(value=track.get("is_default", False))
            var.trace_add("write", lambda *a: self._update_ok_state())
            self._check_vars.append(var)
            ttk.Checkbutton(tracks_frame, text=self._format_track_label(track),
                            variable=var).pack(anchor=tk.W, pady=px(2))

        # Select-by-language shortcut: choices come from the tracks themselves
        langs = sorted({t.get("language", "") for t in self._tracks if t.get("language")})
        self._auto_lang_var = tk.BooleanVar(value=False)
        self._lang_var = tk.StringVar(value=langs[0] if langs else "en")
        if langs:
            lang_frame = ttk.Frame(body)
            lang_frame.pack(fill=tk.X, pady=(0, SP_S))
            ttk.Checkbutton(lang_frame, text="Select all tracks in",
                            variable=self._auto_lang_var,
                            command=self._on_auto_lang_toggle).pack(side=tk.LEFT)
            self._lang_entry = ttk.Combobox(lang_frame, textvariable=self._lang_var,
                                            values=langs, width=8, state="readonly")
            self._lang_entry.state(["disabled"])
            self._lang_entry.pack(side=tk.LEFT, padx=(SP_S, 0))
            self._lang_entry.bind("<<ComboboxSelected>>", self._on_lang_typed)
        else:
            self._lang_entry = None

        self._none_hint = ttk.Label(body, text="", style="Warning.TLabel")
        self._none_hint.pack(anchor=tk.W)

        # Buttons: [Add selected] [Cancel], right-aligned (Windows order)
        btn_frame = ttk.Frame(body)
        btn_frame.pack(fill=tk.X, pady=(SP_S, 0))
        cancel = ttk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        cancel.pack(side=tk.RIGHT)
        self._ok_btn = ttk.Button(btn_frame, text="Add selected", style="Accent.TButton",
                                  command=self._on_ok)
        self._ok_btn.pack(side=tk.RIGHT, padx=(0, SP_S))
        self._update_ok_state()

    def _format_track_label(self, track):
        """Format a human-readable track label."""
        parts = [f"Track {track['index']}: {track['codec'].upper()}"]
        if track.get("language"):
            parts.append(track["language"].capitalize())
        ch = track.get("channels", 0)
        if ch == 1:
            parts.append("mono")
        elif ch == 2:
            parts.append("stereo")
        elif ch == 6:
            parts.append("5.1")
        elif ch > 0:
            parts.append(f"{ch}ch")
        if track.get("title"):
            parts.append(track["title"])
        if track.get("is_default"):
            parts.append("(default)")
        return ", ".join(parts)

    def _update_ok_state(self):
        any_selected = any(v.get() for v in self._check_vars)
        self._ok_btn.state(["!disabled"] if any_selected else ["disabled"])
        self._none_hint.configure(text="" if any_selected else "Tick at least one track.")

    def _on_auto_lang_toggle(self):
        if self._lang_entry is None:
            return
        if self._auto_lang_var.get():
            self._lang_entry.state(["!disabled"])
            self._apply_lang_filter()
        else:
            self._lang_entry.state(["disabled"])

    def _on_lang_typed(self, event=None):
        if self._auto_lang_var.get():
            self._apply_lang_filter()

    def _apply_lang_filter(self):
        """Select tracks matching the language filter."""
        lang = self._lang_var.get().strip().lower()
        if not lang:
            return
        for i, track in enumerate(self._tracks):
            track_lang = track.get("language", "").lower()
            self._check_vars[i].set(track_lang == lang or track_lang.startswith(lang))

    def _on_ok(self):
        if self._auto_lang_var.get():
            self._apply_lang_filter()
        self.result = [i for i, var in enumerate(self._check_vars) if var.get()]
        if not self.result:
            self.bell()  # nothing ticked: keep the dialog open
            return
        self.destroy()

    def _on_cancel(self):
        self.result = []
        self.destroy()

    def _center(self, parent):
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")


def _scale(widget):
    try:
        return widget.winfo_fpixels("1i") / 96.0
    except tk.TclError:
        return 1.0
