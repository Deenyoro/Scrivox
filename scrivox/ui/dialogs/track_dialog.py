"""Audio track selection dialog for multi-track video files."""

import tkinter as tk
from tkinter import ttk

from ..theme import COLORS, FONTS


class TrackDialog(tk.Toplevel):
    """Modal dialog for selecting audio tracks from a multi-track video file.

    Returns list of selected track indices via self.result.
    """

    def __init__(self, parent, filename, tracks):
        """
        Args:
            parent: Parent window.
            filename: Display name of the file.
            tracks: List of track dicts from media.list_audio_tracks().
        """
        super().__init__(parent)
        self.title(f"Audio Tracks: {filename}")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        self.result = []
        self._tracks = tracks
        self._check_vars = []

        self.configure(bg=COLORS["bg"])
        self._build(filename)
        self._center(parent)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda e: self._on_cancel())

    def _build(self, filename):
        # Header
        header = ttk.Label(self, text=f"Select audio tracks from: {filename}",
                           style="Header.TLabel")
        header.pack(padx=16, pady=(12, 8), anchor=tk.W)

        # Track checkboxes
        tracks_frame = ttk.Frame(self)
        tracks_frame.pack(fill=tk.X, padx=16, pady=(0, 8))

        for track in self._tracks:
            var = tk.BooleanVar(value=track.get("is_default", False))
            self._check_vars.append(var)

            label = self._format_track_label(track)
            cb = ttk.Checkbutton(tracks_frame, text=label, variable=var)
            cb.pack(anchor=tk.W, pady=2)

        # Auto-select by language
        lang_frame = ttk.Frame(self)
        lang_frame.pack(fill=tk.X, padx=16, pady=(0, 8))

        self._auto_lang_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lang_frame, text="Auto-select by language:",
                        variable=self._auto_lang_var,
                        command=self._on_auto_lang_toggle).pack(side=tk.LEFT)

        self._lang_entry = ttk.Entry(lang_frame, width=8)
        self._lang_entry.insert(0, "en")
        self._lang_entry.configure(state=tk.DISABLED)
        self._lang_entry.pack(side=tk.LEFT, padx=(8, 0))

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=16, pady=(0, 12))

        ttk.Button(btn_frame, text="Add Selected", style="Accent.TButton",
                   command=self._on_ok).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btn_frame, text="Cancel",
                   command=self._on_cancel).pack(side=tk.RIGHT)

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

    def _on_auto_lang_toggle(self):
        if self._auto_lang_var.get():
            self._lang_entry.configure(state=tk.NORMAL)
            self._apply_lang_filter()
        else:
            self._lang_entry.configure(state=tk.DISABLED)

    def _apply_lang_filter(self):
        """Select tracks matching the language filter."""
        lang = self._lang_entry.get().strip().lower()
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
            # Default to first track if nothing selected
            self.result = [0]
        self.destroy()

    def _on_cancel(self):
        self.result = []
        self.destroy()

    def _center(self, parent):
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"+{x}+{y}")
