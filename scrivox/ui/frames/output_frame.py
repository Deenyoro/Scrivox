"""Step 3 "Save to": output format and destination folder."""

import os
import tkinter as tk
from tkinter import ttk, filedialog

from ...core.constants import OUTPUT_FORMATS
from ..output_paths import FORMAT_DESCRIPTIONS
from ..theme import SP_S, SP_XS, px
from ..widgets import ToolTip, WrappingLabel

SAME_FOLDER = "Same folder as each file"


class OutputFrame(ttk.Frame):
    """Format combo and output folder picker.

    `output_path_var` (an explicit output *file*) is still honoured when set,
    e.g. by older code paths; the visible control picks a *folder*
    (`output_dir_var`), which also works for batches.
    """

    def __init__(self, parent, config_manager=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.config_manager = config_manager

        self.format_var = tk.StringVar(value="txt")
        self.output_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.subtitle_speakers_var = tk.BooleanVar(value=False)

        self._build()
        if config_manager:
            saved = config_manager.get("ui", "output_dir", "") or ""
            if saved and os.path.isdir(saved):
                self.output_dir_var.set(saved)
        self._update_folder_display()

    def _build(self):
        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=(0, SP_XS))
        ttk.Label(row, text="Format").pack(side=tk.LEFT)
        self._fmt_combo = ttk.Combobox(row, textvariable=self.format_var,
                                       values=OUTPUT_FORMATS, state="readonly", width=12)
        self._fmt_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        self._fmt_hint = WrappingLabel(self, text="", style="Dim.TLabel", justify=tk.LEFT)
        self._fmt_hint.pack(fill=tk.X, pady=(0, SP_S))
        self.format_var.trace_add("write", lambda *a: self._on_format_change())

        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=(0, px(2)))
        ttk.Label(row, text="Folder").pack(side=tk.LEFT)
        self._browse_btn = ttk.Button(row, text="Browse\u2026", style="Small.TButton",
                                      command=self._browse_output)
        self._browse_btn.pack(side=tk.RIGHT)
        self._reset_btn = ttk.Button(row, text="Use original folder", style="Small.TButton",
                                     command=self._reset_folder)
        self._folder_display = ttk.Label(self, text=SAME_FOLDER, anchor=tk.W)
        self._folder_display.pack(fill=tk.X, pady=(0, SP_XS))
        self._folder_tip = ToolTip(self._folder_display, "")

        self._sub_cb = ttk.Checkbutton(self, text="Put speaker names in subtitles",
                                       variable=self.subtitle_speakers_var)
        ToolTip(self._sub_cb, "Prefix each subtitle line with the speaker label\n"
                              "(needs Identify speakers)")

        self._save_hint = WrappingLabel(self, text="", style="Dim.TLabel", justify=tk.LEFT)
        self._save_hint.pack(fill=tk.X, pady=(px(2), 0))
        self._on_format_change()

    def _on_format_change(self):
        fmt = self.format_var.get()
        self._fmt_hint.configure(text=FORMAT_DESCRIPTIONS.get(fmt, ""))
        if fmt in ("srt", "vtt"):
            self._sub_cb.pack(anchor=tk.W, pady=(0, SP_XS), before=self._save_hint)
        else:
            self._sub_cb.pack_forget()

    def _update_folder_display(self):
        folder = self.output_dir_var.get()
        if folder:
            self._folder_display.configure(text=_shorten(folder, 44))
            self._folder_tip.text = folder
            self._reset_btn.pack(side=tk.RIGHT, padx=(0, SP_XS), before=self._browse_btn)
            self._save_hint.configure(
                text="Files are saved in this folder, named after the original. "
                     "Existing files are never overwritten.")
        else:
            self._folder_display.configure(text=SAME_FOLDER)
            self._folder_tip.text = ""
            self._reset_btn.pack_forget()
            self._save_hint.configure(
                text="Named after the original, e.g. interview_transcript.txt. "
                     "Existing files are never overwritten.")

    def _reset_folder(self):
        self.output_dir_var.set("")
        if self.config_manager:
            self.config_manager.set("ui", "output_dir", "")
        self._update_folder_display()

    def _browse_output(self):
        initial_dir = self.output_dir_var.get()
        if not initial_dir and self.config_manager:
            initial_dir = self.config_manager.get("ui", "last_output_dir", "")
        path = filedialog.askdirectory(
            parent=self.winfo_toplevel(),
            title="Choose where to save transcripts",
            initialdir=initial_dir or None,
            mustexist=True,
        )
        if path:
            self.output_dir_var.set(os.path.normpath(path))
            if self.config_manager:
                self.config_manager.set("ui", "last_output_dir", os.path.normpath(path))
                self.config_manager.set("ui", "output_dir", os.path.normpath(path))
            self._update_folder_display()

    def set_enabled(self, enabled):
        state = ["!disabled"] if enabled else ["disabled"]
        for w in (self._fmt_combo, self._browse_btn, self._reset_btn, self._sub_cb):
            w.state(state)


def _shorten(path, limit):
    """C:\\Users\\me\\Very\\Long\\Folder -> ...\\Long\\Folder (keeps the end)."""
    if len(path) <= limit:
        return path
    parts = path.replace("/", os.sep).split(os.sep)
    tail = parts[-1]
    for part in reversed(parts[:-1]):
        if len(part) + len(tail) + 5 > limit:
            break
        tail = part + os.sep + tail
    return "\u2026" + os.sep + tail
