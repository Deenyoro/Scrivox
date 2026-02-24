"""Transcript display with copy/save buttons."""

import os
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..theme import COLORS, FONTS


class ResultsFrame(ttk.LabelFrame):
    """Results display with Copy, Save As, and Open Folder buttons."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="RESULTS", **kwargs)
        self._output_path = None
        self._full_text = ""
        self._build()

    def _build(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.text_widget = tk.Text(
            container,
            wrap=tk.WORD,
            font=FONTS["mono_small"],
            bg=COLORS["log_bg"],
            fg=COLORS["fg"],
            insertbackground=COLORS["fg"],
            selectbackground=COLORS["accent"],
            selectforeground=COLORS["button_fg"],
            borderwidth=0,
            highlightthickness=0,
            state=tk.DISABLED,
            height=8,
        )

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL,
                                   command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Buttons
        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._copy_btn = ttk.Button(btn_row, text="Copy", command=self._copy)
        self._copy_btn.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Save As...", command=self._save_as).pack(side=tk.LEFT, padx=(0, 4))
        self._open_btn = ttk.Button(btn_row, text="Open Folder", command=self._open_folder,
                                     state=tk.DISABLED)
        self._open_btn.pack(side=tk.LEFT)

    def show_result(self, text, output_path=None):
        """Display result text and enable buttons."""
        self._output_path = output_path
        self._full_text = text
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        # Show first 5000 chars to avoid UI lag on huge transcripts
        display_text = text
        if len(text) > 5000:
            display_text = text[:5000] + f"\n\n... ({len(text)} total characters, use Save As for full text)"
        self.text_widget.insert("1.0", display_text)
        self.text_widget.configure(state=tk.DISABLED)

        if output_path and os.path.isfile(output_path):
            self._open_btn.configure(state=tk.NORMAL)
        else:
            self._open_btn.configure(state=tk.DISABLED)

    def clear(self):
        """Clear results."""
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.configure(state=tk.DISABLED)
        self._output_path = None
        self._full_text = ""
        self._open_btn.configure(state=tk.DISABLED)

    def _copy(self):
        """Copy full text to clipboard with visual feedback."""
        text = self._full_text
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            # Flash "Copied!" feedback
            self._copy_btn.configure(text="Copied!")
            self.after(1500, lambda: self._copy_btn.configure(text="Copy"))

    def _save_as(self):
        """Save results to a file."""
        path = filedialog.asksaveasfilename(
            title="Save transcript",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("Markdown", "*.md"),
                ("SRT subtitles", "*.srt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._full_text)
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save file:\n{e}")

    def _open_folder(self):
        """Open the folder containing the output file."""
        if self._output_path and os.path.isfile(self._output_path):
            folder = os.path.dirname(os.path.abspath(self._output_path))
            try:
                subprocess.Popen(["explorer", "/select,", os.path.abspath(self._output_path)])
            except Exception:
                os.startfile(folder)
