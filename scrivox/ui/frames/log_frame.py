"""Scrollable read-only log text widget with batched inserts."""

import tkinter as tk
from tkinter import ttk

from ..theme import COLORS, FONTS


class LogFrame(ttk.LabelFrame):
    """Read-only scrolling text log with batched inserts for performance."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="LOG", **kwargs)
        self._buffer = []
        self._flush_id = None
        self._build()

    def _build(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.text_widget = tk.Text(
            container,
            wrap=tk.WORD,
            font=FONTS["mono_small"],
            bg=COLORS["log_bg"],
            fg=COLORS["log_fg"],
            insertbackground=COLORS["fg"],
            selectbackground=COLORS["accent"],
            selectforeground=COLORS["button_fg"],
            borderwidth=0,
            highlightthickness=0,
            state=tk.DISABLED,
            height=10,
        )

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL,
                                   command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def clear(self):
        """Clear the log."""
        self._buffer.clear()
        if self._flush_id is not None:
            self.after_cancel(self._flush_id)
            self._flush_id = None
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.configure(state=tk.DISABLED)

    def append(self, text):
        """Buffer text and schedule a batched flush (thread-safe if called via root.after)."""
        self._buffer.append(text)
        if self._flush_id is None:
            self._flush_id = self.after(32, self._flush)

    def _flush(self):
        """Flush buffered messages in a single NORMAL/insert/see/DISABLED cycle."""
        self._flush_id = None
        if not self._buffer:
            return
        combined = "".join(self._buffer)
        self._buffer.clear()
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, combined)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state=tk.DISABLED)
