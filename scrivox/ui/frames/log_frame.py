"""Scrollable read-only log text widget."""

import tkinter as tk
from tkinter import ttk

from ..theme import COLORS, FONTS


class LogFrame(ttk.LabelFrame):
    """Read-only scrolling text log."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="LOG", **kwargs)
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
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.configure(state=tk.DISABLED)

    def append(self, text):
        """Append text to the log (thread-safe if called via root.after)."""
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state=tk.DISABLED)
