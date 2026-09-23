"""Scrollable read-only log text widget with batched inserts."""

import tkinter as tk
from tkinter import ttk

from ..theme import COLORS, FONTS


class LogFrame(ttk.Frame):
    """Read-only scrolling text log with batched inserts for performance.

    Error lines are shown in red so the reason for a failure stands out.
    """

    MAX_LINES = 5000  # cap growth on long batch runs
    _ERROR_PREFIXES = ("ERROR", "FATAL", "Error:", "Traceback")
    _WARN_PREFIXES = ("Warning", "WARNING")

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._buffer = []
        self._flush_id = None
        self._build()

    def _build(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)
        s = self.winfo_fpixels("1i") / 96.0

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
            padx=int(10 * s), pady=int(8 * s),
            state=tk.DISABLED,
            height=10,
        )
        self.text_widget.tag_configure("error", foreground=COLORS["error"])
        self.text_widget.tag_configure("warning", foreground=COLORS["warning"])
        self.text_widget.tag_configure("current_error", background=COLORS["error_bg"])
        self.text_widget.bind("<1>", lambda e: self.text_widget.focus_set())

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL,
                                   command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def last_error_line(self):
        """Index of the last red line (to jump to after a failure), or None."""
        ranges = self.text_widget.tag_ranges("error")
        return str(ranges[-2]) if ranges else None

    def show_last_error(self):
        idx = self.last_error_line()
        if idx:
            self.text_widget.see(idx)
            self.text_widget.tag_remove("current_error", "1.0", tk.END)
            self.text_widget.tag_add("current_error", f"{idx} linestart", f"{idx} lineend + 1c")

    def clear(self):
        """Clear the log."""
        self._buffer.clear()
        if self._flush_id is not None:
            self.after_cancel(self._flush_id)
            self._flush_id = None
        self.text_widget.configure(state=tk.NORMAL)
        try:
            self.text_widget.delete("1.0", tk.END)
        finally:
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
        try:
            # Only auto-scroll when the user is already at (or near) the
            # bottom — don't yank scrollback away during long batches
            at_bottom = self.text_widget.yview()[1] >= 0.999
            self.text_widget.configure(state=tk.NORMAL)
            for line in combined.splitlines(keepends=True):
                stripped = line.lstrip()
                tag = ()
                if stripped.startswith(self._ERROR_PREFIXES):
                    tag = ("error",)
                elif stripped.startswith(self._WARN_PREFIXES):
                    tag = ("warning",)
                self.text_widget.insert(tk.END, line, tag)
            # Trim oldest lines past the cap
            line_count = int(self.text_widget.index("end-1c").split(".")[0])
            if line_count > self.MAX_LINES:
                self.text_widget.delete("1.0", f"{line_count - self.MAX_LINES}.0")
            if at_bottom:
                self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)
        except tk.TclError:
            pass  # widget destroyed during shutdown
