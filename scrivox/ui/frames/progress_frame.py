"""Progress bar, step label, and elapsed time display."""

import time
import tkinter as tk
from tkinter import ttk

from ..theme import COLORS, FONTS


class ProgressFrame(ttk.LabelFrame):
    """Progress bar with step label and elapsed timer."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="PROGRESS", **kwargs)

        self._step_text = tk.StringVar(value="Ready")
        self._elapsed_text = tk.StringVar(value="")
        self._start_time = None
        self._timer_id = None

        self._build()

    def _build(self):
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Label(row, textvariable=self._step_text, font=FONTS["body"]).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row, textvariable=self._elapsed_text, style="Dim.TLabel").pack(
            side=tk.RIGHT)

        self._progress_bar = ttk.Progressbar(self, mode="determinate", maximum=100)
        self._progress_bar.pack(fill=tk.X, padx=8, pady=(0, 8))

    def reset(self):
        """Reset progress to initial state."""
        self._step_text.set("Ready")
        self._elapsed_text.set("")
        self._progress_bar["value"] = 0
        self._stop_timer()

    def start(self):
        """Start the elapsed timer."""
        self._start_time = time.time()
        self._start_timer()

    def update_step(self, step_num, total_steps, step_name):
        """Update the step label and progress bar."""
        pct = int((step_num / total_steps) * 100)
        self._step_text.set(f"Step {step_num}/{total_steps}: {step_name}")
        self._progress_bar["value"] = pct

    def complete(self, elapsed=None):
        """Mark progress as complete."""
        self._stop_timer()
        self._progress_bar["value"] = 100
        if elapsed is not None:
            self._step_text.set(f"Done in {elapsed:.1f}s")
        else:
            self._step_text.set("Done")

    def set_error(self, message):
        """Show error state."""
        self._stop_timer()
        self._step_text.set(f"Error: {message}")

    def set_cancelled(self):
        """Show cancelled state."""
        self._stop_timer()
        self._step_text.set("Cancelled")

    def _start_timer(self):
        self._update_elapsed()

    def _stop_timer(self):
        if self._timer_id:
            try:
                self.after_cancel(self._timer_id)
            except Exception:
                pass
            self._timer_id = None

    def _update_elapsed(self):
        if self._start_time:
            elapsed = time.time() - self._start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self._elapsed_text.set(f"{mins}:{secs:02d}")
        self._timer_id = self.after(1000, self._update_elapsed)
