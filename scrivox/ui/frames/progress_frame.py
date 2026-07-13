"""Progress bar with file-level and step-level tracking, plus elapsed time."""

import time
import tkinter as tk
from tkinter import ttk

from ..theme import COLORS, FONTS


class ProgressFrame(ttk.LabelFrame):
    """Dual progress bar: file-level (batch) + step-level, with elapsed timer.

    The step bar runs in indeterminate marquee mode when a step reports no
    within-step progress, and switches to determinate (with an ETA estimate)
    once set_step_fraction() data arrives.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="PROGRESS", **kwargs)

        self._file_text = tk.StringVar(value="")
        self._step_text = tk.StringVar(value="Ready")
        self._elapsed_text = tk.StringVar(value="")
        self._start_time = None
        self._timer_id = None
        self._file_bar_shown = False

        # Within-step progress state
        self._step_num = 0
        self._total_steps = 1
        self._step_label = ""
        self._step_start = None
        self._marquee_running = False
        self._cancelling = False

        self._build()

    def _build(self):
        # ── File-level progress (hidden for single-file jobs) ──
        self._file_row = ttk.Frame(self)

        ttk.Label(self._file_row, textvariable=self._file_text,
                  font=FONTS["body"]).pack(fill=tk.X, padx=8, pady=(8, 2))
        self._file_bar = ttk.Progressbar(self._file_row, mode="determinate", maximum=100)
        self._file_bar.pack(fill=tk.X, padx=8, pady=(0, 4))

        # ── Step-level progress ──
        self._step_row = ttk.Frame(self)
        self._step_row.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Label(self._step_row, textvariable=self._step_text, font=FONTS["body"]).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(self._step_row, textvariable=self._elapsed_text, style="Dim.TLabel").pack(
            side=tk.RIGHT)

        self._progress_bar = ttk.Progressbar(self, mode="determinate", maximum=100)
        self._progress_bar.pack(fill=tk.X, padx=8, pady=(0, 8))

    # ── Marquee helpers ──

    def _start_marquee(self):
        if not self._marquee_running:
            self._progress_bar.configure(mode="indeterminate")
            self._progress_bar.start(16)
            self._marquee_running = True

    def _stop_marquee(self):
        if self._marquee_running:
            self._progress_bar.stop()
            self._progress_bar.configure(mode="determinate")
            self._marquee_running = False

    def reset(self):
        """Reset progress to initial state."""
        self._stop_marquee()
        self._file_text.set("")
        self._step_text.set("Ready")
        self._elapsed_text.set("")
        self._progress_bar["value"] = 0
        self._file_bar["value"] = 0
        self._step_num = 0
        self._total_steps = 1
        self._step_label = ""
        self._step_start = None
        self._cancelling = False
        if self._file_bar_shown:
            self._file_row.pack_forget()
            self._file_bar_shown = False
        self._stop_timer()

    def start(self):
        """Start the elapsed timer."""
        self._start_time = time.time()
        self._start_timer()

    def update_file(self, file_num, total_files, filename):
        """Update file-level progress (shown only for batch jobs)."""
        if total_files > 1:
            if not self._file_bar_shown:
                # Show file bar above step bar by repacking in order
                self._step_row.pack_forget()
                self._progress_bar.pack_forget()

                self._file_row.pack(fill=tk.X)
                self._step_row.pack(fill=tk.X, padx=8, pady=(4, 4))
                self._progress_bar.pack(fill=tk.X, padx=8, pady=(0, 8))
                self._file_bar_shown = True

            self._file_text.set(f"File {file_num}/{total_files}: {filename}")
            pct = int(((file_num - 1) / total_files) * 100)
            self._file_bar["value"] = pct

        # Reset step progress for new file
        self._step_text.set("Starting...")
        self._stop_marquee()
        self._progress_bar["value"] = 0

    def update_step(self, step_num, total_steps, step_name):
        """Update the step label; run the bar as a marquee until fraction data arrives."""
        self._step_num = step_num
        self._total_steps = max(total_steps, 1)
        self._step_label = f"Step {step_num}/{total_steps}: {step_name}"
        self._step_start = time.time()
        if not self._cancelling:
            self._step_text.set(self._step_label)
        # No within-step data yet — keep the bar visibly moving
        self._start_marquee()

    def set_step_fraction(self, frac):
        """Report within-step progress (0.0-1.0) for the current step."""
        frac = max(0.0, min(float(frac), 1.0))
        self._stop_marquee()
        # Fill within the current step's slice of the overall bar
        pct = ((self._step_num - 1) + frac) / self._total_steps * 100
        self._progress_bar["value"] = pct

        if self._cancelling:
            return
        eta_text = ""
        if frac > 0.02 and self._step_start is not None:
            elapsed = time.time() - self._step_start
            remaining = elapsed * (1 - frac) / frac
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            eta_text = f" — ETA {mins:02d}:{secs:02d}"
        self._step_text.set(f"{self._step_label}{eta_text}")

    def complete(self, elapsed=None):
        """Mark progress as complete."""
        self._stop_timer()
        self._stop_marquee()
        self._cancelling = False
        self._progress_bar["value"] = 100
        self._file_bar["value"] = 100
        if elapsed is not None:
            self._step_text.set(f"Done in {elapsed:.1f}s")
        else:
            self._step_text.set("Done")

    def set_error(self, message):
        """Show error state."""
        self._stop_timer()
        self._stop_marquee()
        self._cancelling = False
        self._step_text.set(f"Error: {message}")

    def set_cancelling(self):
        """Show cancel-requested state while the pipeline winds down."""
        self._cancelling = True
        self._step_text.set("Cancelling - finishing current operation...")

    def set_cancelled(self):
        """Show cancelled state."""
        self._stop_timer()
        self._stop_marquee()
        self._cancelling = False
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
            self._elapsed_text.set(f"{mins:02d}:{secs:02d}")
        self._timer_id = self.after(1000, self._update_elapsed)
