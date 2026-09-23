"""Run status: a headline, file-level and step-level progress, elapsed time.

The headline doubles as the result line: "Done", or a plain-language error
with a "How to fix" button.
"""

import time
import tkinter as tk
from tkinter import ttk

from ..theme import SP_S, SP_XS, px
from ..widgets import WrappingLabel


class ProgressFrame(ttk.Frame):
    """Dual progress bar: file-level (batch) + step-level, with elapsed timer.

    The step bar runs in indeterminate marquee mode when a step reports no
    within-step progress, and switches to determinate (with an ETA estimate)
    once set_step_fraction() data arrives.
    """

    READY_TEXT = "Ready when you are"
    READY_DETAIL = "Add files, check the options, then press Start transcription."

    def __init__(self, parent, on_progress_change=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_progress_change = on_progress_change or (lambda pct: None)

        self._file_text = tk.StringVar(value="")
        self._step_text = tk.StringVar(value=self.READY_TEXT)
        self._detail_text = tk.StringVar(value=self.READY_DETAIL)
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
        self._file_num = 1
        self._total_files = 1
        self._fix_command = None
        self._current_file = ""
        self._detail_is_status = False

        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self._elapsed_text, style="Dim.TLabel").pack(
            side=tk.RIGHT, anchor=tk.N, pady=(px(3), 0))
        self._headline = WrappingLabel(top, textvariable=self._step_text,
                                       style="Headline.TLabel", justify=tk.LEFT)
        self._headline.pack(side=tk.LEFT, fill=tk.X, expand=True)

        detail_row = ttk.Frame(self)
        detail_row.pack(fill=tk.X, pady=(px(2), SP_S))
        self._fix_btn = ttk.Button(detail_row, text="How to fix\u2026", style="Small.TButton",
                                   command=lambda: self._fix_command and self._fix_command())
        self._detail = WrappingLabel(detail_row, textvariable=self._detail_text,
                                     style="Dim.TLabel", justify=tk.LEFT)
        self._detail.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── File-level progress (hidden for single-file jobs) ──
        self._file_row = ttk.Frame(self)
        ttk.Label(self._file_row, textvariable=self._file_text, style="Dim.TLabel").pack(
            fill=tk.X, pady=(0, px(2)))
        self._file_bar = ttk.Progressbar(self._file_row, mode="determinate", maximum=100,
                                         style="Thin.Horizontal.TProgressbar")
        self._file_bar.pack(fill=tk.X, pady=(0, SP_XS))

        self._step_row = self._detail  # kept name: widget above the step bar
        self._progress_bar = ttk.Progressbar(self, mode="determinate", maximum=100)
        self._progress_bar.pack(fill=tk.X)

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

    def _set_headline(self, text, style="Headline.TLabel", detail="", fix=None):
        self._step_text.set(text)
        self._headline.configure(style=style)
        self._detail_text.set(detail)
        self._fix_command = fix
        if fix:
            self._fix_btn.pack(side=tk.LEFT, padx=(0, SP_S), before=self._detail)
        else:
            self._fix_btn.pack_forget()

    def _restore_detail(self):
        """After a status message (e.g. download) go back to naming the file."""
        if self._detail_is_status:
            self._detail_text.set(self._current_file)
            self._detail_is_status = False

    def overall_percent(self):
        step_pct = float(self._progress_bar["value"]) if not self._marquee_running else None
        if step_pct is None:
            step_pct = ((self._step_num - 1) / self._total_steps * 100) if self._step_num else 0
        return ((self._file_num - 1) + step_pct / 100.0) / max(self._total_files, 1) * 100

    def reset(self):
        """Reset progress to initial state."""
        self._stop_marquee()
        self._file_text.set("")
        self._set_headline(self.READY_TEXT, detail=self.READY_DETAIL)
        self._elapsed_text.set("")
        self._progress_bar["value"] = 0
        self._file_bar["value"] = 0
        self._step_num = 0
        self._total_steps = 1
        self._step_label = ""
        self._step_start = None
        self._cancelling = False
        self._file_num = 1
        self._total_files = 1
        self._hide_file_row()
        self._stop_timer()

    def start(self):
        """Start the elapsed timer."""
        self._start_time = time.time()
        self._set_headline("Starting…", detail="Preparing the first file")
        self._start_timer()

    def update_file(self, file_num, total_files, filename):
        """Update file-level progress (shown only for batch jobs)."""
        self._file_num, self._total_files = file_num, max(total_files, 1)
        if total_files > 1:
            if not self._file_bar_shown:
                self._file_row.pack(fill=tk.X, before=self._progress_bar)
                self._file_bar_shown = True
            self._file_text.set(f"File {file_num} of {total_files}")
            self._file_bar["value"] = int(((file_num - 1) / total_files) * 100)

        # Reset step progress for new file
        if not self._cancelling:
            self._set_headline("Starting…", detail=filename)
        self._current_file = filename
        self._stop_marquee()
        self._progress_bar["value"] = 0
        self._on_progress_change(self.overall_percent())

    def update_step(self, step_num, total_steps, step_name):
        """Update the step label; run the bar as a marquee until fraction data arrives."""
        self._step_num = step_num
        self._total_steps = max(total_steps, 1)
        self._step_label = step_name if total_steps <= 1 else \
            f"{step_name} (step {step_num} of {total_steps})"
        self._step_start = time.time()
        if not self._cancelling:
            self._step_text.set(self._step_label)
            self._restore_detail()
        # No within-step data yet — keep the bar visibly moving
        self._start_marquee()
        self._on_progress_change(self.overall_percent())

    def set_step_fraction(self, frac):
        """Report within-step progress (0.0-1.0) for the current step."""
        frac = max(0.0, min(float(frac), 1.0))
        self._stop_marquee()
        # Fill within the current step's slice of the overall bar
        pct = ((self._step_num - 1) + frac) / self._total_steps * 100
        self._progress_bar["value"] = pct
        self._on_progress_change(self.overall_percent())

        if self._cancelling:
            return
        self._restore_detail()
        eta_text = ""
        if frac > 0.02 and self._step_start is not None:
            elapsed = time.time() - self._step_start
            remaining = elapsed * (1 - frac) / frac
            eta_text = f" — about {_fmt_remaining(remaining)} left"
        self._step_text.set(f"{self._step_label}{eta_text}")

    def set_status(self, text, fraction=None, detail=None):
        """Free-form status for the current step (e.g. a model download)."""
        if self._cancelling:
            return
        self._step_text.set(text)
        if detail is not None:
            self._detail_text.set(detail)
            self._detail_is_status = True
        if fraction is None:
            self._start_marquee()
        else:
            self._stop_marquee()
            self._progress_bar["value"] = max(0.0, min(fraction, 1.0)) * 100

    def _hide_file_row(self):
        if self._file_bar_shown:
            self._file_row.pack_forget()
            self._file_bar_shown = False

    def show_ready(self, count, blocked=False):
        """While idle, say how many files are waiting instead of asking for
        files that are already there."""
        if self._step_text.get() != self.READY_TEXT:
            return  # showing a finished/failed/cancelled run: keep it
        if not count:
            detail = self.READY_DETAIL
        else:
            files = f"{count} file{'s' if count != 1 else ''}"
            detail = (f"{files} added. See the note next to Start transcription."
                      if blocked else f"{files} ready. Press Start transcription.")
        if self._detail_text.get() != detail:
            self._detail_text.set(detail)

    def complete(self, elapsed=None, headline=None, detail="", warning=False):
        """Mark progress as complete. The headline carries the duration, so
        the running timer is cleared (one duration, not two). `warning` is
        for a batch where some files failed: not shown as a success."""
        self._stop_timer()
        self._stop_marquee()
        self._hide_file_row()
        self._cancelling = False
        self._elapsed_text.set("")
        self._progress_bar["value"] = 100
        self._file_bar["value"] = 100
        text = headline or "Done"
        if elapsed is not None:
            text = f"{text} in {_fmt_elapsed(elapsed)}"
        self._set_headline(text, detail=detail, style="HeadlineWarning.TLabel"
                           if warning else "HeadlineSuccess.TLabel")

    def set_error(self, message, detail="", fix=None):
        """Show error state; `fix` adds a "How to fix" button."""
        self._stop_timer()
        self._stop_marquee()
        self._hide_file_row()
        self._cancelling = False
        self._set_headline(message, style="HeadlineError.TLabel",
                           detail=detail or "The log has the technical details.", fix=fix)

    def set_cancelling(self):
        """Show cancel-requested state while the pipeline winds down."""
        self._cancelling = True
        self._set_headline("Cancelling…",
                           detail="Finishing the current operation, this can take a moment.")

    def set_cancelled(self, detail=None):
        """Show cancelled state."""
        self._stop_timer()
        self._stop_marquee()
        self._hide_file_row()
        self._cancelling = False
        self._set_headline("Cancelled", detail=detail or (
            "Nothing more will be processed. Press Start transcription to try again."))

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


def _fmt_remaining(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{max(seconds, 1)} s"
    mins = round(seconds / 60)
    if mins < 60:
        return f"{mins} min"
    return f"{mins // 60} h {mins % 60:02d} min"


def _fmt_elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins} min {secs:02d} s"
    return f"{mins // 60} h {mins % 60:02d} min"
