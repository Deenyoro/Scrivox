"""Job queue frame with multi-file selection and audio track support.

Files appear in the queue immediately; the audio-track probe (ffprobe, up to
15 s per file on a slow or network drive) runs on a worker thread and hands
its result back with after(), so adding many files never freezes the window.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from dataclasses import dataclass
from typing import List

from ...core.constants import VIDEO_EXTENSIONS, AUDIO_EXTENSIONS
from ...core.media import list_audio_tracks
from ..output_paths import is_media_file
from ..theme import COLORS, SP_S, SP_XS
from ..widgets import (DropZone, TreeRowTooltip, TreeTextFitter, WrappingLabel,
                       fit_column_to_labels)
from .. import winnative


@dataclass
class JobConfig:
    """Configuration for a single transcription job."""
    file_path: str
    audio_track: int = 0
    track_label: str = ""       # "Track 0 (English)"
    language_override: str = ""  # Override auto-detect


_STATUS_TEXT = {
    "checking": "Checking\u2026",
    "pending": "Ready",
    "running": "Working\u2026",
    "done": "Done",
    "error": "Failed",
    "cancelled": "Cancelled",
}


class QueueFrame(ttk.Frame):
    """Job table with drop zone, multi-file browse and track selection."""

    def __init__(self, parent, config_manager=None, on_tracks_needed=None,
                 on_change=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.config_manager = config_manager
        self._on_tracks_needed = on_tracks_needed  # callback(filepath, tracks) -> selected indices
        self._on_change = on_change or (lambda: None)
        self._jobs = {}           # tree iid -> JobConfig (ready jobs only)
        self._status = {}         # tree iid -> status key
        self._pending_probes = {}  # placeholder iid -> path
        self._track_queue = []     # (placeholder iid, path, tracks) awaiting a dialog
        self._dialog_open = False
        self._results = queue.Queue()
        self._poll_id = None
        self._enabled = True
        self._skipped = []
        self._build()

    # ── UI ──

    def _build(self):
        self._dnd_available = False

        # Empty state: big drop zone
        self._drop_zone = DropZone(self, on_browse=self.browse_files, dnd_available=False)

        # Filled state: table + button row
        self._list_frame = ttk.Frame(self)
        tree_frame = ttk.Frame(self._list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("file", "track", "status")
        self._tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                  height=4, selectmode="extended")
        self._tree.heading("file", text="File", anchor=tk.W)
        self._tree.heading("track", text="Audio track", anchor=tk.W)
        self._tree.heading("status", text="Status", anchor=tk.W)
        s = winnative_scale(self)
        self._tree.column("file", width=int(170 * s), minwidth=int(90 * s), stretch=True)
        # "Track 1 · jpn" fits whole; longer labels are cut at the end
        fit_column_to_labels(self._tree, "track", ["Track 10 \u00b7 eng"], "Audio track")
        # Wide enough for the longest status ("Cancelled", "Checking...") in
        # the actual font, at any DPI
        fit_column_to_labels(self._tree, "status", _STATUS_TEXT.values(), "Status")
        # Long names get a middle ellipsis that keeps the extension visible
        self._fitter = TreeTextFitter(self._tree, ("file", "track"), keep_start=("track",))

        # Color rows by status
        self._tree.tag_configure("checking", foreground=COLORS["fg_dim"])
        self._tree.tag_configure("running", foreground=COLORS["accent"])
        self._tree.tag_configure("done", foreground=COLORS["success"])
        self._tree.tag_configure("error", foreground=COLORS["error"])
        self._tree.tag_configure("cancelled", foreground=COLORS["fg_dim"])

        self._tree.bind("<Delete>", lambda e: self._remove_selected())
        self._tree.bind("<BackSpace>", lambda e: self._remove_selected())
        self._tree.bind("<Button-3>", self._show_context_menu)
        self._tree.bind("<Shift-F10>", self._show_context_menu)
        try:
            self._tree.bind("<App>", self._show_context_menu)  # Windows menu key
        except tk.TclError:
            pass  # keysym only exists on Windows
        self._tree.bind("<Double-1>", lambda e: self._reveal_selected())
        TreeRowTooltip(self._tree, self._row_tooltip)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        btn_bar = ttk.Frame(self._list_frame)
        btn_bar.pack(fill=tk.X, pady=(SP_S, 0))
        self._add_btn = ttk.Button(btn_bar, text="Add files\u2026", underline=0,
                                   command=self.browse_files)
        self._add_btn.pack(side=tk.LEFT, padx=(0, SP_XS))
        self._remove_btn = ttk.Button(btn_bar, text="Remove", command=self._remove_selected)
        self._remove_btn.pack(side=tk.LEFT, padx=(0, SP_XS))
        self._clear_btn = ttk.Button(btn_bar, text="Clear", command=self._clear_all)
        self._clear_btn.pack(side=tk.LEFT)
        # Queue summary sits beside the buttons (no extra line of height);
        # packed last so it gives way first in a narrow column
        self._hint_label = ttk.Label(btn_bar, text="", style="Dim.TLabel", anchor=tk.E)
        self._hint_label.pack(side=tk.RIGHT, padx=(SP_XS, 0))

        self._notice_label = WrappingLabel(self, text="", style="Warning.TLabel",
                                           justify=tk.LEFT)

        self._menu = tk.Menu(self, tearoff=False)
        self._menu.add_command(label="Show in folder", command=self._reveal_selected)
        self._menu.add_command(label="Remove", accelerator="Del", command=self._remove_selected)
        self._menu.add_separator()
        self._menu.add_command(label="Remove finished", command=self.remove_finished)
        self._menu.add_command(label="Clear queue", command=self._clear_all)

        # Try to enable drag-and-drop, then pick the empty/filled view
        self._setup_dnd()
        self._update_view()

    def _setup_dnd(self):
        """Register the drop zone and the table as file drop targets.

        Needs the tkdnd extension loaded into the root (see ui/app.py); when
        it isn't available the queue quietly works browse-only.
        """
        if not getattr(self.winfo_toplevel(), "dnd_available", False):
            return
        try:
            from tkinterdnd2 import DND_FILES
            for target in (self._drop_zone, self._tree):
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<Drop>>", self._on_drop)
            self._dnd_available = True
        except (ImportError, AttributeError, RuntimeError, tk.TclError):
            self._dnd_available = False
        self._drop_zone.set_dnd_available(self._dnd_available)

    # ── Adding files ──

    @staticmethod
    def parse_drop_data(raw):
        """Split tkdnd drop data: paths with spaces arrive wrapped in braces,
        e.g. `{C:\\path with spaces\\file.mp4} C:\\other.wav`."""
        paths = []
        i = 0
        while i < len(raw):
            if raw[i] == "{":
                end = raw.find("}", i)
                if end == -1:
                    break  # malformed drop data
                paths.append(raw[i + 1:end])
                i = end + 1  # the whitespace branch skips any separator space
            elif raw[i] == " ":
                i += 1
            else:
                end = raw.find(" ", i)
                if end == -1:
                    end = len(raw)
                paths.append(raw[i:end])
                i = end + 1
        return paths

    def _on_drop(self, event):
        """Handle dropped files (folders are expanded one level)."""
        if not self._enabled:
            return
        paths = []
        for path in self.parse_drop_data(event.data):
            if os.path.isdir(path):
                try:
                    paths += sorted(os.path.join(path, f) for f in os.listdir(path)
                                    if is_media_file(f))
                except OSError:
                    pass
            elif os.path.isfile(path):
                paths.append(path)
        if paths and self.config_manager:
            self.config_manager.set("ui", "last_input_dir", os.path.dirname(paths[0]))
        self.add_files(paths)
        return getattr(event, "action", None)

    def browse_files(self):
        """Open multi-file dialog and add selected files to the queue."""
        if not self._enabled:
            return  # queue is locked while a batch is running
        all_exts = sorted(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)
        ext_pattern = " ".join(f"*{e}" for e in all_exts)

        initial_dir = ""
        if self.config_manager:
            initial_dir = self.config_manager.get("ui", "last_input_dir", "")

        paths = filedialog.askopenfilenames(
            parent=self.winfo_toplevel(),
            title="Choose audio or video files",
            initialdir=initial_dir or None,
            filetypes=[
                ("Audio and video", ext_pattern),
                ("Video files", " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))),
                ("Audio files", " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))),
                ("All files", "*.*"),
            ],
        )
        if paths:
            if self.config_manager:
                self.config_manager.set("ui", "last_input_dir", os.path.dirname(paths[0]))
            self.add_files(paths)

    def add_files(self, paths):
        """Queue several files, reporting unsupported ones in one notice."""
        self._skipped = []
        for path in paths:
            self._add_file(path, _batch=True)
        self._show_skipped()

    def _show_skipped(self):
        if not self._skipped:
            self._notice_label.configure(text="")
        else:
            names = ", ".join(os.path.basename(p) for p in self._skipped[:3])
            more = f" and {len(self._skipped) - 3} more" if len(self._skipped) > 3 else ""
            self._notice_label.configure(
                text=f"Skipped {names}{more}: not an audio or video file.")
        self._update_view()

    def _is_queued(self, path, audio_track):
        """True if the same file+track combination is already in the queue."""
        norm = os.path.normcase(os.path.abspath(path))
        if any(os.path.normcase(os.path.abspath(p)) == norm
               for p in self._pending_probes.values()):
            return True
        return any(
            os.path.normcase(os.path.abspath(j.file_path)) == norm
            and j.audio_track == audio_track
            for j in self._jobs.values()
        )

    def _add_file(self, path, _batch=False):
        """Add a file to the queue. Returns immediately: the row shows
        "Checking..." until the track probe finishes in the background."""
        if not _batch:
            self._skipped = []
        if not is_media_file(path):
            self._skipped.append(path)
            if not _batch:
                self._show_skipped()
            return
        if self._is_queued(path, 0):
            return
        iid = self._tree.insert("", tk.END, values=("", "", _STATUS_TEXT["checking"]),
                                tags=("checking",))
        self._fitter.set(iid, "file", os.path.basename(path))
        self._status[iid] = "checking"
        self._pending_probes[iid] = path
        threading.Thread(target=self._probe_worker, args=(iid, path), daemon=True).start()
        self._start_polling()
        if not _batch:
            self._notice_label.configure(text="")
        self._update_view()

    def _probe_worker(self, iid, path):
        # Worker thread: no Tk calls here, only the result queue
        try:
            tracks = list_audio_tracks(path)
        except Exception:  # noqa: BLE001 - a failed probe must still free the row
            tracks = []
        self._results.put((iid, path, tracks))

    def _start_polling(self):
        if self._poll_id is None:
            self._poll_id = self.after(30, self._poll_results)

    def _poll_results(self):
        self._poll_id = None
        try:
            while True:
                iid, path, tracks = self._results.get_nowait()
                self._on_probe_done(iid, path, tracks)
        except queue.Empty:
            pass
        if self._pending_probes:
            self._poll_id = self.after(50, self._poll_results)

    def _on_probe_done(self, iid, path, tracks):
        if iid not in self._pending_probes:
            return  # row was removed while probing
        if len(tracks) > 1 and self._on_tracks_needed:
            self._track_queue.append((iid, path, tracks))
            self._drain_track_queue()
            return
        self._pending_probes.pop(iid, None)
        label = self._format_track_label(tracks[0]) if tracks else ""
        self._make_job(iid, JobConfig(file_path=path, audio_track=0, track_label=label))
        self._update_view()

    def _drain_track_queue(self):
        """Show track dialogs one at a time, in the order files were added."""
        if self._dialog_open or not self._track_queue:
            return
        iid, path, tracks = self._track_queue.pop(0)
        self._dialog_open = True
        try:
            selected = self._on_tracks_needed(path, tracks) or []
        finally:
            self._dialog_open = False
        self._pending_probes.pop(iid, None)
        if self._tree.exists(iid):
            index = self._tree.index(iid)
            self._tree.delete(iid)
            self._status.pop(iid, None)
            self._fitter.forget(iid)
            for idx in selected:
                if self._is_queued(path, idx):
                    continue  # duplicate jobs race on the same output path
                track = tracks[idx] if idx < len(tracks) else tracks[0]
                new_iid = self._tree.insert("", index, values=("", "", ""))
                index += 1
                self._make_job(new_iid, JobConfig(file_path=path, audio_track=idx,
                                                  track_label=self._format_track_label(track)))
        self._update_view()
        if self._track_queue:
            self.after(10, self._drain_track_queue)

    def _make_job(self, iid, job):
        self._jobs[iid] = job
        self._tree.item(iid, values=("", "", _STATUS_TEXT["pending"]), tags=())
        self._fitter.set(iid, "file", os.path.basename(job.file_path))
        self._fitter.set(iid, "track", _short_track(job.track_label) or "\u2014")
        self._status[iid] = "pending"

    def _format_track_label(self, track):
        parts = [f"Track {track['index']}"]
        if track.get("language"):
            parts.append(track["language"])
        if track.get("codec"):
            parts.append(track["codec"].upper())
        return " \u00b7 ".join(parts)

    def _row_tooltip(self, iid):
        job = self._jobs.get(iid)
        path = job.file_path if job else self._pending_probes.get(iid, "")
        if not path:
            return ""
        track = f"\nAudio: {job.track_label}" if job and job.track_label else ""
        return f"{path}{track}\nRight-click for more options"

    # ── Removing ──

    def _remove_selected(self):
        """Remove selected jobs from the queue."""
        if not self._enabled:
            return  # queue is locked while a batch is running
        for iid in self._tree.selection():
            self._forget(iid)
        self._update_view()

    def _forget(self, iid):
        self._jobs.pop(iid, None)
        self._status.pop(iid, None)
        self._pending_probes.pop(iid, None)
        self._fitter.forget(iid)
        self._track_queue = [t for t in self._track_queue if t[0] != iid]
        if self._tree.exists(iid):
            self._tree.delete(iid)

    def _clear_all(self):
        """Remove all jobs from the queue."""
        if not self._enabled:
            return
        for iid in self._tree.get_children():
            self._forget(iid)
        self._notice_label.configure(text="")
        self._update_view()

    def remove_finished(self):
        if not self._enabled:
            return
        for iid, status in list(self._status.items()):
            if status == "done":
                self._forget(iid)
        self._update_view()

    def _reveal_selected(self):
        sel = self._tree.selection()
        job = self._jobs.get(sel[0]) if sel else None
        if job:
            winnative.reveal_in_folder(job.file_path)

    def _show_context_menu(self, event):
        if event.type == tk.EventType.ButtonPress:
            row = self._tree.identify_row(event.y)
            if row and row not in self._tree.selection():
                self._tree.selection_set(row)
            x, y = event.x_root, event.y_root
        else:
            x = self._tree.winfo_rootx() + 20
            y = self._tree.winfo_rooty() + 20
        has_sel = bool(self._tree.selection())
        state = tk.NORMAL if (has_sel and self._enabled) else tk.DISABLED
        self._menu.entryconfigure("Show in folder", state=tk.NORMAL if has_sel else tk.DISABLED)
        self._menu.entryconfigure("Remove", state=state)
        edit = tk.NORMAL if self._enabled else tk.DISABLED
        self._menu.entryconfigure("Remove finished", state=edit)
        self._menu.entryconfigure("Clear queue", state=edit)
        try:
            self._menu.tk_popup(x, y)
        finally:
            self._menu.grab_release()
        return "break"

    # ── View state ──

    def _update_view(self):
        """Show the drop zone when empty, the table when not."""
        rows = len(self._tree.get_children())
        has_rows = bool(rows)
        for w in (self._drop_zone, self._list_frame, self._notice_label):
            w.pack_forget()
        if has_rows:
            # Only as tall as needed (2-4 rows), so step 3 stays in view
            self._tree.configure(height=min(max(rows, 2), 4))
            self._list_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self._drop_zone.pack(fill=tk.X)
        if self._notice_label.cget("text"):
            self._notice_label.pack(fill=tk.X, pady=(SP_XS, 0))

        n = len(self._jobs)
        checking = len(self._pending_probes)
        if checking:
            hint = f"Checking {checking} file{'s' if checking != 1 else ''}\u2026"
        elif n:
            done = sum(1 for s in self._status.values() if s == "done")
            failed = sum(1 for s in self._status.values() if s == "error")
            hint = f"{n} file{'s' if n != 1 else ''}"
            if done:
                hint += f" \u00b7 {done} done"
            if failed:
                hint += f" \u00b7 {failed} failed"
        else:
            hint = ""
        self._hint_label.configure(text=hint)
        self._on_change()

    # Old name kept for callers
    _update_hint = _update_view

    def get_jobs(self) -> List[JobConfig]:
        """Return the ready jobs in table order."""
        return [self._jobs[i] for i in self._tree.get_children() if i in self._jobs]

    def get_job_ids(self):
        return [i for i in self._tree.get_children() if i in self._jobs]

    def get_job_statuses(self):
        return [self._status.get(i, "pending") for i in self.get_job_ids()]

    def set_job_status(self, index, status):
        """Update the status column (and row color tag) for job #index,
        or for a job given by its row id."""
        ids = self.get_job_ids()
        if isinstance(index, int):
            if not 0 <= index < len(ids):
                return
            iid = ids[index]
        else:
            iid = index
        if not self._tree.exists(iid):
            return
        values = list(self._tree.item(iid, "values"))
        values[2] = _STATUS_TEXT.get(status, status)
        tags = (status,) if status in ("running", "done", "error", "cancelled") else ()
        self._tree.item(iid, values=values, tags=tags)
        self._status[iid] = status
        if status == "running":
            self._tree.see(iid)
        self._update_view()

    @property
    def file_path(self):
        """Backward compatibility: return first file path or empty string."""
        jobs = self.get_jobs()
        return jobs[0].file_path if jobs else ""

    @property
    def has_jobs(self):
        return len(self._jobs) > 0

    @property
    def is_checking(self):
        return bool(self._pending_probes) or self._dialog_open

    @property
    def dnd_available(self):
        return self._dnd_available

    def pulse(self):
        """Draw attention to step 1 (Start was pressed with nothing queued)."""
        if self._drop_zone.winfo_manager():
            self._drop_zone.pulse()
        self.focus_add()

    def focus_add(self):
        (self._add_btn if self._tree.get_children() else self._drop_zone).focus_set()

    def set_enabled(self, enabled):
        """Enable/disable queue editing (locked while a batch is running)."""
        self._enabled = enabled
        state = ["!disabled"] if enabled else ["disabled"]
        for btn in (self._add_btn, self._remove_btn, self._clear_btn):
            btn.state(state)


def _short_track(label):
    """'Track 1 · jpn · AC3' -> 'Track 1 · jpn' for the narrow table column
    (the full label is in the row tooltip and the progress text)."""
    parts = [p.strip() for p in label.split("\u00b7")] if label else []
    return " \u00b7 ".join(parts[:2])


def winnative_scale(widget):
    try:
        return widget.winfo_fpixels("1i") / 96.0
    except tk.TclError:
        return 1.0
