"""Job queue frame with multi-file selection and audio track support."""

import os
import tkinter as tk
from tkinter import ttk, filedialog
from dataclasses import dataclass, field
from typing import List

from ...core.constants import VIDEO_EXTENSIONS, AUDIO_EXTENSIONS
from ...core.media import list_audio_tracks
from ..theme import COLORS, FONTS


@dataclass
class JobConfig:
    """Configuration for a single transcription job."""
    file_path: str
    audio_track: int = 0
    track_label: str = ""       # "Track 0 (English)"
    language_override: str = ""  # Override auto-detect


class QueueFrame(ttk.LabelFrame):
    """Job queue table with multi-file browse, track selection, and drag-drop zone."""

    def __init__(self, parent, config_manager=None, on_tracks_needed=None, **kwargs):
        super().__init__(parent, text="JOB QUEUE", **kwargs)
        self.config_manager = config_manager
        self._on_tracks_needed = on_tracks_needed  # callback(filepath, tracks) -> selected indices
        self._jobs: List[JobConfig] = []
        self._build()

    def _build(self):
        # ── Button bar ──
        btn_bar = ttk.Frame(self)
        btn_bar.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Button(btn_bar, text="Add Files...", command=self.browse_files).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_bar, text="Remove", command=self._remove_selected).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_bar, text="Clear All", command=self._clear_all).pack(side=tk.LEFT)

        # ── Treeview (job table) ──
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        columns = ("file", "track", "lang", "status")
        self._tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                   height=5, selectmode="extended")
        self._tree.heading("file", text="File")
        self._tree.heading("track", text="Track")
        self._tree.heading("lang", text="Lang")
        self._tree.heading("status", text="Status")

        self._tree.column("file", width=160, minwidth=100, stretch=True)
        self._tree.column("track", width=80, minwidth=50, stretch=False)
        self._tree.column("lang", width=40, minwidth=35, stretch=False)
        self._tree.column("status", width=60, minwidth=50, stretch=False)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Style the treeview for dark theme
        style = ttk.Style()
        style.configure("Treeview",
                        background=COLORS["bg_secondary"],
                        foreground=COLORS["fg"],
                        fieldbackground=COLORS["bg_secondary"],
                        font=FONTS["small"])
        style.configure("Treeview.Heading",
                        background=COLORS["border"],
                        foreground=COLORS["fg"],
                        font=FONTS["small"])
        style.map("Treeview",
                  background=[("selected", COLORS["selection"])],
                  foreground=[("selected", COLORS["fg_bright"])])

        # ── Drop zone hint ──
        self._hint_label = ttk.Label(self, text="Drag files here or click \"Add Files...\"",
                                      style="Dim.TLabel")
        self._hint_label.pack(padx=8, pady=(0, 6))

        # Try to enable drag-and-drop
        self._setup_dnd()

    def _setup_dnd(self):
        """Try to set up tkinterdnd2 drag-and-drop. Silently fails if not available."""
        try:
            from tkinterdnd2 import DND_FILES
            self._tree.drop_target_register(DND_FILES)
            self._tree.dnd_bind("<<Drop>>", self._on_drop)
        except (ImportError, Exception):
            pass  # tkinterdnd2 not available — browse-only mode

    def _on_drop(self, event):
        """Handle dropped files."""
        # tkinterdnd2 wraps paths with spaces in braces: {C:\path with spaces\file.mp4}
        raw = event.data
        paths = []
        i = 0
        while i < len(raw):
            if raw[i] == "{":
                end = raw.index("}", i)
                paths.append(raw[i + 1:end])
                i = end + 2  # skip closing brace and space
            elif raw[i] == " ":
                i += 1
            else:
                end = raw.find(" ", i)
                if end == -1:
                    end = len(raw)
                paths.append(raw[i:end])
                i = end + 1

        for path in paths:
            if os.path.isfile(path):
                self._add_file(path)

    def browse_files(self):
        """Open multi-file dialog and add selected files to the queue."""
        all_exts = sorted(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)
        ext_pattern = " ".join(f"*{e}" for e in all_exts)

        initial_dir = ""
        if self.config_manager:
            initial_dir = self.config_manager.get("ui", "last_input_dir", "")

        paths = filedialog.askopenfilenames(
            title="Select audio or video files",
            initialdir=initial_dir or None,
            filetypes=[
                ("Media files", ext_pattern),
                ("Video files", " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))),
                ("Audio files", " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))),
                ("All files", "*.*"),
            ],
        )
        if paths:
            if self.config_manager:
                self.config_manager.set("ui", "last_input_dir", os.path.dirname(paths[0]))
            for path in paths:
                self._add_file(path)

    def _add_file(self, path):
        """Add a file to the job queue, prompting for track selection if multi-track video."""
        tracks = list_audio_tracks(path)

        if len(tracks) > 1 and self._on_tracks_needed:
            selected_indices = self._on_tracks_needed(path, tracks)
            if not selected_indices:
                return  # user cancelled
            for idx in selected_indices:
                track = tracks[idx] if idx < len(tracks) else tracks[0]
                label = self._format_track_label(track)
                job = JobConfig(
                    file_path=path,
                    audio_track=idx,
                    track_label=label,
                )
                self._jobs.append(job)
                self._insert_tree_row(job)
        else:
            # Single track or no tracks detected
            label = ""
            if tracks:
                label = self._format_track_label(tracks[0])
            job = JobConfig(file_path=path, audio_track=0, track_label=label)
            self._jobs.append(job)
            self._insert_tree_row(job)

        self._update_hint()

    def _format_track_label(self, track):
        parts = [f"Track {track['index']}"]
        if track.get("language"):
            parts.append(track["language"])
        if track.get("codec"):
            parts.append(track["codec"].upper())
        return " ".join(parts)

    def _insert_tree_row(self, job):
        """Insert a job row into the treeview."""
        filename = os.path.basename(job.file_path)
        track = job.track_label or "\u2014"
        lang = job.language_override or "auto"
        self._tree.insert("", tk.END, values=(filename, track, lang, "Pending"))

    def _remove_selected(self):
        """Remove selected jobs from the queue."""
        selected = self._tree.selection()
        # Get indices (items are in order)
        all_items = self._tree.get_children()
        indices = [all_items.index(item) for item in selected if item in all_items]
        for idx in sorted(indices, reverse=True):
            if idx < len(self._jobs):
                self._jobs.pop(idx)
            self._tree.delete(all_items[idx])
        self._update_hint()

    def _clear_all(self):
        """Remove all jobs from the queue."""
        self._jobs.clear()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._update_hint()

    def _update_hint(self):
        """Show/hide the drop zone hint."""
        if self._jobs:
            self._hint_label.configure(text=f"{len(self._jobs)} job(s) queued")
        else:
            self._hint_label.configure(text="Drag files here or click \"Add Files...\"")

    def get_jobs(self) -> List[JobConfig]:
        """Return the current job list."""
        return list(self._jobs)

    def set_job_status(self, index, status):
        """Update the status column for a job."""
        items = self._tree.get_children()
        if index < len(items):
            item = items[index]
            values = list(self._tree.item(item, "values"))
            status_map = {
                "pending": "Pending",
                "running": "Running...",
                "done": "Done",
                "error": "Error",
            }
            values[3] = status_map.get(status, status)
            self._tree.item(item, values=values)

    @property
    def file_path(self):
        """Backward compatibility: return first file path or empty string."""
        if self._jobs:
            return self._jobs[0].file_path
        return ""

    @property
    def has_jobs(self):
        return len(self._jobs) > 0
