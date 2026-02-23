"""Input file selection frame with media info display."""

import os
import tkinter as tk
from tkinter import ttk, filedialog

from ...core.constants import VIDEO_EXTENSIONS, AUDIO_EXTENSIONS
from ...core.media import get_media_duration, has_video_stream
from ...core.formatter import format_timestamp_human
from ..theme import COLORS, FONTS


class InputFrame(ttk.LabelFrame):
    """File browse + media info display."""

    def __init__(self, parent, config_manager=None, **kwargs):
        super().__init__(parent, text="INPUT FILE", **kwargs)
        self.config_manager = config_manager
        self._file_path = tk.StringVar()
        self._media_info = tk.StringVar(value="No file selected")
        self._is_video = False
        self._duration = None

        self._build()

    def _build(self):
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))

        self._path_entry = ttk.Entry(row, textvariable=self._file_path, state="readonly")
        self._path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        browse_btn = ttk.Button(row, text="Browse...", command=self._browse)
        browse_btn.pack(side=tk.RIGHT)

        info_label = ttk.Label(self, textvariable=self._media_info, style="Dim.TLabel")
        info_label.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _browse(self):
        all_exts = sorted(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)
        ext_pattern = " ".join(f"*{e}" for e in all_exts)

        initial_dir = ""
        if self.config_manager:
            initial_dir = self.config_manager.get("ui", "last_input_dir", "")

        path = filedialog.askopenfilename(
            title="Select audio or video file",
            initialdir=initial_dir or None,
            filetypes=[
                ("Media files", ext_pattern),
                ("Video files", " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))),
                ("Audio files", " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.set_file(path)
            if self.config_manager:
                self.config_manager.set("ui", "last_input_dir", os.path.dirname(path))

    def set_file(self, path):
        """Set the input file path and update media info."""
        self._file_path.set(path)
        self._analyze_file(path)

    def _analyze_file(self, path):
        if not path or not os.path.isfile(path):
            self._media_info.set("No file selected")
            self._is_video = False
            self._duration = None
            return

        ext = os.path.splitext(path)[1].lower()
        self._is_video = has_video_stream(path)
        self._duration = get_media_duration(path)

        size_mb = os.path.getsize(path) / (1024 * 1024)
        media_type = "Video" if self._is_video else "Audio"
        duration_str = format_timestamp_human(self._duration) if self._duration else "unknown"
        self._media_info.set(f"{media_type} ({ext.upper().strip('.')}) | Duration: {duration_str} | {size_mb:.1f} MB")

    @property
    def file_path(self):
        return self._file_path.get()

    @property
    def is_video(self):
        return self._is_video

    @property
    def duration(self):
        return self._duration
