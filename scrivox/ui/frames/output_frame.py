"""Output format dropdown and output path selection."""

import os
import tkinter as tk
from tkinter import ttk, filedialog

from ...core.constants import OUTPUT_FORMATS
from ..theme import COLORS


class OutputFrame(ttk.LabelFrame):
    """Format combo and output path browse."""

    def __init__(self, parent, config_manager=None, **kwargs):
        super().__init__(parent, text="OUTPUT", **kwargs)
        self.config_manager = config_manager

        self.format_var = tk.StringVar(value="txt")
        self.output_path_var = tk.StringVar()

        self._build()

    def _build(self):
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row, text="Format:").pack(side=tk.LEFT)
        fmt_combo = ttk.Combobox(row, textvariable=self.format_var,
                                  values=OUTPUT_FORMATS, state="readonly", width=12)
        fmt_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(row, text="Path:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.output_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4))
        ttk.Button(row, text="...", command=self._browse_output, width=3).pack(side=tk.RIGHT)

        ttk.Label(self, text="Leave path blank for auto-naming or console output",
                  style="Dim.TLabel").pack(padx=8, pady=(0, 6), anchor=tk.W)

    def _browse_output(self):
        fmt = self.format_var.get()
        ext_map = {
            "txt": ("Text files", "*.txt"),
            "md": ("Markdown files", "*.md"),
            "srt": ("SRT subtitles", "*.srt"),
            "vtt": ("VTT subtitles", "*.vtt"),
            "json": ("JSON files", "*.json"),
            "tsv": ("TSV files", "*.tsv"),
        }
        ft_name, ft_pattern = ext_map.get(fmt, ("All files", "*.*"))

        initial_dir = ""
        if self.config_manager:
            initial_dir = self.config_manager.get("ui", "last_output_dir", "")

        path = filedialog.asksaveasfilename(
            title="Save output as",
            initialdir=initial_dir or None,
            defaultextension=f".{fmt}",
            filetypes=[(ft_name, ft_pattern), ("All files", "*.*")],
        )
        if path:
            self.output_path_var.set(path)
            if self.config_manager:
                self.config_manager.set("ui", "last_output_dir", os.path.dirname(path))
