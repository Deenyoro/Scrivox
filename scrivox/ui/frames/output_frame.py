"""Step 3 "Save to": output format, destination folder and (for a single
file) the output name."""

import os
import tkinter as tk
from tkinter import ttk, filedialog

from ...core.constants import OUTPUT_FORMATS
from ..output_paths import default_output_path, format_from_label, format_label
from ..theme import SP_S, SP_XS
from ..widgets import LinkLabel, ToolTip, WrappingLabel, ellipsize

SAME_FOLDER = "Same folder as each file"

_FILETYPES = {
    "txt": ("Text files", "*.txt"),
    "md": ("Markdown", "*.md"),
    "srt": ("SRT subtitles", "*.srt"),
    "vtt": ("WebVTT subtitles", "*.vtt"),
    "json": ("JSON files", "*.json"),
    "tsv": ("TSV files", "*.tsv"),
}


class OutputFrame(ttk.Frame):
    """Format combo, output folder picker and a "Saves as ..." preview.

    `format_var` holds the raw format id ("txt", "srt", ...) as before; the
    combo shows a descriptive label for each. `output_dir_var` is the chosen
    folder ("" = next to each input). `output_path_var` is an explicit output
    *file*: set by "Rename..." for a single-file run, and always honoured.
    """

    def __init__(self, parent, config_manager=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.config_manager = config_manager

        self.format_var = tk.StringVar(value="txt")
        self.output_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.subtitle_speakers_var = tk.BooleanVar(value=False)
        self._fmt_display_var = tk.StringVar(value=format_label("txt"))
        self._jobs = []           # [(file_path, audio_track)] for the name preview
        self._named_for = None    # the single job output_path_var was chosen for
        self._enabled = True

        self._build()
        if config_manager:
            saved = config_manager.get("ui", "output_dir", "") or ""
            if saved and os.path.isdir(saved):
                self.output_dir_var.set(saved)
        self._update_folder_display()

    def _build(self):
        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=(0, SP_XS))
        ttk.Label(row, text="Format").pack(side=tk.LEFT)
        self._fmt_combo = ttk.Combobox(row, textvariable=self._fmt_display_var,
                                       values=[format_label(f) for f in OUTPUT_FORMATS],
                                       state="readonly", width=12, style="Wide.TCombobox")
        self._fmt_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        self._fmt_combo.bind("<<ComboboxSelected>>", self._on_format_picked)
        self.format_var.trace_add("write", lambda *a: self._on_format_change())

        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=(0, SP_XS))
        self._folder_row = row
        ttk.Label(row, text="Folder").pack(side=tk.LEFT)
        self._browse_btn = ttk.Button(row, text="Browse…", style="Small.TButton",
                                      command=self._browse_output)
        self._browse_btn.pack(side=tk.RIGHT)
        self._reset_btn = ttk.Button(row, text="✕", style="Small.TButton", width=2,
                                     command=self._reset_folder)
        ToolTip(self._reset_btn, "Save next to each original file again")
        self._folder_display = ttk.Label(row, text=SAME_FOLDER, anchor=tk.W)
        self._folder_display.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SP_S, SP_XS))
        self._folder_display.bind("<Configure>", lambda e: self._fit_folder_text(), add="+")
        self._folder_tip = ToolTip(self._folder_display, "")

        self._sub_cb = ttk.Checkbutton(self, text="Put speaker names in subtitles",
                                       variable=self.subtitle_speakers_var)
        ToolTip(self._sub_cb, "Prefix each subtitle line with the speaker label\n"
                              "(needs Identify speakers)")

        # "Saves as interview_transcript.txt   Rename..."
        name_row = ttk.Frame(self)
        name_row.pack(fill=tk.X)
        self._name_row = name_row
        self._rename_link = LinkLabel(name_row, text="Rename…", command=self._rename)
        self._default_link = LinkLabel(name_row, text="Use default name",
                                       command=self._clear_name)
        self._save_hint = WrappingLabel(name_row, text="", style="Dim.TLabel", justify=tk.LEFT)
        self._save_hint.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._on_format_change()

    # ── Format ──

    def _on_format_picked(self, event=None):
        fmt = format_from_label(self._fmt_display_var.get())
        if fmt in OUTPUT_FORMATS and fmt != self.format_var.get():
            self.format_var.set(fmt)
        self._fmt_combo.selection_clear()

    def _on_format_change(self):
        fmt = self.format_var.get()
        label = format_label(fmt)
        if self._fmt_display_var.get() != label:
            self._fmt_display_var.set(label)
        if fmt in ("srt", "vtt"):
            self._sub_cb.pack(anchor=tk.W, pady=(0, SP_XS), before=self._name_row)
        else:
            self._sub_cb.pack_forget()
        # A name picked with "Rename..." follows the format
        explicit = self.output_path_var.get()
        if explicit and fmt in OUTPUT_FORMATS:
            base, ext = os.path.splitext(explicit)
            new_ext = f".{fmt}"
            if ext.lower() != new_ext:
                self.output_path_var.set(base + new_ext)
        self._update_name_preview()

    # ── Folder ──

    def _fit_folder_text(self):
        folder = self.output_dir_var.get()
        text = folder or SAME_FOLDER
        width = self._folder_display.winfo_width()
        if folder and width > 20:
            import tkinter.font as tkfont
            font = tkfont.nametofont("TkDefaultFont")
            try:
                font = tkfont.Font(font=ttk.Style(self).lookup("TLabel", "font") or font)
            except tk.TclError:
                pass
            text = ellipsize(folder, font, width - 4, keep_ext=False)
        if self._folder_display.cget("text") != text:
            self._folder_display.configure(text=text)

    def _update_folder_display(self):
        folder = self.output_dir_var.get()
        if folder:
            self._folder_tip.text = folder
            if not self._reset_btn.winfo_manager():
                self._reset_btn.pack(side=tk.RIGHT, padx=(0, SP_XS), before=self._browse_btn)
        else:
            self._folder_tip.text = ""
            self._reset_btn.pack_forget()
        self._fit_folder_text()
        self._update_name_preview()

    def _reset_folder(self):
        self.output_dir_var.set("")
        if self.config_manager:
            self.config_manager.set("ui", "output_dir", "")
        self._update_folder_display()

    def _browse_output(self):
        initial_dir = self.output_dir_var.get()
        if not initial_dir and self.config_manager:
            initial_dir = self.config_manager.get("ui", "last_output_dir", "")
        path = filedialog.askdirectory(
            parent=self.winfo_toplevel(),
            title="Choose where to save transcripts",
            initialdir=initial_dir or None,
            mustexist=True,
        )
        if path:
            self.output_dir_var.set(os.path.normpath(path))
            if self.config_manager:
                self.config_manager.set("ui", "last_output_dir", os.path.normpath(path))
                self.config_manager.set("ui", "output_dir", os.path.normpath(path))
            # A renamed single-file output follows the new folder
            explicit = self.output_path_var.get()
            if explicit:
                self.output_path_var.set(os.path.join(os.path.normpath(path),
                                                      os.path.basename(explicit)))
            self._update_folder_display()

    # ── Output name ──

    def set_jobs(self, jobs):
        """Tell the frame which files are queued ([(path, audio_track)]), for
        the "Saves as ..." line. A name chosen for one file is dropped when
        the queue no longer holds exactly that file."""
        self._jobs = list(jobs)
        if self.output_path_var.get() and (len(self._jobs) != 1
                                           or self._jobs[0] != self._named_for):
            self.output_path_var.set("")
            self._named_for = None
        self._update_name_preview()

    def planned_name(self):
        """Output file name for a single queued file, or None."""
        if len(self._jobs) != 1:
            return None
        explicit = self.output_path_var.get()
        if explicit:
            return explicit
        path, track = self._jobs[0]
        return default_output_path(path, self.format_var.get(), track,
                                   out_dir=self.output_dir_var.get() or None)

    def _update_name_preview(self):
        if not hasattr(self, "_save_hint"):
            return
        self._rename_link.pack_forget()
        self._default_link.pack_forget()
        planned = self.planned_name()
        if planned:
            self._save_hint.configure(text=f"Saves as {os.path.basename(planned)}")
            link = self._default_link if self.output_path_var.get() else self._rename_link
            if self._enabled:
                link.pack(side=tk.RIGHT, anchor=tk.N, padx=(SP_S, 0), before=self._save_hint)
        elif len(self._jobs) > 1:
            fmt = self.format_var.get()
            ext = "_transcript.txt" if fmt == "txt" else f".{fmt}"
            self._save_hint.configure(
                text=f"Each file is saved as <name>{ext}. Nothing is overwritten.")
        else:
            fmt = self.format_var.get()
            example = "interview_transcript.txt" if fmt == "txt" else f"interview.{fmt}"
            self._save_hint.configure(
                text=f"Named after the original, e.g. {example}. Nothing is overwritten.")

    def _rename(self):
        if len(self._jobs) != 1 or not self._enabled:
            return
        planned = self.planned_name()
        fmt = self.format_var.get()
        ft = _FILETYPES.get(fmt, ("All files", "*.*"))
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Save the transcript as",
            initialdir=os.path.dirname(planned),
            initialfile=os.path.basename(planned),
            defaultextension=f".{fmt}",
            filetypes=[ft, ("All files", "*.*")],
        )
        if path:
            self._named_for = self._jobs[0]
            self.output_path_var.set(os.path.normpath(path))
            self._update_name_preview()

    def _clear_name(self):
        self.output_path_var.set("")
        self._named_for = None
        self._update_name_preview()

    def consume_explicit_name(self):
        """A chosen name is used for one run only, so running again never
        overwrites that file."""
        if self.output_path_var.get():
            self.output_path_var.set("")
            self._named_for = None

    def set_enabled(self, enabled):
        self._enabled = enabled
        state = ["!disabled"] if enabled else ["disabled"]
        for w in (self._fmt_combo, self._browse_btn, self._reset_btn, self._sub_cb):
            w.state(state)
        if enabled:
            self._fmt_combo.state(["readonly"])
        self._update_name_preview()
