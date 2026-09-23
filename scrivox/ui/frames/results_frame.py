"""Transcript view with a completion bar: where the file was saved, and one
click to open it, show it in Explorer, copy the text or save a copy."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..theme import COLORS, FONTS, SP_S, SP_XS, px
from ..widgets import ToolTip, TreeTextFitter, fit_column_to_labels
from .. import winnative

# Tk's Text widget handles a few MB fine; only truly huge outputs are cut
MAX_PREVIEW_CHARS = 1_000_000

_FILETYPES = {
    "txt": ("Text files", "*.txt"),
    "md": ("Markdown", "*.md"),
    "srt": ("SRT subtitles", "*.srt"),
    "vtt": ("WebVTT subtitles", "*.vtt"),
    "json": ("JSON files", "*.json"),
    "tsv": ("TSV files", "*.tsv"),
}


class ResultsFrame(ttk.Frame):
    """Results display with Open, Show in folder, Copy and Save as."""

    PLACEHOLDER = ("Your transcript will appear here.\n\n"
                   "It is also saved as a file next to the original "
                   "(or in the folder you chose), and you can open it from the bar below.")

    def __init__(self, parent, config_manager=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.config_manager = config_manager
        self._output_path = None
        self._full_text = ""
        self._format = "txt"
        self._batch = []  # dicts: input, output_path, text, error
        self._build()
        self._show_placeholder()

    def _build(self):
        # Batch results table (only shown for multi-file runs)
        self._batch_frame = ttk.Frame(self)
        self._batch_tree = ttk.Treeview(self._batch_frame, columns=("file", "output", "status"),
                                        show="headings", height=3, selectmode="browse")
        self._batch_tree.heading("file", text="File", anchor=tk.W)
        self._batch_tree.heading("output", text="Saved as", anchor=tk.W)
        self._batch_tree.heading("status", text="Result", anchor=tk.W)
        s = self.winfo_fpixels("1i") / 96.0
        self._batch_tree.column("file", width=int(200 * s), stretch=True)
        self._batch_tree.column("output", width=int(240 * s), stretch=True)
        fit_column_to_labels(self._batch_tree, "status", ("Done", "Failed"), "Result")
        self._batch_fitter = TreeTextFitter(self._batch_tree, ("file", "output"))
        self._batch_tree.tag_configure("error", foreground=COLORS["error"])
        self._batch_tree.tag_configure("done", foreground=COLORS["fg"])
        self._batch_tree.bind("<<TreeviewSelect>>", self._on_batch_select)
        self._batch_tree.bind("<Double-1>", lambda e: self._open_file())
        self._batch_tree.bind("<Return>", lambda e: self._open_file())
        bsb = ttk.Scrollbar(self._batch_frame, orient=tk.VERTICAL, command=self._batch_tree.yview)
        self._batch_tree.configure(yscrollcommand=bsb.set)
        self._batch_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Transcript text
        self._text_container = ttk.Frame(self)
        self._text_container.pack(fill=tk.BOTH, expand=True)
        self.text_widget = tk.Text(
            self._text_container,
            wrap=tk.WORD,
            font=FONTS["mono_small"],
            bg=COLORS["log_bg"],
            fg=COLORS["fg"],
            insertbackground=COLORS["fg"],
            selectbackground=COLORS["accent"],
            selectforeground=COLORS["button_fg"],
            borderwidth=0,
            highlightthickness=0,
            padx=int(10 * s), pady=int(8 * s),
            state=tk.DISABLED,
            height=8,
        )
        scrollbar = ttk.Scrollbar(self._text_container, orient=tk.VERTICAL,
                                  command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Read-only but selectable/copyable with the keyboard
        self.text_widget.bind("<1>", lambda e: self.text_widget.focus_set())

        # Dim style for the empty-state placeholder text
        self.text_widget.tag_configure("placeholder", foreground=COLORS["fg_dim"],
                                       font=FONTS["body"])

        # ── Completion / action bar ──
        bar = ttk.Frame(self, style="Bar.TFrame", padding=(px(12), px(8)))
        bar.pack(fill=tk.X, pady=(SP_S, 0))
        self._bar = bar
        self._saved_label = ttk.Label(bar, text="", style="BarDim.TLabel", anchor=tk.W)
        self._saved_label.pack(fill=tk.X, pady=(0, SP_XS))
        self._saved_tip = ToolTip(self._saved_label, "")

        btn_row = ttk.Frame(bar, style="Bar.TFrame")
        self._btn_row = btn_row  # shown once there is a result
        self._openfile_btn = ttk.Button(btn_row, text="Open", command=self._open_file,
                                        state=tk.DISABLED)
        self._openfile_btn.pack(side=tk.LEFT, padx=(0, SP_XS))
        self._open_btn = ttk.Button(btn_row, text="Show in folder", command=self._open_folder,
                                    state=tk.DISABLED)
        self._open_btn.pack(side=tk.LEFT, padx=(0, SP_XS))
        self._copy_btn = ttk.Button(btn_row, text="Copy", command=self._copy,
                                    state=tk.DISABLED)
        self._copy_btn.pack(side=tk.LEFT, padx=(0, SP_XS))
        self._save_btn = ttk.Button(btn_row, text="Save as…", command=self._save_as,
                                    state=tk.DISABLED)
        self._save_btn.pack(side=tk.LEFT)

    def _show_placeholder(self):
        """Show dim placeholder text in the empty results pane."""
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert("1.0", self.PLACEHOLDER, "placeholder")
        self.text_widget.configure(state=tk.DISABLED)
        self._saved_label.configure(text="No transcript yet", style="BarDim.TLabel")
        self._saved_tip.text = ""
        self._btn_row.pack_forget()

    def _set_text(self, text):
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        display_text = text
        if len(text) > MAX_PREVIEW_CHARS:
            display_text = (text[:MAX_PREVIEW_CHARS]
                            + f"\n\n… preview ends here ({len(text):,} characters in total). "
                              "Open the saved file to see everything.")
        self.text_widget.insert("1.0", display_text)
        self.text_widget.configure(state=tk.DISABLED)
        self.text_widget.yview_moveto(0)

    def _update_bar(self):
        path = self._output_path
        saved = bool(path and os.path.isfile(path))
        has_text = bool(self._full_text)
        if saved:
            self._saved_label.configure(text=f"✓  Saved as {os.path.basename(path)}",
                                        style="BarSuccess.TLabel")
            self._saved_tip.text = path
        elif has_text:
            self._saved_label.configure(text="Not saved to a file. Use Save as to keep it.",
                                        style="BarDim.TLabel")
            self._saved_tip.text = ""
        state = tk.NORMAL if saved else tk.DISABLED
        self._openfile_btn.configure(state=state)
        self._open_btn.configure(state=state)
        text_state = tk.NORMAL if has_text else tk.DISABLED
        self._copy_btn.configure(state=text_state)
        self._save_btn.configure(state=text_state)
        if (saved or has_text) and not self._btn_row.winfo_manager():
            self._btn_row.pack(fill=tk.X)
        elif not (saved or has_text):
            self._btn_row.pack_forget()

    def show_result(self, text, output_path=None, fmt=None):
        """Display one result and enable the actions."""
        self._batch = []
        self._batch_frame.pack_forget()
        self._output_path = output_path
        self._full_text = text
        if fmt:
            self._format = fmt
        self._set_text(text)
        self._update_bar()

    def show_batch(self, items, fmt=None):
        """Show a multi-file run: a table of results plus a preview of the
        selected one. `items` are dicts with input, output_path, text, error."""
        if fmt:
            self._format = fmt
        self._batch = list(items)
        for iid in self._batch_tree.get_children():
            self._batch_fitter.forget(iid)
        self._batch_tree.delete(*self._batch_tree.get_children())
        for i, item in enumerate(self._batch):
            if item.get("error"):
                output, status, tag = "—", "Failed", "error"
            else:
                out = item.get("output_path")
                output = os.path.basename(out) if out else "(not saved)"
                status, tag = "Done", "done"
            iid = self._batch_tree.insert("", tk.END, iid=str(i), values=("", "", status),
                                          tags=(tag,))
            self._batch_fitter.set(iid, "file", item["input"])
            self._batch_fitter.set(iid, "output", output)
        self._batch_frame.pack(fill=tk.X, pady=(0, SP_S), before=self._text_container)
        first_ok = next((i for i, it in enumerate(self._batch) if not it.get("error")), 0)
        if self._batch:
            self._batch_tree.selection_set(str(first_ok))
            self._batch_tree.focus(str(first_ok))
            self._select_batch_item(first_ok)
        ok = sum(1 for it in self._batch if not it.get("error"))
        if self._output_path and ok:
            self._saved_label.configure(
                text=f"✓  {ok} of {len(self._batch)} files saved · "
                     f"selected: {os.path.basename(self._output_path)}",
                style="BarSuccess.TLabel")

    def _on_batch_select(self, event=None):
        sel = self._batch_tree.selection()
        if sel:
            self._select_batch_item(int(sel[0]))

    def _select_batch_item(self, index):
        item = self._batch[index]
        self._output_path = item.get("output_path")
        self._full_text = item.get("text") or ""
        if item.get("error"):
            self._set_text(f"This file failed:\n\n{item['error']}")
        else:
            self._set_text(self._full_text)
        self._update_bar()
        if self._batch:
            ok = sum(1 for it in self._batch if not it.get("error"))
            if self._output_path and os.path.isfile(self._output_path):
                self._saved_label.configure(
                    text=f"✓  {ok} of {len(self._batch)} files saved · "
                         f"selected: {os.path.basename(self._output_path)}",
                    style="BarSuccess.TLabel")

    def clear(self):
        """Clear results (restores the empty-state placeholder)."""
        self._batch = []
        self._batch_frame.pack_forget()
        self._show_placeholder()
        self._output_path = None
        self._full_text = ""
        self._update_bar()
        self._saved_label.configure(text="No transcript yet", style="BarDim.TLabel")
        self._copy_btn.configure(text="Copy")

    @property
    def output_path(self):
        return self._output_path

    def _copy(self):
        """Copy full text to clipboard with visual feedback."""
        text = self._full_text
        if not text:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
        except tk.TclError:
            return
        # Flash "Copied!" feedback
        self._copy_btn.configure(text="Copied!")
        self.after(1500, lambda: self._copy_btn.configure(text="Copy"))

    def _save_as(self):
        """Save a copy of the shown result, defaulting to its name and format."""
        if not self._full_text:
            messagebox.showinfo("Scrivox", "Transcribe a file first.",
                                parent=self.winfo_toplevel())
            return
        fmt = self._format if self._format in _FILETYPES else "txt"
        initial_dir, initial_file = "", ""
        if self._output_path:
            initial_dir = os.path.dirname(os.path.abspath(self._output_path))
            initial_file = os.path.basename(self._output_path)
        elif self.config_manager:
            initial_dir = self.config_manager.get("ui", "last_output_dir", "")
        filetypes = [_FILETYPES[fmt]] + [ft for k, ft in _FILETYPES.items() if k != fmt]
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Save a copy of the transcript",
            initialdir=initial_dir or None,
            initialfile=initial_file or None,
            defaultextension=f".{fmt}",
            filetypes=filetypes + [("All files", "*.*")],
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._full_text)
            except Exception as e:
                messagebox.showerror("Scrivox", f"Couldn't save the file:\n{e}",
                                     parent=self.winfo_toplevel())
                return
            if self.config_manager:
                self.config_manager.set("ui", "last_output_dir", os.path.dirname(path))

    def _open_file(self):
        if self._output_path and os.path.isfile(self._output_path):
            winnative.open_path(self._output_path)

    def _open_folder(self):
        """Open the folder containing the output file, with the file selected."""
        if self._output_path and os.path.isfile(self._output_path):
            winnative.reveal_in_folder(self._output_path)
