"""Settings window: AI service keys and advanced options, out of the way of
the main three-step flow. Created once (hidden) so its variables always
exist; showing it just brings the window up on the requested tab."""

import os
import tkinter as tk
from tkinter import ttk

from .. import winnative
from ..theme import COLORS, px


class SettingsDialog(tk.Toplevel):
    TAB_KEYS = "keys"
    TAB_ADVANCED = "advanced"

    def __init__(self, parent, config_manager, show_keys=True, show_diarization=True):
        super().__init__(parent)
        self.withdraw()
        self.title("Scrivox Settings")
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self.bind("<Escape>", lambda e: (self.hide(), "break")[1])
        self._config_manager = config_manager
        self._parent = parent
        self._placed = False

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=px(12), pady=(px(12), 0))

        from ..frames.models_frame import ModelsFrame
        self.api_frame = None
        self._tabs = {}
        if show_keys:
            from ..frames.api_frame import ApiFrame
            self.api_frame = ApiFrame(self.notebook, config_manager=config_manager)
            self.notebook.add(self.api_frame, text="AI services", underline=0)
            self._tabs[self.TAB_KEYS] = self.api_frame
        self.models_frame = ModelsFrame(self.notebook, show_diarization=show_diarization)
        self.notebook.add(self.models_frame, text="Advanced", underline=0)
        self._tabs[self.TAB_ADVANCED] = self.models_frame

        btns = ttk.Frame(self, padding=(px(12), px(12)))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Open settings folder",
                   command=self._open_config_folder).pack(side=tk.LEFT)
        self._done = ttk.Button(btns, text="Done", style="Accent.TButton", command=self.hide)
        self._done.pack(side=tk.RIGHT)
        self.bind("<Return>", self._on_return)
        self.notebook.enable_traversal()

    def _on_return(self, event):
        # Enter in an entry field confirms, like a normal dialog
        if not isinstance(event.widget, (ttk.Button,)):
            self.hide()
            return "break"

    def _open_config_folder(self):
        path = getattr(self._config_manager, "path", "")
        folder = os.path.dirname(path) if path else ""
        if folder:
            winnative.reveal_in_folder(path if os.path.exists(path) else folder)

    def show(self, tab=None):
        if tab in self._tabs:
            self.notebook.select(self._tabs[tab])
        if not self._placed:
            self.update_idletasks()
            s = self._parent.winfo_fpixels("1i") / 96.0
            w = min(max(self.winfo_reqwidth(), int(540 * s)), int(640 * s))
            h = self.winfo_reqheight()
            x = self._parent.winfo_rootx() + (self._parent.winfo_width() - w) // 2
            y = self._parent.winfo_rooty() + max(0, (self._parent.winfo_height() - h) // 3)
            self.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")
            self.minsize(int(460 * s), int(360 * s))
            self._placed = True
        self.deiconify()
        winnative.use_dark_title_bar(self)
        self.lift()
        self.focus_set()
        current = self.notebook.select()
        if current:
            target = self.nametowidget(current)
            if target is self.api_frame:
                target = self.api_frame.first_empty_field()
            self._focus_later(target, 50)

    def _focus_later(self, widget, ms):
        """Focus `widget` once the window is shown; the timer is cancelled if
        the dialog goes away first (no stale Tcl callbacks on exit)."""
        if getattr(self, "_focus_after", None):
            self.after_cancel(self._focus_after)

        def _go():
            self._focus_after = None
            try:
                widget.focus_set()
            except tk.TclError:
                pass
        self._focus_after = self.after(ms, _go)
        if not getattr(self, "_destroy_bound", False):
            self.bind("<Destroy>", self._on_destroy, add="+")
            self._destroy_bound = True

    def _on_destroy(self, event):
        if event.widget is self and getattr(self, "_focus_after", None):
            try:
                self.after_cancel(self._focus_after)
            except tk.TclError:
                pass
            self._focus_after = None

    def hide(self):
        self.withdraw()
        try:
            self._parent.focus_set()
        except tk.TclError:
            pass
        self.event_generate("<<SettingsClosed>>")

    def focus_widget(self, widget):
        """Open on the tab containing `widget` and focus it."""
        for tab in self._tabs.values():
            w = widget
            while w is not None:
                if w is tab:
                    self.show()
                    self.notebook.select(tab)
                    self._focus_later(widget, 60)
                    return True
                w = w.master
        return False

