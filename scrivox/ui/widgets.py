"""Reusable UI widgets for Scrivox."""

import tkinter as tk
from tkinter import ttk


class AutocompleteCombobox(ttk.Combobox):
    """Combobox with type-to-filter autocomplete popup.

    As you type, a popup shows matching items below the entry. Click an item
    or use Up/Down + Enter to select. The popup never steals focus, so you
    can keep typing to refine matches. Escape closes the popup. The native
    dropdown arrow still works for browsing the full (or filtered) list.

    multi_value=True: comma-separated input where filtering and selection
    operate on the token after the last comma.
    """

    # Popup colors — imported lazily from theme on first use
    _theme_loaded = False
    _colors = {
        "bg": "#313244",
        "fg": "#cdd6f4",
        "select_bg": "#89b4fa",
        "select_fg": "#1e1e2e",
        "border": "#45475a",
    }

    def __init__(self, master=None, **kwargs):
        self._all_values = list(kwargs.pop("values", []))
        self._multi_value = kwargs.pop("multi_value", False)
        super().__init__(master, values=self._all_values, **kwargs)
        self._debounce_id = None
        self._selecting = False  # guard against re-entrant filtering
        self._multi_prefix = ""  # text before the last comma in multi_value mode
        self._popup = None
        self._listbox = None
        self._popup_active = False
        self.bind("<KeyRelease>", self._on_key)
        self.bind("<Escape>", self._on_escape)
        self.bind("<<ComboboxSelected>>", self._on_combo_select)
        self._load_theme()

    @classmethod
    def _load_theme(cls):
        if cls._theme_loaded:
            return
        try:
            from .theme import COLORS
            cls._colors = {
                "bg": COLORS["entry_bg"],
                "fg": COLORS["entry_fg"],
                "select_bg": COLORS["accent"],
                "select_fg": COLORS["button_fg"],
                "border": COLORS["border"],
            }
        except Exception:
            pass
        cls._theme_loaded = True

    # ── Key handling ──

    def _on_key(self, event):
        # Popup navigation
        if event.keysym == "Down" and self._popup_active:
            self._navigate(1)
            return "break"
        if event.keysym == "Up" and self._popup_active:
            self._navigate(-1)
            return "break"
        if event.keysym == "Return" and self._popup_active:
            self._select_highlighted()
            return "break"
        # Ignore other nav/modifier keys
        if event.keysym in ("Up", "Down", "Left", "Right", "Return",
                            "Tab", "Escape", "Shift_L", "Shift_R",
                            "Control_L", "Control_R", "Alt_L", "Alt_R"):
            return
        if self._selecting:
            return
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(50, self._filter)

    # ── Filtering ──

    def _filter(self):
        self._debounce_id = None
        if self._selecting:
            return
        full = self.get()
        if self._multi_value and "," in full:
            text = full.rsplit(",", 1)[1].strip().lower()
            self._multi_prefix = full.rsplit(",", 1)[0].strip()
        else:
            text = full.lower()
            self._multi_prefix = ""
        if not text:
            self._close_popup()
            self["values"] = self._all_values
            return
        filtered = [v for v in self._all_values if text in v.lower()]
        self["values"] = filtered if filtered else self._all_values
        if filtered:
            self._show_popup(filtered)
        else:
            self._close_popup()

    # ── Custom autocomplete popup ──

    def _show_popup(self, items):
        if self._popup is not None:
            # Update existing popup content
            self._listbox.delete(0, tk.END)
            for item in items:
                self._listbox.insert(tk.END, item)
            self._position_popup(items)
            self._popup_active = True
            return
        self._popup = tk.Toplevel(self)
        self._popup.wm_overrideredirect(True)
        self._popup.wm_attributes('-topmost', True)
        # Prevent the popup from stealing focus
        self._popup.wm_attributes('-disabled', True)

        c = self._colors
        self._listbox = tk.Listbox(
            self._popup,
            selectmode=tk.SINGLE,
            activestyle='none',
            exportselection=False,
            bg=c["bg"], fg=c["fg"],
            selectbackground=c["select_bg"],
            selectforeground=c["select_fg"],
            highlightthickness=0,
            borderwidth=1,
            relief='solid',
            font=("Segoe UI", 9),
        )
        self._listbox.pack(fill=tk.BOTH, expand=True)

        for item in items:
            self._listbox.insert(tk.END, item)
        self._position_popup(items)
        self._popup_active = True

        # Bind clicks — need to handle manually since popup is disabled
        self._popup.wm_attributes('-disabled', False)
        self._listbox.bind("<Button-1>", self._on_popup_click)
        self._listbox.bind("<ButtonRelease-1>", self._on_popup_release)
        # Re-focus entry after any interaction with popup
        self._listbox.bind("<FocusIn>", lambda e: self.after_idle(self.focus_set))

    def _position_popup(self, items):
        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        width = self.winfo_width()
        num_visible = min(len(items), 8)
        # Estimate item height from font
        item_height = 20
        height = num_visible * item_height + 4
        self._popup.wm_geometry(f"{width}x{height}+{x}+{y}")

    def _navigate(self, direction):
        if not self._listbox:
            return
        sel = self._listbox.curselection()
        if sel:
            idx = sel[0] + direction
        else:
            idx = 0 if direction > 0 else self._listbox.size() - 1
        if 0 <= idx < self._listbox.size():
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(idx)
            self._listbox.see(idx)

    def _select_highlighted(self):
        if not self._listbox:
            return
        sel = self._listbox.curselection()
        if not sel:
            return
        self._apply_selection(self._listbox.get(sel[0]))

    def _on_popup_click(self, event):
        # Identify which item was clicked
        idx = self._listbox.nearest(event.y)
        if 0 <= idx < self._listbox.size():
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(idx)

    def _on_popup_release(self, event):
        sel = self._listbox.curselection()
        if not sel:
            return
        self._apply_selection(self._listbox.get(sel[0]))
        # Return focus to entry
        self.focus_set()
        self.icursor(tk.END)

    def _apply_selection(self, selected):
        self._selecting = True
        try:
            if self._multi_value and self._multi_prefix:
                new_text = f"{self._multi_prefix}, {selected}"
            else:
                new_text = selected
            self.set(new_text)
            self.icursor(tk.END)
            self._multi_prefix = ""
        finally:
            self._selecting = False
        self._close_popup()
        self["values"] = self._all_values

    def _close_popup(self, event=None):
        if self._popup is not None:
            self._popup.destroy()
            self._popup = None
            self._listbox = None
        self._popup_active = False

    # ── Native combobox dropdown selection ──

    def _on_combo_select(self, event):
        """Handle selection via the native dropdown (arrow button)."""
        if self._selecting:
            return
        self._selecting = True
        try:
            selected = self.get()
            if self._multi_value and self._multi_prefix:
                new_text = f"{self._multi_prefix}, {selected}"
                self.set(new_text)
                self.icursor(tk.END)
            self._multi_prefix = ""
        finally:
            self._selecting = False
        self["values"] = self._all_values

    def _on_escape(self, event=None):
        self._close_popup()
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        self["values"] = self._all_values
        self._multi_prefix = ""

    def _restore(self, event=None):
        """Restore full values (called externally or on Escape)."""
        self._on_escape(event)

    def set_values(self, values):
        """Update the full values list."""
        self._all_values = list(values)
        self["values"] = self._all_values
