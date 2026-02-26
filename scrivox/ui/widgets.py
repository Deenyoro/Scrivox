"""Reusable UI widgets for Scrivox."""

import tkinter as tk
from tkinter import ttk


class AutocompleteCombobox(ttk.Combobox):
    """Combobox with type-to-filter: case-insensitive substring matching.

    - KeyRelease triggers 50ms debounced filter
    - Escape or focus-out restores full list
    - state="normal" preserved — custom text always allowed
    - multi_value=True: comma-separated input where autocomplete and selection
      operate on the token after the last comma
    """

    def __init__(self, master=None, **kwargs):
        self._all_values = list(kwargs.pop("values", []))
        self._multi_value = kwargs.pop("multi_value", False)
        super().__init__(master, values=self._all_values, **kwargs)
        self._debounce_id = None
        self._selecting = False  # guard against re-entrant filtering
        self._multi_prefix = ""  # text before the last comma in multi_value mode
        self.bind("<KeyRelease>", self._on_key)
        self.bind("<Escape>", self._restore)
        self.bind("<FocusOut>", self._restore)
        if self._multi_value:
            self.bind("<<ComboboxSelected>>", self._on_multi_select)

    def _on_key(self, event):
        # Ignore navigation/modifier keys
        if event.keysym in ("Up", "Down", "Left", "Right", "Return",
                            "Tab", "Escape", "Shift_L", "Shift_R",
                            "Control_L", "Control_R", "Alt_L", "Alt_R"):
            return
        if self._selecting:
            return
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(50, self._filter)

    def _filter(self):
        self._debounce_id = None
        if self._selecting:
            return
        full = self.get()
        if self._multi_value and "," in full:
            text = full.rsplit(",", 1)[1].strip().lower()
            # Save prefix so _on_multi_select can reconstruct the full value
            self._multi_prefix = full.rsplit(",", 1)[0].strip()
        else:
            text = full.lower()
            self._multi_prefix = ""
        if not text:
            self["values"] = self._all_values
            return
        filtered = [v for v in self._all_values if text in v.lower()]
        self["values"] = filtered if filtered else self._all_values
        # Keep dropdown open while typing
        self.event_generate("<Down>")

    def _on_multi_select(self, event):
        """Handle dropdown selection in multi_value mode: append to existing text."""
        self._selecting = True
        try:
            selected = self.get()  # Tk replaced the full text with the selection
            prefix = self._multi_prefix
            if prefix:
                new_text = f"{prefix}, {selected}"
            else:
                new_text = selected
            self.set(new_text)
            self.icursor(tk.END)
            self._multi_prefix = ""
        finally:
            self._selecting = False
        self["values"] = self._all_values

    def _restore(self, event=None):
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        self["values"] = self._all_values

    def set_values(self, values):
        """Update the full values list."""
        self._all_values = list(values)
        self["values"] = self._all_values
