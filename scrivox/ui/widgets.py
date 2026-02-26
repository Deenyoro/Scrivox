"""Reusable UI widgets for Scrivox."""

import tkinter as tk
from tkinter import ttk


class AutocompleteCombobox(ttk.Combobox):
    """Combobox with type-to-filter: case-insensitive substring matching.

    - KeyRelease triggers 50ms debounced filter
    - Escape restores full list; FocusOut restores after a short delay
    - state="normal" preserved — custom text always allowed
    - multi_value=True: comma-separated input where autocomplete and selection
      operate on the token after the last comma
    """

    def __init__(self, master=None, **kwargs):
        self._all_values = list(kwargs.pop("values", []))
        self._multi_value = kwargs.pop("multi_value", False)
        super().__init__(master, values=self._all_values, **kwargs)
        self._debounce_id = None
        self._restore_id = None
        self._selecting = False  # guard against re-entrant filtering
        self._multi_prefix = ""  # text before the last comma in multi_value mode
        self.bind("<KeyRelease>", self._on_key)
        self.bind("<Escape>", self._restore)
        self.bind("<FocusOut>", self._schedule_restore)
        self.bind("<<ComboboxSelected>>", self._on_select)

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
            # Save prefix so _on_select can reconstruct the full value
            self._multi_prefix = full.rsplit(",", 1)[0].strip()
        else:
            text = full.lower()
            self._multi_prefix = ""
        if not text:
            self["values"] = self._all_values
            return
        filtered = [v for v in self._all_values if text in v.lower()]
        self["values"] = filtered if filtered else self._all_values
        # Open dropdown to show filtered results
        self._selecting = True
        try:
            self.event_generate("<Down>")
        finally:
            self._selecting = False
        # Restore user's typed text — the Down key overwrites it with the
        # highlighted item, which corrupts subsequent keystrokes and breaks
        # multi-value comma prefixes.
        self.set(full)
        self.icursor(tk.END)

    def _on_select(self, event):
        """Handle dropdown selection for both single and multi-value modes."""
        # Cancel any pending FocusOut restore — the user clicked an item
        if self._restore_id is not None:
            self.after_cancel(self._restore_id)
            self._restore_id = None
        if self._selecting:
            return
        self._selecting = True
        try:
            selected = self.get()  # Tk already replaced text with clicked item
            if self._multi_value and self._multi_prefix:
                new_text = f"{self._multi_prefix}, {selected}"
                self.set(new_text)
                self.icursor(tk.END)
            self._multi_prefix = ""
        finally:
            self._selecting = False
        self["values"] = self._all_values

    def _schedule_restore(self, event=None):
        """Delayed restore on FocusOut — avoids racing with dropdown clicks."""
        if self._restore_id is not None:
            self.after_cancel(self._restore_id)
        self._restore_id = self.after(200, self._do_restore)

    def _do_restore(self):
        self._restore_id = None
        self["values"] = self._all_values
        self._multi_prefix = ""

    def _restore(self, event=None):
        """Immediate restore on Escape."""
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        if self._restore_id is not None:
            self.after_cancel(self._restore_id)
            self._restore_id = None
        self["values"] = self._all_values
        self._multi_prefix = ""

    def set_values(self, values):
        """Update the full values list."""
        self._all_values = list(values)
        self["values"] = self._all_values
