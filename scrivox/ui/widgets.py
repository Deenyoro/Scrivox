"""Reusable UI widgets for Scrivox."""

from tkinter import ttk


class AutocompleteCombobox(ttk.Combobox):
    """Combobox with type-to-filter: case-insensitive substring matching.

    - KeyRelease triggers 50ms debounced filter
    - Escape or focus-out restores full list
    - state="normal" preserved — custom text always allowed
    """

    def __init__(self, master=None, **kwargs):
        self._all_values = list(kwargs.pop("values", []))
        super().__init__(master, values=self._all_values, **kwargs)
        self._debounce_id = None
        self.bind("<KeyRelease>", self._on_key)
        self.bind("<Escape>", self._restore)
        self.bind("<FocusOut>", self._restore)

    def _on_key(self, event):
        # Ignore navigation/modifier keys
        if event.keysym in ("Up", "Down", "Left", "Right", "Return",
                            "Tab", "Escape", "Shift_L", "Shift_R",
                            "Control_L", "Control_R", "Alt_L", "Alt_R"):
            return
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(50, self._filter)

    def _filter(self):
        self._debounce_id = None
        text = self.get().lower()
        if not text:
            self["values"] = self._all_values
            return
        filtered = [v for v in self._all_values if text in v.lower()]
        self["values"] = filtered if filtered else self._all_values
        # Keep dropdown open while typing
        self.event_generate("<Down>")

    def _restore(self, event=None):
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        self["values"] = self._all_values

    def set_values(self, values):
        """Update the full values list."""
        self._all_values = list(values)
        self["values"] = self._all_values
