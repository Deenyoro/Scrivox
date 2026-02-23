"""Thread-safe stdout redirect to a Tkinter text widget."""

import io
import tkinter as tk


class LogRedirect(io.TextIOBase):
    """Captures writes to stdout and routes them to a Tk text widget via root.after()."""

    def __init__(self, text_widget, root, original_stdout=None):
        super().__init__()
        self.text_widget = text_widget
        self.root = root
        self.original_stdout = original_stdout

    def write(self, text):
        if not text:
            return 0
        # Also write to original stdout (visible in console)
        if self.original_stdout:
            try:
                self.original_stdout.write(text)
                self.original_stdout.flush()
            except Exception:
                pass
        # Schedule append on the main thread
        try:
            self.root.after(0, self._append, text)
        except Exception:
            pass  # widget may be destroyed
        return len(text)

    def _append(self, text):
        try:
            self.text_widget.configure(state=tk.NORMAL)
            # Handle \r (carriage return) for progress updates
            if text.startswith("\r"):
                # Delete current line and replace
                self.text_widget.delete("end-1c linestart", "end-1c")
                text = text.lstrip("\r")
            self.text_widget.insert(tk.END, text)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)
        except Exception:
            pass

    def flush(self):
        if self.original_stdout:
            try:
                self.original_stdout.flush()
            except Exception:
                pass
