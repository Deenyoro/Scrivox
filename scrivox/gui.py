"""GUI entry point for Scrivox."""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
if getattr(sys, 'frozen', False):
    _dotenv_base = os.path.dirname(sys.executable)
else:
    _dotenv_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_dotenv_base, ".env"))


def _startup_failure(details):
    """The window could not be built: save the traceback and say so plainly,
    instead of PyInstaller's raw "Unhandled exception" box (or nothing)."""
    log_path = None
    try:
        from .config import _get_config_dir
        log_path = os.path.join(_get_config_dir(), "scrivox_error.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"--- startup failure ---\n{details}\n")
    except (ImportError, OSError):
        log_path = None
    import tkinter as tk
    from tkinter import messagebox
    try:
        root = tk.Tk()
        root.withdraw()
        msg = "Scrivox couldn't start."
        if log_path:
            msg += f"\n\nDetails were saved to:\n{log_path}"
        msg += "\n\n" + details.strip().splitlines()[-1]
        messagebox.showerror("Scrivox", msg, parent=root)
        root.destroy()
    except tk.TclError:
        log_path = log_path or ""  # no display at all: the log file is all we can do
    try:
        sys.stderr.write(details)
    except (AttributeError, OSError, ValueError):
        return


def launch_gui():
    """Create and run the Scrivox GUI application."""
    try:
        from .ui.app import ScrivoxApp
        app = ScrivoxApp()
    except Exception:  # noqa: BLE001 - any startup error gets the friendly report
        import traceback
        _startup_failure(traceback.format_exc())
        raise SystemExit(1)
    app.mainloop()
