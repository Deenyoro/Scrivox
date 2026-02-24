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


def launch_gui():
    """Create and run the Scrivox GUI application."""
    from .ui.app import ScrivoxApp
    app = ScrivoxApp()
    app.mainloop()
