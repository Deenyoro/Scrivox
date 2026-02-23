"""GUI entry point for Scrivox."""

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


def launch_gui():
    """Create and run the Scrivox GUI application."""
    from .ui.app import ScrivoxApp
    app = ScrivoxApp()
    app.mainloop()
