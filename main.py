"""Scrivox - Dual-mode entry point.

No args  -> launch GUI (no console window)
With args -> run CLI (attaches to parent console if available)
"""

import os
import sys

# PyInstaller console=False builds set sys.stdout/stderr to None.
# Libraries (pyannote, torch, etc.) crash when they try to print/write.
# Redirect to devnull so nothing breaks.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


def _attach_console():
    """Attach to the parent process console on Windows (for windowed exe).

    When built with console=False, the exe has no console by default.
    If launched from a terminal (cmd/powershell), we attach to that console
    so CLI output is visible. If double-clicked, this silently fails and
    the GUI launches without a console window.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Try to attach to parent process console
            if kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS = -1
                # Reopen stdout/stderr to the attached console
                sys.stdout = open("CONOUT$", "w", closefd=False)
                sys.stderr = open("CONOUT$", "w", closefd=False)
        except Exception:
            pass


def main():
    if len(sys.argv) > 1:
        # CLI mode — attach to parent console for output
        _attach_console()
        from scrivox.cli import run_cli
        run_cli()
    else:
        # GUI mode — no console window needed
        from scrivox.gui import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
