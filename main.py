"""Scrivox - Dual-mode entry point.

No args  -> launch GUI (no console window)
With args -> run CLI (attaches to parent console if available)
"""

import os
import sys

# PyInstaller console=False builds set sys.stdout/stderr to None.
# Libraries (pyannote, torch, etc.) crash when they try to print/write.
# Redirect to devnull so nothing breaks. Remember which streams were only
# devnull placeholders so _attach_console knows they are safe to rebind.
_STDOUT_WAS_NULL = sys.stdout is None
_STDERR_WAS_NULL = sys.stderr is None
if _STDOUT_WAS_NULL:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
if _STDERR_WAS_NULL:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")


def _stream_invalid(stream):
    """True if a stream is None or has no usable file descriptor."""
    if stream is None:
        return True
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return True
    return False


def _attach_console():
    """Attach to the parent process console on Windows (for windowed exe).

    When built with console=False, the exe has no console by default.
    If launched from a terminal (cmd/powershell), we attach to that console
    so CLI output is visible. If double-clicked, this silently fails and
    the GUI launches without a console window.

    Only streams that were the devnull placeholder (or are invalid) get
    rebound to CONOUT$ — rebinding unconditionally would clobber real
    redirections like `Scrivox.exe file.mp4 > out.txt`.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Try to attach to parent process console
            if kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS = -1
                # Reopen stdout/stderr to the attached console.
                # closefd=False is illegal with a filename and would raise,
                # leaving CLI output bound to devnull.
                if _STDOUT_WAS_NULL or _stream_invalid(sys.stdout):
                    sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
                if _STDERR_WAS_NULL or _stream_invalid(sys.stderr):
                    sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
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
