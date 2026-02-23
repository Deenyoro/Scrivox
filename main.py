"""Scrivox - Dual-mode entry point.

No args  → launch GUI
With args → run CLI
"""

import sys


def main():
    if len(sys.argv) > 1:
        # CLI mode
        from scrivox.cli import run_cli
        run_cli()
    else:
        # GUI mode
        from scrivox.gui import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
