"""Small help and error dialogs written for non-technical users."""

import tkinter as tk
from tkinter import ttk

from .. import winnative
from ..theme import COLORS, FONTS, SP_L, SP_S, SP_XS, px

FFMPEG_WINGET = "winget install --id Gyan.FFmpeg -e"

HELP_TOPICS = {
    "ffmpeg": (
        "Install ffmpeg",
        "Scrivox uses the free ffmpeg tool to read audio and video files.",
        [
            ("Open the Start menu, type “Terminal” (or “PowerShell”) "
             "and open it."),
            "Paste this command and press Enter:",
            "@cmd",
            "When it finishes, close Scrivox and open it again.",
        ],
        ("Prefer a manual install? Download a Windows build from ffmpeg.org, unzip it, "
         "and add its “bin” folder to your PATH."),
    ),
    "gpu": (
        "An NVIDIA graphics card is needed",
        "Scrivox transcribes on an NVIDIA graphics card (CUDA). None was found on this PC.",
        [
            ("If this PC has an NVIDIA card, install the latest driver from nvidia.com "
             "or the NVIDIA app, then restart Scrivox."),
            ("On a laptop, plug in the charger and set Windows graphics settings to "
             "“High performance” for Scrivox."),
            "PCs with only Intel or AMD graphics can't run Scrivox's transcription.",
        ],
        "",
    ),
}


def _dialog_frame(win, title):
    win.title(title)
    win.configure(bg=COLORS["bg"])
    win.resizable(False, False)
    body = ttk.Frame(win, padding=(px(20), px(16), px(20), px(16)))
    body.pack(fill=tk.BOTH, expand=True)
    return body


def _place(win, parent):
    win.update_idletasks()
    w, h = win.winfo_reqwidth(), win.winfo_reqheight()
    x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
    y = parent.winfo_rooty() + max(0, (parent.winfo_height() - h) // 3)
    win.geometry(f"+{max(0, x)}+{max(0, y)}")
    win.deiconify()
    winnative.use_dark_title_bar(win)


class FixHelpDialog(tk.Toplevel):
    """Step-by-step fix for a missing prerequisite, with a "Check again"."""

    def __init__(self, parent, topic, on_recheck=None):
        super().__init__(parent)
        self.withdraw()
        self.transient(parent)
        title, intro, steps, footer = HELP_TOPICS[topic]
        body = _dialog_frame(self, f"{title} - Scrivox")
        wrap = int(440 * parent.winfo_fpixels("1i") / 96.0)

        ttk.Label(body, text=title, style="CardTitle.TLabel").pack(anchor=tk.W)
        ttk.Label(body, text=intro, style="Dim.TLabel", wraplength=wrap,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(SP_XS, SP_S))
        n = 0
        for step in steps:
            if step == "@cmd":
                row = ttk.Frame(body)
                row.pack(fill=tk.X, pady=(0, SP_S), padx=(px(22), 0))
                cmd = ttk.Entry(row, font=FONTS["mono"])
                cmd.insert(0, FFMPEG_WINGET)
                cmd.state(["readonly"])
                cmd.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self._copy_btn = ttk.Button(row, text="Copy", style="Small.TButton",
                                            command=lambda: self._copy(FFMPEG_WINGET))
                self._copy_btn.pack(side=tk.LEFT, padx=(SP_XS, 0))
                continue
            n += 1
            ttk.Label(body, text=f"{n}.  {step}", wraplength=wrap,
                      justify=tk.LEFT).pack(anchor=tk.W, pady=(0, SP_XS))
        if footer:
            ttk.Label(body, text=footer, style="Dim.TLabel", wraplength=wrap,
                      justify=tk.LEFT).pack(anchor=tk.W, pady=(SP_S, 0))

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(SP_L, 0))
        ttk.Button(btns, text="Close", command=self.destroy).pack(side=tk.RIGHT)
        if on_recheck:
            recheck = ttk.Button(btns, text="Check again", style="Accent.TButton",
                                 command=lambda: (self.destroy(), on_recheck()))
            recheck.pack(side=tk.RIGHT, padx=(0, SP_S))
            recheck.focus_set()
        self.bind("<Escape>", lambda e: (self.destroy(), "break")[1])
        _place(self, parent)

    def _copy(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        self._copy_btn.configure(text="Copied")
        self.after(1500, lambda: self._copy_btn.winfo_exists()
                   and self._copy_btn.configure(text="Copy"))


class ErrorReportDialog(tk.Toplevel):
    """Friendly "something went wrong" with the details one click away."""

    def __init__(self, parent, details, log_path=None):
        super().__init__(parent)
        self.withdraw()
        self.transient(parent)
        body = _dialog_frame(self, "Scrivox")
        wrap = int(440 * parent.winfo_fpixels("1i") / 96.0)
        ttk.Label(body, text="Something went wrong", style="CardTitle.TLabel").pack(anchor=tk.W)
        msg = ("Scrivox hit an unexpected problem. Your queue and settings are still "
               "here, so you can try again.")
        if log_path:
            msg += f"\n\nDetails were saved to {log_path}"
        ttk.Label(body, text=msg, style="Dim.TLabel", wraplength=wrap,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(SP_XS, SP_S))
        self._details = details
        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(SP_S, 0))
        ok = ttk.Button(btns, text="OK", style="Accent.TButton", command=self.destroy)
        ok.pack(side=tk.RIGHT)
        self._copy_btn = ttk.Button(btns, text="Copy details", command=self._copy)
        self._copy_btn.pack(side=tk.RIGHT, padx=(0, SP_S))
        self.bind("<Escape>", lambda e: (self.destroy(), "break")[1])
        self.bind("<Return>", lambda e: (self.destroy(), "break")[1])
        _place(self, parent)
        ok.focus_set()

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._details)
        self._copy_btn.configure(text="Copied")
