"""Theme configuration: colors, fonts, and ttk.Style setup."""

import tkinter as tk
from tkinter import ttk


# ── Color palette ──
COLORS = {
    "bg": "#1e1e2e",
    "bg_secondary": "#252535",
    "bg_input": "#2a2a3d",
    "fg": "#cdd6f4",
    "fg_dim": "#8a8ea8",
    "fg_bright": "#ffffff",
    "accent": "#89b4fa",
    "accent_hover": "#74c7ec",
    "success": "#a6e3a1",
    "warning": "#f9e2af",
    "error": "#f38ba8",
    "border": "#45475a",
    "selection": "#313244",
    "button_bg": "#89b4fa",
    "button_fg": "#1e1e2e",
    "button_hover": "#74c7ec",
    "cancel_bg": "#f38ba8",
    "cancel_fg": "#1e1e2e",
    "entry_bg": "#313244",
    "entry_fg": "#cdd6f4",
    "log_bg": "#11111b",
    "log_fg": "#a6adc8",
    "progress_trough": "#313244",
    "progress_bar": "#89b4fa",
    "frame_header": "#b4befe",
}

# ── Fonts ──
FONTS = {
    "heading": ("Segoe UI", 11, "bold"),
    "body": ("Segoe UI", 10),
    "small": ("Segoe UI", 9),
    "mono": ("Cascadia Code", 10),
    "mono_small": ("Cascadia Code", 9),
    "button": ("Segoe UI", 10, "bold"),
    "title": ("Segoe UI", 14, "bold"),
}


def _resolve_mono_font():
    """Find the best available monospace font."""
    import tkinter.font as tkfont
    available = tkfont.families()
    for candidate in ("Cascadia Code", "Consolas", "Courier New"):
        if candidate in available:
            return candidate
    return "TkFixedFont"


def configure_theme(root):
    """Apply the dark theme to all ttk widgets."""
    # Resolve monospace font with fallback chain
    mono = _resolve_mono_font()
    FONTS["mono"] = (mono, 10)
    FONTS["mono_small"] = (mono, 9)

    style = ttk.Style(root)
    style.theme_use("clam")

    # ── Global ──
    root.configure(bg=COLORS["bg"])
    root.option_add("*TCombobox*Listbox.background", COLORS["entry_bg"])
    root.option_add("*TCombobox*Listbox.foreground", COLORS["entry_fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", COLORS["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", COLORS["button_fg"])

    # ── TFrame ──
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Secondary.TFrame", background=COLORS["bg_secondary"])
    style.configure("Card.TFrame", background=COLORS["bg_secondary"])

    # ── TLabel ──
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["fg"],
                     font=FONTS["body"])
    style.configure("Header.TLabel", background=COLORS["bg"],
                     foreground=COLORS["frame_header"], font=FONTS["heading"])
    style.configure("Dim.TLabel", background=COLORS["bg"],
                     foreground=COLORS["fg_dim"], font=FONTS["small"])
    style.configure("Success.TLabel", background=COLORS["bg"],
                     foreground=COLORS["success"], font=FONTS["body"])
    style.configure("Error.TLabel", background=COLORS["bg"],
                     foreground=COLORS["error"], font=FONTS["body"])
    style.configure("Title.TLabel", background=COLORS["bg"],
                     foreground=COLORS["accent"], font=FONTS["title"])

    # ── TButton ──
    style.configure("TButton", background=COLORS["button_bg"],
                     foreground=COLORS["button_fg"], font=FONTS["button"],
                     padding=(12, 6), borderwidth=0)
    style.map("TButton",
              background=[("active", COLORS["button_hover"]),
                          ("disabled", COLORS["border"])],
              foreground=[("disabled", COLORS["fg_dim"])])

    style.configure("Accent.TButton", background=COLORS["accent"],
                     foreground=COLORS["button_fg"], font=FONTS["button"],
                     padding=(16, 10))
    style.map("Accent.TButton",
              background=[("active", COLORS["accent_hover"]),
                          ("disabled", COLORS["border"])])

    style.configure("Cancel.TButton", background=COLORS["cancel_bg"],
                     foreground=COLORS["cancel_fg"], font=FONTS["button"],
                     padding=(16, 8))
    style.map("Cancel.TButton",
              background=[("active", COLORS["error"]),
                          ("disabled", COLORS["border"])])

    # ── TCheckbutton ──
    style.configure("TCheckbutton", background=COLORS["bg"],
                     foreground=COLORS["fg"], font=FONTS["body"])
    style.map("TCheckbutton",
              background=[("active", COLORS["bg"])],
              foreground=[("disabled", COLORS["fg_dim"])])

    # ── TCombobox ──
    style.configure("TCombobox", fieldbackground=COLORS["entry_bg"],
                     background=COLORS["entry_bg"], foreground=COLORS["entry_fg"],
                     arrowcolor=COLORS["fg"], borderwidth=1)
    style.map("TCombobox",
              fieldbackground=[("readonly", COLORS["entry_bg"])],
              foreground=[("readonly", COLORS["entry_fg"])])

    # ── TEntry ──
    style.configure("TEntry", fieldbackground=COLORS["entry_bg"],
                     foreground=COLORS["entry_fg"], borderwidth=1,
                     insertcolor=COLORS["fg"])

    # ── Progressbar ──
    style.configure("TProgressbar", troughcolor=COLORS["progress_trough"],
                     background=COLORS["progress_bar"], borderwidth=0,
                     thickness=8)

    # ── TLabelframe ──
    style.configure("TLabelframe", background=COLORS["bg"],
                     foreground=COLORS["frame_header"], bordercolor=COLORS["border"],
                     relief="groove", borderwidth=1)
    style.configure("TLabelframe.Label", background=COLORS["bg"],
                     foreground=COLORS["frame_header"], font=FONTS["heading"])

    # ── Separator ──
    style.configure("TSeparator", background=COLORS["border"])

    # ── PanedWindow ──
    style.configure("TPanedwindow", background=COLORS["bg"])

    # ── Scrollbar ──
    style.configure("Vertical.TScrollbar",
                     background=COLORS["border"],
                     troughcolor=COLORS["bg_secondary"],
                     borderwidth=0, arrowsize=0)
    style.map("Vertical.TScrollbar",
              background=[("active", COLORS["fg_dim"])])

    return style
