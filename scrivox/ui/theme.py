"""Theme configuration: colors, fonts, spacing and ttk.Style setup.

Scrivox deliberately uses its own dark theme (built on ttk's "clam" engine,
which ships with every Tk) rather than an extra theme package, so nothing has
to be bundled. Sizes that Tk does not scale by itself (row heights, indicator
images, scrollbar widths) are derived from the window's DPI.
"""

import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# ── Color palette ──
COLORS = {
    "bg": "#1e1e2e",
    "bg_secondary": "#252535",
    "bg_input": "#2a2a3d",
    "bg_card": "#222232",
    "fg": "#cdd6f4",
    "fg_dim": "#9399b2",
    "fg_disabled": "#6c7086",
    "fg_bright": "#ffffff",
    "accent": "#89b4fa",
    "accent_hover": "#a6c8ff",
    "accent_pressed": "#74a0e8",
    "success": "#a6e3a1",
    "warning": "#f9e2af",
    "warning_bg": "#3a3326",
    "error": "#f38ba8",
    "error_bg": "#3b2433",
    "border": "#45475a",
    "border_strong": "#585b70",
    "selection": "#313244",
    "button_bg": "#313244",
    "button_fg": "#1e1e2e",
    "button_hover": "#3d3f55",
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

# ── Fonts (points, so Tk scales them with the display DPI) ──
FONTS = {
    "heading": ("Segoe UI", 11, "bold"),
    "body": ("Segoe UI", 10),
    "small": ("Segoe UI", 9),
    "mono": ("Cascadia Code", 10),
    "mono_small": ("Cascadia Code", 9),
    "button": ("Segoe UI", 10),
    "button_bold": ("Segoe UI", 10, "bold"),
    "title": ("Segoe UI", 14, "bold"),
    "headline": ("Segoe UI", 12, "bold"),
}


def px(n):
    """Spacing in *points* for an `n`-pixel gap at 100% (96 dpi).

    Tk converts points with the current `tk scaling`, so an 8 px gap becomes
    12 px at 150% without any manual math. Use for padx/pady/padding.
    """
    return f"{n * 0.75:g}p"


# Spacing scale (8 px grid)
SP_XS, SP_S, SP_M, SP_L = px(4), px(8), px(12), px(16)


def _resolve_family(candidates, fallback):
    available = set(tkfont.families())
    for candidate in candidates:
        if candidate in available:
            return candidate
    return fallback


def _resolve_mono_font():
    """Find the best available monospace font."""
    return _resolve_family(("Cascadia Mono", "Cascadia Code", "Consolas",
                            "DejaVu Sans Mono", "Courier New"), "TkFixedFont")


def _configure_named_fonts(root):
    """Point Tk's named fonts at Segoe UI so every widget (including dialogs,
    menus and message boxes Tk builds itself) uses the Windows UI font."""
    default = tkfont.nametofont("TkDefaultFont", root=root)
    ui_family = _resolve_family(("Segoe UI", "Noto Sans", "DejaVu Sans"),
                                default.actual("family"))
    mono = _resolve_mono_font()

    for name, size, weight in (("TkDefaultFont", 10, "normal"),
                               ("TkTextFont", 10, "normal"),
                               ("TkMenuFont", 10, "normal"),
                               ("TkHeadingFont", 10, "bold"),
                               ("TkCaptionFont", 11, "bold"),
                               ("TkSmallCaptionFont", 9, "normal"),
                               ("TkIconFont", 10, "normal"),
                               ("TkTooltipFont", 9, "normal")):
        try:
            tkfont.nametofont(name, root=root).configure(
                family=ui_family, size=size, weight=weight)
        except tk.TclError:
            continue  # named font missing in this Tk build
    try:
        tkfont.nametofont("TkFixedFont", root=root).configure(family=mono, size=10)
    except tk.TclError:
        mono = "TkFixedFont"

    for key, spec in list(FONTS.items()):
        family = mono if key.startswith("mono") else ui_family
        FONTS[key] = (family,) + tuple(spec[1:])


# ── Indicator images (check mark / radio dot that scale with DPI) ──

def _indicator_images(root, size):
    """Build PhotoImages for checkbox and radio indicators with Pillow.

    clam draws an "X" in its checkboxes and ignores DPI; these images draw a
    real check mark at the right pixel size. Returns None when Pillow is not
    available, and the stock clam indicator is used instead.
    """
    try:
        import base64
        import io

        from PIL import Image, ImageDraw
    except (ImportError, OSError):
        return None

    ss = 4  # supersample for smooth edges
    s = size * ss
    c = COLORS

    def png(draw_fn):
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(img))
        img = img.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return tk.PhotoImage(master=root, data=base64.b64encode(buf.getvalue()))

    lw = max(ss, int(s * 0.07))
    radius = int(s * 0.22)
    inset = lw

    def box(fill, outline):
        def fn(d):
            d.rounded_rectangle([inset, inset, s - inset - 1, s - inset - 1],
                                radius=radius, fill=fill, outline=outline, width=lw)
        return fn

    def checked(fill, mark):
        def fn(d):
            box(fill, fill)(d)
            pts = [(s * 0.26, s * 0.52), (s * 0.43, s * 0.69), (s * 0.75, s * 0.33)]
            d.line(pts, fill=mark, width=int(s * 0.11), joint="curve")
        return fn

    def circle(fill, outline, dot=None):
        def fn(d):
            d.ellipse([inset, inset, s - inset - 1, s - inset - 1],
                      fill=fill, outline=outline, width=lw)
            if dot:
                r = s * 0.2
                d.ellipse([s / 2 - r, s / 2 - r, s / 2 + r, s / 2 + r], fill=dot)
        return fn

    return {
        "cb_off": png(box(c["entry_bg"], c["border_strong"])),
        "cb_off_hover": png(box(c["entry_bg"], c["accent"])),
        "cb_off_dis": png(box(c["bg_secondary"], c["border"])),
        "cb_on": png(checked(c["accent"], c["button_fg"])),
        "cb_on_hover": png(checked(c["accent_hover"], c["button_fg"])),
        "cb_on_dis": png(checked(c["border"], c["fg_disabled"])),
        "rb_off": png(circle(c["entry_bg"], c["border_strong"])),
        "rb_off_hover": png(circle(c["entry_bg"], c["accent"])),
        "rb_off_dis": png(circle(c["bg_secondary"], c["border"])),
        "rb_on": png(circle(c["accent"], c["accent"], c["button_fg"])),
        "rb_on_hover": png(circle(c["accent_hover"], c["accent_hover"], c["button_fg"])),
        "rb_on_dis": png(circle(c["border"], c["border"], c["fg_disabled"])),
    }


def ui_scale(root):
    try:
        return root.winfo_fpixels("1i") / 96.0
    except tk.TclError:
        return 1.0


def configure_theme(root):
    """Apply the dark theme to all ttk widgets."""
    _configure_named_fonts(root)
    scale = ui_scale(root)
    c = COLORS

    style = ttk.Style(root)
    style.theme_use("clam")

    # ── Global defaults (inherited by every style) ──
    root.configure(bg=c["bg"])
    style.configure(".", background=c["bg"], foreground=c["fg"],
                    fieldbackground=c["entry_bg"], bordercolor=c["border"],
                    lightcolor=c["bg"], darkcolor=c["bg"], troughcolor=c["bg_secondary"],
                    selectbackground=c["accent"], selectforeground=c["button_fg"],
                    insertcolor=c["fg"], focuscolor=c["accent"], font=FONTS["body"])
    style.map(".", foreground=[("disabled", c["fg_disabled"])])

    # Classic (non-ttk) widgets Tk creates itself: dropdown lists, menus
    for pattern, value in (
            ("*TCombobox*Listbox.background", c["entry_bg"]),
            ("*TCombobox*Listbox.foreground", c["entry_fg"]),
            ("*TCombobox*Listbox.selectBackground", c["accent"]),
            ("*TCombobox*Listbox.selectForeground", c["button_fg"]),
            ("*TCombobox*Listbox.font", FONTS["body"])):
        root.option_add(pattern, value)
    if sys.platform != "win32":
        # The Windows menubar is drawn natively (and follows the OS theme)
        for pattern, value in (("*Menu.background", c["bg_secondary"]),
                               ("*Menu.foreground", c["fg"]),
                               ("*Menu.activeBackground", c["selection"]),
                               ("*Menu.activeForeground", c["fg_bright"]),
                               ("*Menu.relief", "flat"),
                               ("*Menu.borderWidth", 0),
                               ("*Menu.activeBorderWidth", 0)):
            root.option_add(pattern, value)

    # ── Frames ──
    style.configure("TFrame", background=c["bg"])
    style.configure("Secondary.TFrame", background=c["bg_secondary"])
    style.configure("Card.TFrame", background=c["bg"], borderwidth=1, relief="solid",
                    bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"])
    style.configure("Bar.TFrame", background=c["bg_secondary"])
    style.configure("Warning.TFrame", background=c["warning_bg"])
    style.configure("ErrorBanner.TFrame", background=c["error_bg"])

    # ── Labels ──
    style.configure("TLabel", background=c["bg"], foreground=c["fg"], font=FONTS["body"])
    style.configure("Header.TLabel", background=c["bg"],
                    foreground=c["frame_header"], font=FONTS["heading"])
    style.configure("CardTitle.TLabel", background=c["bg"],
                    foreground=c["fg_bright"], font=FONTS["heading"])
    style.configure("Dim.TLabel", background=c["bg"],
                    foreground=c["fg_dim"], font=FONTS["small"])
    style.configure("Success.TLabel", background=c["bg"],
                    foreground=c["success"], font=FONTS["body"])
    style.configure("Error.TLabel", background=c["bg"],
                    foreground=c["error"], font=FONTS["body"])
    style.configure("SmallError.TLabel", background=c["bg"],
                    foreground=c["error"], font=FONTS["small"])
    style.configure("Warning.TLabel", background=c["bg"],
                    foreground=c["warning"], font=FONTS["small"])
    style.configure("Title.TLabel", background=c["bg"],
                    foreground=c["accent"], font=FONTS["title"])
    style.configure("Headline.TLabel", background=c["bg"],
                    foreground=c["fg_bright"], font=FONTS["headline"])
    style.configure("HeadlineError.TLabel", background=c["bg"],
                    foreground=c["error"], font=FONTS["headline"])
    style.configure("HeadlineSuccess.TLabel", background=c["bg"],
                    foreground=c["success"], font=FONTS["headline"])
    style.configure("Link.TLabel", background=c["bg"], foreground=c["accent"],
                    font=FONTS["small"] + ("underline",))
    style.map("Link.TLabel", foreground=[("active", c["accent_hover"])])
    style.configure("Bar.TLabel", background=c["bg_secondary"], foreground=c["fg"])
    style.configure("BarDim.TLabel", background=c["bg_secondary"],
                    foreground=c["fg_dim"], font=FONTS["small"])
    style.configure("BarSuccess.TLabel", background=c["bg_secondary"],
                    foreground=c["success"], font=FONTS["button_bold"])
    style.configure("Banner.TLabel", background=c["warning_bg"], foreground=c["warning"],
                    font=FONTS["body"])
    style.configure("BannerTitle.TLabel", background=c["warning_bg"], foreground=c["warning"],
                    font=FONTS["button_bold"])
    style.configure("BannerDim.TLabel", background=c["warning_bg"], foreground=c["fg"],
                    font=FONTS["small"])
    style.configure("BannerIcon.TLabel", background=c["warning_bg"], foreground=c["warning"],
                    font=(FONTS["body"][0], 14, "bold"))

    # ── Buttons ──
    # Neutral buttons for everything except the one primary action per view.
    # Disabled buttons keep a readable label (fg_disabled on a flat slab)
    # instead of disappearing into their background.
    flat = {"borderwidth": 1, "relief": "flat", "focusthickness": 1}
    style.configure("TButton", background=c["button_bg"], foreground=c["fg"],
                    font=FONTS["button"], padding=(px(12), px(5)),
                    bordercolor=c["border"], lightcolor=c["button_bg"],
                    darkcolor=c["button_bg"], focuscolor=c["accent"], **flat)
    style.map("TButton",
              background=[("disabled", c["bg_secondary"]), ("pressed", c["selection"]),
                          ("active", c["button_hover"])],
              lightcolor=[("disabled", c["bg_secondary"]), ("active", c["button_hover"])],
              darkcolor=[("disabled", c["bg_secondary"]), ("active", c["button_hover"])],
              bordercolor=[("focus", c["accent"]), ("disabled", c["border"])],
              foreground=[("disabled", c["fg_disabled"])])
    style.configure("Secondary.TButton", background=c["button_bg"], foreground=c["fg"],
                    font=FONTS["button"], padding=(px(12), px(5)),
                    bordercolor=c["border"], lightcolor=c["button_bg"],
                    darkcolor=c["button_bg"], **flat)
    style.map("Secondary.TButton",
              background=[("disabled", c["bg_secondary"]), ("pressed", c["selection"]),
                          ("active", c["button_hover"])],
              lightcolor=[("disabled", c["bg_secondary"]), ("active", c["button_hover"])],
              darkcolor=[("disabled", c["bg_secondary"]), ("active", c["button_hover"])],
              bordercolor=[("focus", c["accent"]), ("disabled", c["bg_secondary"])],
              foreground=[("disabled", c["fg_disabled"])])
    style.configure("Small.TButton", background=c["button_bg"], foreground=c["fg"],
                    font=FONTS["small"], padding=(px(8), px(2)),
                    bordercolor=c["border"], lightcolor=c["button_bg"],
                    darkcolor=c["button_bg"], **flat)
    style.map("Small.TButton", **style.map("Secondary.TButton"))

    style.configure("Accent.TButton", background=c["accent"], foreground=c["button_fg"],
                    font=FONTS["button_bold"], padding=(px(16), px(8)),
                    bordercolor=c["accent"], lightcolor=c["accent"], darkcolor=c["accent"],
                    focuscolor=c["button_fg"], **flat)
    style.map("Accent.TButton",
              background=[("disabled", c["selection"]), ("pressed", c["accent_pressed"]),
                          ("active", c["accent_hover"])],
              lightcolor=[("disabled", c["selection"]), ("active", c["accent_hover"])],
              darkcolor=[("disabled", c["selection"]), ("active", c["accent_hover"])],
              bordercolor=[("focus", c["fg_bright"]), ("disabled", c["selection"]),
                           ("active", c["accent_hover"])],
              foreground=[("disabled", c["fg_disabled"])])

    # Primary action that can't run yet: same size and place, but a quiet
    # slab with a dim label, so it doesn't invite a click. It stays
    # focusable/clickable; a click explains what's missing.
    style.configure("AccentBlocked.TButton", background=c["selection"],
                    foreground=c["fg_disabled"], font=FONTS["button_bold"],
                    padding=(px(16), px(8)), bordercolor=c["border"],
                    lightcolor=c["selection"], darkcolor=c["selection"],
                    focuscolor=c["accent"], **flat)
    style.map("AccentBlocked.TButton",
              background=[("pressed", c["button_hover"]), ("active", c["button_hover"])],
              lightcolor=[("active", c["button_hover"])],
              darkcolor=[("active", c["button_hover"])],
              bordercolor=[("focus", c["accent"])],
              foreground=[("active", c["fg_dim"])])

    # Section disclosure ("▸ Extras"): reads as a heading, acts as a button
    style.configure("Disclosure.TButton", background=c["bg"], foreground=c["frame_header"],
                    font=FONTS["heading"], padding=(0, px(3), px(4), px(3)), anchor="w",
                    bordercolor=c["bg"], lightcolor=c["bg"], darkcolor=c["bg"],
                    focuscolor=c["accent"], borderwidth=1, relief="flat", focusthickness=1)
    style.map("Disclosure.TButton",
              background=[("active", c["bg"]), ("pressed", c["bg"]), ("disabled", c["bg"])],
              lightcolor=[("active", c["bg"])], darkcolor=[("active", c["bg"])],
              bordercolor=[("focus", c["accent"])],
              foreground=[("disabled", c["fg_disabled"]), ("active", c["accent_hover"])])

    # Cancel: danger *outline* so it reads as "stop", not as another go button
    style.configure("Cancel.TButton", background=c["bg"], foreground=c["error"],
                    font=FONTS["button_bold"], padding=(px(16), px(8)),
                    bordercolor=c["error"], lightcolor=c["bg"], darkcolor=c["bg"],
                    borderwidth=1, relief="solid", focusthickness=1)
    style.map("Cancel.TButton",
              background=[("disabled", c["bg"]), ("pressed", c["error_bg"]),
                          ("active", c["error_bg"])],
              lightcolor=[("active", c["error_bg"])],
              darkcolor=[("active", c["error_bg"])],
              bordercolor=[("disabled", c["border"])],
              foreground=[("disabled", c["fg_disabled"])])

    style.configure("Banner.TButton", background=c["warning_bg"], foreground=c["warning"],
                    bordercolor=c["warning"], lightcolor=c["warning_bg"],
                    darkcolor=c["warning_bg"], padding=(px(10), px(3)),
                    font=FONTS["small"], borderwidth=1, relief="solid")
    style.map("Banner.TButton", background=[("active", "#4a4230")],
              lightcolor=[("active", "#4a4230")], darkcolor=[("active", "#4a4230")])

    # ── Check / radio buttons ──
    ind = round(16 * scale)
    images = _indicator_images(root, ind)
    for cls in ("TCheckbutton", "TRadiobutton"):
        style.configure(cls, background=c["bg"], foreground=c["fg"], font=FONTS["body"],
                        indicatorbackground=c["entry_bg"], indicatorforeground=c["accent"],
                        upperbordercolor=c["border_strong"], lowerbordercolor=c["border_strong"],
                        indicatormargin=(0, 0, px(6), 0), padding=(px(1), px(2)),
                        focuscolor=c["accent"])
        style.map(cls, background=[("active", c["bg"])],
                  foreground=[("disabled", c["fg_disabled"]), ("active", c["fg_bright"])],
                  indicatorbackground=[("selected", c["accent"])])
    if images:
        root._scrivox_indicator_images = images  # keep references alive
        i = images
        gap = round(8 * scale)  # space between indicator and text
        style.element_create(
            "Scrivox.Checkbutton.indicator", "image", i["cb_off"],
            ("disabled", "selected", i["cb_on_dis"]), ("disabled", i["cb_off_dis"]),
            ("active", "selected", i["cb_on_hover"]), ("selected", i["cb_on"]),
            ("active", i["cb_off_hover"]),
            sticky="w", width=ind + gap)
        style.element_create(
            "Scrivox.Radiobutton.indicator", "image", i["rb_off"],
            ("disabled", "selected", i["rb_on_dis"]), ("disabled", i["rb_off_dis"]),
            ("active", "selected", i["rb_on_hover"]), ("selected", i["rb_on"]),
            ("active", i["rb_off_hover"]),
            sticky="w", width=ind + gap)
        for cls, elem in (("TCheckbutton", "Scrivox.Checkbutton.indicator"),
                          ("TRadiobutton", "Scrivox.Radiobutton.indicator")):
            base = cls[1:]
            style.layout(cls, [(f"{base}.padding", {"sticky": "nswe", "children": [
                (elem, {"side": "left", "sticky": "w"}),
                (f"{base}.focus", {"side": "left", "sticky": "w", "children": [
                    (f"{base}.label", {"sticky": "nswe"})]})]})])
            style.configure(cls, padding=(0, px(2), px(2), px(2)))

    # ── Entry / Combobox / Spinbox ──
    field = {"fieldbackground": c["entry_bg"], "foreground": c["entry_fg"],
             "bordercolor": c["border"], "lightcolor": c["entry_bg"],
             "darkcolor": c["entry_bg"], "insertcolor": c["fg"], "padding": (px(6), px(3))}
    field_map = {
        "bordercolor": [("invalid", c["error"]), ("focus", c["accent"]),
                        ("hover", c["border_strong"])],
        "lightcolor": [("focus", c["entry_bg"])],
        "fieldbackground": [("disabled", c["bg_secondary"]), ("readonly", c["entry_bg"])],
        "foreground": [("disabled", c["fg_disabled"])]}
    style.configure("TEntry", **field)
    style.map("TEntry", **field_map)
    arrow = max(10, round(12 * scale))
    style.configure("TCombobox", background=c["entry_bg"], arrowcolor=c["fg_dim"],
                    arrowsize=arrow, **field)
    style.map("TCombobox", background=[("active", c["button_hover"])],
              arrowcolor=[("disabled", c["fg_disabled"]), ("active", c["fg"])],
              selectbackground=[("readonly", "!focus", c["entry_bg"]),
                                ("readonly", "focus", c["entry_bg"])],
              selectforeground=[("readonly", c["entry_fg"])],
              **field_map)
    # Dropdown lists with descriptions: the list may be wider than the field
    # (it opens over the right-hand pane, like a native Windows dropdown)
    style.configure("Wide.TCombobox", postoffset=(0, 0, round(110 * scale), 0))
    style.configure("TSpinbox", background=c["entry_bg"], arrowcolor=c["fg_dim"],
                    arrowsize=max(8, round(10 * scale)), **field)
    style.map("TSpinbox", background=[("active", c["button_hover"])],
              arrowcolor=[("disabled", c["fg_disabled"]), ("active", c["fg"])],
              **field_map)

    # ── Progressbar ──
    # (clam sizes the bar from -arrowsize, not -thickness)
    bar_h = max(6, round(8 * scale))
    style.configure("TProgressbar", troughcolor=c["progress_trough"],
                    background=c["progress_bar"], borderwidth=0,
                    bordercolor=c["progress_trough"], lightcolor=c["progress_bar"],
                    darkcolor=c["progress_bar"], arrowsize=bar_h, thickness=bar_h)
    thin_h = max(4, round(4 * scale))
    style.configure("Thin.Horizontal.TProgressbar", troughcolor=c["progress_trough"],
                    background=c["frame_header"], borderwidth=0,
                    bordercolor=c["progress_trough"], lightcolor=c["frame_header"],
                    darkcolor=c["frame_header"], arrowsize=thin_h, thickness=thin_h)

    # ── Labelframe ──
    style.configure("TLabelframe", background=c["bg"], bordercolor=c["border"],
                    lightcolor=c["border"], darkcolor=c["border"],
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=c["bg"],
                    foreground=c["frame_header"], font=FONTS["heading"])

    # ── Notebook ──
    style.configure("TNotebook", background=c["bg"], borderwidth=0,
                    bordercolor=c["border"], lightcolor=c["bg"], darkcolor=c["bg"],
                    tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=c["bg"], foreground=c["fg_dim"],
                    padding=(px(14), px(6)), bordercolor=c["bg"],
                    lightcolor=c["bg"], darkcolor=c["bg"], font=FONTS["body"],
                    focuscolor=c["accent"])
    style.map("TNotebook.Tab",
              background=[("selected", c["bg_secondary"]), ("active", c["bg_secondary"])],
              foreground=[("selected", c["fg_bright"]), ("active", c["fg"])],
              lightcolor=[("selected", c["accent"])],
              bordercolor=[("selected", c["border"])],
              expand=[("selected", (0, 0, 0, 0))])

    # ── Separator / Panedwindow / Sizegrip ──
    style.configure("TSeparator", background=c["border"])
    style.configure("TPanedwindow", background=c["bg"])
    style.configure("Sash", sashthickness=max(6, round(6 * scale)),
                    gripcount=0, background=c["bg"], bordercolor=c["bg"],
                    lightcolor=c["bg"], darkcolor=c["bg"])

    # ── Scrollbars: slim, arrow-less, like Windows 11 overlay bars ──
    sb_width = max(8, round(10 * scale))
    for orient in ("Vertical", "Horizontal"):
        style.layout(f"{orient}.TScrollbar", [
            (f"{orient}.Scrollbar.trough", {"sticky": "nswe" if orient == "Vertical" else "ew",
                                            "children": [
                (f"{orient}.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
        style.configure(f"{orient}.TScrollbar", background=c["border"],
                        troughcolor=c["bg"], bordercolor=c["bg"],
                        lightcolor=c["border"], darkcolor=c["border"],
                        borderwidth=0, width=sb_width, arrowsize=sb_width,
                        gripcount=0)
        style.map(f"{orient}.TScrollbar",
                  background=[("pressed", c["fg_dim"]), ("active", c["border_strong"])],
                  lightcolor=[("pressed", c["fg_dim"]), ("active", c["border_strong"])],
                  darkcolor=[("pressed", c["fg_dim"]), ("active", c["border_strong"])])

    # ── Treeview ──
    body_font = tkfont.Font(root=root, font=FONTS["body"])
    row_h = int(body_font.metrics("linespace") * 1.6)
    style.configure("Treeview", background=c["bg_secondary"], foreground=c["fg"],
                    fieldbackground=c["bg_secondary"], font=FONTS["body"],
                    rowheight=row_h, bordercolor=c["border"],
                    lightcolor=c["bg_secondary"], darkcolor=c["bg_secondary"],
                    borderwidth=0)
    style.map("Treeview",
              background=[("selected", "focus", c["selection"]), ("selected", c["selection"])],
              foreground=[("selected", c["fg_bright"])])
    style.configure("Treeview.Heading", background=c["bg"], foreground=c["fg_dim"],
                    font=FONTS["small"], relief="flat", borderwidth=0,
                    bordercolor=c["bg"], lightcolor=c["bg"], darkcolor=c["bg"],
                    padding=(px(6), px(4)))
    style.map("Treeview.Heading", background=[("active", c["bg_secondary"])])
    # Drop the heading cell borders/indicator arrow for a flat header row
    style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

    return style
