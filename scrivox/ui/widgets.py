"""Reusable UI widgets for Scrivox."""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from .theme import COLORS, FONTS, SP_S, px


class ToolTip:
    """Simple tooltip that appears on hover (or keyboard focus) after a delay."""

    DELAY_MS = 500

    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self._tipwindow = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    def _schedule(self, event=None):
        self._unschedule()
        if self._text:
            self._after_id = self._widget.after(self.DELAY_MS, self._show)

    def _unschedule(self):
        if self._after_id:
            try:
                self._widget.after_cancel(self._after_id)
            except tk.TclError:
                pass  # widget already destroyed
            self._after_id = None

    def _show(self, event=None):
        self._after_id = None
        if self._tipwindow or not self._text:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tipwindow = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self._text, justify=tk.LEFT,
                         bg=COLORS["bg_secondary"], fg=COLORS["fg"],
                         font=FONTS["small"], relief=tk.SOLID, borderwidth=1,
                         padx=6, pady=4, wraplength=int(360 * _scale(tw)))
        label.pack()

    def _hide(self, event=None):
        self._unschedule()
        if self._tipwindow:
            self._tipwindow.destroy()
            self._tipwindow = None


def _scale(widget):
    try:
        return widget.winfo_fpixels("1i") / 96.0
    except tk.TclError:
        return 1.0


class TreeRowTooltip:
    """Tooltip for a Treeview showing a per-row text (e.g. the full path)."""

    def __init__(self, tree, text_for_row):
        self._tree = tree
        self._text_for_row = text_for_row
        self._row = None
        self._tip = ToolTip(tree, "")
        tree.bind("<Motion>", self._on_motion, add="+")

    def _on_motion(self, event):
        row = self._tree.identify_row(event.y)
        if row == self._row:
            return
        self._row = row
        self._tip._hide()
        self._tip.text = self._text_for_row(row) if row else ""
        if self._tip.text:
            self._tip._schedule()


class LinkLabel(ttk.Label):
    """Clickable text that looks like a hyperlink and is keyboard-reachable."""

    def __init__(self, parent, text, command, **kwargs):
        kwargs.setdefault("style", "Link.TLabel")
        super().__init__(parent, text=text, cursor="hand2", takefocus=True, **kwargs)
        self._command = command
        self.bind("<Button-1>", lambda e: self._command())
        self.bind("<Return>", lambda e: self._command())
        self.bind("<space>", lambda e: self._command())
        self.bind("<Enter>", lambda e: self.state(["active"]))
        self.bind("<Leave>", lambda e: self.state(["!active"]))


class WrappingLabel(ttk.Label):
    """Label whose wraplength follows its own width, so hint text wraps
    instead of being clipped when the column is narrow or the DPI is high."""

    def __init__(self, parent, **kwargs):
        # A modest starting width so an unwrapped long text never forces its
        # container (or a dialog) to be very wide before the first layout
        kwargs.setdefault("wraplength", int(280 * _scale(parent)))
        super().__init__(parent, **kwargs)
        self.bind("<Configure>", self._rewrap, add="+")

    def _rewrap(self, event):
        if event.width > 20:
            self.configure(wraplength=event.width - 2)


class StepCard(ttk.Frame):
    """A numbered, bordered section: (1) Files, (2) Options, (3) Save to.

    Children go into `.body`.
    """

    def __init__(self, parent, number, title, subtitle="", **kwargs):
        super().__init__(parent, style="Card.TFrame", padding=(px(12), px(8)), **kwargs)
        header = ttk.Frame(self)
        header.pack(fill=tk.X, pady=(0, px(6)))

        badge_font = FONTS["button_bold"]
        size = int(tkfont.Font(font=badge_font).metrics("linespace") * 1.25)
        badge = tk.Canvas(header, width=size, height=size, bg=COLORS["bg"],
                          highlightthickness=0, borderwidth=0)
        badge.create_oval(1, 1, size - 1, size - 1, fill=COLORS["accent"], outline="")
        badge.create_text(size / 2, size / 2, text=str(number),
                          fill=COLORS["button_fg"], font=badge_font)
        badge.pack(side=tk.LEFT, padx=(0, SP_S))
        self.title_label = ttk.Label(header, text=title, style="CardTitle.TLabel")
        self.title_label.pack(side=tk.LEFT)
        self.header = header
        if subtitle:
            ttk.Label(header, text=subtitle, style="Dim.TLabel").pack(
                side=tk.LEFT, padx=(SP_S, 0), pady=(px(2), 0))
        # Short live status beside the title (e.g. "2 files"); the header
        # always has room for it, unlike a crowded button row
        self.note_label = ttk.Label(header, text="", style="Dim.TLabel")

        self.body = ttk.Frame(self)
        self.body.pack(fill=tk.BOTH, expand=True)

    def set_note(self, text):
        """Show `text` beside the title, or hide the note when empty."""
        if self.note_label.cget("text") != text:
            self.note_label.configure(text=text)
        if text and not self.note_label.winfo_manager():
            self.note_label.pack(side=tk.LEFT, padx=(SP_S, 0), pady=(px(2), 0),
                                 after=self.title_label)
        elif not text and self.note_label.winfo_manager():
            self.note_label.pack_forget()


class DropZone(tk.Canvas):
    """Compact dashed "drop files here" area shown while the queue is empty.

    Kept short (about 80 px at 100%) so steps 1-3 and Start fit on one
    screen; `pulse()` flashes it to point the user here when Start is
    pressed with nothing queued.
    """

    HEIGHT = 80

    def __init__(self, parent, on_browse, dnd_available, **kwargs):
        s = _scale(parent)
        super().__init__(parent, height=int(self.HEIGHT * s), bg=COLORS["bg"],
                         highlightthickness=0, borderwidth=0, cursor="hand2",
                         takefocus=True, **kwargs)
        self._on_browse = on_browse
        self._dnd = dnd_available
        self._hover = False
        self._pulse_left = 0
        self._pulse_on = False
        self._pulse_id = None
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>", lambda e: self._on_browse())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<Return>", lambda e: self._on_browse())
        self.bind("<space>", lambda e: self._on_browse())
        self.bind("<FocusIn>", lambda e: self._draw())
        self.bind("<FocusOut>", lambda e: self._draw())
        self.bind("<Destroy>", lambda e: self._stop_pulse(), add="+")

    def set_dnd_available(self, available):
        self._dnd = available
        self._draw()

    def _set_hover(self, hover):
        self._hover = hover
        self._draw()

    def pulse(self, times=3):
        """Flash the border a few times (drawing the eye to step 1)."""
        self._stop_pulse()
        self._pulse_left = times * 2
        self._pulse_step()

    def _pulse_step(self):
        self._pulse_id = None
        if self._pulse_left <= 0:
            self._pulse_on = False
            self._draw()
            return
        self._pulse_on = not self._pulse_on
        self._pulse_left -= 1
        self._draw()
        self._pulse_id = self.after(180, self._pulse_step)

    def _stop_pulse(self):
        if self._pulse_id is not None:
            try:
                self.after_cancel(self._pulse_id)
            except tk.TclError:
                pass
            self._pulse_id = None
        self._pulse_on = False

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10:
            return
        s = _scale(self)
        m = int(2 * s)
        try:
            focused = self.focus_get() is self
        except (KeyError, tk.TclError):
            focused = False
        if self._pulse_on:
            color, fill, width = COLORS["accent_hover"], COLORS["bg_secondary"], 2.5
        else:
            color = COLORS["accent"] if (self._hover or focused) else COLORS["border_strong"]
            fill = COLORS["bg_secondary"] if self._hover else COLORS["bg"]
            width = 1.5
        self.create_rectangle(m, m, w - m, h - m, outline=color,
                              dash=(int(6 * s), int(4 * s)), width=max(1, int(width * s)),
                              fill=fill)
        main = ("Drop audio or video files here" if self._dnd
                else "Click to choose audio or video files")
        sub = "or click to browse  \u00b7  Ctrl+O" if self._dnd else "or press Ctrl+O"
        body_font = tkfont.Font(font=FONTS["body"])
        small_font = tkfont.Font(font=FONTS["small"])
        g = 9 * s  # glyph half-size
        gap = 14 * s
        text_w = max(body_font.measure(main), small_font.measure(sub))
        group_w = 2.4 * g + gap + text_w
        left = max(m + 8 * s, (w - group_w) / 2)
        cy = h / 2
        # "Upload" glyph (arrow into a tray) to the left of the text
        gx = left + 1.2 * g
        self.create_line(gx, cy - g * 1.2, gx, cy + g * 0.3, fill=COLORS["accent"],
                         width=max(2, int(2 * s)), arrow=tk.LAST,
                         arrowshape=(int(7 * s), int(8 * s), int(4 * s)))
        self.create_line(gx - g * 1.2, cy + g * 0.1, gx - g * 1.2, cy + g * 0.9,
                         gx + g * 1.2, cy + g * 0.9, gx + g * 1.2, cy + g * 0.1,
                         fill=COLORS["accent"], width=max(2, int(2 * s)))
        tx = left + 2.4 * g + gap
        line_h = body_font.metrics("linespace")
        self.create_text(tx, cy - line_h * 0.45, text=main, anchor=tk.W,
                         fill=COLORS["fg"], font=FONTS["body"])
        self.create_text(tx, cy + line_h * 0.6, text=sub, anchor=tk.W,
                         fill=COLORS["fg_dim"], font=FONTS["small"])


def ellipsize(text, font, max_px, keep_ext=True):
    """Shorten `text` to fit `max_px` with a middle ellipsis, keeping the
    end (and the file extension) visible: "Team meeting 2024…(final).wav".

    `font` is a tkinter.font.Font.
    """
    if max_px <= 0 or font.measure(text) <= max_px:
        return text
    ell = "\u2026"
    ext_len = 0
    if keep_ext:
        dot = text.rfind(".")
        if 0 < dot and len(text) - dot <= 6:
            ext_len = len(text) - dot
    total = len(text)
    # Keep about a third of the characters at the end (at least the
    # extension), shrinking both sides until it fits
    for keep in range(total - 1, 0, -1):
        tail = min(keep - 1, max(ext_len, keep // 3))
        head = keep - tail
        candidate = text[:head].rstrip() + ell + (text[total - tail:] if tail else "")
        if font.measure(candidate) <= max_px:
            return candidate
    return ell + (text[total - ext_len:] if ext_len else "")


def ellipsize_end(text, font, max_px):
    """'Track 1 · japanese' -> 'Track 1 · jap…' to fit `max_px`."""
    if max_px <= 0 or font.measure(text) <= max_px:
        return text
    for keep in range(len(text) - 1, 0, -1):
        candidate = text[:keep].rstrip() + "\u2026"
        if font.measure(candidate) <= max_px:
            return candidate
    return "\u2026"


class TreeTextFitter:
    """Keeps Treeview cells readable in narrow columns: long values get a
    middle ellipsis (the extension stays) and are re-fitted on resize.

    Call `set(iid, column, full_text)` instead of writing the cell directly.
    """

    def __init__(self, tree, columns, keep_start=()):
        self._tree = tree
        self._columns = tuple(columns)
        self._keep_start = set(keep_start)  # columns cut at the end instead
        self._full = {}  # (iid, column) -> full text
        self._after = None
        self._font = None
        tree.bind("<Configure>", self._schedule, add="+")
        tree.bind("<ButtonRelease-1>", self._schedule, add="+")  # column drag

    def font(self):
        if self._font is None:
            name = ttk.Style(self._tree).lookup("Treeview", "font") or "TkDefaultFont"
            try:
                self._font = tkfont.nametofont(name)
            except tk.TclError:
                self._font = tkfont.Font(font=name)
        return self._font

    def set(self, iid, column, text):
        self._full[(iid, column)] = text
        self._apply(iid, column)

    def full(self, iid, column):
        return self._full.get((iid, column), "")

    def forget(self, iid):
        for col in self._columns:
            self._full.pop((iid, col), None)

    def _apply(self, iid, column):
        if not self._tree.exists(iid):
            return
        try:
            width = int(self._tree.column(column, "width"))
        except tk.TclError:
            return
        pad = int(14 * _scale(self._tree))
        text = self._full.get((iid, column), "")
        if column in self._keep_start:
            fitted = ellipsize_end(text, self.font(), width - pad)
        else:
            fitted = ellipsize(text, self.font(), width - pad)
        self._tree.set(iid, column, fitted)

    def _schedule(self, event=None):
        if self._after is None:
            try:
                self._after = self._tree.after_idle(self.refit)
            except tk.TclError:
                pass

    def refit(self):
        self._after = None
        for (iid, column) in list(self._full):
            if self._tree.exists(iid):
                self._apply(iid, column)
            else:
                self._full.pop((iid, column), None)


def fit_column_to_labels(tree, column, labels, heading=""):
    """Width a fixed column so its longest label (or heading) never clips."""
    name = ttk.Style(tree).lookup("Treeview", "font") or "TkDefaultFont"
    try:
        font = tkfont.nametofont(name)
    except tk.TclError:
        font = tkfont.Font(font=name)
    widest = max([font.measure(t) for t in labels] + [font.measure(heading)])
    width = widest + int(20 * _scale(tree))
    tree.column(column, width=width, minwidth=width, stretch=False)
    return width


def call_in_ui(widget, fn, *args):
    """Schedule fn(*args) on the Tk thread from a worker thread, through the
    app's call queue when there is one (ScrivoxApp.call_soon)."""
    root = widget._root()
    call_soon = getattr(root, "call_soon", None)
    if call_soon is not None:
        call_soon(fn, *args)
    else:
        widget.after(0, fn, *args)


def set_state_recursive(widget, enabled, skip=()):
    """Enable/disable every interactive ttk widget under `widget`.

    Uses ttk state flags so a readonly combobox stays readonly when it is
    re-enabled.
    """
    for child in widget.winfo_children():
        if child in skip:
            continue
        if isinstance(child, (ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Entry,
                              ttk.Combobox, ttk.Spinbox, ttk.Scale)):
            try:
                child.state(["!disabled"] if enabled else ["disabled"])
            except tk.TclError:
                pass
        elif isinstance(child, LinkLabel):
            child.configure(takefocus=enabled)
        set_state_recursive(child, enabled, skip)


class AutocompleteCombobox(ttk.Combobox):
    """Combobox with type-to-filter autocomplete popup.

    As you type, a popup shows matching items below the entry. Click an item
    or use Up/Down + Enter to select. The popup never steals focus, so you
    can keep typing to refine matches. Escape closes the popup. The native
    dropdown arrow still works for browsing the full (or filtered) list.

    multi_value=True: comma-separated input where filtering and selection
    operate on the token after the last comma.
    """

    # Popup colors — imported lazily from theme on first use
    _theme_loaded = False
    _colors = {
        "bg": "#313244",
        "fg": "#cdd6f4",
        "select_bg": "#89b4fa",
        "select_fg": "#1e1e2e",
        "border": "#45475a",
    }

    def __init__(self, master=None, **kwargs):
        self._all_values = list(kwargs.pop("values", []))
        self._multi_value = kwargs.pop("multi_value", False)
        # Optional {list label: stored value}: the list can describe each
        # entry ("large-v3 - most accurate - 3.1 GB") while the field and
        # its variable keep the plain value ("large-v3")
        self._display_map = dict(kwargs.pop("display_map", None) or {})
        self._user_postcommand = kwargs.pop("postcommand", None)
        super().__init__(master, values=self._all_values,
                         postcommand=self._on_post, **kwargs)
        self._debounce_id = None
        self._selecting = False  # guard against re-entrant filtering
        self._multi_prefix = ""  # text before the last comma in multi_value mode
        self._popup = None
        self._listbox = None
        self._popup_active = False
        self.bind("<KeyRelease>", self._on_key)
        self.bind("<Escape>", self._on_escape)
        self.bind("<<ComboboxSelected>>", self._on_combo_select)
        self._load_theme()

    @classmethod
    def _load_theme(cls):
        if cls._theme_loaded:
            return
        try:
            from .theme import COLORS
            cls._colors = {
                "bg": COLORS["entry_bg"],
                "fg": COLORS["entry_fg"],
                "select_bg": COLORS["accent"],
                "select_fg": COLORS["button_fg"],
                "border": COLORS["border"],
            }
        except Exception:
            pass
        cls._theme_loaded = True

    # ── Key handling ──

    def _on_key(self, event):
        # Popup navigation
        if event.keysym == "Down" and self._popup_active:
            self._navigate(1)
            return "break"
        if event.keysym == "Up" and self._popup_active:
            self._navigate(-1)
            return "break"
        if event.keysym == "Return" and self._popup_active:
            self._select_highlighted()
            return "break"
        # Ignore other nav/modifier keys
        if event.keysym in ("Up", "Down", "Left", "Right", "Return",
                            "Tab", "Escape", "Shift_L", "Shift_R",
                            "Control_L", "Control_R", "Alt_L", "Alt_R"):
            return
        if self._selecting:
            return
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(50, self._filter)

    # ── Filtering ──

    def _filter(self):
        self._debounce_id = None
        if self._selecting:
            return
        full = self.get()
        if self._multi_value and "," in full:
            text = full.rsplit(",", 1)[1].strip().lower()
            self._multi_prefix = full.rsplit(",", 1)[0].strip()
        else:
            text = full.lower()
            self._multi_prefix = ""
        if not text:
            self._close_popup()
            self["values"] = self._all_values
            return
        filtered = [v for v in self._all_values if text in v.lower()]
        self["values"] = filtered if filtered else self._all_values
        if filtered:
            self._show_popup(filtered)
        else:
            self._close_popup()

    # ── Custom autocomplete popup ──

    def _show_popup(self, items):
        if self._popup is not None:
            # Update existing popup content
            self._listbox.delete(0, tk.END)
            for item in items:
                self._listbox.insert(tk.END, item)
            self._position_popup(items)
            self._popup_active = True
            return
        self._popup = tk.Toplevel(self)
        self._popup.wm_overrideredirect(True)
        self._popup.wm_attributes('-topmost', True)
        # Prevent the popup from stealing focus
        self._popup.wm_attributes('-disabled', True)

        c = self._colors
        self._listbox = tk.Listbox(
            self._popup,
            selectmode=tk.SINGLE,
            activestyle='none',
            exportselection=False,
            bg=c["bg"], fg=c["fg"],
            selectbackground=c["select_bg"],
            selectforeground=c["select_fg"],
            highlightthickness=0,
            borderwidth=1,
            relief='solid',
            font="TkDefaultFont",
        )
        self._listbox.pack(fill=tk.BOTH, expand=True)

        for item in items:
            self._listbox.insert(tk.END, item)
        self._position_popup(items)
        self._popup_active = True

        # Bind clicks — need to handle manually since popup is disabled
        self._popup.wm_attributes('-disabled', False)
        self._listbox.bind("<Button-1>", self._on_popup_click)
        self._listbox.bind("<ButtonRelease-1>", self._on_popup_release)
        # Re-focus entry after any interaction with popup
        self._listbox.bind("<FocusIn>", lambda e: self.after_idle(self.focus_set))

    def _position_popup(self, items):
        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        width = self.winfo_width()
        num_visible = min(len(items), 8)
        # Row height from the actual font, so the popup fits at any DPI
        try:
            import tkinter.font as tkfont
            item_height = tkfont.nametofont("TkDefaultFont").metrics("linespace") + 2
        except tk.TclError:
            item_height = 20
        height = num_visible * item_height + 4
        self._popup.wm_geometry(f"{width}x{height}+{x}+{y}")

    def _navigate(self, direction):
        if not self._listbox:
            return
        sel = self._listbox.curselection()
        if sel:
            idx = sel[0] + direction
        else:
            idx = 0 if direction > 0 else self._listbox.size() - 1
        if 0 <= idx < self._listbox.size():
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(idx)
            self._listbox.see(idx)

    def _select_highlighted(self):
        if not self._listbox:
            return
        sel = self._listbox.curselection()
        if not sel:
            return
        self._apply_selection(self._listbox.get(sel[0]))

    def _on_popup_click(self, event):
        # Identify which item was clicked
        idx = self._listbox.nearest(event.y)
        if 0 <= idx < self._listbox.size():
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(idx)

    def _on_popup_release(self, event):
        sel = self._listbox.curselection()
        if not sel:
            return
        self._apply_selection(self._listbox.get(sel[0]))
        # Return focus to entry
        self.focus_set()
        self.icursor(tk.END)

    def _apply_selection(self, selected):
        selected = self._display_map.get(selected, selected)
        self._selecting = True
        try:
            if self._multi_value and self._multi_prefix:
                new_text = f"{self._multi_prefix}, {selected}"
            else:
                new_text = selected
            self.set(new_text)
            self.icursor(tk.END)
            self._multi_prefix = ""
        finally:
            self._selecting = False
        self._close_popup()
        self["values"] = self._all_values

    def _close_popup(self, event=None):
        if self._popup is not None:
            self._popup.destroy()
            self._popup = None
            self._listbox = None
        self._popup_active = False

    # ── Native combobox dropdown ──

    def _on_post(self):
        """Runs just before the native list opens. Tk then highlights
        `current()`, which is -1 when the field holds a plain value ("large-v3")
        and the list holds descriptions, so it would mark the first row. Put
        the highlight on the row for the current value once the list is up."""
        if self._user_postcommand is not None:
            self._user_postcommand()
        lb = self._popdown_listbox()
        if lb is not None:
            try:
                # A little inner padding so rows don't touch the list's edge
                # (set before Tk sizes the list, so no row gets cut off)
                self.tk.call(lb, "configure", "-borderwidth",
                             self.winfo_pixels("3p"), "-relief", "flat")
            except tk.TclError:
                pass
        self.after_idle(self._mark_current_row)

    def _popdown_listbox(self):
        try:
            popdown = self.tk.call("ttk::combobox::PopdownWindow", self)
            return f"{popdown}.f.l"
        except tk.TclError:
            return None

    def current_row(self):
        """Index in the list of the row for the field's value, or -1."""
        text = self.get()
        values = list(self.cget("values"))
        if text in values:
            return values.index(text)
        for i, label in enumerate(values):
            if self._display_map.get(label) == text:
                return i
        return -1

    def _mark_current_row(self):
        lb = self._popdown_listbox()
        if lb is None:
            return
        try:
            idx = self.current_row()
            self.tk.call(lb, "selection", "clear", 0, "end")
            if idx >= 0:
                self.tk.call(lb, "selection", "set", idx)
                self.tk.call(lb, "activate", idx)
                self.tk.call(lb, "see", idx)
        except tk.TclError:
            pass

    # ── Native combobox dropdown selection ──

    def _on_combo_select(self, event):
        """Handle selection via the native dropdown (arrow button)."""
        if self._selecting:
            return
        self._selecting = True
        try:
            selected = self.get()
            if selected in self._display_map:
                selected = self._display_map[selected]
                self.set(selected)
                self.icursor(tk.END)
                self.selection_clear()
            if self._multi_value and self._multi_prefix:
                new_text = f"{self._multi_prefix}, {selected}"
                self.set(new_text)
                self.icursor(tk.END)
            self._multi_prefix = ""
        finally:
            self._selecting = False
        self["values"] = self._all_values

    def _on_escape(self, event=None):
        had_popup = self._popup_active
        self._close_popup()
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        self["values"] = self._all_values
        self._multi_prefix = ""
        # Consume the event only when Escape actually dismissed a popup — the
        # app binds <Escape> globally to Cancel, and dismissing autocomplete
        # must not cancel a running batch. With no popup open, let it through
        # so Escape still works as Cancel while the combobox has focus.
        if had_popup:
            return "break"

    def _restore(self, event=None):
        """Restore full values (called externally or on Escape)."""
        self._on_escape(event)

    def set_values(self, values):
        """Update the full values list."""
        self._all_values = list(values)
        self["values"] = self._all_values
