"""ScrivoxApp - main application window assembling all frames with threading.

Layout ("calm three-step studio"):

    +- Setup ---------------------+- Output ------------------------------+
    | (1) Files   drop zone/queue | [preflight banner, only when needed]  |
    | (2) Options model, extras   | headline + progress                   |
    | (3) Save to format, folder  | [Transcript | Log]                    |
    | ...                         |                                       |
    | "Ready: 2 files"            | Saved as x.txt [Open][Show in folder] |
    | [   Start transcription   ] |                                       |
    +-----------------------------+---------------------------------------+

All long work (ffprobe, torch import, the pipeline) runs on worker threads;
they only talk to Tk through after().
"""

import functools
import os
import queue
import re
import shutil
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import ttk, messagebox

# tkinterdnd2 is optional: when its native tkdnd library can't load (missing
# DLL, unsupported platform), the app must still start, browse-only.
try:
    from tkinterdnd2 import TkinterDnD as _TkDnD
    _DnDWrapper = _TkDnD.DnDWrapper
    _dnd_require = _TkDnD._require
except (ImportError, AttributeError):
    _DnDWrapper = object
    _dnd_require = None

from .. import __version__, __app_name__
from ..config import ConfigManager
from ..core.constants import OUTPUT_FORMATS
from ..core.features import has_diarization, get_variant_name
from ..core.pipeline import (
    PipelineConfig, PipelineCancelled, PipelineError, TranscriptionPipeline,
)
from . import winnative
from .output_paths import (
    FIX_FFMPEG, FIX_GPU, FIX_KEYS, MODEL_INFO, explain_error, format_size,
    plan_output_paths,
)
from .theme import COLORS, SP_M, SP_S, SP_XS, configure_theme, px, ui_scale
from .widgets import LinkLabel, StepCard, ToolTip, WrappingLabel, set_state_recursive
from .frames.queue_frame import QueueFrame
from .frames.settings_frame import SettingsFrame
from .frames.output_frame import OutputFrame
from .frames.progress_frame import ProgressFrame
from .frames.log_frame import LogFrame
from .frames.results_frame import ResultsFrame

START_TEXT = "Start transcription"
ERROR_LOG_NAME = "scrivox_error.log"

# Preflight banner text per problem: (title, explanation)
_ISSUE_TEXT = {
    FIX_FFMPEG: ("ffmpeg isn't installed",
                 ("Scrivox needs this free tool to read audio and video. "
                  "Transcription will fail until it is installed.")),
    FIX_GPU: ("No NVIDIA graphics card found",
              ("Transcription runs on an NVIDIA GPU with CUDA. "
               "Update the driver if this PC has one.")),
}


class _RootBase(tk.Tk, _DnDWrapper):
    """tk.Tk with drag-and-drop methods when tkinterdnd2 is installed.

    Unlike TkinterDnD.Tk, a tkdnd load failure is not fatal: it just leaves
    `dnd_available` False.
    """

    def __init__(self):
        tk.Tk.__init__(self)
        self.dnd_available = False
        if _dnd_require is not None:
            try:
                self.TkdndVersion = _dnd_require(self)
                self.dnd_available = True
            except Exception as e:  # RuntimeError('Unable to load tkdnd library.')
                self._dnd_error = f"{type(e).__name__}: {e}"


def restart_command(main_module=None):
    """(argv, cwd) that starts this Scrivox again. The frozen exe restarts
    itself; from source, `python -m scrivox.gui` must be relaunched with -m
    (running gui.py as a script breaks its relative imports)."""
    if getattr(sys, "frozen", False):
        return [sys.executable], os.path.dirname(sys.executable) or None
    main = main_module if main_module is not None else sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None and getattr(spec, "name", ""):
        return [sys.executable, "-m", spec.name], os.getcwd()
    script = os.path.abspath(sys.argv[0])
    return [sys.executable, script], os.path.dirname(script) or None


class ScrivoxApp(_RootBase):
    """Main Scrivox application window."""

    def __init__(self):
        # DPI awareness and taskbar identity must be set BEFORE the Tk window
        # exists, otherwise Windows bitmap-stretches the UI into a blur
        winnative.enable_dpi_awareness()
        winnative.set_app_user_model_id()

        super().__init__()
        self.withdraw()  # build off-screen, show once laid out

        # Worker threads never touch Tk: they queue calls that the Tk thread
        # runs (see call_soon); Tk calls from other threads can crash on Windows
        self._ui_calls = queue.Queue()
        self._drain_id = None
        self._drain_ui_calls()

        # UI scale factor relative to standard 96 DPI
        self._ui_scale = ui_scale(self)

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.config_manager = ConfigManager()
        self._error_log_path = os.path.join(os.path.dirname(self.config_manager.path),
                                            ERROR_LOG_NAME)
        self._install_exception_handlers()
        self._variant = get_variant_name()

        # Title with variant name for Lite/Full
        title = f"{__app_name__} v{__version__}"
        if self._variant != "Regular":
            title += f" ({self._variant})"
        self._base_title = title
        self.title(title)

        # Theme + icon
        configure_theme(self)
        self._set_icon()

        # Pipeline state
        self._pipeline = None
        self._pipeline_thread = None
        self._cancel = threading.Event()
        self._last_switch = 0.0
        self._save_after_id = None
        self._readiness_after_id = None
        self._hint_flash_id = None
        self._closing = False
        self._download_active = False
        self._preflight_issues = []
        self._loading = True
        self._run_jobs = []

        self._build_menu()
        self._build_ui()
        self._set_geometry()

        # Route stray library prints (model downloads, tqdm, ffmpeg helpers)
        # into the log. In a windowed exe they would otherwise vanish.
        from .log_redirect import LogRedirect
        sys.stdout = LogRedirect(self.log_frame, self, original_stdout=self._original_stdout)
        sys.stderr = LogRedirect(self.log_frame, self, original_stdout=self._original_stderr)

        self._load_saved_settings()
        self.settings_frame.set_extras_open(
            bool(self.config_manager.get("ui", "extras_open", False)))
        self._setup_keyboard_shortcuts()
        self._watch_settings()
        self._loading = False
        self._refresh_readiness()

        self.deiconify()
        winnative.use_dark_title_bar(self)
        if self.config_manager.get("ui", "zoomed", False) or self._start_zoomed:
            self._zoom()
        # Kept so closing right after launch cancels them cleanly
        self._startup_after = [
            self.after(30, lambda: self._place_sash(self._initial_sash())),
            self.after(100, self._run_preflight_checks),
            self.after(150, self.queue_frame.focus_add),
        ]

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Thread hand-off ──

    def call_soon(self, fn, *args):
        """Run fn(*args) on the Tk thread shortly. Safe from any thread."""
        self._ui_calls.put((fn, args))

    def _drain_ui_calls(self):
        try:
            for _ in range(1000):
                fn, args = self._ui_calls.get_nowait()
                try:
                    fn(*args)
                except tk.TclError:
                    # Expected when the target widget is already gone (window
                    # closing); anything else is a real bug worth logging
                    owner = getattr(fn, "__self__", None)
                    try:
                        alive = isinstance(owner, tk.Misc) and bool(owner.winfo_exists())
                    except tk.TclError:
                        alive = False
                    if alive:
                        self._log_exception(*sys.exc_info())
                except Exception:
                    self.report_callback_exception(*sys.exc_info())
        except queue.Empty:
            pass
        try:
            self._drain_id = self.after(25, self._drain_ui_calls)
        except tk.TclError:
            self._drain_id = None

    # ── Robustness ──

    def _install_exception_handlers(self):
        """Log unexpected errors to scrivox_error.log and tell the user,
        instead of failing silently (a windowed exe has no console)."""

        def _thread_hook(args):
            if args.exc_type is SystemExit:
                return
            self._report_exception(args.exc_type, args.exc_value, args.exc_traceback,
                                   from_thread=True)

        sys.excepthook = lambda t, v, tb: self._report_exception(t, v, tb)
        threading.excepthook = _thread_hook

    def report_callback_exception(self, exc, val, tb):  # Tk callback errors
        self._report_exception(exc, val, tb)

    def _log_exception(self, exc, val, tb):
        """Append a traceback to scrivox_error.log (and the real stderr).
        Returns (details, log path or None)."""
        details = "".join(traceback.format_exception(exc, val, tb))
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_path = None
        try:
            with open(self._error_log_path, "a", encoding="utf-8") as f:
                f.write(f"--- {stamp} Scrivox {__version__} ({self._variant}) ---\n{details}\n")
            log_path = self._error_log_path
        except OSError:
            pass
        try:
            self._original_stderr.write(details)
        except Exception:
            pass
        return details, log_path

    def _report_exception(self, exc, val, tb, from_thread=False):
        details, log_path = self._log_exception(exc, val, tb)

        def _show():
            # One dialog at a time: a repeating error must not stack windows
            if getattr(self, "_error_dialog", None) is not None:
                try:
                    if self._error_dialog.winfo_exists():
                        return
                except tk.TclError:
                    pass
            from .dialogs.help_dialog import ErrorReportDialog
            try:
                self._error_dialog = ErrorReportDialog(self, details, log_path)
            except tk.TclError:
                self._error_dialog = None
        try:
            if from_thread:
                self.call_soon(_show)
            else:
                _show()
        except (RuntimeError, tk.TclError):
            pass  # interpreter shutting down

    def _set_icon(self):
        """Window/taskbar icon (and the default for every dialog)."""
        if getattr(sys, "frozen", False):
            base = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)),
                                "assets")
        else:
            base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "assets")
        ico = os.path.join(base, "scrivox.ico")
        png = os.path.join(base, "scrivox.png")
        if sys.platform == "win32" and os.path.isfile(ico):
            try:
                self.iconbitmap(default=ico)
                return
            except tk.TclError:
                pass
        if os.path.isfile(png):
            try:
                self._icon_image = tk.PhotoImage(file=png)
                self.iconphoto(True, self._icon_image)
            except tk.TclError:
                pass

    # ── Geometry ──

    def _set_geometry(self):
        """Size the window to fit the work area (taskbar excluded), at any DPI."""
        s = self._ui_scale
        wx, wy, ww, wh = winnative.work_area(self)
        chrome = int(40 * s)  # title bar + borders
        avail_w, avail_h = ww, max(wh - chrome, 200)
        self.minsize(min(int(780 * s), avail_w), min(int(520 * s), avail_h))
        self._start_zoomed = False

        saved = self.config_manager.get("ui", "geometry", "")
        m = re.match(r"^(\d+)x(\d+)([+-]-?\d+)([+-]-?\d+)$", saved or "")
        if m:
            w, h = min(int(m.group(1)), avail_w), min(int(m.group(2)), avail_h)
            x, y = int(m.group(3)), int(m.group(4))
            # Recover if the saved position is off-screen (monitor removed,
            # resolution changed): keep the whole window inside the work area
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            if x < -50 or y < -50 or x > sw - 100 or y > sh - 100:
                x = wx + (ww - w) // 2
                y = wy + (avail_h - h) // 2
            x = max(wx, min(x, wx + ww - w))
            y = max(wy, min(y, wy + avail_h - h))
            self.geometry(f"{w}x{h}+{x}+{y}")
            return

        # First run: 1100x750 at 100% (scaled), never larger than the screen
        want_w, want_h = int(1100 * s), int(750 * s)
        w = min(want_w, int(avail_w * 0.94))
        h = min(want_h, int(avail_h * 0.94))
        x = wx + (ww - w) // 2
        y = wy + (avail_h - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        # Small or high-scale screens: start maximized so nothing is cut off
        self._start_zoomed = want_w > avail_w or want_h > avail_h

    def _zoom(self):
        try:
            self.state("zoomed")
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                pass

    # ── Menu ──

    def _build_menu(self):
        menubar = tk.Menu(self)
        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="Add files…", underline=0, accelerator="Ctrl+O",
                           command=self._browse_files)
        m_file.add_separator()
        m_file.add_command(label=START_TEXT, underline=0, accelerator="Ctrl+Enter",
                           command=self._start_pipeline)
        m_file.add_command(label="Cancel", underline=0, accelerator="Esc",
                           command=lambda: self._cancel_pipeline(confirm=True))
        m_file.add_separator()
        m_file.add_command(label="Show last result in folder", underline=5,
                           command=lambda: self.results_frame._open_folder())
        m_file.add_separator()
        m_file.add_command(label="Exit", underline=1, accelerator="Alt+F4",
                           command=self._on_close)
        menubar.add_cascade(label="File", underline=0, menu=m_file)

        m_edit = tk.Menu(menubar, tearoff=False)
        m_edit.add_command(label="Copy transcript", underline=0, accelerator="Ctrl+Shift+C",
                           command=lambda: self.results_frame._copy())
        m_edit.add_command(label="Clear log", underline=6, accelerator="Ctrl+L",
                           command=lambda: self.log_frame.clear())
        m_edit.add_separator()
        m_edit.add_command(label="Remove finished files", underline=0,
                           command=lambda: self.queue_frame.remove_finished())
        menubar.add_cascade(label="Edit", underline=0, menu=m_edit)

        m_tools = tk.Menu(menubar, tearoff=False)
        m_tools.add_command(label="Settings…", underline=0, accelerator="Ctrl+,",
                            command=lambda: self.open_settings())
        menubar.add_cascade(label="Tools", underline=0, menu=m_tools)

        m_help = tk.Menu(menubar, tearoff=False)
        m_help.add_command(label="Installing ffmpeg", underline=0,
                           command=lambda: self._show_fix(FIX_FFMPEG))
        m_help.add_command(label="Graphics card requirements", underline=0,
                           command=lambda: self._show_fix(FIX_GPU))
        m_help.add_separator()
        m_help.add_command(label="Open settings folder", underline=5,
                           command=self._open_config_folder)
        m_help.add_separator()
        m_help.add_command(label=f"About {__app_name__}", underline=0, command=self._about)
        menubar.add_cascade(label="Help", underline=0, menu=m_help)
        self.configure(menu=menubar)
        self._file_menu = m_file

    # ── Layout ──

    def _build_ui(self):
        s = self._ui_scale

        # ── Status bar (bottom) ──
        status = ttk.Frame(self, style="Bar.TFrame", padding=(px(12), px(3)))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_bar = ttk.Label(status, text="", style="BarDim.TLabel", anchor=tk.W)
        self._status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._version_label = ttk.Label(status, text=f"v{__version__} · {self._variant}",
                                        style="BarDim.TLabel")
        self._version_label.pack(side=tk.RIGHT)

        # ── Resizable two-column body ──
        self._paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=COLORS["bg"],
                                     sashwidth=int(8 * s), borderwidth=0, sashrelief=tk.FLAT,
                                     opaqueresize=True, showhandle=False)
        self._paned.pack(fill=tk.BOTH, expand=True, padx=px(8), pady=(px(8), px(8)))

        left_panel = ttk.Frame(self._paned)
        right_panel = ttk.Frame(self._paned)
        self._paned.add(left_panel, minsize=int(360 * s), stretch="never")
        self._paned.add(right_panel, minsize=int(380 * s), stretch="always")
        self._left_panel = left_panel

        self._build_action_bar(left_panel)
        self._build_setup_column(left_panel)
        self._build_output_column(right_panel)

        # Settings window (hidden until asked for); its frames hold the
        # API-key and advanced variables, so it is created up front
        from .dialogs.settings_dialog import SettingsDialog
        self.settings_dialog = SettingsDialog(self, self.config_manager,
                                              show_keys=has_diarization(),
                                              show_diarization=has_diarization())
        self.api_frame = self.settings_dialog.api_frame
        self.models_frame = self.settings_dialog.models_frame
        self.settings_dialog.bind("<<SettingsClosed>>", lambda e: self._refresh_readiness())

    def _build_action_bar(self, parent):
        bar = ttk.Frame(parent, padding=(0, px(8), px(12), 0))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._action_hint = WrappingLabel(bar, text="", style="Dim.TLabel", anchor=tk.W,
                                          justify=tk.LEFT)
        self._action_hint.pack(fill=tk.X, pady=(0, SP_XS))
        self._action_hint.bind("<Button-1>", lambda e: self._fix_first_problem())
        self._start_btn = ttk.Button(bar, text=START_TEXT, style="Accent.TButton",
                                     command=self._on_primary, underline=0)
        self._start_btn.pack(fill=tk.X)
        # One slot for Start/Cancel: the running state swaps label and style
        self._cancel_btn = self._start_btn

    def _build_setup_column(self, parent):
        wrap = ttk.Frame(parent)
        wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        left_canvas = tk.Canvas(wrap, bg=COLORS["bg"], highlightthickness=0, borderwidth=0)
        left_scrollbar = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=left_canvas.yview)
        self._left_inner = ttk.Frame(left_canvas, padding=(0, 0, px(4), 0))
        self._left_canvas = left_canvas
        self._left_scrollbar = left_scrollbar

        self._scroll_update_id = None

        def _update_scrollregion(event=None):
            if self._scroll_update_id:
                left_canvas.after_cancel(self._scroll_update_id)
            self._scroll_update_id = left_canvas.after(20, _apply_scrollregion)

        def _apply_scrollregion():
            self._scroll_update_id = None
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            # Hide the scrollbar when everything fits
            needs = self._left_inner.winfo_reqheight() > left_canvas.winfo_height() + 1
            if needs and not left_scrollbar.winfo_manager():
                left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=left_canvas)
            elif not needs and left_scrollbar.winfo_manager():
                left_scrollbar.pack_forget()
                left_canvas.yview_moveto(0)

        self._left_inner.bind("<Configure>", _update_scrollregion)

        def _on_canvas_configure(event):
            left_canvas.itemconfigure(self._canvas_window, width=event.width)
            _update_scrollregion()

        self._canvas_window = left_canvas.create_window((0, 0), window=self._left_inner,
                                                        anchor=tk.NW)
        left_canvas.bind("<Configure>", _on_canvas_configure)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse wheel scrolls the setup column wherever the pointer is over
        # it, including over comboboxes and spinboxes, whose own wheel
        # bindings would otherwise silently change the selected value
        self._mousewheel_bound = True
        self.bind_all("<MouseWheel>", self._on_wheel, add="+")
        for cls in ("TCombobox", "TSpinbox"):
            self.bind_class(cls, "<MouseWheel>", self._on_value_widget_wheel)
            if sys.platform.startswith("linux"):
                self.bind_class(cls, "<Button-4>", self._on_value_widget_wheel)
                self.bind_class(cls, "<Button-5>", self._on_value_widget_wheel)
        if sys.platform.startswith("linux"):
            self.bind_all("<Button-4>", self._on_wheel, add="+")
            self.bind_all("<Button-5>", self._on_wheel, add="+")
        self.bind_all("<FocusIn>", self._ensure_focus_visible, add="+")

        inner = self._left_inner
        card1 = StepCard(inner, 1, "Files")
        card1.pack(fill=tk.X, pady=(0, SP_S))
        self.queue_frame = QueueFrame(
            card1.body, config_manager=self.config_manager,
            on_tracks_needed=self._show_track_dialog,
            on_change=self._schedule_readiness,
        )
        self.queue_frame.pack(fill=tk.BOTH, expand=True)

        card2 = StepCard(inner, 2, "Options")
        card2.pack(fill=tk.X, pady=(0, SP_S))
        # Everything else (AI keys, subtitle tuning, hardware) is one click
        # away in the card's header, not a line of its own
        more = LinkLabel(card2.header, text="More settings…", command=self.open_settings)
        more.pack(side=tk.RIGHT)
        ToolTip(more, ("AI service keys, speaker model, subtitle timing, accuracy and "
                       "graphics card options (Ctrl+,)") if has_diarization()
                else "Subtitle timing, accuracy and graphics card options (Ctrl+,)")
        self._more_link = more
        self.settings_frame = SettingsFrame(card2.body, on_setup_keys=self.open_settings)
        self.settings_frame.pack(fill=tk.X)
        self.settings_frame.bind("<<ExtrasChanged>>", lambda e: self._schedule_readiness())
        self.settings_frame.bind("<<ExtrasToggled>>", lambda e: self._on_setting_changed())

        card3 = StepCard(inner, 3, "Save to")
        card3.pack(fill=tk.X, pady=(0, SP_XS))
        self.output_frame = OutputFrame(card3.body, config_manager=self.config_manager)
        self.output_frame.pack(fill=tk.X)
        self._cards = (card1, card2, card3)

    def _build_output_column(self, parent):
        body = ttk.Frame(parent, padding=(px(12), 0, 0, 0))
        body.pack(fill=tk.BOTH, expand=True)

        # Preflight banner (hidden unless something blocks transcription)
        self._banner = ttk.Frame(body, style="Warning.TFrame", padding=(px(12), px(10)))
        self._banner_rows = ttk.Frame(self._banner, style="Warning.TFrame")
        self._banner_rows.pack(fill=tk.X)

        self.progress_frame = ProgressFrame(body, on_progress_change=self._on_overall_progress)
        self.progress_frame.pack(fill=tk.X, pady=(0, SP_M))

        self._notebook = ttk.Notebook(body)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        self.results_frame = ResultsFrame(self._notebook, config_manager=self.config_manager,
                                          padding=(0, px(8), 0, 0))
        self.log_frame = LogFrame(self._notebook, padding=(0, px(8), 0, 0))
        self._notebook.add(self.results_frame, text="Transcript", underline=0)
        self._notebook.add(self.log_frame, text="Log", underline=0)
        self._notebook.enable_traversal()

    # ── Scrolling ──

    def _in_setup_column(self, widget):
        w = widget
        while w is not None:
            if w is self._left_canvas:
                return True
            w = getattr(w, "master", None)
        return False

    def _scroll_setup(self, event):
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
            step *= max(1, abs(event.delta) // 120)
        if self._left_scrollbar.winfo_manager():
            self._left_canvas.yview_scroll(step, "units")

    def _on_wheel(self, event):
        widget = event.widget
        if isinstance(widget, str):
            try:
                widget = self.nametowidget(widget)
            except KeyError:
                return
        # Widgets that scroll themselves keep their own wheel behaviour
        if isinstance(widget, (tk.Text, ttk.Treeview, tk.Listbox)):
            return
        if self._in_setup_column(widget):
            self._scroll_setup(event)

    def _on_value_widget_wheel(self, event):
        """Wheel over a combobox/spinbox scrolls the panel, never the value."""
        if self._in_setup_column(event.widget):
            self._scroll_setup(event)
        return "break"

    def _ensure_focus_visible(self, event):
        """Keyboard users tabbing through the setup column see the focus."""
        widget = event.widget
        if isinstance(widget, str) or not self._in_setup_column(widget) \
                or widget is self._left_canvas:
            return
        try:
            inner_h = max(self._left_inner.winfo_height(), 1)
            top = widget.winfo_rooty() - self._left_inner.winfo_rooty()
            bottom = top + widget.winfo_height()
            view_top, view_bottom = (f * inner_h for f in self._left_canvas.yview())
            pad = int(24 * self._ui_scale)
            if top < view_top:
                self._left_canvas.yview_moveto(max(0, top - pad) / inner_h)
            elif bottom > view_bottom:
                visible = view_bottom - view_top
                self._left_canvas.yview_moveto(max(0, bottom + pad - visible) / inner_h)
        except tk.TclError:
            pass

    # ── Keyboard ──

    def _setup_keyboard_shortcuts(self):
        """Bind keyboard shortcuts (also listed in the menus)."""
        self.bind_all("<Control-o>", lambda e: self._browse_files())
        self.bind_all("<Control-O>", lambda e: self._browse_files())
        self.bind_all("<Control-Return>", lambda e: self._start_pipeline())
        self.bind_all("<Escape>", lambda e: self._cancel_pipeline(confirm=True))
        self.bind_all("<Control-l>", lambda e: self.log_frame.clear())
        self.bind_all("<Control-comma>", lambda e: self.open_settings())
        self.bind_all("<Control-Shift-C>", lambda e: self.results_frame._copy())
        self.bind_all("<F1>", lambda e: self._show_fix(FIX_FFMPEG))
        # Mnemonics for the underlined letters on the main buttons
        self.bind_all("<Alt-a>", lambda e: self._browse_files())
        self.bind_all("<Alt-s>", lambda e: self._start_pipeline())

    def _browse_files(self):
        # Queue is locked while running; Ctrl+O must respect that too
        if not self._is_running:
            self.queue_frame.browse_files()

    # ── Preflight ──

    def _run_preflight_checks(self, on_done=None):
        """Check ffmpeg and GPU availability without blocking the window
        (`import torch` alone takes several seconds). `on_done(issues)` is
        called on the Tk thread afterwards (used by "Check again")."""
        self._status_bar.configure(text="Checking graphics card and ffmpeg…")

        def _check():
            issues = []
            parts = []
            # "Check again": pick up programs installed since Scrivox started
            # (e.g. winget install ffmpeg). Windows only updates the registry,
            # not the PATH this process inherited. Not done at startup, where
            # the inherited PATH is already current.
            if on_done is not None:
                winnative.refresh_path()

            # Determine CUDA source label
            if getattr(sys, "frozen", False):
                _flag = os.path.join(sys._MEIPASS, '..', 'use_system_cuda')
            else:
                _flag = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__)))),
                    'use_system_cuda')
            cuda_source = "system" if os.path.isfile(_flag) else "bundled"

            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    cuda_ver = torch.version.cuda or "unknown"
                    parts.append(f"GPU: {gpu_name} · CUDA {cuda_ver} ({cuda_source})")
                else:
                    parts.append("GPU: no NVIDIA card found")
                    issues.append(FIX_GPU)
            except Exception as e:
                parts.append(f"GPU: unavailable ({type(e).__name__})")
                issues.append(FIX_GPU)

            missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
            if missing:
                parts.append("ffmpeg: not installed")
                issues.insert(0, FIX_FFMPEG)
            else:
                parts.append("ffmpeg: ready")

            try:
                self.call_soon(self._apply_preflight, "   ·   ".join(parts), issues)
                if on_done is not None:
                    self.call_soon(on_done, issues)
            except (RuntimeError, tk.TclError):
                pass  # window closed during startup checks

        threading.Thread(target=_check, daemon=True).start()


    def _apply_preflight(self, status_text, issues):
        self._status_bar.configure(text=status_text)
        self._preflight_issues = issues
        for child in self._banner_rows.winfo_children():
            child.destroy()
        if not issues:
            self._banner.pack_forget()
            self._refresh_readiness()
            return
        for i, topic in enumerate(issues):
            title, detail = _ISSUE_TEXT[topic]
            row = ttk.Frame(self._banner_rows, style="Warning.TFrame")
            row.pack(fill=tk.X, pady=(SP_S if i else 0, 0))
            ttk.Label(row, text="⚠", style="BannerIcon.TLabel").pack(
                side=tk.LEFT, anchor=tk.N, padx=(0, SP_S))
            ttk.Button(row, text="How to fix", style="Banner.TButton",
                       command=lambda t=topic: self._show_fix(t)).pack(
                side=tk.RIGHT, anchor=tk.N, padx=(SP_S, 0))
            text = ttk.Frame(row, style="Warning.TFrame")
            text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            WrappingLabel(text, text=title, style="BannerTitle.TLabel",
                          justify=tk.LEFT).pack(fill=tk.X)
            WrappingLabel(text, text=detail, style="BannerDim.TLabel",
                          justify=tk.LEFT).pack(fill=tk.X)
        self._banner.pack(fill=tk.X, pady=(0, SP_M), before=self.progress_frame)
        self._refresh_readiness()

    def _show_fix(self, topic):
        if topic == FIX_KEYS:
            self.open_settings("keys")
            return
        from .dialogs.help_dialog import FixHelpDialog
        FixHelpDialog(self, topic,
                      on_recheck=lambda done: self._run_preflight_checks(on_done=done),
                      on_restart=self._restart)

    def _restart(self):
        """Start a fresh Scrivox and close this one. A program installed
        while Scrivox was open is then certainly found."""
        if self._is_running:
            messagebox.showinfo("Scrivox", "Scrivox can restart once the current "
                                           "transcription has finished.", parent=self)
            return
        cmd, cwd = restart_command()
        self._save_current_settings()  # the new window starts from them
        env = dict(os.environ)
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"  # a fully separate new process
        try:
            import subprocess
            subprocess.Popen(cmd, env=env, close_fds=True, cwd=cwd)
        except OSError as e:
            messagebox.showerror("Scrivox", f"Couldn't restart Scrivox:\n{e}\n\n"
                                            "Close it and open it again from the Start menu.",
                                 parent=self)
            return
        self._on_close()

    # ── Dialogs ──

    def open_settings(self, tab=None):
        self.settings_dialog.show(tab)

    def _open_config_folder(self):
        path = self.config_manager.path
        winnative.reveal_in_folder(path if os.path.exists(path) else os.path.dirname(path))

    def _about(self):
        messagebox.showinfo(
            f"About {__app_name__}",
            f"{__app_name__} {__version__} ({self._variant})\n\n"
            "Transcribe audio and video on your NVIDIA graphics card, with optional "
            "speaker labels, summaries and translation.\n\n"
            "Developed by Deenyoro at KawaConnect LLC.",
            parent=self)

    def _show_track_dialog(self, filepath, tracks):
        """Show track selection dialog. Returns list of selected track indices."""
        from .dialogs.track_dialog import TrackDialog
        filename = os.path.basename(filepath)
        dialog = TrackDialog(self, filename, tracks)
        self.wait_window(dialog)
        return dialog.result

    # ── Settings persistence ──

    def _load_saved_settings(self):
        """Load last-used settings from config."""
        settings = self.config_manager.get_last_settings()
        self.settings_frame.load_settings(settings)
        if self.models_frame:
            self.models_frame.load_settings(settings)
        fmt = settings.get("output_format", "txt")
        if fmt in OUTPUT_FORMATS:
            self.output_frame.format_var.set(fmt)
        self.output_frame.subtitle_speakers_var.set(
            bool(settings.get("subtitle_speakers", False)))

    def _initial_sash(self):
        sash = self.config_manager.get("ui", "sash", None)
        return sash if isinstance(sash, int) and sash > 0 else int(440 * self._ui_scale)

    def _place_sash(self, x):
        try:
            self.update_idletasks()
            total = self._paned.winfo_width()
            if total < 100:  # not laid out yet
                self.after(50, lambda: self._place_sash(x))
                return
            x = max(int(360 * self._ui_scale), min(x, total - int(380 * self._ui_scale)))
            self._paned.sash_place(0, x, 1)
        except tk.TclError:
            pass

    def _watch_settings(self):
        """Save settings shortly after any change, so a crash or power cut
        doesn't lose them."""
        watched = list(self.settings_frame.all_vars())
        watched += [self.output_frame.format_var, self.output_frame.subtitle_speakers_var]
        if self.models_frame:
            watched += self.models_frame.all_vars()
        if self.api_frame:
            watched += [self.api_frame.hf_token_var, self.api_frame.openrouter_key_var,
                        self.api_frame.anthropic_key_var, self.api_frame.provider_var,
                        self.api_frame.custom_base_var]
        for var in watched:
            var.trace_add("write", lambda *a: self._on_setting_changed())
        self.output_frame.output_dir_var.trace_add("write", lambda *a: self._schedule_readiness())

    def _on_setting_changed(self):
        if self._loading or self._closing:
            return
        self._schedule_readiness()
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
        self._save_after_id = self.after(1000, self._autosave)

    def _autosave(self):
        self._save_after_id = None
        try:
            self._save_current_settings()
        except Exception as e:  # never let a save problem interrupt the user
            self._on_progress(f"Warning: couldn't save settings ({type(e).__name__}: {e})")

    def _save_current_settings(self):
        """Save current settings to config."""
        settings = self.settings_frame.get_settings_dict()
        if self.models_frame:
            settings.update(self.models_frame.get_settings_dict())
        settings["output_format"] = self.output_frame.format_var.get()
        settings["subtitle_speakers"] = self.output_frame.subtitle_speakers_var.get()
        self.config_manager.save_last_settings(**settings)
        if self.api_frame:
            self.api_frame.save_to_config()
        try:
            zoomed = self.state() == "zoomed"
        except tk.TclError:
            zoomed = False
        self.config_manager.set("ui", "zoomed", zoomed)
        self.config_manager.set("ui", "extras_open", self.settings_frame.extras_open)
        if not zoomed:
            self.config_manager.set("ui", "geometry", self.geometry())
        try:
            self.config_manager.set("ui", "sash", int(self._paned.sash_coord(0)[0]))
        except (tk.TclError, IndexError):
            pass
        self.config_manager.save()

        # Write or delete the use_system_cuda flag file
        if getattr(sys, "frozen", False):
            flag_path = os.path.join(sys._MEIPASS, '..', 'use_system_cuda')
        else:
            flag_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                'use_system_cuda')
        try:
            if self.models_frame.use_system_cuda_var.get():
                if not os.path.isfile(flag_path):
                    with open(flag_path, 'w'):
                        pass  # empty flag file
            else:
                if os.path.isfile(flag_path):
                    os.remove(flag_path)
        except OSError:
            pass

    # ── Readiness / validation ──

    def _missing_keys(self):
        missing = set()
        if not self.api_frame:
            return missing
        if not self.api_frame.get_hf_token() and not self.api_frame._has_bundled:
            missing.add("hf")
        base = self.api_frame.get_api_base() or ""
        is_local = any(h in base for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not self.api_frame.get_openrouter_key() and not is_local:
            missing.add("llm")
        return missing

    def _problems(self, include_files=True):
        """Everything that blocks a run, as (message, fix) where fix is a
        widget to focus or a callable."""
        problems = []
        s = self.settings_frame
        if include_files:
            if self.queue_frame.is_checking:
                problems.append(("Checking the added files…", None))
            elif not self.queue_frame.has_jobs:
                problems.append(("Add at least one audio or video file to start",
                                 self.queue_frame.focus_add))
            for job in self.queue_frame.get_jobs():
                if not os.path.isfile(job.file_path):
                    problems.append((f"File not found: {os.path.basename(job.file_path)}",
                                     None))
                    break
        if not self.api_frame and (s.diarize_var.get() or s.vision_var.get()
                                   or s.summarize_var.get() or s.translate_var.get()):
            problems.append(("Extras aren't available in the Lite build", None))
        for msg, target in s.problems(self._missing_keys()):
            if target == "keys":
                target = functools.partial(self.open_settings, "keys")
            problems.append((msg, target))
        if self.models_frame:
            for msg, widget in self.models_frame.problems():
                problems.append((msg, widget))
        out_dir = self.output_frame.output_dir_var.get()
        if out_dir and not os.path.isdir(out_dir):
            problems.append(("The save folder no longer exists. Choose another one",
                             self.output_frame._browse_btn))
        return problems

    def _schedule_readiness(self):
        if self._readiness_after_id is None and not self._closing:
            self._readiness_after_id = self.after(120, self._refresh_readiness)

    def _refresh_readiness(self):
        """Update the line above the Start button and the missing-key links."""
        if self._readiness_after_id is not None:
            # Called directly while a scheduled refresh is pending: that one
            # is now redundant (and must not outlive the window)
            try:
                self.after_cancel(self._readiness_after_id)
            except tk.TclError:
                pass
            self._readiness_after_id = None
        if self._closing:
            return
        if self._loading:
            return
        self.settings_frame.set_key_hints(self._missing_keys())
        if self._is_running:
            return
        jobs = self.queue_frame.get_jobs()
        self.output_frame.set_jobs([(j.file_path, j.audio_track) for j in jobs])
        problems = self._problems()
        self._problems_cache = problems
        self.progress_frame.show_ready(
            len(jobs), blocked=bool(problems) or FIX_FFMPEG in self._preflight_issues)
        # Start looks unavailable while something blocks the run (it stays
        # focusable: pressing it explains what's missing)
        self._start_btn.configure(style="AccentBlocked.TButton" if problems
                                  else "Accent.TButton")
        if self._hint_flash_id is not None:
            if problems:
                return  # a refused Start is being explained; don't overwrite it
            self._cancel_hint_flash()
        if problems:
            msg, fix = problems[0]
            about_files = fix is None or fix == self.queue_frame.focus_add
            if len(problems) > 1 and not about_files:
                msg = f"{len(problems)} things to fix before starting. First: {msg[0].lower()}{msg[1:]}"
            self._action_hint.configure(
                text=msg, style="Dim.TLabel" if about_files else "SmallError.TLabel",
                cursor="" if fix is None else "hand2")
            return
        self._problems_cache = []
        fmt = self.output_frame.format_var.get()
        where = "in your chosen folder" if self.output_frame.output_dir_var.get() \
            else "next to each file" if len(jobs) > 1 else "next to the original"
        ready = f"Ready: {len(jobs)} file{'s' if len(jobs) != 1 else ''} → .{fmt} {where}"
        if FIX_FFMPEG in self._preflight_issues:
            ready = "ffmpeg is missing, so transcription will fail. See \u201cHow to fix\u201d."
            self._action_hint.configure(text=ready, style="Warning.TLabel", cursor="")
            return
        self._action_hint.configure(text=ready, style="Dim.TLabel", cursor="")

    def _fix_first_problem(self):
        problems = getattr(self, "_problems_cache", None) or self._problems()
        if not problems:
            return
        fix = problems[0][1]
        if fix is None:
            return
        if fix == self.queue_frame.focus_add:
            self.queue_frame.pulse()
        elif callable(fix) and not isinstance(fix, tk.Misc):
            fix()
        elif isinstance(fix, tk.Misc):
            if self.settings_dialog.focus_widget(fix):
                return
            self.settings_frame.reveal(fix)  # opens Extras if it's in there
            self.update_idletasks()
            fix.focus_set()

    def _validate(self):
        """Validate inputs before starting; problems are shown inline next
        to the Start button (no pop-ups) and the first one gets focus."""
        problems = self._problems()
        if not problems:
            return True
        self._problems_cache = problems
        self._refresh_readiness()
        self.bell()
        self._flash_hint()
        self._fix_first_problem()
        return False

    def _flash_hint(self):
        """A refused Start must visibly react: the reason turns red for ~2 s."""
        if self._hint_flash_id is not None:
            self.after_cancel(self._hint_flash_id)
        text = self._action_hint.cget("text")
        if text and not text.startswith("\u26a0"):
            text = "\u26a0  " + text
        self._action_hint.configure(text=text, style="SmallError.TLabel")
        self._hint_flash_id = self.after(2000, self._end_hint_flash)

    def _cancel_hint_flash(self):
        if self._hint_flash_id is not None:
            self.after_cancel(self._hint_flash_id)
            self._hint_flash_id = None

    def _end_hint_flash(self):
        self._cancel_hint_flash()
        self._refresh_readiness()

    # ── Run ──

    def _build_config(self, job=None):
        """Build PipelineConfig from current GUI state and optional job."""
        s = self.settings_frame

        # Determine file path and audio track
        file_path = job.file_path if job else self.queue_frame.file_path
        audio_track = job.audio_track if job else 0
        language = job.language_override if (job and job.language_override) else s.get_language_code()

        # Get advanced settings from models frame
        adv = self.models_frame.get_settings_dict() if self.models_frame else {}

        return PipelineConfig(
            input_path=file_path,
            model=s.model_var.get().strip() or "large-v3",
            language=language,
            diarize=s.diarize_var.get(),
            vision=s.vision_var.get(),
            summarize=s.summarize_var.get(),
            diarization_model=(self.models_frame.get_diarization_model()
                               if self.models_frame else "pyannote/speaker-diarization-community-1"),
            num_speakers=s.get_int_or_none(s.num_speakers_var),
            min_speakers=s.get_int_or_none(s.min_speakers_var),
            max_speakers=s.get_int_or_none(s.max_speakers_var),
            speaker_names=s.get_speaker_names(),
            vision_interval=s._safe_float(s.vision_interval_var.get(), 60.0),
            vision_model=s.vision_model_var.get(),
            vision_workers=s._safe_int(s.vision_workers_var.get(), 4),
            vision_change_threshold=s._safe_int(s.vision_change_threshold_var.get(), 0),
            summary_model=s.summary_model_var.get(),
            translate=s.translate_var.get(),
            translate_all=s.translate_all_var.get(),
            translate_to=s.get_translate_to_codes(),
            translation_model=s.translation_model_var.get(),
            output_format=self.output_frame.format_var.get(),
            output_path=self.output_frame.output_path_var.get() or None,
            subtitle_speakers=self.output_frame.subtitle_speakers_var.get(),
            subtitle_max_chars=adv.get("subtitle_max_chars", 84),
            subtitle_max_duration=adv.get("subtitle_max_duration", 4.0),
            subtitle_max_gap=adv.get("subtitle_max_gap", 0.8),
            subtitle_min_chars=adv.get("subtitle_min_chars", 15),
            confidence_threshold=adv.get("confidence_threshold", 0.50),
            api_base=(self.api_frame.get_api_base() if self.api_frame else None),
            hf_token=(self.api_frame.get_hf_token() if self.api_frame else None),
            openrouter_key=(self.api_frame.get_openrouter_key() if self.api_frame else None),
            audio_track=audio_track,
        )

    @property
    def _is_running(self):
        return self._pipeline_thread is not None and self._pipeline_thread.is_alive()

    def _on_primary(self):
        """The single Start/Cancel button."""
        # Ignore a double-click landing on the freshly swapped button
        if time.monotonic() - self._last_switch < 0.8:
            return
        if self._is_running:
            self._cancel_pipeline()
        else:
            self._start_pipeline()

    def _set_running(self, running):
        """Swap Start/Cancel and lock inputs during pipeline execution."""
        self._last_switch = time.monotonic()
        self._cancel_hint_flash()
        if running:
            self._start_btn.configure(text="Cancel", style="Cancel.TButton", underline=-1)
            self._start_btn.state(["!disabled"])
            self._action_hint.configure(
                text="Working… options are locked until this run finishes.",
                style="Dim.TLabel", cursor="")
        else:
            self._start_btn.configure(text=START_TEXT, style="Accent.TButton", underline=0)
            self._start_btn.state(["!disabled"])
            self.title(self._base_title)
        self._file_menu.entryconfigure(START_TEXT, state=tk.DISABLED if running else tk.NORMAL)
        self._file_menu.entryconfigure("Cancel", state=tk.NORMAL if running else tk.DISABLED)
        # Lock the queue and options while running: configs were captured at
        # Start, so edits would silently apply to nothing
        self.queue_frame.set_enabled(not running)
        set_state_recursive(self.settings_frame, not running)
        self.output_frame.set_enabled(not running)
        set_state_recursive(self.settings_dialog.notebook, not running)
        if not running:
            self._refresh_readiness()

    def _on_overall_progress(self, pct):
        if self._is_running and not self._download_active:
            label = f"{int(pct)}%" if pct >= 1 else "Working…"
            self.title(f"{label} · {self._base_title}")

    def _on_progress(self, msg):
        """Thread-safe progress callback: schedules log append on main thread."""
        try:
            self.call_soon(self.log_frame.append, msg + "\n")
        except (RuntimeError, tk.TclError):
            pass

    def _on_step(self, step_num, total_steps, step_name):
        """Thread-safe step update callback."""
        try:
            self.call_soon(self.progress_frame.update_step, step_num, total_steps,
                       _friendly_step(step_name))
        except (RuntimeError, tk.TclError):
            pass

    def _on_fraction(self, frac):
        """Thread-safe within-step progress callback."""
        try:
            self.call_soon(self._apply_fraction, frac)
        except (RuntimeError, tk.TclError):
            pass

    def _apply_fraction(self, frac):
        self._download_active = False  # transcription itself is under way
        self.progress_frame.set_step_fraction(frac)

    def _on_download(self, model, done):
        """Thread-safe first-run model download progress (done=None: the
        download finished and the model is loading)."""
        try:
            self.call_soon(self._show_download, model, done)
        except (RuntimeError, tk.TclError):
            pass

    def _show_download(self, model, done):
        if done is None:
            self._download_active = False
            self.progress_frame.set_status(
                "Loading the speech model…", None,
                f"{model} is downloaded. Getting it ready on the graphics card.")
            self.title(f"Loading… · {self._base_title}")
            return
        self._download_active = True
        total = MODEL_INFO.get(model, (None, None))[1]
        text = "Downloading the speech model (first run only)"
        if total:
            detail = f"{model}: {format_size(done)} of about {format_size(total)}"
            frac = min(done / total, 0.99)
            self.title(f"Downloading {int(frac * 100)}% · {self._base_title}")
        else:
            detail = f"{model}: {format_size(done)} so far"
            frac = None
            self.title(f"Downloading… · {self._base_title}")
        self.progress_frame.set_status(text, frac, detail)

    def _choose_jobs(self):
        """All jobs, or only the unfinished ones if some are already done."""
        ids = self.queue_frame.get_job_ids()
        jobs = self.queue_frame.get_jobs()
        statuses = self.queue_frame.get_job_statuses()
        done = [i for i, st in enumerate(statuses) if st == "done"]
        if done and len(done) < len(jobs):
            new = len(jobs) - len(done)
            answer = messagebox.askyesnocancel(
                "Scrivox",
                f"{len(done)} file{'s are' if len(done) != 1 else ' is'} already transcribed.\n\n"
                f"Transcribe only the {new} new file{'s' if new != 1 else ''}?\n"
                "(Choose No to transcribe everything again.)",
                parent=self)
            if answer is None:
                return None
            if answer:
                keep = [i for i in range(len(jobs)) if i not in done]
                return [(ids[i], jobs[i]) for i in keep]
        return list(zip(ids, jobs))

    def _start_pipeline(self):
        """Validate inputs and start the pipeline in a background thread."""
        if self._is_running:
            return  # Ctrl+Return can fire while a batch is already running
        if not self._validate():
            return
        selected = self._choose_jobs()
        if not selected:
            return

        self._save_current_settings()
        self._cancel.clear()

        # Prepare UI
        self._set_running(True)
        self.log_frame.clear()
        self.results_frame.clear()
        self.progress_frame.reset()
        self.progress_frame.start()
        self._notebook.select(self.results_frame)

        iids = [iid for iid, _ in selected]
        jobs = [job for _, job in selected]
        total_jobs = len(jobs)
        for iid in iids:
            self.queue_frame.set_job_status(iid, "pending")

        # Build every job's config NOW, on the main thread: Tk variables must
        # not be read from the worker thread, and snapshotting up front means
        # mid-run widget edits can't silently change later jobs.
        fmt = self.output_frame.format_var.get()
        out_paths = plan_output_paths(
            [(j.file_path, j.audio_track) for j in jobs], fmt,
            explicit_output=self.output_frame.output_path_var.get(),
            out_dir=self.output_frame.output_dir_var.get())
        configs = []
        for job, out_path in zip(jobs, out_paths):
            config = self._build_config(job)
            config.output_path = out_path
            configs.append(config)
        self._run_fmt = fmt
        self._download_active = False
        # A name picked with "Rename..." is for one successful run only
        # (dropped in _on_batch_complete), so running again never
        # overwrites that file; a failed or cancelled run keeps it
        self._run_explicit_name = self.output_frame.output_path_var.get()

        def _run_batch():
            results = []   # (index, PipelineResult)
            errors = []    # (index, message)
            try:
                for i, (job, config) in enumerate(zip(jobs, configs)):
                    if self._cancel.is_set():
                        self.call_soon(self._on_pipeline_cancelled, results, errors, jobs)
                        return

                    # Update file-level progress
                    filename = os.path.basename(job.file_path)
                    track_info = f" ({job.track_label})" if job.track_label else ""
                    self.call_soon(self.progress_frame.update_file,
                               i + 1, total_jobs, f"{filename}{track_info}")
                    self.call_soon(self.queue_frame.set_job_status, iids[i], "running")

                    # Add job separator in log
                    if i > 0:
                        self._on_progress("\n" + "=" * 60)
                        self._on_progress(f"  JOB {i + 1}/{total_jobs}")
                        self._on_progress("=" * 60)

                    pipeline = _make_pipeline(
                        config,
                        on_progress=self._on_progress,
                        on_step=self._on_step,
                        cancel_event=self._cancel,
                        on_fraction=self._on_fraction,
                        on_download=self._on_download,
                    )
                    self._pipeline = pipeline

                    try:
                        result = pipeline.run()
                        self.call_soon(self.queue_frame.set_job_status, iids[i], "done")
                        results.append((i, result))
                    except PipelineCancelled:
                        self.call_soon(self.queue_frame.set_job_status, iids[i], "cancelled")
                        self.call_soon(self._on_pipeline_cancelled, results, errors, jobs)
                        return
                    except PipelineError as e:
                        self.call_soon(self.queue_frame.set_job_status, iids[i], "error")
                        self._on_progress(f"ERROR: {e}")
                        errors.append((i, str(e)))
                        continue
                    except Exception as e:
                        self.call_soon(self.queue_frame.set_job_status, iids[i], "error")
                        self._on_progress(f"ERROR: {type(e).__name__}: {e}")
                        errors.append((i, f"{type(e).__name__}: {e}"))
                        continue

                self.call_soon(self._on_batch_complete, results, errors, jobs)
            except Exception as e:
                # Anything unexpected must still release the UI, otherwise
                # the Start button stays in Cancel mode until app restart
                self._on_progress(f"FATAL: {type(e).__name__}: {e}")
                errors.append((-1, f"{type(e).__name__}: {e}"))
                try:
                    self.call_soon(self._on_batch_complete, results, errors, jobs)
                except (RuntimeError, tk.TclError):
                    pass

        self._pipeline_thread = threading.Thread(target=_run_batch, daemon=True)
        self._pipeline_thread.start()

    def _cancel_pipeline(self, confirm=False):
        """Request pipeline cancellation.

        Args:
            confirm: Ask before cancelling (used by the global Escape binding
                so a stray keypress can't kill a long run).
        """
        if not self._is_running or self._cancel.is_set():
            return  # Escape can fire while idle
        if confirm and not messagebox.askyesno(
                "Scrivox", "Stop the running transcription?\n\n"
                           "Files already finished are kept.", parent=self):
            return
        self._cancel.set()
        self._cancel_btn.configure(text="Cancelling…")
        self._cancel_btn.state(["disabled"])
        self.progress_frame.set_cancelling()
        if self._pipeline:
            self._pipeline.cancel()

    def _notify_finished(self):
        """Bell + taskbar flash, so a long run finishing in the background
        is noticed."""
        try:
            self.bell()
        except tk.TclError:
            pass
        winnative.flash_taskbar(self)

    def _show_results(self, results, errors, jobs):
        fmt = getattr(self, "_run_fmt", None)
        if len(jobs) <= 1 and results:
            last = results[-1][1]
            self.results_frame.show_result(last.output_text, last.output_path, fmt=fmt)
            for tr in last.translated_outputs:
                if tr.get("output_path"):
                    self._on_progress(
                        f"Translation ({tr['lang_name']}) saved to: {tr['output_path']}")
            return
        if len(jobs) > 1:
            by_index = {i: r for i, r in results}
            err_by_index = dict(errors)
            items = []
            for i, job in enumerate(jobs):
                name = os.path.basename(job.file_path)
                if job.track_label:
                    name += f" ({job.track_label})"
                if i in by_index:
                    r = by_index[i]
                    items.append({"input": name, "output_path": r.output_path,
                                  "text": r.output_text, "error": None})
                elif i in err_by_index:
                    items.append({"input": name, "output_path": None, "text": "",
                                  "error": err_by_index[i]})
            if items:
                self.results_frame.show_batch(items, fmt=fmt)

    def _on_batch_complete(self, results, errors, jobs=None):
        """Called on main thread when all jobs finish."""
        jobs = jobs or []
        self._set_running(False)
        self._pipeline = None
        self._notify_finished()
        explicit = getattr(self, "_run_explicit_name", "")
        self._run_explicit_name = ""
        if explicit and any(os.path.normcase(os.path.abspath(r.output_path or ""))
                            == os.path.normcase(os.path.abspath(explicit))
                            for _, r in results):
            self.output_frame.consume_explicit_name()

        if not results:
            first = errors[0][1] if errors else "No files were transcribed"
            headline, fix = explain_error(first)
            detail = "The log has the technical details."
            if len(errors) > 1:
                detail = f"All {len(errors)} files failed. " + detail
            self.progress_frame.set_error(
                headline, detail=detail,
                fix=(lambda: self._show_fix(fix)) if fix else None)
            self._show_results(results, errors, jobs)
            self._flush_log()
            self._notebook.select(self.log_frame)
            self.log_frame.show_last_error()
            return

        total_elapsed = sum(r.elapsed for _, r in results)
        if errors:
            headline, fix = explain_error(errors[0][1])
            self.progress_frame.complete(
                elapsed=total_elapsed,
                headline=f"{len(results)} of {len(results) + len(errors)} files done",
                detail=f"Failed: {headline} See the Log tab for details.",
                warning=True)
        else:
            n = len(results)
            last = results[-1][1]
            where = os.path.dirname(last.output_path) if last.output_path else ""
            detail = f"Saved in {where}" if where else ""
            if n > 1:
                detail = f"{n} files transcribed. " + detail
            self.progress_frame.complete(elapsed=total_elapsed, detail=detail)
        self._show_results(results, errors, jobs)
        self._notebook.select(self.results_frame)

    def _flush_log(self):
        lf = self.log_frame
        if lf._flush_id is not None:
            lf.after_cancel(lf._flush_id)
            lf._flush_id = None
        lf._flush()

    def _on_pipeline_cancelled(self, results=None, errors=None, jobs=None):
        """Called on main thread when pipeline is cancelled."""
        self._set_running(False)
        if self._download_active:
            # huggingface_hub can't stop mid-file, so say so instead of
            # pretending the bandwidth/disk use has stopped
            self.progress_frame.set_cancelled(
                detail="The speech model keeps downloading in the background, so the "
                       "next start continues where it left off.")
            self._download_active = False
        else:
            self.progress_frame.set_cancelled()
        self._on_progress("Pipeline cancelled by user.")
        self._pipeline = None
        if results:
            self._show_results(results, errors or [], jobs or [])

    def _on_close(self):
        """Handle window close."""
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            if not messagebox.askyesno(
                    "Scrivox", "A transcription is still running.\n\nStop it and exit?",
                    parent=self):
                return
            self._cancel.set()
            if self._pipeline:
                self._pipeline.cancel()
            # Give the worker a few seconds to notice the cancel and clean up
            # (temp WAVs, keyframe dirs, ffmpeg children) before tearing down
            try:
                self.progress_frame.set_cancelling()
                self._cancel_btn.configure(text="Cancelling…")
            except tk.TclError:
                pass
            deadline = time.time() + 3.0
            while self._pipeline_thread.is_alive() and time.time() < deadline:
                try:
                    self.update()
                except tk.TclError:
                    break  # window already destroyed
                time.sleep(0.05)

        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
            self._save_after_id = None
        self._closing = True
        self._save_current_settings()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        # Cancel pending timers on the widget that created them (cancelling
        # through another widget leaves a stale command that Tk 9 rejects
        # when the owner is destroyed)
        for after_id in getattr(self, "_startup_after", ()):
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                pass
        self._startup_after = []
        for owner, attr in ((self, "_drain_id"), (self, "_readiness_after_id"),
                            (self, "_hint_flash_id"), (self, "_save_after_id"),
                            (self._left_canvas, "_scroll_update_id")):
            after_id = getattr(self, attr, None)
            if after_id is not None:
                try:
                    owner.after_cancel(after_id)
                except tk.TclError:
                    pass
                setattr(self, attr, None)
        self.destroy()


def _make_pipeline(config, on_download=None, **kwargs):
    """Create the pipeline, passing the download callback only if supported
    (keeps drop-in pipeline replacements working)."""
    try:
        return TranscriptionPipeline(config, on_download=on_download, **kwargs)
    except TypeError:
        return TranscriptionPipeline(config, **kwargs)


_STEP_NAMES = {
    "Transcribing": "Transcribing speech",
    "Diarizing": "Identifying speakers",
    "Analyzing keyframes": "Describing on-screen content",
    "Generating summary": "Writing the summary",
    "Formatting output": "Saving",
}


def _friendly_step(name):
    return _STEP_NAMES.get(name, name)
