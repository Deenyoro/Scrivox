"""ScrivoxApp - main application window assembling all frames with threading."""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from .. import __version__, __app_name__
from ..config import ConfigManager
from ..core.constants import OUTPUT_FORMATS
from ..core.pipeline import (
    PipelineConfig, PipelineCancelled, PipelineError, PipelineResult,
    TranscriptionPipeline,
)
from .theme import configure_theme, COLORS, FONTS
from .log_redirect import LogRedirect
from .frames.input_frame import InputFrame
from .frames.settings_frame import SettingsFrame
from .frames.api_frame import ApiFrame
from .frames.output_frame import OutputFrame
from .frames.progress_frame import ProgressFrame
from .frames.log_frame import LogFrame
from .frames.results_frame import ResultsFrame


class ScrivoxApp(tk.Tk):
    """Main Scrivox application window."""

    def __init__(self):
        super().__init__()

        self.config_manager = ConfigManager()

        self.title(f"{__app_name__} v{__version__}")
        self.minsize(900, 600)
        self._set_geometry()

        # Theme
        configure_theme(self)

        # Icon
        self._set_icon()

        # Pipeline state
        self._pipeline = None
        self._pipeline_thread = None
        self._original_stdout = sys.stdout

        self._build_ui()
        self._load_saved_settings()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _set_icon(self):
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "..", "assets", "scrivox.ico")
        if os.path.isfile(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception:
                pass

    def _set_geometry(self):
        saved = self.config_manager.get("ui", "geometry", "")
        if saved:
            try:
                self.geometry(saved)
                return
            except Exception:
                pass
        # Default: centered 1100x750
        self.geometry("1100x750")
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 1100) // 2
        y = (self.winfo_screenheight() - 750) // 2
        self.geometry(f"1100x750+{x}+{y}")

    def _build_ui(self):
        # ── Title bar ──
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, padx=12, pady=(10, 4))
        ttk.Label(title_frame, text=f"{__app_name__}", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_frame, text=f"v{__version__}",
                  style="Dim.TLabel").pack(side=tk.LEFT, padx=(6, 0), pady=(4, 0))

        # ── Main paned window ──
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # ── Left panel: Settings ──
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)

        # Scrollable left panel
        left_canvas = tk.Canvas(left_panel, bg=COLORS["bg"], highlightthickness=0,
                                 borderwidth=0)
        left_scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL,
                                        command=left_canvas.yview)
        self._left_inner = ttk.Frame(left_canvas)

        self._left_inner.bind("<Configure>",
                               lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=self._left_inner, anchor=tk.NW)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mousewheel to left panel
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        # Input frame
        self.input_frame = InputFrame(self._left_inner, config_manager=self.config_manager)
        self.input_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        # Settings frame
        self.settings_frame = SettingsFrame(self._left_inner)
        self.settings_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        # Output frame
        self.output_frame = OutputFrame(self._left_inner, config_manager=self.config_manager)
        self.output_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        # API Keys frame
        self.api_frame = ApiFrame(self._left_inner, config_manager=self.config_manager)
        self.api_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        # ── Start / Cancel buttons ──
        btn_frame = ttk.Frame(self._left_inner)
        btn_frame.pack(fill=tk.X, padx=4, pady=(4, 8))

        self._start_btn = ttk.Button(btn_frame, text="START", style="Accent.TButton",
                                      command=self._start_pipeline)
        self._start_btn.pack(fill=tk.X, pady=(0, 4))

        self._cancel_btn = ttk.Button(btn_frame, text="Cancel", style="Cancel.TButton",
                                       command=self._cancel_pipeline, state=tk.DISABLED)
        self._cancel_btn.pack(fill=tk.X)

        # ── Right panel: Progress + Log + Results ──
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)

        # Progress
        self.progress_frame = ProgressFrame(right_panel)
        self.progress_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        # Log
        self.log_frame = LogFrame(right_panel)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Results
        self.results_frame = ResultsFrame(right_panel)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Set sash position
        self.update_idletasks()
        saved_sash = self.config_manager.get("ui", "sash_position")
        if saved_sash:
            try:
                paned.sashpos(0, int(saved_sash))
            except Exception:
                pass

        self._paned = paned

    def _load_saved_settings(self):
        """Load last-used settings from config."""
        settings = self.config_manager.get_last_settings()
        self.settings_frame.load_settings(settings)
        fmt = settings.get("output_format", "txt")
        if fmt in OUTPUT_FORMATS:
            self.output_frame.format_var.set(fmt)

    def _save_current_settings(self):
        """Save current settings to config."""
        settings = self.settings_frame.get_settings_dict()
        settings["output_format"] = self.output_frame.format_var.get()
        self.config_manager.save_last_settings(**settings)
        self.api_frame.save_to_config()
        self.config_manager.set("ui", "geometry", self.geometry())
        try:
            self.config_manager.set("ui", "sash_position", self._paned.sashpos(0))
        except Exception:
            pass
        self.config_manager.save()

    def _validate(self):
        """Validate inputs before starting pipeline."""
        if not self.input_frame.file_path:
            messagebox.showerror("Error", "Please select an input file.")
            return False
        if not os.path.isfile(self.input_frame.file_path):
            messagebox.showerror("Error", f"File not found: {self.input_frame.file_path}")
            return False

        settings = self.settings_frame
        if settings.diarize_var.get() and not self.api_frame.get_hf_token():
            messagebox.showerror("Error",
                                 "Diarization requires a HuggingFace token.\n"
                                 "Enter it in the API Keys section.")
            return False
        if (settings.vision_var.get() or settings.summarize_var.get()) and not self.api_frame.get_openrouter_key():
            messagebox.showerror("Error",
                                 "Vision/Summary requires an OpenRouter API key.\n"
                                 "Enter it in the API Keys section.")
            return False

        return True

    def _build_config(self):
        """Build PipelineConfig from current GUI state."""
        s = self.settings_frame
        return PipelineConfig(
            input_path=self.input_frame.file_path,
            model=s.model_var.get(),
            language=s.language_var.get() or None,
            diarize=s.diarize_var.get(),
            vision=s.vision_var.get(),
            summarize=s.summarize_var.get(),
            num_speakers=s.get_int_or_none(s.num_speakers_var),
            min_speakers=s.get_int_or_none(s.min_speakers_var),
            max_speakers=s.get_int_or_none(s.max_speakers_var),
            speaker_names=s.get_speaker_names(),
            vision_interval=int(s.vision_interval_var.get() or 60),
            vision_model=s.vision_model_var.get(),
            vision_workers=int(s.vision_workers_var.get() or 4),
            summary_model=s.summary_model_var.get(),
            output_format=self.output_frame.format_var.get(),
            output_path=self.output_frame.output_path_var.get() or None,
            hf_token=self.api_frame.get_hf_token(),
            openrouter_key=self.api_frame.get_openrouter_key(),
        )

    def _set_running(self, running):
        """Enable/disable controls during pipeline execution."""
        state = tk.DISABLED if running else tk.NORMAL
        self._start_btn.configure(state=state)
        cancel_state = tk.NORMAL if running else tk.DISABLED
        self._cancel_btn.configure(state=cancel_state)

    def _on_progress(self, msg):
        """Thread-safe progress callback: schedules log append on main thread."""
        try:
            self.after(0, self.log_frame.append, msg + "\n")
        except Exception:
            pass

    def _on_step(self, step_num, total_steps, step_name):
        """Thread-safe step update callback."""
        try:
            self.after(0, self.progress_frame.update_step, step_num, total_steps, step_name)
        except Exception:
            pass

    def _start_pipeline(self):
        """Validate inputs and start the pipeline in a background thread."""
        if not self._validate():
            return

        self._save_current_settings()

        # Prepare UI
        self._set_running(True)
        self.log_frame.clear()
        self.results_frame.clear()
        self.progress_frame.reset()
        self.progress_frame.start()

        config = self._build_config()
        self._pipeline = TranscriptionPipeline(
            config,
            on_progress=self._on_progress,
            on_step=self._on_step,
        )

        def _run():
            try:
                result = self._pipeline.run()
                self.after(0, self._on_pipeline_complete, result)
            except PipelineCancelled:
                self.after(0, self._on_pipeline_cancelled)
            except PipelineError as e:
                self.after(0, self._on_pipeline_error, str(e))
            except Exception as e:
                self.after(0, self._on_pipeline_error, f"{type(e).__name__}: {e}")

        self._pipeline_thread = threading.Thread(target=_run, daemon=True)
        self._pipeline_thread.start()

    def _cancel_pipeline(self):
        """Request pipeline cancellation."""
        if self._pipeline:
            self._pipeline.cancel()
            self._cancel_btn.configure(state=tk.DISABLED)

    def _on_pipeline_complete(self, result: PipelineResult):
        """Called on main thread when pipeline finishes successfully."""
        self._set_running(False)
        self.progress_frame.complete(elapsed=result.elapsed)
        self.results_frame.show_result(result.output_text, result.output_path)
        self._pipeline = None

    def _on_pipeline_cancelled(self):
        """Called on main thread when pipeline is cancelled."""
        self._set_running(False)
        self.progress_frame.set_cancelled()
        self._on_progress("Pipeline cancelled by user.")
        self._pipeline = None

    def _on_pipeline_error(self, error_msg):
        """Called on main thread when pipeline encounters an error."""
        self._set_running(False)
        self.progress_frame.set_error(error_msg)
        self._on_progress(f"ERROR: {error_msg}")
        messagebox.showerror("Pipeline Error", error_msg)
        self._pipeline = None

    def _on_close(self):
        """Handle window close."""
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            if not messagebox.askyesno("Confirm", "A transcription is in progress. Cancel and exit?"):
                return
            if self._pipeline:
                self._pipeline.cancel()

        self._save_current_settings()
        self.destroy()
