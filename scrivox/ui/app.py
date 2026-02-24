"""ScrivoxApp - main application window assembling all frames with threading."""

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from .. import __version__, __app_name__
from ..config import ConfigManager
from ..core.constants import OUTPUT_FORMATS
from ..core.features import has_diarization, has_advanced_features, get_variant_name
from ..core.pipeline import (
    PipelineConfig, PipelineCancelled, PipelineError, PipelineResult,
    TranscriptionPipeline,
)
from .theme import configure_theme, COLORS, FONTS
from .frames.queue_frame import QueueFrame
from .frames.settings_frame import SettingsFrame
from .frames.output_frame import OutputFrame
from .frames.progress_frame import ProgressFrame
from .frames.log_frame import LogFrame
from .frames.results_frame import ResultsFrame


class ScrivoxApp(tk.Tk):
    """Main Scrivox application window."""

    def __init__(self):
        super().__init__()

        self.config_manager = ConfigManager()
        self._variant = get_variant_name()

        # Title with variant name for Lite/Full
        title = f"{__app_name__} v{__version__}"
        if self._variant != "Regular":
            title += f" ({self._variant})"
        self.title(title)
        self.minsize(900, 600)
        self._set_geometry()

        # Theme
        configure_theme(self)

        # Icon
        self._set_icon()

        # Pipeline state
        self._pipeline = None
        self._pipeline_thread = None
        self._cancel = threading.Event()
        self._original_stdout = sys.stdout

        self._build_ui()
        self._load_saved_settings()
        self._setup_keyboard_shortcuts()
        self._run_preflight_checks()

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
        version_text = f"v{__version__}"
        if self._variant != "Regular":
            version_text += f" ({self._variant})"
        ttk.Label(title_frame, text=version_text,
                  style="Dim.TLabel").pack(side=tk.LEFT, padx=(6, 0), pady=(4, 0))

        # ── Main container ──
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # ── Left panel: Settings (fixed width, scrollable) ──
        left_panel = ttk.Frame(main_frame, width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        left_canvas = tk.Canvas(left_panel, bg=COLORS["bg"], highlightthickness=0,
                                 borderwidth=0)
        left_scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL,
                                        command=left_canvas.yview)
        self._left_inner = ttk.Frame(left_canvas)
        self._left_canvas = left_canvas

        # Debounced scrollregion update
        self._scroll_update_id = None

        def _update_scrollregion(event=None):
            if self._scroll_update_id:
                left_canvas.after_cancel(self._scroll_update_id)
            self._scroll_update_id = left_canvas.after(
                16, lambda: left_canvas.configure(scrollregion=left_canvas.bbox("all")))

        self._left_inner.bind("<Configure>", _update_scrollregion)

        # Sync inner frame width to canvas width
        def _on_canvas_configure(event):
            left_canvas.itemconfigure(self._canvas_window, width=event.width)

        self._canvas_window = left_canvas.create_window(
            (0, 0), window=self._left_inner, anchor=tk.NW)
        left_canvas.bind("<Configure>", _on_canvas_configure)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scoped mousewheel — only scroll left panel when mouse is over it
        self._mousewheel_bound = False

        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            if not self._mousewheel_bound:
                left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
                self._mousewheel_bound = True

        def _unbind_mousewheel(event):
            if self._mousewheel_bound:
                left_canvas.unbind_all("<MouseWheel>")
                self._mousewheel_bound = False

        left_canvas.bind("<Enter>", _bind_mousewheel)
        left_canvas.bind("<Leave>", _unbind_mousewheel)

        # ── Start / Cancel buttons (top of left panel) ──
        btn_frame = ttk.Frame(self._left_inner)
        btn_frame.pack(fill=tk.X, padx=4, pady=(4, 6))

        self._start_btn = ttk.Button(btn_frame, text="Start", style="Accent.TButton",
                                      command=self._start_pipeline)
        self._start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self._cancel_btn = ttk.Button(btn_frame, text="Cancel", style="Cancel.TButton",
                                       command=self._cancel_pipeline, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Queue frame
        self.queue_frame = QueueFrame(
            self._left_inner, config_manager=self.config_manager,
            on_tracks_needed=self._show_track_dialog,
        )
        self.queue_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        # Settings frame
        self.settings_frame = SettingsFrame(self._left_inner)
        self.settings_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        # Output frame
        self.output_frame = OutputFrame(self._left_inner, config_manager=self.config_manager)
        self.output_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        # API Keys frame (conditional on advanced features)
        if has_diarization():
            from .frames.api_frame import ApiFrame
            self.api_frame = ApiFrame(self._left_inner, config_manager=self.config_manager)
            self.api_frame.pack(fill=tk.X, padx=4, pady=(0, 6))
        else:
            self.api_frame = None

        # Advanced settings frame (always shown)
        from .frames.models_frame import ModelsFrame
        self.models_frame = ModelsFrame(self._left_inner,
                                         show_diarization=has_diarization())
        self.models_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        # ── Separator ──
        ttk.Separator(main_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4)

        # ── Right panel: Progress + Log + Results ──
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Progress
        self.progress_frame = ProgressFrame(right_panel)
        self.progress_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        # Log
        self.log_frame = LogFrame(right_panel)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Results
        self.results_frame = ResultsFrame(right_panel)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # ── Status bar ──
        self._status_bar = ttk.Label(self, text="", style="Dim.TLabel", anchor=tk.W)
        self._status_bar.pack(fill=tk.X, padx=12, pady=(0, 4))

    def _setup_keyboard_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.bind_all("<Control-o>", lambda e: self.queue_frame.browse_files())
        self.bind_all("<Control-Return>", lambda e: self._start_pipeline())
        self.bind_all("<Escape>", lambda e: self._cancel_pipeline())
        self.bind_all("<Control-l>", lambda e: self.log_frame.clear())

    def _run_preflight_checks(self):
        """Check ffmpeg and GPU availability on startup, update status bar."""
        parts = []

        # Determine CUDA source label
        if getattr(sys, "frozen", False):
            _flag = os.path.join(sys._MEIPASS, '..', 'use_system_cuda')
        else:
            _flag = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                'use_system_cuda')
        cuda_source = "system" if os.path.isfile(_flag) else "bundled"

        # GPU check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cuda_ver = torch.version.cuda or "unknown"
                parts.append(f"GPU: {gpu_name} | CUDA {cuda_ver} ({cuda_source})")
            else:
                parts.append("GPU: No CUDA device found")
        except Exception:
            parts.append("GPU: torch not available")

        # ffmpeg check
        try:
            subprocess.run(["ffmpeg", "-version"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            parts.append("ffmpeg: NOT FOUND")

        parts.append(f"{__app_name__} v{__version__} ({self._variant})")
        self._status_bar.configure(text=" | ".join(parts))

    def _show_track_dialog(self, filepath, tracks):
        """Show track selection dialog. Returns list of selected track indices."""
        from .dialogs.track_dialog import TrackDialog
        filename = os.path.basename(filepath)
        dialog = TrackDialog(self, filename, tracks)
        self.wait_window(dialog)
        return dialog.result

    def _load_saved_settings(self):
        """Load last-used settings from config."""
        settings = self.config_manager.get_last_settings()
        self.settings_frame.load_settings(settings)
        if self.models_frame:
            self.models_frame.load_settings(settings)
        fmt = settings.get("output_format", "txt")
        if fmt in OUTPUT_FORMATS:
            self.output_frame.format_var.set(fmt)

    def _save_current_settings(self):
        """Save current settings to config."""
        settings = self.settings_frame.get_settings_dict()
        if self.models_frame:
            settings.update(self.models_frame.get_settings_dict())
        settings["output_format"] = self.output_frame.format_var.get()
        self.config_manager.save_last_settings(**settings)
        if self.api_frame:
            self.api_frame.save_to_config()
        self.config_manager.set("ui", "geometry", self.geometry())
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
                with open(flag_path, 'w') as f:
                    pass  # empty flag file
            else:
                if os.path.isfile(flag_path):
                    os.remove(flag_path)
        except OSError:
            pass

    def _validate(self):
        """Validate inputs before starting pipeline."""
        if not self.queue_frame.has_jobs:
            messagebox.showerror("Error", "Please add at least one file to the queue.")
            return False

        # Validate all job files exist
        for job in self.queue_frame.get_jobs():
            if not os.path.isfile(job.file_path):
                messagebox.showerror("Error", f"File not found: {job.file_path}")
                return False

        settings = self.settings_frame
        if settings.diarize_var.get():
            if not self.api_frame:
                messagebox.showerror("Error",
                                     "Diarization is not available in the Lite build.")
                return False
            if not self.api_frame.get_hf_token() and not self.api_frame._has_bundled:
                messagebox.showerror("Error",
                                     "Diarization requires a HuggingFace token.\n"
                                     "Enter it in the API Keys section.")
                return False

        if settings.vision_var.get() or settings.summarize_var.get():
            if not self.api_frame:
                messagebox.showerror("Error",
                                     "Vision/Summary features are not available in the Lite build.")
                return False
            # Ollama (local) doesn't require an API key
            is_local = "localhost" in (self.api_frame.get_api_base() or "")
            if not self.api_frame.get_openrouter_key() and not is_local:
                messagebox.showerror("Error",
                                     "Vision/Summary requires an LLM API key.\n"
                                     "Enter it in the API Keys section.")
                return False

        return True

    def _build_config(self, job=None):
        """Build PipelineConfig from current GUI state and optional job."""
        s = self.settings_frame

        # Determine file path and audio track
        file_path = job.file_path if job else self.queue_frame.file_path
        audio_track = job.audio_track if job else 0
        language = job.language_override if (job and job.language_override) else (s.language_var.get() or None)

        # Get advanced settings from models frame
        adv = self.models_frame.get_settings_dict() if self.models_frame else {}

        return PipelineConfig(
            input_path=file_path,
            model=s.model_var.get(),
            language=language,
            diarize=s.diarize_var.get(),
            vision=s.vision_var.get(),
            summarize=s.summarize_var.get(),
            diarization_model=(self.models_frame.get_diarization_model()
                               if self.models_frame else "pyannote/speaker-diarization-3.1"),
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
        self._cancel.clear()

        # Prepare UI
        self._set_running(True)
        self.log_frame.clear()
        self.results_frame.clear()
        self.progress_frame.reset()
        self.progress_frame.start()

        jobs = self.queue_frame.get_jobs()

        def _run_batch():
            results = []
            total_jobs = len(jobs)

            for i, job in enumerate(jobs):
                if self._cancel.is_set():
                    break

                # Update file-level progress
                filename = os.path.basename(job.file_path)
                track_info = f" ({job.track_label})" if job.track_label else ""
                self.after(0, self.progress_frame.update_file,
                           i + 1, total_jobs, f"{filename}{track_info}")
                self.after(0, self.queue_frame.set_job_status, i, "running")

                # Add job separator in log
                if i > 0:
                    self._on_progress("\n" + "=" * 60)
                    self._on_progress(f"  JOB {i + 1}/{total_jobs}")
                    self._on_progress("=" * 60)

                config = self._build_config(job)

                # For batch with no explicit output path, auto-generate per-job
                if total_jobs > 1 and not self.output_frame.output_path_var.get():
                    base = os.path.splitext(job.file_path)[0]
                    fmt = self.output_frame.format_var.get()
                    ext_map = {"txt": ".txt", "md": ".md", "srt": ".srt",
                               "vtt": ".vtt", "json": ".json", "tsv": ".tsv"}
                    suffix = f"_track{job.audio_track}" if job.audio_track > 0 else ""
                    config.output_path = f"{base}{suffix}{ext_map.get(fmt, '.txt')}"

                pipeline = TranscriptionPipeline(
                    config,
                    on_progress=self._on_progress,
                    on_step=self._on_step,
                )
                self._pipeline = pipeline

                try:
                    result = pipeline.run()
                    self.after(0, self.queue_frame.set_job_status, i, "done")
                    results.append(result)
                except PipelineCancelled:
                    self.after(0, self.queue_frame.set_job_status, i, "error")
                    self.after(0, self._on_pipeline_cancelled)
                    return
                except PipelineError as e:
                    self.after(0, self.queue_frame.set_job_status, i, "error")
                    self._on_progress(f"ERROR: {e}")
                    # Continue to next job on error
                    continue
                except Exception as e:
                    self.after(0, self.queue_frame.set_job_status, i, "error")
                    self._on_progress(f"ERROR: {type(e).__name__}: {e}")
                    continue

            self.after(0, self._on_batch_complete, results)

        self._pipeline_thread = threading.Thread(target=_run_batch, daemon=True)
        self._pipeline_thread.start()

    def _cancel_pipeline(self):
        """Request pipeline cancellation."""
        self._cancel.set()
        if self._pipeline:
            self._pipeline.cancel()
            self._cancel_btn.configure(state=tk.DISABLED)

    def _on_batch_complete(self, results):
        """Called on main thread when all jobs finish."""
        self._set_running(False)
        self._pipeline = None

        if not results:
            self.progress_frame.set_error("No jobs completed successfully")
            return

        # Show last result (or summary for multi-job)
        last = results[-1]
        total_elapsed = sum(r.elapsed for r in results)
        self.progress_frame.complete(elapsed=total_elapsed)

        if len(results) == 1:
            self.results_frame.show_result(last.output_text, last.output_path)
        else:
            summary_lines = [f"Batch complete: {len(results)} job(s) finished\n"]
            for r in results:
                path = r.output_path or "(console)"
                summary_lines.append(f"  - {r.metadata.get('input_file', '?')} -> {path}")
            summary_lines.append(f"\nTotal time: {total_elapsed:.1f}s")
            summary_text = "\n".join(summary_lines)
            self.results_frame.show_result(summary_text, last.output_path)

    def _on_pipeline_cancelled(self):
        """Called on main thread when pipeline is cancelled."""
        self._set_running(False)
        self.progress_frame.set_cancelled()
        self._on_progress("Pipeline cancelled by user.")
        self._pipeline = None

    def _on_close(self):
        """Handle window close."""
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            if not messagebox.askyesno("Confirm", "A transcription is in progress. Cancel and exit?"):
                return
            self._cancel.set()
            if self._pipeline:
                self._pipeline.cancel()

        self._save_current_settings()
        self.destroy()
