"""Behaviour tests for the Tk GUI (need a display: run under xvfb-run on Linux).

Skipped automatically when there is no display or the heavy runtime
dependencies (torch, Pillow) aren't installed. Nothing here touches the
network, ffmpeg or a GPU: the pipeline and the ffprobe track probe are
replaced with fakes.

Run with:  xvfb-run -a python -m unittest discover -s tests

Run them under Tk 8.6 as well (python.org Python 3.11 on Windows, which the
.exe ships with): widget metrics differ from Tk 9, so layout checks here
avoid hard-coded pixel budgets.
"""

import os
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest import mock

try:
    _root = tk.Tk()
    _root.destroy()
    HAVE_DISPLAY = True
except tk.TclError:
    HAVE_DISPLAY = False

try:
    import PIL  # noqa: F401
    import torch  # noqa: F401  (scrivox.core.pipeline needs it)
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


class _FakeResult:
    def __init__(self, text, path):
        self.output_text = text
        self.output_path = path
        self.translated_outputs = []
        self.elapsed = 0.1
        self.metadata = {}


@unittest.skipUnless(HAVE_DISPLAY and HAVE_DEPS, "needs a display, torch and Pillow")
class GuiTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.cfg_dir = os.path.join(self.dir, "cfg")
        os.mkdir(self.cfg_dir)
        self._patches = [mock.patch("scrivox.config._get_config_dir", lambda: self.cfg_dir)]
        # The GUI's startup probe imports torch and looks for ffmpeg; keep it
        # quiet and deterministic
        self._patches.append(mock.patch("scrivox.ui.app.ScrivoxApp._run_preflight_checks",
                                        lambda self: None))
        for p in self._patches:
            p.start()
        self._hooks = (sys.excepthook, threading.excepthook, sys.stdout, sys.stderr)
        self.app = None

    def make_app(self):
        from scrivox.ui.app import ScrivoxApp
        self.app = ScrivoxApp()
        self.pump(0.2)
        return self.app

    def tearDown(self):
        if self.app is not None:
            try:
                self.app._on_close()
            except tk.TclError:
                self.app = None  # already destroyed by the test
        sys.excepthook, threading.excepthook, sys.stdout, sys.stderr = self._hooks
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def pump(self, seconds=0.1, until=None):
        end = time.time() + seconds
        while time.time() < end:
            self.app.update()
            if until is not None and until():
                return True
            time.sleep(0.01)
        return until() if until is not None else True

    def media(self, name="interview.wav"):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(b"RIFF")
        return path

    def add_ready_job(self, path):
        import scrivox.ui.frames.queue_frame as qf
        with mock.patch.object(qf, "list_audio_tracks", lambda p: []):
            self.app.queue_frame._add_file(path)
            self.assertTrue(self.pump(3, until=lambda: self.app.queue_frame.has_jobs
                                      and not self.app.queue_frame.is_checking))


class DefaultRunSavesTests(GuiTestCase):
    def _run_with_fake_pipeline(self):
        import scrivox.ui.app as appmod
        seen = {}

        class FakePipeline:
            def __init__(self, config, **kwargs):
                self.config = config
                seen["config"] = config
                seen["kwargs"] = kwargs

            def cancel(self):
                pass

            def run(self):
                # Same rule as the real pipeline: write only when a path is set
                if self.config.output_path:
                    with open(self.config.output_path, "w", encoding="utf-8") as f:
                        f.write("hello")
                return _FakeResult("hello", self.config.output_path)

        with mock.patch.object(appmod, "TranscriptionPipeline", FakePipeline):
            self.app._start_pipeline()
            self.assertTrue(self.pump(5, until=lambda: not self.app._is_running))
            self.pump(0.2)
        return seen

    def test_default_txt_run_writes_file_and_enables_open(self):
        # Regression for the audit's #1: txt + no extras + blank path saved
        # nothing, while the UI claimed output is saved next to the input
        self.make_app()
        media = self.media()
        self.add_ready_job(media)
        seen = self._run_with_fake_pipeline()
        expected = os.path.join(self.dir, "interview_transcript.txt")
        self.assertEqual(seen["config"].output_path, expected)
        self.assertTrue(os.path.isfile(expected))
        self.assertEqual(str(self.app.results_frame._open_btn.cget("state")), "normal")
        self.assertIsNotNone(seen["kwargs"].get("on_download"))

    def test_rerun_does_not_overwrite_previous_result(self):
        self.make_app()
        media = self.media()
        earlier = os.path.join(self.dir, "interview_transcript.txt")
        with open(earlier, "w") as f:
            f.write("keep me")
        self.add_ready_job(media)
        seen = self._run_with_fake_pipeline()
        self.assertEqual(seen["config"].output_path,
                         os.path.join(self.dir, "interview_transcript (2).txt"))
        with open(earlier) as f:
            self.assertEqual(f.read(), "keep me")


class ValidationTests(GuiTestCase):
    def test_empty_queue_shows_inline_hint_not_modal(self):
        self.make_app()
        with mock.patch("tkinter.messagebox.showerror") as showerror:
            self.app._start_pipeline()
        showerror.assert_not_called()
        self.assertFalse(self.app._is_running)
        self.assertIn("Add at least one", self.app._action_hint.cget("text"))

    def test_missing_token_reported_inline(self):
        self.make_app()
        if self.app.api_frame is None:
            self.skipTest("Lite build: no diarization")
        self.add_ready_job(self.media())
        self.app.api_frame.hf_token_var.set("")
        self.app.api_frame._has_bundled = False
        sf = self.app.settings_frame
        sf.diarize_var.set(True)
        sf._toggle_diarize()
        with mock.patch("tkinter.messagebox.showerror") as showerror:
            self.app._start_pipeline()
        showerror.assert_not_called()
        self.assertFalse(self.app._is_running)
        self.assertIn("Hugging Face token", self.app._action_hint.cget("text"))


class QueueTests(GuiTestCase):
    def test_track_probe_runs_off_the_tk_thread(self):
        # Regression for the audit's #5: ffprobe ran on the Tk thread and
        # froze the window for up to 15 s per file
        self.make_app()
        import scrivox.ui.frames.queue_frame as qf
        probe_threads = []

        def slow_probe(path):
            probe_threads.append(threading.current_thread())
            time.sleep(0.4)
            return []

        with mock.patch.object(qf, "list_audio_tracks", slow_probe):
            t0 = time.monotonic()
            self.app.queue_frame._add_file(self.media())
            self.assertLess(time.monotonic() - t0, 0.2)
            self.assertTrue(self.app.queue_frame.is_checking)
            self.assertTrue(self.pump(3, until=lambda: self.app.queue_frame.has_jobs))
        self.assertIsNot(probe_threads[0], threading.main_thread())

    def test_non_media_file_rejected_with_notice(self):
        self.make_app()
        notes = os.path.join(self.dir, "notes.txt")
        with open(notes, "w") as f:
            f.write("x")
        self.app.queue_frame.add_files([notes])
        self.pump(0.1)
        self.assertFalse(self.app.queue_frame.has_jobs)
        self.assertIn("notes.txt", self.app.queue_frame._notice_label.cget("text"))


class WheelTests(GuiTestCase):
    def test_wheel_over_combobox_scrolls_panel_not_value(self):
        # Regression for the audit's #6
        self.make_app()
        self.app.geometry("900x520")
        self.pump(0.3)
        combo = self.app.settings_frame._model_combo
        before = self.app.settings_frame.model_var.get()
        combo.event_generate("<MouseWheel>", delta=-120, x=5, y=5)
        self.pump(0.1)
        self.assertEqual(self.app.settings_frame.model_var.get(), before)


class RobustnessTests(GuiTestCase):
    def test_app_starts_when_tkdnd_cannot_load(self):
        # Regression for the audit's #7: TkinterDnD.Tk() raised RuntimeError
        # and the app died before showing a window
        import scrivox.ui.app as appmod

        def broken(_root):
            raise RuntimeError("Unable to load tkdnd library.")

        with mock.patch.object(appmod, "_dnd_require", broken):
            app = self.make_app()
        self.assertFalse(app.dnd_available)
        self.assertFalse(app.queue_frame.dnd_available)

    def test_callback_exception_is_logged(self):
        app = self.make_app()
        with mock.patch("scrivox.ui.dialogs.help_dialog.ErrorReportDialog") as dlg:
            try:
                raise ValueError("boom")
            except ValueError:
                app.report_callback_exception(*sys.exc_info())
        dlg.assert_called_once()
        with open(os.path.join(self.cfg_dir, "scrivox_error.log"), encoding="utf-8") as f:
            self.assertIn("ValueError: boom", f.read())


class SettingsCompatTests(GuiTestCase):
    def test_old_settings_file_loads_and_language_round_trips(self):
        import json
        old = {"last_settings": {"model": "medium", "language": "en", "output_format": "srt"},
               "ui": {"geometry": "1000x700+10+10"}}
        with open(os.path.join(self.cfg_dir, "scrivox_config.json"), "w") as f:
            json.dump(old, f)
        app = self.make_app()
        sf = app.settings_frame
        self.assertEqual(sf.model_var.get(), "medium")
        self.assertEqual(sf.language_var.get(), "English (en)")
        self.assertEqual(sf.get_language_code(), "en")
        self.assertEqual(app.output_frame.format_var.get(), "srt")
        sf.language_var.set("Auto-detect")
        self.assertIsNone(sf.get_language_code())
        self.assertEqual(sf.get_settings_dict()["language"], "")


class StartButtonTests(GuiTestCase):
    def test_start_looks_unavailable_until_a_file_is_ready(self):
        self.make_app()
        self.pump(0.3)
        self.assertEqual(self.app._start_btn.cget("style"), "AccentBlocked.TButton")
        self.assertNotIn("disabled", self.app._start_btn.state())  # still focusable
        self.add_ready_job(self.media())
        self.pump(0.3)
        self.assertEqual(self.app._start_btn.cget("style"), "Accent.TButton")

    def test_refused_start_visibly_reacts(self):
        # Regression: a refused click only rang the bell; the window looked
        # exactly the same before and after
        self.make_app()
        self.pump(0.3)
        with mock.patch.object(self.app.queue_frame._drop_zone, "pulse") as pulse:
            self.app._start_btn.invoke()
        pulse.assert_called_once()
        self.assertEqual(self.app._action_hint.cget("style"), "SmallError.TLabel")
        self.app._end_hint_flash()
        self.assertEqual(self.app._action_hint.cget("style"), "Dim.TLabel")


class ExtrasTests(GuiTestCase):
    def test_extras_start_folded_and_open_when_an_extra_is_switched_on(self):
        self.make_app()
        if self.app.api_frame is None:
            self.skipTest("Lite build: no extras")
        sf = self.app.settings_frame
        self.assertFalse(sf.extras_open)
        self.assertFalse(sf._extras.winfo_ismapped())
        sf.diarize_var.set(True)
        sf._toggle_diarize()
        self.pump(0.1)
        self.assertTrue(sf.extras_open)

    def test_fixing_a_problem_inside_folded_extras_opens_them(self):
        self.make_app()
        if self.app.api_frame is None:
            self.skipTest("Lite build: no extras")
        sf = self.app.settings_frame
        sf.set_extras_open(False)
        sf.reveal(sf._min_entry)
        self.assertTrue(sf.extras_open)

    def test_extras_summary_ignores_clicks_while_running(self):
        self.make_app()
        sf = self.app.settings_frame
        sf.set_extras_open(False)
        self.app._set_running(True)
        try:
            sf._on_summary_click()
            self.assertFalse(sf.extras_open)
        finally:
            self.app._set_running(False)
        sf._on_summary_click()
        self.assertTrue(sf.extras_open)

    def test_extras_state_is_remembered(self):
        import json
        self.make_app()
        self.app.settings_frame.set_extras_open(True)
        self.app._on_close()
        self.app = None
        with open(os.path.join(self.cfg_dir, "scrivox_config.json")) as f:
            self.assertTrue(json.load(f)["ui"]["extras_open"])
        self.make_app()
        self.assertTrue(self.app.settings_frame.extras_open)


class DropdownLabelTests(GuiTestCase):
    def test_format_list_describes_each_format_but_stores_the_id(self):
        self.make_app()
        of = self.app.output_frame
        values = of._fmt_combo.cget("values")
        self.assertTrue(any("subtitles" in v for v in values))
        srt = next(v for v in values if v.startswith("srt"))
        of._fmt_combo.set(srt)
        of._fmt_combo.event_generate("<<ComboboxSelected>>")
        self.assertEqual(of.format_var.get(), "srt")
        self.assertEqual(self.app._save_current_settings() or
                         self.app.config_manager.get_last_settings()["output_format"], "srt")
        of.format_var.set("vtt")
        self.assertTrue(of._fmt_combo.get().startswith("vtt"))

    def test_model_list_shows_size_but_field_keeps_model_name(self):
        self.make_app()
        sf = self.app.settings_frame
        combo = sf._model_combo
        label = next(v for v in combo.cget("values") if v.startswith("medium"))
        self.assertIn("GB", label)
        combo.set(label)
        combo.event_generate("<<ComboboxSelected>>")
        self.assertEqual(sf.model_var.get(), "medium")
        # Opening the list marks the current model, not the first row
        values = list(combo.cget("values"))
        self.assertEqual(combo.current_row(), values.index(label))
        self.assertNotEqual(values.index(label), 0)
        self.app.tk.call("ttk::combobox::Post", combo)
        try:
            self.pump(0.2)
            lb = combo._popdown_listbox()
            self.assertEqual([int(i) for i in self.app.tk.splitlist(
                self.app.tk.call(lb, "curselection"))], [values.index(label)])
        finally:
            self.app.tk.call("ttk::combobox::Unpost", combo)


class QueueDisplayTests(GuiTestCase):
    def test_long_names_keep_their_extension_and_status_fits(self):
        import tkinter.font as tkfont
        self.make_app()
        self.app.geometry("900x600")
        path = self.media("Team meeting 2024-03-14 (final) with the whole department.wav")
        self.add_ready_job(path)
        self.pump(0.3)
        qf = self.app.queue_frame
        iid = qf.get_job_ids()[0]
        shown = qf._tree.set(iid, "file")
        self.assertTrue(shown.endswith(".wav"), shown)
        self.assertIn("\u2026", shown)
        font = tkfont.nametofont("TkDefaultFont")
        self.assertGreaterEqual(int(qf._tree.column("status", "width")),
                                font.measure("Cancelled"))

    def test_queue_count_sits_in_the_step_header(self):
        # The count used to squeeze in beside Add/Remove/Clear, where at
        # 125-150% scaling it never fit; the "Files" header always has room.
        # No pixel assumptions, so this holds on Tk 8.6 and Tk 9 alike.
        self.make_app()
        self.app.geometry("1366x700")
        card = self.app._cards[0]
        qf = self.app.queue_frame
        self.assertFalse(card.note_label.winfo_manager())  # empty queue: no note
        for name in ("a.wav", "b.wav", "c.wav"):
            self.add_ready_job(self.media(name))
        qf.set_job_status(0, "done")
        qf.set_job_status(1, "error")
        self.pump(0.3)
        self.assertEqual(card.note_label.cget("text"), "3 files")
        self.assertTrue(card.note_label.winfo_ismapped())
        self.assertGreaterEqual(card.note_label.winfo_width(),
                                card.note_label.winfo_reqwidth())
        qf._clear_all()
        self.pump(0.1)
        self.assertFalse(card.note_label.winfo_manager())

    def test_ellipsize_keeps_extension(self):
        import tkinter.font as tkfont

        from scrivox.ui.widgets import ellipsize
        self.make_app()
        font = tkfont.nametofont("TkDefaultFont")
        text = "Team meeting 2024-03-14 (final).wav"
        out = ellipsize(text, font, font.measure(text) // 2)
        self.assertTrue(out.endswith(".wav"))
        self.assertLessEqual(font.measure(out), font.measure(text) // 2)
        self.assertEqual(ellipsize("a.wav", font, 1000), "a.wav")


class OutputNameTests(GuiTestCase):
    def test_single_file_shows_planned_name_and_rename_is_one_shot(self):
        self.make_app()
        path = self.media()
        self.add_ready_job(path)
        self.pump(0.3)
        of = self.app.output_frame
        self.assertIn("interview_transcript.txt", of._save_hint.cget("text"))
        chosen = os.path.join(self.dir, "Board minutes.txt")
        with mock.patch("tkinter.filedialog.asksaveasfilename", return_value=chosen):
            of._rename()
        self.assertEqual(of.output_path_var.get(), chosen)
        self.assertIn("Board minutes.txt", of._save_hint.cget("text"))
        # The next run uses it; afterwards the default name is back so a
        # second run can't overwrite it
        self.app.output_frame.consume_explicit_name()
        self.assertEqual(of.output_path_var.get(), "")

    def test_chosen_name_never_overwrites_when_format_or_folder_changes(self):
        self.make_app()
        self.add_ready_job(self.media())
        self.pump(0.3)
        of = self.app.output_frame
        chosen = os.path.join(self.dir, "Board minutes.txt")
        with open(chosen, "w") as f:
            f.write("confirmed in Save As")
        taken = os.path.join(self.dir, "Board minutes.srt")
        with open(taken, "w") as f:
            f.write("keep me")
        with mock.patch("tkinter.filedialog.asksaveasfilename", return_value=chosen):
            of._rename()
        of.format_var.set("srt")
        self.assertEqual(of.output_path_var.get(),
                         os.path.join(self.dir, "Board minutes (2).srt"))
        # Back to the format the user confirmed in Save As: their exact choice
        of.format_var.set("txt")
        self.assertEqual(of.output_path_var.get(), chosen)
        other = os.path.join(self.dir, "other")
        os.mkdir(other)
        with open(os.path.join(other, "Board minutes.txt"), "w") as f:
            f.write("keep me too")
        with mock.patch("tkinter.filedialog.askdirectory", return_value=other):
            of._browse_output()
        self.assertEqual(of.output_path_var.get(),
                         os.path.join(other, "Board minutes (2).txt"))

    def test_chosen_name_survives_a_failed_run_and_is_used_once(self):
        import scrivox.ui.app as appmod
        from scrivox.core.pipeline import PipelineError
        self.make_app()
        self.add_ready_job(self.media())
        self.pump(0.3)
        of = self.app.output_frame
        chosen = os.path.join(self.dir, "Board minutes.txt")
        with mock.patch("tkinter.filedialog.asksaveasfilename", return_value=chosen):
            of._rename()
        outcome = {"fail": True}

        class FakePipeline:
            def __init__(self, config, **kwargs):
                self.config = config

            def cancel(self):
                pass

            def run(self):
                if outcome["fail"]:
                    raise PipelineError("ffmpeg failed")
                with open(self.config.output_path, "w") as f:
                    f.write("hello")
                return _FakeResult("hello", self.config.output_path)

        with mock.patch.object(appmod, "TranscriptionPipeline", FakePipeline):
            self.app._start_pipeline()
            self.assertTrue(self.pump(5, until=lambda: not self.app._is_running))
            self.pump(0.2)
            self.assertEqual(of.output_path_var.get(), chosen)
            outcome["fail"] = False
            self.app._start_pipeline()
            self.assertTrue(self.pump(5, until=lambda: not self.app._is_running))
            self.pump(0.2)
        self.assertTrue(os.path.isfile(chosen))
        self.assertEqual(of.output_path_var.get(), "")

    def test_chosen_name_is_dropped_when_the_queue_changes(self):
        self.make_app()
        self.add_ready_job(self.media())
        self.pump(0.3)
        of = self.app.output_frame
        with mock.patch("tkinter.filedialog.asksaveasfilename",
                        return_value=os.path.join(self.dir, "x.txt")):
            of._rename()
        self.add_ready_job(self.media("second.wav"))
        self.pump(0.4)
        self.assertEqual(of.output_path_var.get(), "")


class ProgressTests(GuiTestCase):
    def test_download_then_loading_state_and_title(self):
        self.make_app()
        with mock.patch.object(type(self.app), "_is_running", new=True):
            self.app._show_download("large-v3", 1_550_000_000)
            self.assertIn("Downloading", self.app.progress_frame._step_text.get())
            self.assertTrue(self.app.title().startswith("Downloading 50%"))
            self.app._show_download("large-v3", None)
            self.assertIn("Loading the speech model", self.app.progress_frame._step_text.get())
            self.assertTrue(self.app.title().startswith("Loading"))

    def test_completion_shows_one_duration_and_no_stale_file_row(self):
        self.make_app()
        pf = self.app.progress_frame
        pf.start()
        pf.update_file(2, 2, "b.wav")
        self.assertTrue(pf._file_bar_shown)
        pf.complete(elapsed=4.3)
        self.assertFalse(pf._file_bar_shown)
        self.assertEqual(pf._elapsed_text.get(), "")
        self.assertIn("4.3 s", pf._step_text.get())
        pf.update_file(1, 2, "a.wav")
        pf.set_error("Oops")
        self.assertFalse(pf._file_bar_shown)

    def test_idle_pane_counts_queued_files_and_partial_failure_warns(self):
        self.make_app()
        pf = self.app.progress_frame
        self.assertEqual(pf._detail_text.get(), pf.READY_DETAIL)
        self.add_ready_job(self.media())
        self.add_ready_job(self.media("b.wav"))
        self.app._refresh_readiness()
        self.assertEqual(pf._detail_text.get(), "2 files ready. Press Start transcription.")
        pf.complete(elapsed=3, headline="1 of 2 files done", detail="Failed: x", warning=True)
        self.assertEqual(str(pf._headline.cget("style")), "HeadlineWarning.TLabel")
        self.app._refresh_readiness()
        self.assertEqual(pf._detail_text.get(), "Failed: x")  # the result stays
        pf.complete(elapsed=3)
        self.assertEqual(str(pf._headline.cget("style")), "HeadlineSuccess.TLabel")

    def test_blocked_queue_is_not_called_ready(self):
        self.make_app()
        pf = self.app.progress_frame
        self.add_ready_job(self.media())
        self.add_ready_job(self.media("b.wav"))
        from scrivox.ui.app import FIX_FFMPEG
        self.app._preflight_issues = [FIX_FFMPEG]
        self.app._refresh_readiness()
        self.assertEqual(pf._step_text.get(), "2 files added")
        self.assertIn("Install ffmpeg", pf._detail_text.get())
        self.app._preflight_issues = []
        self.app._refresh_readiness()
        self.assertEqual(pf._step_text.get(), pf.READY_TEXT)

    def test_error_clears_timer_and_gives_way_when_the_queue_changes(self):
        self.make_app()
        pf = self.app.progress_frame
        self.add_ready_job(self.media())
        self.app._refresh_readiness()
        pf.start()
        self.assertEqual(pf._elapsed_text.get(), "00:00")
        pf.set_error("ffmpeg isn't installed")
        self.assertEqual(pf._elapsed_text.get(), "")
        self.app._refresh_readiness()  # same files: the error stays
        self.assertEqual(pf._step_text.get(), "ffmpeg isn't installed")
        self.add_ready_job(self.media("b.wav"))
        self.app._refresh_readiness()
        self.assertEqual(pf._step_text.get(), pf.READY_TEXT)
        self.assertEqual(str(pf._headline.cget("style")), "Headline.TLabel")

    def test_close_stops_a_running_marquee(self):
        self.make_app()
        pf = self.app.progress_frame
        pf.start()
        pf.update_step(1, 1, "Transcribing speech")
        self.assertTrue(pf._marquee_running)
        with mock.patch.object(pf, "_stop_marquee", wraps=pf._stop_marquee) as stop:
            self.app._on_close()
        self.assertTrue(stop.called)
        self.app = None

    def test_cancel_during_download_says_it_continues(self):
        self.make_app()
        self.app._download_active = True
        self.app._on_pipeline_cancelled()
        self.assertIn("background", self.app.progress_frame._detail_text.get())


class FixDialogTests(GuiTestCase):
    def test_check_again_offers_restart_when_still_missing(self):
        from scrivox.ui.dialogs.help_dialog import FixHelpDialog
        self.make_app()
        restarted = []
        dlg = FixHelpDialog(self.app, "ffmpeg", on_recheck=lambda done: done(["ffmpeg"]),
                            on_restart=lambda: restarted.append(True))
        dlg._recheck_btn.invoke()
        self.pump(0.1)
        self.assertEqual(dlg._restart_btn.cget("style"), "Accent.TButton")
        self.assertTrue(dlg._restart_btn.winfo_ismapped())
        dlg._restart_btn.invoke()
        self.assertEqual(restarted, [True])

    def test_check_again_confirms_when_fixed(self):
        from scrivox.ui.dialogs.help_dialog import FixHelpDialog
        self.make_app()
        dlg = FixHelpDialog(self.app, "ffmpeg", on_recheck=lambda done: done([]),
                            on_restart=lambda: None)
        dlg._recheck_btn.invoke()
        self.pump(0.1)
        self.assertIn("installed", dlg._result.cget("text"))
        self.assertEqual(dlg._close_btn.cget("text"), "Done")
        dlg.destroy()

    def test_preflight_rereads_path_and_reports_back(self):
        import scrivox.ui.app as appmod
        self.make_app()
        results = []
        with mock.patch.object(appmod.winnative, "refresh_path") as refresh, \
                mock.patch.object(appmod.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self._real_preflight(on_done=results.append)
            self.assertTrue(self.pump(10, until=lambda: bool(results)))
        refresh.assert_called_once()
        self.assertNotIn("ffmpeg", results[0])

    def test_startup_preflight_leaves_path_alone(self):
        # Rebuilding PATH at every launch could demote the bundled CUDA DLLs
        import scrivox.ui.app as appmod
        self.make_app()
        with mock.patch.object(appmod.winnative, "refresh_path") as refresh, \
                mock.patch.object(appmod.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self._real_preflight()
            self.assertTrue(self.pump(10, until=lambda: not str(
                self.app._status_bar.cget("text")).startswith("Checking")))
        refresh.assert_not_called()

    def _real_preflight(self, **kw):
        # setUp stubs the startup check out; run the real one here
        for p in self._patches:
            if getattr(p, "attribute", None) == "_run_preflight_checks":
                p.stop()
                self._patches.remove(p)
                break
        self.app._run_preflight_checks(**kw)



@unittest.skipUnless(HAVE_DEPS, "needs torch and Pillow")
class RestartCommandTests(unittest.TestCase):
    def test_module_launch_restarts_with_dash_m(self):
        import types

        from scrivox.ui.app import restart_command
        main = types.SimpleNamespace(__spec__=types.SimpleNamespace(name="scrivox.gui"))
        cmd, cwd = restart_command(main)
        self.assertEqual(cmd, [sys.executable, "-m", "scrivox.gui"])
        self.assertEqual(cwd, os.getcwd())

    def test_script_launch_restarts_the_script(self):
        import types

        from scrivox.ui.app import restart_command
        with mock.patch.object(sys, "argv", ["main.py"]):
            cmd, cwd = restart_command(types.SimpleNamespace(__spec__=None))
        self.assertEqual(cmd, [sys.executable, os.path.abspath("main.py")])
        self.assertEqual(cwd, os.path.dirname(os.path.abspath("main.py")))


if __name__ == "__main__":
    unittest.main()
