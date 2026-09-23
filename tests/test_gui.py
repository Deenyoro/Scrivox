"""Behaviour tests for the Tk GUI (need a display: run under xvfb-run on Linux).

Skipped automatically when there is no display or the heavy runtime
dependencies (torch, Pillow) aren't installed. Nothing here touches the
network, ffmpeg or a GPU: the pipeline and the ffprobe track probe are
replaced with fakes.

Run with:  xvfb-run -a python -m unittest discover -s tests
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


if __name__ == "__main__":
    unittest.main()
