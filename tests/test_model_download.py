"""Tests for the first-run model download helper (no network: faster-whisper's
download function is replaced with a fake)."""

import threading
import time
import unittest
from unittest import mock

try:
    import faster_whisper.utils  # noqa: F401

    from scrivox.core import transcriber
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


class _Cancelled(Exception):
    pass


@unittest.skipUnless(HAVE_DEPS, "needs torch and faster-whisper")
class EnsureModelDownloadedTests(unittest.TestCase):
    def test_cached_model_skips_download(self):
        calls = []

        def fake_download(name, local_files_only=False, **kw):
            calls.append(local_files_only)
            return "/cache/model"

        with mock.patch("faster_whisper.utils.download_model", fake_download):
            self.assertFalse(transcriber.ensure_model_downloaded("tiny", lambda n: None))
        self.assertEqual(calls, [True])

    def test_download_reports_progress(self):
        reports = []

        def fake_download(name, local_files_only=False, **kw):
            if local_files_only:
                raise OSError("not cached")
            time.sleep(0.8)
            return "/cache/model"

        with mock.patch("faster_whisper.utils.download_model", fake_download):
            self.assertTrue(transcriber.ensure_model_downloaded("tiny", reports.append))
        self.assertGreaterEqual(len(reports), 2)

    def test_cancel_stops_waiting_for_download(self):
        release = threading.Event()

        def fake_download(name, local_files_only=False, **kw):
            if local_files_only:
                raise OSError("not cached")
            release.wait(10)
            return "/cache/model"

        cancel = threading.Event()

        def should_cancel():
            if cancel.is_set():
                raise _Cancelled()

        threading.Timer(0.3, cancel.set).start()
        t0 = time.monotonic()
        with mock.patch("faster_whisper.utils.download_model", fake_download), \
                self.assertRaises(_Cancelled):
            transcriber.ensure_model_downloaded("tiny", lambda n: None, should_cancel)
        self.assertLess(time.monotonic() - t0, 3)
        release.set()

    def test_finish_is_signalled_with_none(self):
        reports = []

        def fake_download(name, local_files_only=False, **kw):
            if local_files_only:
                raise OSError("not cached")
            time.sleep(0.6)
            return "/cache/model"

        with mock.patch("faster_whisper.utils.download_model", fake_download):
            transcriber.ensure_model_downloaded("tiny", reports.append)
        self.assertIsNone(reports[-1])
        self.assertTrue(all(isinstance(r, int) for r in reports[:-1]))

    def test_restart_after_cancel_reattaches_to_running_download(self):
        # Cancel only stops waiting; a second Start must not launch a second
        # concurrent multi-GB download of the same model
        release = threading.Event()
        started = []

        def fake_download(name, local_files_only=False, **kw):
            if local_files_only:
                raise OSError("not cached")
            started.append(name)
            release.wait(10)
            return "/cache/model"

        cancel = threading.Event()

        def should_cancel():
            if cancel.is_set():
                raise _Cancelled()

        with mock.patch("faster_whisper.utils.download_model", fake_download):
            threading.Timer(0.3, cancel.set).start()
            with self.assertRaises(_Cancelled):
                transcriber.ensure_model_downloaded("small", lambda n: None, should_cancel)
            self.assertTrue(transcriber.download_in_progress("small"))
            threading.Timer(0.3, release.set).start()
            self.assertTrue(transcriber.ensure_model_downloaded("small", lambda n: None))
        self.assertEqual(started, ["small"])
        self.assertFalse(transcriber.download_in_progress("small"))

    def test_unknown_model_name_is_left_to_whisper(self):
        def fake_download(name, local_files_only=False, **kw):
            raise ValueError("Invalid model size")

        with mock.patch("faster_whisper.utils.download_model", fake_download):
            self.assertFalse(transcriber.ensure_model_downloaded("nope", lambda n: None))


if __name__ == "__main__":
    unittest.main()
