"""Tests for the GUI's output naming and error explanations (no Tk needed).

Run with:  python -m unittest discover -s tests
"""

import os
import tempfile
import unittest

from scrivox.ui.output_paths import (
    FIX_FFMPEG,
    FIX_GPU,
    FIX_KEYS,
    default_output_path,
    describe_model,
    explain_error,
    format_size,
    is_media_file,
    plan_output_paths,
    unique_path,
)


class DefaultOutputPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.media = os.path.join(self.dir, "interview.wav")
        open(self.media, "wb").close()

    def tearDown(self):
        self._tmp.cleanup()

    def test_txt_gets_transcript_suffix_next_to_input(self):
        # Regression: with the default txt format and no extras, the GUI used
        # to pass output_path=None and nothing was written to disk at all
        self.assertEqual(default_output_path(self.media, "txt"),
                         os.path.join(self.dir, "interview_transcript.txt"))

    def test_other_formats_use_input_stem(self):
        for fmt in ("md", "srt", "vtt", "json", "tsv"):
            self.assertEqual(default_output_path(self.media, fmt),
                             os.path.join(self.dir, f"interview.{fmt}"))

    def test_audio_track_suffix(self):
        self.assertEqual(default_output_path(self.media, "srt", audio_track=2),
                         os.path.join(self.dir, "interview_track2.srt"))

    def test_existing_file_is_never_overwritten(self):
        taken = os.path.join(self.dir, "interview_transcript.txt")
        with open(taken, "w") as f:
            f.write("earlier work")
        self.assertEqual(default_output_path(self.media, "txt"),
                         os.path.join(self.dir, "interview_transcript (2).txt"))
        with open(os.path.join(self.dir, "interview_transcript (2).txt"), "w"):
            pass
        self.assertEqual(default_output_path(self.media, "txt"),
                         os.path.join(self.dir, "interview_transcript (3).txt"))

    def test_out_dir(self):
        out = os.path.join(self.dir, "out")
        os.mkdir(out)
        self.assertEqual(default_output_path(self.media, "vtt", out_dir=out),
                         os.path.join(out, "interview.vtt"))


class PlanOutputPathsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _p(self, *parts):
        return os.path.join(self.dir, *parts)

    def test_same_stem_in_two_folders_into_one_out_dir(self):
        os.mkdir(self._p("a"))
        os.mkdir(self._p("b"))
        os.mkdir(self._p("out"))
        jobs = [(self._p("a", "meeting.mp4"), 0), (self._p("b", "meeting.mp4"), 0)]
        paths = plan_output_paths(jobs, "srt", out_dir=self._p("out"))
        self.assertEqual(paths, [self._p("out", "meeting.srt"), self._p("out", "meeting (2).srt")])

    def test_each_track_gets_its_own_file(self):
        jobs = [(self._p("talk.mkv"), 0), (self._p("talk.mkv"), 2)]
        self.assertEqual(plan_output_paths(jobs, "txt"),
                         [self._p("talk_transcript.txt"), self._p("talk_track2_transcript.txt")])

    def test_explicit_path_single_job_used_verbatim(self):
        chosen = self._p("notes.md")
        self.assertEqual(plan_output_paths([(self._p("x.wav"), 0)], "md", explicit_output=chosen),
                         [chosen])

    def test_explicit_path_batch_is_made_unique_per_job(self):
        chosen = self._p("minutes.md")
        jobs = [(self._p("x.wav"), 0), (self._p("y.wav"), 0)]
        self.assertEqual(plan_output_paths(jobs, "md", explicit_output=chosen),
                         [self._p("minutes_x.md"), self._p("minutes_y.md")])

    def test_unique_path_respects_taken(self):
        cand = self._p("a.txt")
        self.assertEqual(unique_path(cand, taken=[cand]), self._p("a (2).txt"))


class ExplainErrorTests(unittest.TestCase):
    def test_ffmpeg(self):
        headline, fix = explain_error("'ffmpeg' not found. Install ffmpeg and ensure it's in your PATH.")
        self.assertIn("ffmpeg isn't installed", headline)
        self.assertEqual(fix, FIX_FFMPEG)

    def test_gpu(self):
        self.assertEqual(explain_error("CUDA GPU not available. This tool requires an NVIDIA GPU.")[1],
                         FIX_GPU)

    def test_keys(self):
        self.assertEqual(explain_error("Diarization requires HF_TOKEN in .env, config, or "
                                       "huggingface-cli login")[1], FIX_KEYS)
        self.assertEqual(explain_error("Vision/Summary/Translation requires an LLM API key "
                                       "in .env or config")[1], FIX_KEYS)

    def test_unknown_error_keeps_first_line(self):
        headline, fix = explain_error("Something odd\nwith details")
        self.assertEqual((headline, fix), ("Something odd", None))


class SmallHelpersTests(unittest.TestCase):
    def test_is_media_file(self):
        self.assertTrue(is_media_file("C:/x/Movie.MKV"))
        self.assertTrue(is_media_file("a.m4a"))
        self.assertFalse(is_media_file("notes.txt"))

    def test_format_size(self):
        self.assertEqual(format_size(3.1e9), "3.1 GB")
        self.assertEqual(format_size(484e6), "484 MB")
        self.assertEqual(format_size(512), "512 bytes")

    def test_describe_model(self):
        self.assertIn("3.1 GB", describe_model("large-v3"))
        self.assertEqual(describe_model("my/custom-model"), "Custom model name or folder")


if __name__ == "__main__":
    unittest.main()
