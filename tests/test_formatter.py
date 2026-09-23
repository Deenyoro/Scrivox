"""Unit tests for scrivox.core.formatter (no GPU, network or ffmpeg needed).

Run with:  python -m unittest discover -s tests
"""

import unittest

from scrivox.core.formatter import format_output, format_timestamp


class FormatTimestampTests(unittest.TestCase):
    def test_srt_and_vtt_separators(self):
        self.assertEqual(format_timestamp(3723.5, "srt"), "01:02:03,500")
        self.assertEqual(format_timestamp(3723.5, "vtt"), "01:02:03.500")

    def test_float_fractions_are_not_truncated_down(self):
        # 2.3 * 1000 == 2299.999... in binary floating point; the old
        # truncating code emitted ",299" for these.
        self.assertEqual(format_timestamp(2.3), "00:00:02,300")
        self.assertEqual(format_timestamp(4.35), "00:00:04,350")
        self.assertEqual(format_timestamp(1.15), "00:00:01,150")
        self.assertEqual(format_timestamp(1.001), "00:00:01,001")

    def test_rounding_carries_into_seconds_minutes_hours(self):
        self.assertEqual(format_timestamp(59.9996), "00:01:00,000")
        self.assertEqual(format_timestamp(3599.9999), "01:00:00,000")

    def test_zero_and_negative_clamp(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")
        self.assertEqual(format_timestamp(-0.0004), "00:00:00,000")


def _seg(start, end, text, **extra):
    seg = {"start": start, "end": end, "text": text}
    seg.update(extra)
    return seg


class SubtitleOutputTests(unittest.TestCase):
    def test_srt_cue_timing(self):
        out = format_output([_seg(0.0, 2.3, "Hello there.")], "srt")
        self.assertEqual(out.splitlines()[:3],
                         ["1", "00:00:00,000 --> 00:00:02,300", "Hello there."])

    def test_vtt_language_tag_kept(self):
        out = format_output(
            [_seg(0.0, 2.0, "Bonjour", language="fr")], "vtt",
            metadata={"detected_language": "en"},
        )
        self.assertIn("<lang fr>Bonjour</lang>", out)


class OtherFormatTests(unittest.TestCase):
    def test_tsv_escapes_tabs_and_newlines(self):
        out = format_output([_seg(1.0, 2.5, "a\tb\nc", language="en")], "tsv")
        self.assertEqual(out.splitlines(),
                         ["start\tend\tlanguage\ttext", "1.000\t2.500\ten\ta\\tb c"])

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            format_output([], "docx")


if __name__ == "__main__":
    unittest.main()
