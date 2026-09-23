"""Headless tests for the Windows helpers that don't need Windows."""

import os
import unittest

from scrivox.ui import winnative


class MergePathTests(unittest.TestCase):
    def test_registry_path_first_then_extra_then_process_only_entries(self):
        sep = os.pathsep
        # (no drive letters: on Linux ":" is the PATH separator)
        current = sep.join(["\\Python", "\\Windows"])
        machine = sep.join(["\\Windows", "\\Windows\\System32"])
        user = "\\Users\\me\\AppData\\Local\\Microsoft\\WinGet\\Packages\\ffmpeg\\bin"
        links = "\\Users\\me\\AppData\\Local\\Microsoft\\WinGet\\Links"
        merged = winnative.merge_path(current, machine, user, [links]).split(sep)
        self.assertEqual(merged[0], "\\Windows")
        self.assertIn(user, merged)
        self.assertIn(links, merged)
        self.assertEqual(merged[-1], "\\Python")
        self.assertEqual(len(merged), len(set(merged)))

    def test_duplicates_ignore_trailing_slash_and_blank_entries(self):
        sep = os.pathsep
        merged = winnative.merge_path(sep.join(["/opt/bin/", ""]), "/opt/bin", "", [])
        self.assertEqual(merged, "/opt/bin")

    def test_refresh_path_is_a_no_op_off_windows(self):
        if winnative.IS_WINDOWS:
            self.skipTest("Windows")
        before = os.environ.get("PATH")
        self.assertFalse(winnative.refresh_path())
        self.assertEqual(os.environ.get("PATH"), before)


if __name__ == "__main__":
    unittest.main()
