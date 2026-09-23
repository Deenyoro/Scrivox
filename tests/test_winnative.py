"""Headless tests for the Windows helpers that don't need Windows."""

import os
import unittest

from scrivox.ui import winnative


class MergePathTests(unittest.TestCase):
    def test_process_entries_stay_first_and_new_folders_are_appended(self):
        sep = os.pathsep
        # (no drive letters: on Linux ":" is the PATH separator)
        # The frozen build's runtime hook puts the bundled torch/lib first;
        # a system CUDA folder in the registry must not overtake it
        torch_lib = "\\Scrivox\\_internal\\torch\\lib"
        cuda = "\\CUDA\\v12.6\\bin"
        current = sep.join([torch_lib, "\\Python", "\\Windows"])
        machine = sep.join([cuda, "\\Windows", "\\Windows\\System32"])
        user = "\\Users\\me\\AppData\\Local\\Microsoft\\WinGet\\Packages\\ffmpeg\\bin"
        links = "\\Users\\me\\AppData\\Local\\Microsoft\\WinGet\\Links"
        merged = winnative.merge_path(current, machine, user, [links]).split(sep)
        self.assertEqual(merged[:3], [torch_lib, "\\Python", "\\Windows"])
        self.assertEqual(merged[3:], [cuda, "\\Windows\\System32", user, links])

    def test_unchanged_when_registry_adds_nothing(self):
        sep = os.pathsep
        current = sep.join(["/b", "/a"])
        self.assertEqual(winnative.merge_path(current, "/a", "/b", []), current)

    def test_duplicates_ignore_trailing_slash_and_blank_entries(self):
        sep = os.pathsep
        merged = winnative.merge_path(sep.join(["/opt/bin/", ""]), "/opt/bin", "", [])
        self.assertEqual(merged, "/opt/bin/")

    def test_refresh_path_is_a_no_op_off_windows(self):
        if winnative.IS_WINDOWS:
            self.skipTest("Windows")
        before = os.environ.get("PATH")
        self.assertFalse(winnative.refresh_path())
        self.assertEqual(os.environ.get("PATH"), before)


if __name__ == "__main__":
    unittest.main()
