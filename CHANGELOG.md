# Changelog

All notable changes to Scrivox are listed here. Versions match the git tags
(`vX.Y.Z`) and `scrivox.__version__`, which the window title, status bar and
About dialog show.

## [1.8.2] - 2026-09-24

Everything below changed since v1.8.1. Note: v1.8.1 was tagged while
`scrivox/__init__.py` still said 1.8.0; this release brings the app's own
version number back in line with the tag.

### Fixed
- SRT/VTT timestamps are rounded to whole milliseconds instead of truncated, so
  cues no longer start and end 1 ms early (e.g. 2.3 s was written as ",299");
  exact half-milliseconds round up.
- WebVTT cue text, speaker names in `<v ...>` and `<lang>` values now escape
  `&`, `<` and `>`, so text like "if a < b" or a literal "-->" no longer breaks
  subtitles in players. SRT output is unchanged.
- "Check again" in the ffmpeg How to fix dialog now finds ffmpeg right after
  `winget install` (re-reads PATH from the registry and winget's Links folder)
  and reports the result; "Restart Scrivox" is offered if it is still missing.
- The bundled CUDA DLLs stay first on PATH, so a system CUDA/cuDNN install can
  no longer be picked up instead of the ones Scrivox ships.
- The whole winget command (including the trailing `-e`) is visible in the
  How to fix dialog.
- A name chosen with "Rename..." can no longer silently overwrite an existing
  file after switching format or folder, and it survives a failed or cancelled
  run.
- Cancelling the first-run model download no longer starts a second
  concurrent download on the next Start; the next run re-attaches to it.
- Closing the window is clean on Tk 9 ("can't delete Tcl command") and Tk 8.6
  (no background error about a destroyed progress bar).
- "Restart Scrivox" works when run from source with `python -m scrivox.gui`.
- The model list highlights the current model instead of "tiny".
- The Extras summary is only clickable when Extras can open; Lite shows a
  plain note instead of an empty section.
- The queue file count no longer clips; it now sits in the Files header.
- Dark title bar: correct ctypes argument types for window handles and an
  immediate repaint on Windows 10.

### Changed
- Main window rebuilt as a three-step flow (1 Files, 2 Options, 3 Save to)
  with one Start/Cancel button, fitting on one screen at 1366x768 @100% and
  1920x1080 @150%.
- Every GUI run now saves a file next to the original (or in the chosen
  folder), numbered instead of overwriting; previously the default run saved
  nothing. The CLI is unchanged.
- Completion bar with Saved as / Open / Show in folder / Copy / Save as, a
  results table for batches, bell, taskbar flash and progress in the title.
- Preflight banner with "How to fix" steps for missing ffmpeg or GPU; failed
  runs show a plain headline and jump to the error line in the Log tab.
- Inline validation above Start instead of error pop-ups; a refused Start is
  visibly highlighted and focuses the field to fix.
- API keys and advanced options moved to a Settings window (Ctrl+,); menu bar
  with keyboard shortcuts.
- The status pane says how many files are ready, does not say "Ready" while a
  run cannot start (e.g. ffmpeg missing), shows partial batch failures in the
  warning colour, and clears stale errors when the queue changes.
- Model download progress is shown on first run ("large-v3: 800 MB of about
  3.1 GB"), then "Loading the speech model...", and Cancel takes effect
  immediately.
- Model and format lists describe each entry; stored settings values are
  unchanged and old settings files load as before.
- Track detection (ffprobe) runs off the UI thread; mouse wheel over a
  combobox scrolls the panel instead of changing the value.

### Added
- Finished dark theme: Segoe UI fonts, readable disabled buttons, DPI-scaled
  check marks and radio dots, slim scrollbars, red borders on invalid fields.
- Windows integration: DPI awareness, taskbar app id, dark title bar, taskbar
  flash, open file / show in Explorer.
- App icon (`assets/scrivox.ico`, `assets/scrivox.png`) bundled into the build.
- Tk errors and thread errors are written to `scrivox_error.log` with a
  friendly dialog; a drag-and-drop (tkdnd) load failure falls back to browsing.
- Offline unit test suite in `tests/` (`python -m unittest discover -s
  tests`): timestamp formatting, subtitle/TSV output, LLM client retries and
  429 Retry-After handling, translation parsing, model download, output
  naming, GUI behaviour and Windows helpers. No API keys or network needed.

### Changed (developer)
- `test_setup.py` no longer runs its environment checks (CUDA, model loads,
  paid OpenRouter call) when a test runner imports it; running it as a script
  is unchanged.
- Removed 13 unused imports and an unused variable flagged by ruff.
- README: three-step window, where settings live, build.py builds all three
  variants when no flag is given (`--regular` documented), how to run the
  unit tests (also under Tk 8.6).
