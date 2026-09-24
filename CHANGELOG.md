# Changelog

All notable changes to Scrivox are listed here. Versions match the git tags
(`vX.Y.Z`) and `scrivox.__version__`, which the window title, status bar and
About dialog show.

## [1.8.3] - 2026-09-24

Build and release changes only; the app itself is unchanged.

### Added
- GitLab CI pipeline (`.gitlab-ci.yml`) that replaces the GitHub Actions
  release workflow. It runs on `v*` tags and on manual web/API runs, never on
  plain pushes.
  - **test** (Linux, Python 3.11, Tk 8.6): unit and GUI tests under Xvfb with
    CPU-only torch. For a release it also checks that `scrivox.__version__`
    matches the tag, that this file has an entry for the version, and that a
    `RELEASE_VERSION` run was started on that tag.
  - **build-windows-lite / -regular / -full** (Windows runner): Python 3.11.9
    + Tk, Inno Setup 6.7.3 and 7-Zip from pinned, SHA-256-checked downloads
    (`ci/tools-windows.ps1`); CUDA 12.6 torch, `build.py --clean --<variant>`,
    bundle checks (`ci/check_build.py`), the portable `.7z` and the installer
    (`ci/build-windows.ps1`). On release runs each job uploads its files to
    the GitLab Package Registry (`ci/publish-windows.ps1`).
  - **release**: creates or updates the GitLab release with links to those
    files.
- README: "Building / Releases (GitLab CI)" section.

### Changed
- Releases are built and published on GitLab only (GitHub Actions is
  disabled).
- An installer of 2 GiB or more (over GitHub's release asset limit, which
  applies because releases are mirrored to GitHub) is still left out of the
  release, as before; that variant ships its `.7z` portable only.
- Publishing never uploads a file whose name is already in that version's
  package (builds are not reproducible, so the copy would differ). To
  republish, release a new patch version; the README explains this.

### Notes
- Scrivox-Full is built only once an `HF_TOKEN` CI/CD variable is added to
  the GitLab project (it downloads the gated pyannote models). If that
  variable is protected, the `v*` tags must be protected too. Until then a
  release has Lite and Regular only, and its notes say Full was not built.

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
