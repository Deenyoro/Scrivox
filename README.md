# Scrivox

GPU-accelerated transcription suite with speaker diarization, SRT subtitle generation, visual context analysis, and meeting summarization.

Built by **Deenyoro** at **KawaConnect LLC**.

---

## Features

- **Transcription** — faster-whisper on CUDA with float16 precision
- **Speaker Diarization** — identify who said what via pyannote
- **SRT/VTT Subtitles** — subtitle files with optional speaker labels
- **Multi-Language** — auto-detect or specify any language code
- **Vision Analysis** — extract video keyframes and describe them with vision LLMs
- **Meeting Summary** — structured summaries with action items and key points
- **6 Output Formats** — txt, md, srt, vtt, json, tsv
- **GUI + CLI** — Tkinter desktop app or full command-line interface
- **Portable Config** — JSON config stored next to the executable

## Quick Start

### GUI

Double-click `Scrivox.exe` or run without arguments:

```
python main.py
```

### CLI

```bash
# Basic transcription
python main.py meeting.mp3

# Diarized SRT subtitles
python main.py video.mp4 --diarize --format srt -o subtitles.srt

# Full pipeline: diarize + vision + summary
python main.py meeting.mp4 --all --format md -o minutes.md

# Custom speaker names
python main.py meeting.mp4 --diarize --speaker-names "Alice,Bob,Charlie"

# Different language
python main.py audio.wav --language fr

# All features with JSON output
python main.py video.mp4 --all -f json -o report.json
```

### Dictation (Real-Time)

```bash
python dictate.py
# Hold Ctrl+Shift to record, release to transcribe and type
# Ctrl+Shift+Q to quit
```

## Installation

### From Release (Recommended)

1. Download the latest release from [Releases](https://github.com/Deenyoro/Scrivox/releases)
2. Extract the `.7z` archive
3. Copy `.env.example` to `.env` and add your API keys
4. Run `Scrivox.exe`

Two variants are available:

| Variant | Description |
|---------|-------------|
| **Scrivox-Full** | Diarization models pre-bundled, works immediately |
| **Scrivox-Lite** | Smaller download, provide your own HuggingFace token |

### From Source

```bash
git clone https://github.com/Deenyoro/Scrivox.git
cd Scrivox
pip install -r requirements.txt
python main.py
```

### Custom Models

You can provide your own models by placing them in a `models/` directory next to the executable:

```
Scrivox/
  Scrivox.exe
  models/
    whisper/
      large-v3/          # Custom Whisper model (CTranslate2 format)
    hub/                  # Custom HuggingFace models (diarization)
```

## Requirements

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with CUDA support
- **ffmpeg** in PATH ([download](https://ffmpeg.org/download.html))

## API Keys

Add to `.env` file or enter in the GUI:

| Key | Required For | Get One |
|-----|-------------|---------|
| `HF_TOKEN` | Speaker diarization (Lite variant) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `OPENROUTER_API_KEY` | Vision analysis, meeting summaries | [openrouter.ai/keys](https://openrouter.ai/keys) |

For diarization, you must also accept the model license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

## CLI Reference

```
python main.py <input> [options]

Options:
  --model MODEL           Whisper model: tiny, base, small, medium, large-v3
  --language LANG         Language code (en, fr, ja, etc.) or auto-detect
  --format FORMAT         Output: txt, md, srt, vtt, json, tsv
  --output PATH, -o PATH  Output file path

  --diarize               Enable speaker diarization
  --num-speakers N        Exact speaker count (if known)
  --min-speakers N        Minimum speakers expected
  --max-speakers N        Maximum speakers expected
  --speaker-names NAMES   Comma-separated names: "Alice,Bob"

  --vision                Analyze video keyframes with vision LLM
  --vision-interval SEC   Seconds between keyframes (default: 60)
  --vision-model MODEL    Vision model (default: google/gemini-2.5-flash)

  --subtitle-speakers     Show speaker labels in SRT/VTT (off by default)

  --summarize             Generate meeting summary
  --summary-model MODEL   Summary model (default: google/gemini-2.5-flash)

  --all                   Enable diarize + vision + summarize
  --clear-cache           Force re-transcription
```

## Building from Source

```bash
# Install build dependencies
pip install pyinstaller

# Build both variants (Full requires HF_TOKEN env var)
python build.py --clean

# Build only Lite variant (no models needed)
python build.py --lite

# Build only Full variant (downloads diarization models)
set HF_TOKEN=hf_your_token_here
python build.py --full
```

## Project Structure

```
scrivox/
  __init__.py              Package metadata
  cli.py                   CLI entry point (argparse -> pipeline)
  gui.py                   GUI entry point
  config.py                JSON config manager
  core/
    constants.py           Models, extensions, defaults
    torch_compat.py        PyTorch 2.6+ compatibility
    media.py               ffmpeg utilities
    transcriber.py         Whisper transcription + post-processing
    diarizer.py            Speaker diarization
    vision.py              Keyframe extraction + vision LLM
    summarizer.py          Meeting summary generation
    formatter.py           Output formatting (6 formats)
    pipeline.py            Pipeline orchestrator (PipelineConfig -> run)
  ui/
    app.py                 Main application window
    theme.py               Dark theme configuration
    log_redirect.py        Thread-safe stdout -> log widget
    frames/
      input_frame.py       File selection + media info
      settings_frame.py    Model, language, feature toggles
      api_frame.py         API key management
      output_frame.py      Format + output path
      progress_frame.py    Progress bar + elapsed timer
      log_frame.py         Scrollable log display
      results_frame.py     Transcript display + copy/save
```

---

**KawaConnect LLC** | Built by Deenyoro
