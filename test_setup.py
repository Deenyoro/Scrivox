"""Verify the full Scrivox GPU + Diarization + Vision + Dictation pipeline."""

import os
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import torch
from scrivox.core.torch_compat import _allow_unsafe_torch_load


def check(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


print("=" * 60)
print("  Scrivox Full Setup Test")
print("=" * 60)

all_ok = True

# ── Core dependencies ──
print("\n  -- Core Dependencies --")
all_ok &= check("PyTorch CUDA", lambda: torch.cuda.get_device_name(0))
all_ok &= check("VRAM", lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
all_ok &= check("faster-whisper", lambda: f"v{__import__('faster_whisper').__version__}")
all_ok &= check("numpy", lambda: f"v{__import__('numpy').__version__}")
all_ok &= check("python-dotenv", lambda: "ok" if __import__("dotenv") else "ok")
all_ok &= check("requests", lambda: f"v{__import__('requests').__version__}")

# ── Audio / Dictation dependencies ──
print("\n  -- Audio / Dictation --")
all_ok &= check("sounddevice", lambda: __import__("sounddevice").query_devices(kind="input")["name"])
all_ok &= check("keyboard", lambda: "ok" if __import__("keyboard").is_pressed else "ok")
all_ok &= check("pynput (typing)", lambda: "ok" if __import__("pynput.keyboard", fromlist=["Controller"]).Controller else "ok")

# ── Diarization dependencies ──
print("\n  -- Diarization --")
all_ok &= check("pyannote.audio", lambda: "ok" if __import__("pyannote.audio") else "ok")

# ── System tools ──
print("\n  -- System Tools --")

def check_ffmpeg():
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
    version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
    return version_line.strip()

def check_ffprobe():
    result = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, timeout=5)
    version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
    return version_line.strip()

all_ok &= check("ffmpeg", check_ffmpeg)
all_ok &= check("ffprobe", check_ffprobe)

# ── Credentials ──
print("\n  -- Credentials --")
hf_token = os.environ.get("HF_TOKEN")
openrouter_key = os.environ.get("OPENROUTER_API_KEY")

if hf_token:
    print(f"  [OK] HF_TOKEN: set ({hf_token[:8]}...)")
else:
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print(f"  [OK] HF_TOKEN: cached via huggingface-cli ({token[:8]}...)")
            hf_token = token
        else:
            print(f"  [WARN] HF_TOKEN: not set (diarization will not work)")
    except Exception:
        print(f"  [WARN] HF_TOKEN: not set (diarization will not work)")

if openrouter_key:
    print(f"  [OK] OPENROUTER_API_KEY: set ({openrouter_key[:12]}...)")
else:
    print(f"  [WARN] OPENROUTER_API_KEY: not set (vision/summary will not work)")

# ── Scrivox module import test ──
print("\n  -- Scrivox Module --")
all_ok &= check("scrivox package", lambda: f"v{__import__('scrivox').__version__}")
all_ok &= check("scrivox.core.pipeline", lambda: "ok" if __import__("scrivox.core.pipeline") else "ok")

# ── GPU transcription test ──
print("\n  -- Functional Tests --")
print("  Testing faster-whisper GPU transcription...")
try:
    from faster_whisper import WhisperModel
    import numpy as np
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    silence = np.zeros(16000, dtype=np.float32)
    segments, info = model.transcribe(silence, beam_size=1)
    list(segments)
    print(f"  [OK] GPU transcription works (large-v3, float16)")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  [FAIL] GPU transcription: {e}")
    all_ok = False

# ── Diarization test ──
if hf_token:
    print("  Testing pyannote diarization pipeline...")
    try:
        from pyannote.audio import Pipeline
        with _allow_unsafe_torch_load():
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        pipeline.to(torch.device("cuda"))
        print(f"  [OK] Diarization pipeline loaded on GPU")
        del pipeline
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [FAIL] Diarization: {e}")
        all_ok = False
else:
    print("  [SKIP] Diarization test (no HF_TOKEN)")

# ── Vision API test ──
if openrouter_key:
    print("  Testing OpenRouter API connectivity...")
    try:
        import requests
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {openrouter_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"  [OK] OpenRouter API reachable (key valid)")
        else:
            print(f"  [WARN] OpenRouter API returned {resp.status_code}")
    except Exception as e:
        print(f"  [FAIL] OpenRouter API: {e}")
        all_ok = False
else:
    print("  [SKIP] OpenRouter API test (no key)")

# ── Results ──
print()
print("=" * 60)
if all_ok:
    print("  ALL TESTS PASSED!")
else:
    print("  SOME TESTS FAILED - check errors above")

print()
print("  USAGE:")
print()
print("  DICTATION (real-time, type into any window):")
print("    python dictate.py")
print("    python dictate.py --model base    (faster, less accurate)")
print("    Hold Ctrl+Shift to record, release to transcribe & type")
print()
print("  TRANSCRIBE FILES (CLI):")
print("    python main.py meeting.mp3")
print("    python main.py meeting.wav --diarize")
print("    python main.py meeting.mp4 --all")
print("    python main.py video.mp4 --diarize --format srt -o subtitles.srt")
print("    python main.py meeting.mp4 --all --format md -o minutes.md")
print()
print("  GUI:")
print("    python main.py       (no arguments launches the GUI)")
print()

if not all_ok:
    sys.exit(1)
