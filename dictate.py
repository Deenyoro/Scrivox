"""
Whisper GPU Dictation - Real-time speech-to-text with keyboard typing output.

Uses faster-whisper on GPU for fast transcription.

Usage:
    python dictate.py [options]

Controls:
    Hold Ctrl+Shift   - Record audio (release to transcribe & type)
    Ctrl+Shift+Q      - Quit

Options:
    --model       Model size: tiny, base, small, medium, large-v3 (default: large-v3)
    --language    Language code, e.g. en (default: auto-detect)
    --device-id   Audio input device ID (default: system default)
    --list-devices  List available audio input devices and exit
    --hotkey      Hotkey combo to hold for recording (default: ctrl+shift)
    --copy-mode   Copy to clipboard instead of typing
"""

import argparse
import re
import sys
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def _clean_dictation_text(text):
    """Remove Whisper hallucination artifacts from dictated text."""
    # Remove repeated phrases (2+ word phrases repeated 2+ times), restart after each collapse
    changed = True
    while changed:
        changed = False
        words = text.split()
        if len(words) < 4:
            break
        for phrase_len in range(len(words) // 2, 1, -1):
            for start in range(len(words) - phrase_len * 2 + 1):
                phrase = words[start:start + phrase_len]
                repeats = 1
                pos = start + phrase_len
                while pos + phrase_len <= len(words) and words[pos:pos + phrase_len] == phrase:
                    repeats += 1
                    pos += phrase_len
                if repeats >= 2:
                    before = words[:start]
                    after = words[pos:]
                    words = before + phrase + after
                    text = " ".join(words)
                    changed = True
                    break
            if changed:
                break

    # Strip stray non-ASCII characters (foreign script hallucinations)
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u2018\u2019\u201C\u201D\u2014\u2013]+', '', text)
    text = text.strip()

    if not text:
        return text

    # Capitalize first letter and pronoun I
    text = text[0].upper() + text[1:]
    text = re.sub(r'\bi\b', 'I', text)

    return text


def list_audio_devices():
    print("\nAvailable audio INPUT devices:\n")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " <-- DEFAULT" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}] {d['name']}{marker}")
    print()


def type_text(text):
    from pynput.keyboard import Controller, Key
    kb = Controller()
    time.sleep(0.15)
    for char in text:
        if char == "\n":
            kb.press(Key.enter)
            kb.release(Key.enter)
        else:
            kb.type(char)


def main():
    parser = argparse.ArgumentParser(description="Whisper GPU Dictation")
    parser.add_argument("--model", default="large-v3",
                        help="Model: tiny, base, small, medium, large-v3 (default: large-v3)")
    parser.add_argument("--language", default=None,
                        help="Language code e.g. 'en' (default: auto-detect)")
    parser.add_argument("--device-id", type=int, default=None,
                        help="Audio input device ID")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio input devices and exit")
    parser.add_argument("--hotkey", default="ctrl+shift",
                        help="Hotkey to hold for recording (default: ctrl+shift)")
    parser.add_argument("--copy-mode", action="store_true",
                        help="Copy to clipboard instead of typing")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # Check CUDA availability before loading model
    try:
        import torch
        if not torch.cuda.is_available():
            print("Error: CUDA GPU not available. This tool requires an NVIDIA GPU.", file=sys.stderr)
            print("  Check that you have NVIDIA drivers and CUDA toolkit installed.", file=sys.stderr)
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        print("Error: PyTorch not installed. Run: pip install torch", file=sys.stderr)
        sys.exit(1)

    # Load model
    from faster_whisper import WhisperModel
    print(f"Loading faster-whisper '{args.model}' on CUDA (float16)...")
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    print("Model loaded!\n")

    # State
    recording = False
    frames = []

    def audio_callback(indata, frame_count, time_info, status):
        try:
            if status:
                print(f"  [audio status: {status}]", file=sys.stderr)
            if recording:
                frames.append(indata.copy())
        except Exception:
            pass  # never crash the audio stream

    # Validate audio device
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
            device=args.device_id, blocksize=1024, callback=audio_callback,
        )
    except sd.PortAudioError as e:
        print(f"Error: Could not open audio device: {e}", file=sys.stderr)
        if args.device_id is not None:
            print("  Try --list-devices to see available devices.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  WHISPER GPU DICTATION")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Language: {args.language or 'auto-detect'}")
    print(f"  GPU:      {gpu_name}")
    print(f"  Hotkey:   Hold {args.hotkey.upper()} to record")
    print(f"  Output:   {'Clipboard' if args.copy_mode else 'Type into active window'}")
    print(f"  Quit:     Ctrl+Shift+Q")
    print("=" * 60)
    print("\nReady! Hold the hotkey and speak...\n")

    import keyboard
    stream.start()

    try:
        while True:
            keyboard.wait(args.hotkey, suppress=False)

            recording = True
            frames.clear()
            print("  ** RECORDING... (release hotkey to stop)")

            while keyboard.is_pressed(args.hotkey):
                time.sleep(0.02)

            recording = False
            duration = len(frames) * 1024 / SAMPLE_RATE if frames else 0

            if duration < 0.3:
                print("  (too short, skipped)\n")
                continue

            print(f"  Recording: {duration:.1f}s - Transcribing...")

            audio_np = np.concatenate(frames, axis=0).flatten().astype(np.float32) / 32768.0
            segments, _ = model.transcribe(
                audio_np, beam_size=5,
                language=args.language or None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            text = _clean_dictation_text(text)

            if text:
                print(f"  >> {text}")
                if args.copy_mode:
                    import pyperclip
                    pyperclip.copy(text)
                    print("  (copied to clipboard)\n")
                else:
                    type_text(text)
                    print("  (typed into active window)\n")
            else:
                print("  (no speech detected)\n")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        stream.close()
    print("Goodbye!")


if __name__ == "__main__":
    import keyboard
    keyboard.add_hotkey("ctrl+shift+q", lambda: sys.exit(0))
    main()
