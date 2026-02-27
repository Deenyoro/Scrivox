"""Speaker diarization using pyannote.audio with bundled model support."""

import os
import sys
import threading
import time

import torch

from .torch_compat import _allow_unsafe_torch_load
from .media import extract_wav


def _get_bundled_models_dir():
    """Check for bundled models directory next to the exe or project root.

    Looks for a 'models' directory containing pre-downloaded HuggingFace models.
    Users can also place their own models here to avoid needing an HF token.

    Returns the path if found, or None.
    """
    if getattr(sys, "frozen", False):
        # PyInstaller exe — check next to the exe
        base = os.path.dirname(sys.executable)
    else:
        # Dev mode — check project root
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    models_dir = os.path.join(base, "models")
    if os.path.isdir(models_dir):
        return models_dir
    return None


def _setup_bundled_cache():
    """Point HuggingFace Hub cache to bundled models directory if available.

    MUST be called BEFORE importing pyannote.audio — huggingface_hub reads
    HF_HUB_OFFLINE at import time and caches it as a module-level constant.

    Returns True if bundled models were found and configured.
    """
    models_dir = _get_bundled_models_dir()
    if models_dir:
        # Set HF cache env vars so huggingface_hub finds the pre-downloaded models
        os.environ["HF_HOME"] = models_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(models_dir, "hub")
        # Force offline mode so pyannote doesn't try to access HuggingFace
        os.environ["HF_HUB_OFFLINE"] = "1"
        return True
    return False


def _force_hf_offline():
    """Force huggingface_hub into offline mode even if already imported.

    huggingface_hub reads HF_HUB_OFFLINE once at import time into a module
    constant. Sub-modules copy it via `from .constants import HF_HUB_OFFLINE`,
    creating local bindings that aren't updated by os.environ changes.
    We must patch every module that cached the old value.
    """
    try:
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_OFFLINE = True
    except ImportError:
        return

    for mod in sys.modules.values():
        if (mod is not None
                and getattr(mod, '__package__', None)
                and getattr(mod, '__package__', '').startswith('huggingface_hub')
                and hasattr(mod, 'HF_HUB_OFFLINE')):
            mod.HF_HUB_OFFLINE = True


def diarize_audio(audio_path, hf_token, num_speakers=None, min_speakers=None,
                  max_speakers=None, diarization_model=None, audio_track=0,
                  on_progress=print):
    """Run speaker diarization on audio. Returns list of speaker segments.

    If bundled models are found in a 'models/' directory next to the exe,
    they are used directly and no HF token is needed for download.
    """
    from .constants import DEFAULT_DIARIZATION_MODEL

    if not diarization_model:
        diarization_model = DEFAULT_DIARIZATION_MODEL

    # MUST set up bundled cache BEFORE importing pyannote.audio —
    # that import triggers huggingface_hub which reads HF_HUB_OFFLINE
    # env var at import time and caches it as a module constant.
    has_bundled = _setup_bundled_cache()

    from pyannote.audio import Pipeline

    # Force-patch all huggingface_hub modules that cached the offline
    # constant (covers case where huggingface_hub was already imported
    # before _setup_bundled_cache set the env var)
    if has_bundled:
        _force_hf_offline()

    wav_path = None
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in (".wav", ".wave"):
        wav_path = extract_wav(audio_path, track_index=audio_track,
                               on_progress=on_progress)
        diarize_input = wav_path
    else:
        diarize_input = audio_path

    try:
        if has_bundled:
            on_progress("Using bundled diarization models...")
        else:
            on_progress("Downloading diarization models (first run only)...")

        on_progress(f"Loading {diarization_model} on CUDA...")
        with _allow_unsafe_torch_load():
            if has_bundled:
                # Bundled models — load from local cache, no token needed
                pipeline = Pipeline.from_pretrained(diarization_model)
            else:
                # Download from HF Hub — needs token
                # pyannote >= 3.3 uses 'token'; older versions use 'use_auth_token'
                try:
                    pipeline = Pipeline.from_pretrained(
                        diarization_model, token=hf_token,
                    )
                except TypeError:
                    pipeline = Pipeline.from_pretrained(
                        diarization_model, use_auth_token=hf_token,
                    )
        pipeline.to(torch.device("cuda"))

        on_progress("Running diarization... (this may take several minutes)")
        t0 = time.time()

        stop_progress = threading.Event()

        def progress_ticker():
            while not stop_progress.is_set():
                elapsed = time.time() - t0
                on_progress(f"  Diarizing... {elapsed:.0f}s elapsed")
                stop_progress.wait(5)

        ticker = threading.Thread(target=progress_ticker, daemon=True)
        ticker.start()

        try:
            diarization = pipeline(
                diarize_input,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        finally:
            stop_progress.set()
            ticker.join(timeout=5)

        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        elapsed = time.time() - t0
        speakers = set(s["speaker"] for s in speaker_segments)
        on_progress(f"Diarization done in {elapsed:.1f}s ({len(speakers)} speakers detected)")

        del pipeline
        torch.cuda.empty_cache()

        return speaker_segments

    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def assign_speakers(transcript_segments, speaker_segments, speaker_names=None):
    """Assign speaker labels to transcript segments via overlap detection."""
    EPSILON = 1e-6
    for seg in transcript_segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for spk in speaker_segments:
            overlap_start = max(seg["start"], spk["start"])
            overlap_end = min(seg["end"], spk["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap + EPSILON:
                best_overlap = overlap
                best_speaker = spk["speaker"]

        # If no overlap found, fall back to nearest speaker in time
        if best_speaker == "UNKNOWN" and speaker_segments:
            min_gap = float('inf')
            for spk in speaker_segments:
                gap = max(0, max(seg["start"], spk["start"]) - min(seg["end"], spk["end"]))
                if gap < min_gap:
                    min_gap = gap
                    best_speaker = spk["speaker"]

        seg["speaker"] = best_speaker

    speaker_map = {}
    counter = 1
    for seg in transcript_segments:
        if seg["speaker"] not in speaker_map and seg["speaker"] != "UNKNOWN":
            if speaker_names and counter <= len(speaker_names):
                speaker_map[seg["speaker"]] = speaker_names[counter - 1]
            else:
                speaker_map[seg["speaker"]] = f"SPEAKER_{counter:02d}"
            counter += 1
        seg["speaker"] = speaker_map.get(seg["speaker"], "UNKNOWN")

    return transcript_segments


def rename_speakers(segments, speaker_names):
    """Re-map existing SPEAKER_XX labels to custom names (for cached data)."""
    seen = []
    for seg in segments:
        spk = seg.get("speaker", "")
        if spk and spk != "UNKNOWN" and spk not in seen:
            seen.append(spk)

    rename_map = {}
    for i, old_name in enumerate(seen):
        if i < len(speaker_names):
            rename_map[old_name] = speaker_names[i]

    if not rename_map:
        return segments

    for seg in segments:
        spk = seg.get("speaker", "")
        if spk in rename_map:
            seg["speaker"] = rename_map[spk]

    return segments
