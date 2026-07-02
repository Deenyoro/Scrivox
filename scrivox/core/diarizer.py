"""Speaker diarization using pyannote.audio with bundled model support."""

import os
import sys
import threading
import time

import torch

from .torch_compat import _allow_unsafe_torch_load
from .media import extract_wav


def _get_bundled_models_dir():
    """Check for bundled diarization models next to the exe or project root.

    Only counts a 'models' directory that actually contains HuggingFace hub
    models (models/hub/models--*). A bare 'models/whisper/' directory for
    custom Whisper models must NOT trigger the bundled-diarization path.

    Returns the path if found, or None.
    """
    if getattr(sys, "frozen", False):
        # PyInstaller exe — check next to the exe
        base = os.path.dirname(sys.executable)
    else:
        # Dev mode — check project root
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    models_dir = os.path.join(base, "models")
    hub_dir = os.path.join(models_dir, "hub")
    try:
        if os.path.isdir(hub_dir) and any(
            e.startswith("models--") for e in os.listdir(hub_dir)
        ):
            return models_dir
    except OSError:
        pass
    return None


_HF_ENV_KEYS = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE")


def _apply_bundled_hf_env(models_dir):
    """Point HF Hub at bundled models and force offline mode.

    Returns the previous env values so the caller can restore them after
    diarization — a permanent HF_HUB_OFFLINE=1 would break later Whisper
    model downloads in a long-lived GUI process. Applied for the whole
    diarization run (load AND inference), since pyannote components may
    defer hub/cache resolution past construction.
    """
    saved = {k: os.environ.get(k) for k in _HF_ENV_KEYS}
    os.environ["HF_HOME"] = models_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(models_dir, "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    return saved


def _restore_hf_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _resolve_snapshot(model_id, hub_dir):
    """Resolve a HuggingFace model ID to its local snapshot directory.

    Reads the refs/main file to get the commit hash, then returns the
    full path to the snapshot directory containing the model files.
    """
    if "/" not in model_id:
        raise ValueError(f"Invalid model ID (expected 'org/name'): {model_id}")
    org, name = model_id.split("/", 1)
    model_dir = os.path.join(hub_dir, f"models--{org}--{name.replace('/', '--')}")
    refs_file = os.path.join(model_dir, "refs", "main")
    if not os.path.isfile(refs_file):
        raise FileNotFoundError(
            f"Bundled model '{model_id}' not found in {hub_dir}\n"
            f"Expected refs file: {refs_file}")
    with open(refs_file, encoding="utf-8") as f:
        commit_hash = f.read().strip()
    return os.path.join(model_dir, "snapshots", commit_hash)


def _load_bundled_pipeline(diarization_model, hub_dir, on_progress=print):
    """Load a pyannote pipeline entirely from local files, bypassing HF Hub.

    pyannote 4.0's community-1 model is self-contained: segmentation, embedding,
    and PLDA are bundled as subfolders. Pipeline.from_pretrained() natively handles
    local directories — it reads config.yaml and expands $model/subfolder paths
    to local subfolder paths automatically. No config rewriting needed.
    """
    from pyannote.audio import Pipeline

    pipeline_dir = _resolve_snapshot(diarization_model, hub_dir)
    config_path = os.path.join(pipeline_dir, "config.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Bundled model config not found: {config_path}\n"
            f"Expected bundled model: {diarization_model}")

    on_progress(f"  Loading bundled pipeline: {diarization_model}")
    with _allow_unsafe_torch_load():
        pipeline = Pipeline.from_pretrained(pipeline_dir)

    return pipeline


def diarize_audio(audio_path, hf_token, num_speakers=None, min_speakers=None,
                  max_speakers=None, diarization_model=None, audio_track=0,
                  on_progress=print):
    """Run speaker diarization on audio. Returns list of speaker segments.

    If bundled models are found in a 'models/' directory next to the exe,
    they are loaded directly from disk — no HF token or network needed.
    """
    from .constants import DEFAULT_DIARIZATION_MODEL

    if not diarization_model:
        diarization_model = DEFAULT_DIARIZATION_MODEL

    # Auto-upgrade saved configs from pyannote 3.x model ID
    if diarization_model == "pyannote/speaker-diarization-3.1":
        diarization_model = DEFAULT_DIARIZATION_MODEL
        on_progress("Upgraded diarization model from 3.1 to community-1 (pyannote 4.0)")

    bundled_models_dir = _get_bundled_models_dir()
    has_bundled = bundled_models_dir is not None
    _saved_hf_env = _apply_bundled_hf_env(bundled_models_dir) if has_bundled else None

    wav_path = None
    ext = os.path.splitext(audio_path)[1].lower()
    try:
        if ext not in (".wav", ".wave"):
            wav_path = extract_wav(audio_path, track_index=audio_track,
                                   on_progress=on_progress)
            wav_file = wav_path
        else:
            wav_file = audio_path

        # Pre-load audio as a waveform tensor so pyannote never touches
        # torchcodec's AudioDecoder (broken on Windows / PyInstaller).
        # Use soundfile instead of torchaudio — torchaudio 2.10+ delegates to
        # torchcodec internally, which fails in bundled builds.
        import soundfile as sf
        data, sample_rate = sf.read(wav_file, dtype="float32")
        if data.ndim > 1:
            # User-supplied multi-channel WAV (extracted WAVs are already mono):
            # downmix so the tensor is (1, samples) as pyannote expects.
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data).unsqueeze(0)  # (samples,) -> (1, samples)
        diarize_input = {"waveform": waveform, "sample_rate": sample_rate}

        if has_bundled:
            on_progress("Using bundled diarization models...")
        else:
            on_progress("Downloading diarization models (first run only)...")

        on_progress(f"Loading {diarization_model} on CUDA...")
        if has_bundled:
            # Bundled models — resolve local paths and load directly from disk.
            # This bypasses HuggingFace Hub entirely: no cache lookups, no
            # token checks, no network access.
            bundled_hub = os.path.join(bundled_models_dir, "hub")
            pipeline = _load_bundled_pipeline(
                diarization_model, bundled_hub, on_progress=on_progress,
            )
        else:
            # Download from HF Hub — needs token
            from pyannote.audio import Pipeline
            with _allow_unsafe_torch_load():
                pipeline = Pipeline.from_pretrained(
                    diarization_model, token=hf_token,
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
            result = pipeline(
                diarize_input,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        finally:
            stop_progress.set()
            ticker.join(timeout=5)

        # pyannote 4.0 returns DiarizeOutput; use exclusive_speaker_diarization
        # (no overlapping speech) for cleaner transcription alignment.
        # Fall back to the result itself for pyannote 3.x Annotation objects.
        if hasattr(result, "exclusive_speaker_diarization"):
            annotation = result.exclusive_speaker_diarization
        elif hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        else:
            annotation = result

        speaker_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
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
        if _saved_hf_env is not None:
            _restore_hf_env(_saved_hf_env)
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
