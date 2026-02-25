"""Pipeline orchestrator: PipelineConfig dataclass + TranscriptionPipeline."""

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from .constants import (
    AUDIO_EXTENSIONS, DEFAULT_DIARIZATION_MODEL, DEFAULT_SUMMARY_MODEL,
    DEFAULT_TRANSLATION_MODEL, DEFAULT_VISION_MODEL,
    LANGUAGE_CODE_TO_NAME, TRANSLATION_CODE_TO_NAME,
    VIDEO_EXTENSIONS,
)
from .media import check_ffmpeg, extract_wav, get_media_duration, has_video_stream
from .transcriber import clean_transcription, transcribe_audio
from .diarizer import assign_speakers, diarize_audio, rename_speakers, _get_bundled_models_dir
from .vision import analyze_keyframes, extract_keyframes
from .summarizer import generate_meeting_summary
from .translator import translate_segments
from .formatter import format_output, format_timestamp_human


@dataclass
class PipelineConfig:
    """All settings needed to run a transcription pipeline."""
    input_path: str
    model: str = "large-v3"
    language: Optional[str] = None

    # Features
    diarize: bool = False
    vision: bool = False
    summarize: bool = False
    translate: bool = False
    translate_to: Optional[str] = None  # target language code e.g. "ar"
    translation_model: str = DEFAULT_TRANSLATION_MODEL

    # Diarization
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    speaker_names: Optional[List[str]] = None

    # Vision
    vision_interval: int = 60
    vision_model: str = DEFAULT_VISION_MODEL
    vision_workers: int = 4

    # Summary
    summary_model: str = DEFAULT_SUMMARY_MODEL

    # Output
    output_format: str = "txt"
    output_path: Optional[str] = None
    subtitle_speakers: bool = False

    # Subtitle tuning
    subtitle_max_chars: int = 84
    subtitle_max_duration: float = 4.0
    subtitle_max_gap: float = 0.8
    subtitle_min_chars: int = 15

    # Post-processing tuning
    confidence_threshold: float = 0.50

    # LLM API
    api_base: Optional[str] = None

    # Credentials
    hf_token: Optional[str] = None
    openrouter_key: Optional[str] = None

    # Multi-track
    audio_track: int = 0  # Audio stream index for multi-track videos

    # Cache
    clear_cache: bool = False


class PipelineError(Exception):
    """Raised when the pipeline encounters a fatal error."""
    pass


class PipelineCancelled(Exception):
    """Raised when the pipeline is cancelled by the user."""
    pass


@dataclass
class PipelineResult:
    """Results from a completed pipeline run."""
    segments: list = field(default_factory=list)
    visual_context: Optional[list] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    output_text: str = ""
    output_path: Optional[str] = None
    translated_segments: Optional[list] = None
    translated_output_text: str = ""
    translated_output_path: Optional[str] = None
    elapsed: float = 0.0


class TranscriptionPipeline:
    """Orchestrates the full transcription pipeline.

    Both CLI and GUI create a PipelineConfig and call pipeline.run().
    The pipeline communicates via on_progress callbacks and never calls print() directly.
    """

    STEPS_BASE = ["Transcribe"]
    STEPS_DIARIZE = ["Diarize"]
    STEPS_VISION = ["Vision Analysis"]
    STEPS_SUMMARY = ["Meeting Summary"]
    STEPS_TRANSLATE = ["Translate"]
    STEPS_OUTPUT = ["Format Output"]

    def __init__(self, config: PipelineConfig, on_progress: Callable = print,
                 on_step: Optional[Callable] = None):
        """
        Args:
            config: Pipeline configuration
            on_progress: Callback for log messages (default: print)
            on_step: Callback for step changes: on_step(step_num, total_steps, step_name)
        """
        self.config = config
        self.on_progress = on_progress
        self.on_step = on_step or (lambda *a: None)
        self._cancel = threading.Event()

    def cancel(self):
        """Request pipeline cancellation. Checked between steps."""
        self._cancel.set()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise PipelineCancelled("Pipeline cancelled by user")

    def _count_steps(self):
        steps = 1  # transcribe
        if self.config.diarize:
            steps += 1
        if self.config.vision:
            steps += 1
        if self.config.summarize:
            steps += 1
        if self.config.translate:
            steps += 1
        steps += 1  # format output
        return steps

    def run(self) -> PipelineResult:
        """Execute the full pipeline. Returns PipelineResult."""
        cfg = self.config
        total_steps = self._count_steps()
        current_step = 0

        # ── Validate ──
        if not os.path.isfile(cfg.input_path):
            raise PipelineError(f"File not found: {cfg.input_path}")

        ext = os.path.splitext(cfg.input_path)[1].lower()
        if ext not in VIDEO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
            self.on_progress(f"Warning: Unrecognized file extension '{ext}'. Attempting transcription anyway.")

        check_ffmpeg(on_progress=self.on_progress)

        if not torch.cuda.is_available():
            raise PipelineError("CUDA GPU not available. This tool requires an NVIDIA GPU.")

        # Resolve credentials
        hf_token = cfg.hf_token or os.environ.get("HF_TOKEN")
        openrouter_key = cfg.openrouter_key or os.environ.get("OPENROUTER_API_KEY")

        if cfg.diarize and not hf_token:
            try:
                from huggingface_hub import HfFolder
                hf_token = HfFolder.get_token()
            except Exception:
                pass
        # Bundled models don't need an HF token
        has_bundled = _get_bundled_models_dir() is not None
        if cfg.diarize and not hf_token and not has_bundled:
            raise PipelineError("Diarization requires HF_TOKEN in .env, config, or huggingface-cli login")

        is_local = cfg.api_base and any(h in cfg.api_base for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if (cfg.vision or cfg.summarize or cfg.translate) and not openrouter_key and not is_local:
            raise PipelineError("Vision/Summary/Translation requires an LLM API key in .env or config")

        # Check video for vision
        is_video = has_video_stream(cfg.input_path)
        if cfg.vision and not is_video:
            self.on_progress("Warning: --vision requires a video file, skipping keyframe extraction")
            cfg.vision = False

        # ── Banner ──
        self.on_progress("=" * 60)
        self.on_progress("  SCRIVOX TRANSCRIPTION")
        if cfg.diarize:
            self.on_progress("  + SPEAKER DIARIZATION")
        if cfg.vision:
            self.on_progress("  + VISUAL CONTEXT (keyframe analysis)")
        if cfg.summarize:
            self.on_progress("  + MEETING SUMMARY")
        if cfg.translate and cfg.translate_to:
            target_name = TRANSLATION_CODE_TO_NAME.get(cfg.translate_to, cfg.translate_to)
            self.on_progress(f"  + TRANSLATION ({target_name})")
        self.on_progress("=" * 60)
        self.on_progress(f"  Input:    {cfg.input_path}")
        self.on_progress(f"  Model:    {cfg.model}")
        self.on_progress(f"  Language: {cfg.language or 'auto-detect'}")
        diarize_line = f"  Diarize:  {cfg.diarize}"
        if cfg.diarize and cfg.speaker_names:
            diarize_line += f" (speakers: {', '.join(cfg.speaker_names)})"
        self.on_progress(diarize_line)
        vision_line = f"  Vision:   {cfg.vision}"
        if cfg.vision:
            vision_line += f" (every {cfg.vision_interval}s, model: {cfg.vision_model})"
        self.on_progress(vision_line)
        if cfg.summarize:
            self.on_progress(f"  Summary:  True (model: {cfg.summary_model})")
        self.on_progress(f"  Format:   {cfg.output_format}")
        self.on_progress(f"  GPU:      {torch.cuda.get_device_name(0)}")
        self.on_progress("=" * 60)
        self.on_progress("")

        total_t0 = time.time()

        # ── Cache ──
        cache_path = cfg.input_path + ".whisper_cache.json"
        if cfg.clear_cache and os.path.exists(cache_path):
            os.remove(cache_path)
            self.on_progress("Cleared cached results.")

        # Step 1: Transcribe
        current_step += 1
        self.on_step(current_step, total_steps, "Transcribing")
        self._check_cancel()

        cache_hit = False
        segments = []
        detected_lang = cfg.language
        if os.path.exists(cache_path) and not cfg.clear_cache:
            self.on_progress("Loading cached transcription + diarization...")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                segments = cache["segments"]
                cached_model = cache.get("model")
                cached_language = cache.get("language")
                cached_diarized = cache.get("diarized", False)
                cached_diarize_params = cache.get("diarize_params", {})

                current_diarize_params = {
                    "num_speakers": cfg.num_speakers,
                    "min_speakers": cfg.min_speakers,
                    "max_speakers": cfg.max_speakers,
                }

                if cached_model and cached_model != cfg.model:
                    self.on_progress(f"  Cache used model '{cached_model}', now using '{cfg.model}' \u2014 re-transcribing.")
                elif cached_language and cfg.language and cached_language != cfg.language:
                    self.on_progress(f"  Cache used language '{cached_language}', now using '{cfg.language}' \u2014 re-transcribing.")
                elif cfg.diarize and not cached_diarized:
                    self.on_progress("  Cache was not diarized, but diarization requested \u2014 re-transcribing.")
                elif cfg.diarize and cached_diarized and cached_diarize_params != current_diarize_params:
                    self.on_progress("  Diarization parameters changed \u2014 re-transcribing.")
                else:
                    self.on_progress(f"Loaded {len(segments)} segments from cache")
                    detected_lang = cache.get("detected_language", cfg.language)
                    cache_hit = True
            except (json.JSONDecodeError, KeyError) as e:
                self.on_progress(f"Warning: Cache file corrupt ({e}), re-transcribing.")

        if not cache_hit:
            # Pre-extract audio for non-default tracks
            transcribe_path = cfg.input_path
            _pre_extracted_wav = None
            if cfg.audio_track != 0:
                _pre_extracted_wav = extract_wav(
                    cfg.input_path, track_index=cfg.audio_track,
                    on_progress=self.on_progress,
                )
                transcribe_path = _pre_extracted_wav

            segments, info = transcribe_audio(
                transcribe_path, cfg.model, cfg.language,
                on_progress=self.on_progress,
            )
            detected_lang = cfg.language or info.language

            before_count = len(segments)
            segments = clean_transcription(segments, language=detected_lang,
                                           confidence_threshold=cfg.confidence_threshold)
            removed = before_count - len(segments)
            if removed > 0:
                self.on_progress(f"Post-processing: removed {removed} hallucinated/non-speech segments")

            self._check_cancel()

            # Step 2: Diarize
            if cfg.diarize:
                current_step += 1
                self.on_step(current_step, total_steps, "Diarizing")
                self.on_progress("")
                diarize_path = _pre_extracted_wav or cfg.input_path
                speaker_segments = diarize_audio(
                    diarize_path, hf_token,
                    num_speakers=cfg.num_speakers,
                    min_speakers=cfg.min_speakers,
                    max_speakers=cfg.max_speakers,
                    diarization_model=cfg.diarization_model,
                    audio_track=cfg.audio_track,
                    on_progress=self.on_progress,
                )
                segments = assign_speakers(segments, speaker_segments, cfg.speaker_names)

            # Save cache
            cache_data = {
                "segments": segments,
                "model": cfg.model,
                "language": cfg.language,
                "detected_language": detected_lang,
                "diarized": cfg.diarize,
                "diarize_params": {
                    "num_speakers": cfg.num_speakers,
                    "min_speakers": cfg.min_speakers,
                    "max_speakers": cfg.max_speakers,
                },
            }
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False)
                self.on_progress(f"Cached transcription to {cache_path}")
            except OSError as e:
                self.on_progress(f"Warning: Could not save cache: {e}")
            finally:
                # Clean up pre-extracted WAV
                if _pre_extracted_wav and os.path.exists(_pre_extracted_wav):
                    os.remove(_pre_extracted_wav)
        else:
            # Advance step counter for skipped diarize
            if cfg.diarize:
                current_step += 1

        # Apply speaker names to cached data
        if cache_hit and cfg.speaker_names:
            segments = rename_speakers(segments, cfg.speaker_names)

        self._check_cancel()

        # Step 3: Vision
        visual_context = None
        tmpdir = None
        try:
            if cfg.vision:
                current_step += 1
                self.on_step(current_step, total_steps, "Analyzing keyframes")
                self.on_progress("")
                keyframes, tmpdir = extract_keyframes(
                    cfg.input_path,
                    interval_secs=cfg.vision_interval,
                    on_progress=self.on_progress,
                )
                if keyframes:
                    self._check_cancel()
                    visual_context = analyze_keyframes(
                        keyframes, openrouter_key, cfg.vision_model,
                        cfg.vision_workers, api_base=cfg.api_base,
                        on_progress=self.on_progress,
                    )

            self._check_cancel()

            # Step 4: Meeting Summary
            summary = None
            if cfg.summarize:
                current_step += 1
                self.on_step(current_step, total_steps, "Generating summary")
                self.on_progress("")
                summary = generate_meeting_summary(
                    segments, openrouter_key, cfg.summary_model,
                    diarized=cfg.diarize, visual_context=visual_context,
                    api_base=cfg.api_base,
                    on_progress=self.on_progress,
                )
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Step 5: Translate
        translated_segments = None
        if cfg.translate and cfg.translate_to:
            self._check_cancel()
            current_step += 1
            target_name = TRANSLATION_CODE_TO_NAME.get(cfg.translate_to, cfg.translate_to)
            self.on_step(current_step, total_steps, f"Translating to {target_name}")
            self.on_progress("")

            source_name = LANGUAGE_CODE_TO_NAME.get(detected_lang, detected_lang) if detected_lang else None
            translated_segments = translate_segments(
                segments, target_name, openrouter_key, cfg.translation_model,
                source_language=source_name, api_base=cfg.api_base,
                on_progress=self.on_progress, cancel_event=self._cancel,
            )

        self._check_cancel()

        # Step 6: Format output
        current_step += 1
        self.on_step(current_step, total_steps, "Formatting output")

        duration = get_media_duration(cfg.input_path)
        metadata = {
            "input_file": os.path.basename(cfg.input_path),
            "model": cfg.model,
            "language": cfg.language or "auto-detect",
            "detected_language": detected_lang,
            "diarized": cfg.diarize,
            "vision": cfg.vision,
            "summarized": cfg.summarize,
            "segments_count": len(segments),
            "duration_seconds": duration,
            "gpu": torch.cuda.get_device_name(0),
        }

        output_text = format_output(
            segments, cfg.output_format,
            diarized=cfg.diarize,
            visual_context=visual_context,
            summary=summary,
            metadata=metadata,
            subtitle_speakers=cfg.subtitle_speakers,
            subtitle_max_chars=cfg.subtitle_max_chars,
            subtitle_max_duration=cfg.subtitle_max_duration,
            subtitle_max_gap=cfg.subtitle_max_gap,
            subtitle_min_chars=cfg.subtitle_min_chars,
        )

        # Format translated output (if translation was done)
        translated_output_text = ""
        if translated_segments:
            translated_output_text = format_output(
                translated_segments, cfg.output_format,
                diarized=cfg.diarize,
                visual_context=visual_context,
                summary=summary,
                metadata=metadata,
                subtitle_speakers=cfg.subtitle_speakers,
                subtitle_max_chars=cfg.subtitle_max_chars,
                subtitle_max_duration=cfg.subtitle_max_duration,
                subtitle_max_gap=cfg.subtitle_max_gap,
                subtitle_min_chars=cfg.subtitle_min_chars,
            )

        total_elapsed = time.time() - total_t0
        self.on_progress(f"\nTotal time: {total_elapsed:.1f}s")
        self.on_progress(f"Segments: {len(segments)}")
        if visual_context:
            self.on_progress(f"Keyframes analyzed: {len(visual_context)}")
        if summary:
            self.on_progress("Meeting summary: generated")
        if translated_segments:
            self.on_progress(f"Translation: {TRANSLATION_CODE_TO_NAME.get(cfg.translate_to, cfg.translate_to)}")

        # Determine output path
        output_path = cfg.output_path
        if not output_path:
            base = os.path.splitext(cfg.input_path)[0]
            if cfg.output_format != "txt":
                ext_map = {"md": ".md", "srt": ".srt", "vtt": ".vtt", "json": ".json", "tsv": ".tsv"}
                output_path = base + ext_map[cfg.output_format]
            elif cfg.diarize or cfg.vision or cfg.summarize or cfg.translate:
                output_path = base + "_transcript.txt"
            if output_path:
                self.on_progress(f"Auto-saving to: {output_path}")

        # Write original output
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output_text)
                self.on_progress(f"Saved to: {output_path}")
            except OSError as e:
                self.on_progress(f"Error: Could not write output file: {e}")

        # Write translated output (e.g. file.ar.srt)
        translated_output_path = None
        if translated_segments and output_path:
            base_no_ext, ext = os.path.splitext(output_path)
            translated_output_path = f"{base_no_ext}.{cfg.translate_to}{ext}"
            try:
                with open(translated_output_path, "w", encoding="utf-8") as f:
                    f.write(translated_output_text)
                self.on_progress(f"Saved translation to: {translated_output_path}")
            except OSError as e:
                self.on_progress(f"Error: Could not write translated output: {e}")
                translated_output_path = None

        return PipelineResult(
            segments=segments,
            visual_context=visual_context,
            summary=summary,
            metadata=metadata,
            output_text=output_text,
            output_path=output_path,
            translated_segments=translated_segments,
            translated_output_text=translated_output_text,
            translated_output_path=translated_output_path,
            elapsed=total_elapsed,
        )
