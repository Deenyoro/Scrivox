"""CLI entry point: argparse -> PipelineConfig -> pipeline.run()."""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
if getattr(sys, 'frozen', False):
    _dotenv_base = os.path.dirname(sys.executable)
else:
    _dotenv_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_dotenv_base, ".env"))

from .core.constants import (
    DEFAULT_DIARIZATION_MODEL, DEFAULT_TRANSLATION_MODEL, DEFAULT_VISION_MODEL,
    DEFAULT_SUMMARY_MODEL, OUTPUT_FORMATS,
)
from .core.features import has_diarization, has_advanced_features, get_variant_name
from .core.pipeline import PipelineConfig, PipelineError, TranscriptionPipeline


def build_parser():
    """Build the argparse parser for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Scrivox - GPU Transcription + Diarization + Vision + Summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py meeting.mp3
  python main.py meeting.mp4 --diarize
  python main.py meeting.mp4 --all --format md -o minutes.md
  python main.py video.mp4 --diarize --format srt -o subtitles.srt
  python main.py video.mp4 --diarize --speaker-names "Alice,Bob"
  python main.py meeting.mp3 --translate-to ar,fr,ja --format srt
  python main.py meeting.mp4 --all --translate-to fr --translate-all
  python main.py video.mp4 --list-tracks
  python main.py video.mp4 --audio-track 1 --language ja
        """,
    )
    parser.add_argument("input", help="Audio or video file to transcribe")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model: tiny, base, small, medium, large-v3 (default: large-v3)")
    parser.add_argument("--language", default=None,
                        help="Primary language code e.g. 'ko', 'ja', 'en' (default: auto-detect). "
                             "Sets the main language; other languages in mixed content are still detected.")
    parser.add_argument("--all", action="store_true",
                        help="Enable all features: diarize + vision + summarize")

    # Diarization options
    diarize_group = parser.add_argument_group("Speaker Diarization")
    diarize_group.add_argument("--diarize", action="store_true",
                               help="Enable speaker diarization (requires HF_TOKEN)")
    diarize_group.add_argument("--num-speakers", type=int, default=None,
                               help="Exact number of speakers (if known)")
    diarize_group.add_argument("--min-speakers", type=int, default=None,
                               help="Minimum number of speakers expected")
    diarize_group.add_argument("--max-speakers", type=int, default=None,
                               help="Maximum number of speakers expected")
    diarize_group.add_argument("--speaker-names", default=None,
                               help="Comma-separated speaker names, e.g. 'Alice,Bob,Charlie'")
    diarize_group.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL,
                               help=f"Diarization model ID or local path (default: {DEFAULT_DIARIZATION_MODEL})")

    # Vision options
    vision_group = parser.add_argument_group("Vision Analysis")
    vision_group.add_argument("--vision", action="store_true",
                              help="Extract keyframes and describe with vision LLM (video only)")
    vision_group.add_argument("--vision-interval", type=int, default=60,
                              help="Seconds between keyframe captures (default: 60)")
    vision_group.add_argument("--vision-model", default=DEFAULT_VISION_MODEL,
                              help=f"Vision LLM model (default: {DEFAULT_VISION_MODEL})")
    vision_group.add_argument("--vision-workers", type=int, default=4,
                              help="Concurrent vision API requests (default: 4)")

    # Summary options
    summary_group = parser.add_argument_group("Meeting Summary")
    summary_group.add_argument("--summarize", action="store_true",
                               help="Generate meeting summary with key points and action items")
    summary_group.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL,
                               help=f"LLM model for summary (default: {DEFAULT_SUMMARY_MODEL})")

    # Translation options
    translate_group = parser.add_argument_group("Translation")
    translate_group.add_argument("--translate-to", default=None,
                                 help="Translate to target language(s). Comma-separated for multiple: "
                                      "'ar' or 'ar,fr,ja'. Produces additional output files per target language.")
    translate_group.add_argument("--translate-all", action="store_true",
                                 help="Also translate summary, vision descriptions, and document headers "
                                      "(not just transcript segments)")
    translate_group.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL,
                                 help=f"LLM model for translation (default: {DEFAULT_TRANSLATION_MODEL})")

    # Multi-track options
    track_group = parser.add_argument_group("Audio Tracks")
    track_group.add_argument("--audio-track", type=int, default=0,
                             help="Audio stream index for multi-track videos (default: 0)")
    track_group.add_argument("--list-tracks", action="store_true",
                             help="List audio tracks in the file and exit")

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", default=None,
                              help="Output file path (default: print to console)")
    output_group.add_argument("--format", "-f", default="txt",
                              choices=OUTPUT_FORMATS,
                              help="Output format (default: txt)")
    output_group.add_argument("--subtitle-speakers", action="store_true",
                              help="Include speaker labels in SRT/VTT subtitles (off by default)")
    output_group.add_argument("--subtitle-max-chars", type=int, default=84,
                              help="Max characters per subtitle cue (default: 84)")
    output_group.add_argument("--subtitle-max-duration", type=float, default=4.0,
                              help="Max seconds per subtitle cue (default: 4.0)")
    output_group.add_argument("--subtitle-max-gap", type=float, default=0.8,
                              help="Max gap in seconds to merge across (default: 0.8)")
    output_group.add_argument("--subtitle-min-chars", type=int, default=15,
                              help="Min characters per cue when splitting (default: 15)")
    output_group.add_argument("--confidence-threshold", type=float, default=0.50,
                              help="Min avg word probability to keep a segment (default: 0.50)")

    # Cache / credential options
    parser.add_argument("--clear-cache", action="store_true",
                        help="Force re-transcription, ignoring cached results")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (overrides .env / cached login)")
    parser.add_argument("--api-key", default=None,
                        help="LLM API key for vision/summary/translation (overrides .env)")
    parser.add_argument("--openrouter-key", default=None,
                        help="Alias for --api-key (backward compat)")
    parser.add_argument("--anthropic-key", default=None,
                        help="Anthropic API key (uses Anthropic Messages API). "
                             "Overrides --api-key and sets API base to Anthropic.")
    parser.add_argument("--api-base", default=None,
                        help="LLM API base URL (default: OpenRouter). "
                             "Examples: https://api.openai.com/v1/chat/completions, "
                             "http://localhost:11434/v1/chat/completions, "
                             "https://api.anthropic.com/v1/messages")

    return parser


def run_cli(argv=None):
    """Parse CLI args and run the pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle --list-tracks
    if args.list_tracks:
        from .core.media import list_audio_tracks
        if not os.path.isfile(args.input):
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        tracks = list_audio_tracks(args.input)
        if not tracks:
            print("No audio tracks found.")
        else:
            for t in tracks:
                parts = [f"Track {t['index']}: {t['codec'].upper()}"]
                if t.get("language"):
                    parts.append(t["language"].capitalize())
                ch = t.get("channels", 0)
                if ch == 1:
                    parts.append("mono")
                elif ch == 2:
                    parts.append("stereo")
                elif ch == 6:
                    parts.append("5.1")
                elif ch > 0:
                    parts.append(f"{ch}ch")
                if t.get("title"):
                    parts.append(t["title"])
                if t.get("is_default"):
                    parts.append("(default)")
                print("  " + ", ".join(parts))
        sys.exit(0)

    # Expand --all with feature availability checks
    if args.all:
        args.diarize = has_diarization()
        args.vision = has_advanced_features()
        args.summarize = has_advanced_features()

        if not has_diarization():
            variant = get_variant_name()
            print(f"Notice: --all in {variant} build enables transcription only "
                  f"(diarization/vision/summary not available).", file=sys.stderr)

    # Feature availability checks for explicit flags
    if args.diarize and not has_diarization():
        print(f"Error: --diarize is not available in the {get_variant_name()} build. "
              "Use the Regular or Full build for diarization.", file=sys.stderr)
        sys.exit(1)
    if args.vision and not has_advanced_features():
        print(f"Error: --vision is not available in the {get_variant_name()} build. "
              "Use the Regular or Full build for vision analysis.", file=sys.stderr)
        sys.exit(1)
    if args.summarize and not has_advanced_features():
        print(f"Error: --summarize is not available in the {get_variant_name()} build. "
              "Use the Regular or Full build for meeting summaries.", file=sys.stderr)
        sys.exit(1)
    if args.translate_to and not has_advanced_features():
        print(f"Error: --translate-to is not available in the {get_variant_name()} build. "
              "Use the Regular or Full build for translation.", file=sys.stderr)
        sys.exit(1)

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate speaker count args
    for arg_name, arg_val in [("--num-speakers", args.num_speakers),
                               ("--min-speakers", args.min_speakers),
                               ("--max-speakers", args.max_speakers)]:
        if arg_val is not None and arg_val < 1:
            print(f"Error: {arg_name} must be at least 1", file=sys.stderr)
            sys.exit(1)
    if args.min_speakers is not None and args.max_speakers is not None:
        if args.min_speakers > args.max_speakers:
            print("Error: --min-speakers cannot be greater than --max-speakers", file=sys.stderr)
            sys.exit(1)
    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        print("Warning: --num-speakers overrides --min-speakers/--max-speakers", file=sys.stderr)

    if args.vision_interval < 1:
        print("Error: --vision-interval must be at least 1 second", file=sys.stderr)
        sys.exit(1)

    # Parse speaker names
    speaker_names = None
    if args.speaker_names:
        speaker_names = [name.strip() for name in args.speaker_names.split(",") if name.strip()]

    # Build config — resolve API key and base URL
    api_key = args.api_key or args.openrouter_key
    api_base = args.api_base

    # --anthropic-key overrides: use Anthropic's Messages API
    if args.anthropic_key:
        api_key = args.anthropic_key
        if not api_base:
            from .core.constants import LLM_PROVIDERS
            api_base = LLM_PROVIDERS["Anthropic"]

    config = PipelineConfig(
        input_path=args.input,
        model=args.model,
        language=args.language,
        diarize=args.diarize,
        vision=args.vision,
        summarize=args.summarize,
        diarization_model=args.diarization_model,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        speaker_names=speaker_names,
        vision_interval=args.vision_interval,
        vision_model=args.vision_model,
        vision_workers=args.vision_workers,
        summary_model=args.summary_model,
        translate=bool(args.translate_to),
        translate_all=args.translate_all,
        translate_to=args.translate_to,
        translation_model=args.translation_model,
        output_format=args.format,
        output_path=args.output,
        subtitle_speakers=args.subtitle_speakers,
        subtitle_max_chars=args.subtitle_max_chars,
        subtitle_max_duration=args.subtitle_max_duration,
        subtitle_max_gap=args.subtitle_max_gap,
        subtitle_min_chars=args.subtitle_min_chars,
        confidence_threshold=args.confidence_threshold,
        api_base=api_base,
        hf_token=args.hf_token,
        openrouter_key=api_key,
        audio_track=args.audio_track,
        clear_cache=args.clear_cache,
    )

    # Run pipeline
    try:
        pipeline = TranscriptionPipeline(config, on_progress=print)
        result = pipeline.run()
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print translated output paths
    for tr in result.translated_outputs:
        if tr.get("output_path"):
            print(f"Translation ({tr['lang_name']}) saved to: {tr['output_path']}")

    # Print to console if no output file
    if not result.output_path:
        print("\n" + "=" * 60)
        print("  TRANSCRIPT")
        print("=" * 60)
        print(result.output_text)
        for tr in result.translated_outputs:
            if tr.get("output_text"):
                print("\n" + "=" * 60)
                print(f"  TRANSLATION ({tr['lang_name']})")
                print("=" * 60)
                print(tr["output_text"])
        print()
