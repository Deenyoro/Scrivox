"""CLI entry point: argparse -> PipelineConfig -> pipeline.run()."""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from .core.constants import DEFAULT_VISION_MODEL, DEFAULT_SUMMARY_MODEL, OUTPUT_FORMATS
from .core.pipeline import PipelineConfig, PipelineError, TranscriptionPipeline


def build_parser():
    """Build the argparse parser for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Scrivox - GPU Transcription + Diarization + Vision + Summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py meeting.mp3
  python main.py meeting.mp4 --diarize
  python main.py meeting.mp4 --all
  python main.py meeting.mp4 --diarize --vision --summarize
  python main.py video.mp4 --diarize --format srt -o subtitles.srt
  python main.py video.mp4 --diarize --speaker-names "Alice,Bob"
  python main.py meeting.mp4 --all --format md -o minutes.md
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

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", default=None,
                              help="Output file path (default: print to console)")
    output_group.add_argument("--format", "-f", default="txt",
                              choices=OUTPUT_FORMATS,
                              help="Output format (default: txt)")
    output_group.add_argument("--subtitle-speakers", action="store_true",
                              help="Include speaker labels in SRT/VTT subtitles (off by default)")

    # Cache / credential options
    parser.add_argument("--clear-cache", action="store_true",
                        help="Force re-transcription, ignoring cached results")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (overrides .env / cached login)")
    parser.add_argument("--openrouter-key", default=None,
                        help="OpenRouter API key (overrides .env)")

    return parser


def run_cli(argv=None):
    """Parse CLI args and run the pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Expand --all
    if args.all:
        args.diarize = True
        args.vision = True
        args.summarize = True

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

    # Build config
    config = PipelineConfig(
        input_path=args.input,
        model=args.model,
        language=args.language,
        diarize=args.diarize,
        vision=args.vision,
        summarize=args.summarize,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        speaker_names=speaker_names,
        vision_interval=args.vision_interval,
        vision_model=args.vision_model,
        vision_workers=args.vision_workers,
        summary_model=args.summary_model,
        output_format=args.format,
        output_path=args.output,
        subtitle_speakers=args.subtitle_speakers,
        hf_token=args.hf_token,
        openrouter_key=args.openrouter_key,
        clear_cache=args.clear_cache,
    )

    # Run pipeline
    try:
        pipeline = TranscriptionPipeline(config, on_progress=print)
        result = pipeline.run()
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print to console if no output file
    if not result.output_path:
        print("\n" + "=" * 60)
        print("  TRANSCRIPT")
        print("=" * 60)
        print(result.output_text)
        print()
