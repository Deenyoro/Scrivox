"""Output formatting: timestamps and multi-format transcript output."""

import json
import textwrap


def format_timestamp(seconds, fmt="srt"):
    """Convert seconds to SRT or VTT timestamp format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    if fmt == "vtt":
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_human(seconds):
    """Human-readable timestamp: M:SS or H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _merge_subtitle_segments(segments, max_chars=84, max_duration=4.0, max_gap=0.8):
    """Merge consecutive segments into subtitle-sized blocks.

    Args:
        max_chars: Max characters per subtitle block (two lines of ~42 chars)
        max_duration: Max seconds a single subtitle should stay on screen
        max_gap: Max gap in seconds to allow merging across
    """
    if not segments:
        return []

    merged = []
    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"].strip(),
        "speaker": segments[0].get("speaker", ""),
        "language": segments[0].get("language", ""),
    }

    for seg in segments[1:]:
        same_speaker = seg.get("speaker", "") == current["speaker"]
        gap = seg["start"] - current["end"]
        combined_text = current["text"] + " " + seg["text"].strip()
        combined_duration = seg["end"] - current["start"]

        if (same_speaker and gap <= max_gap
                and len(combined_text) <= max_chars
                and combined_duration <= max_duration):
            current["text"] = combined_text
            current["end"] = seg["end"]
        else:
            merged.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "speaker": seg.get("speaker", ""),
                "language": seg.get("language", ""),
            }

    merged.append(current)
    return merged


def _wrap_subtitle_text(text, max_line=42):
    """Wrap subtitle text to two lines for readability."""
    if len(text) <= max_line:
        return text
    lines = textwrap.wrap(text, width=max_line,
                           break_long_words=False, break_on_hyphens=False)
    return "\n".join(lines[:2])


def format_output(segments, fmt="txt", diarized=False, visual_context=None,
                  summary=None, metadata=None, subtitle_speakers=False):
    """Format transcript segments into the requested output format.

    Args:
        segments: List of transcript segment dicts
        fmt: Output format (txt, md, srt, vtt, json, tsv)
        diarized: Whether segments have speaker labels
        visual_context: Optional list of visual context entries
        summary: Optional meeting summary string
        metadata: Optional metadata dict
        subtitle_speakers: Show speaker labels in SRT/VTT subtitles (default: off)
    """
    lines = []

    if fmt == "txt":
        if summary:
            lines.append(summary)
            lines.append("\n" + "=" * 60)
            lines.append("  FULL TRANSCRIPT")
            lines.append("=" * 60 + "\n")

        vis_idx = 0
        vis = visual_context or []
        current_speaker = None

        for seg in segments:
            while vis_idx < len(vis) and vis[vis_idx]["timestamp"] <= seg["start"]:
                ts = format_timestamp_human(vis[vis_idx]["timestamp"])
                lines.append(f"\n--- [{ts}] SCREEN: {vis[vis_idx]['description']} ---\n")
                vis_idx += 1

            ts = format_timestamp_human(seg["start"])
            if diarized and seg.get("speaker") != current_speaker:
                current_speaker = seg.get("speaker", "")
                lines.append(f"\n[{ts}] [{current_speaker}]")
            elif not diarized:
                lines.append(f"[{ts}] {seg['text']}")
                continue
            lines.append(seg["text"])

        while vis_idx < len(vis):
            ts = format_timestamp_human(vis[vis_idx]["timestamp"])
            lines.append(f"\n--- [{ts}] SCREEN: {vis[vis_idx]['description']} ---\n")
            vis_idx += 1

        return "\n".join(lines).strip()

    elif fmt == "md":
        if metadata:
            duration = metadata.get("duration_seconds")
            duration_str = format_timestamp_human(duration) if duration else "unknown"
            lines.append("# Meeting Transcript")
            lines.append("")
            lines.append(f"- **File:** {metadata.get('input_file', 'unknown')}")
            lines.append(f"- **Duration:** {duration_str}")
            lines.append(f"- **Model:** {metadata.get('model', 'unknown')}")
            lines.append(f"- **Language:** {metadata.get('language', 'unknown')}")
            if diarized:
                speakers = set(seg.get("speaker", "") for seg in segments)
                speakers.discard("")
                speakers.discard("UNKNOWN")
                lines.append(f"- **Speakers:** {', '.join(sorted(speakers)) if speakers else 'unknown'}")
            lines.append("")

        if summary:
            lines.append("---")
            lines.append("")
            lines.append(summary)
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("## Full Transcript")
            lines.append("")

        vis_idx = 0
        vis = visual_context or []
        current_speaker = None

        for seg in segments:
            while vis_idx < len(vis) and vis[vis_idx]["timestamp"] <= seg["start"]:
                ts = format_timestamp_human(vis[vis_idx]["timestamp"])
                lines.append(f"\n> *[{ts}] {vis[vis_idx]['description']}*\n")
                vis_idx += 1

            ts = format_timestamp_human(seg["start"])
            if diarized:
                if seg.get("speaker") != current_speaker:
                    current_speaker = seg.get("speaker", "")
                    lines.append(f"\n**[{ts}] {current_speaker}:**")
                lines.append(f"{seg['text']}")
            else:
                lines.append(f"`{ts}` {seg['text']}")

        while vis_idx < len(vis):
            ts = format_timestamp_human(vis[vis_idx]["timestamp"])
            lines.append(f"\n> *[{ts}] {vis[vis_idx]['description']}*\n")
            vis_idx += 1

        return "\n".join(lines).strip()

    elif fmt == "srt":
        # Sort segments by start time and merge into proper subtitle blocks
        sorted_segs = sorted(segments, key=lambda x: x["start"])
        merged = _merge_subtitle_segments(sorted_segs)
        for i, seg in enumerate(merged, 1):
            start_ts = format_timestamp(seg["start"], "srt")
            end_ts = format_timestamp(seg["end"], "srt")
            if subtitle_speakers and diarized and seg.get("speaker"):
                text = _wrap_subtitle_text(f"[{seg['speaker']}] {seg['text']}")
            else:
                text = _wrap_subtitle_text(seg["text"])
            lines.append(f"{i}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    elif fmt == "vtt":
        lines.append("WEBVTT")
        lines.append("")
        primary_lang = (metadata or {}).get("detected_language", "")
        sorted_segs = sorted(segments, key=lambda x: x["start"])
        merged = _merge_subtitle_segments(sorted_segs)
        for seg in merged:
            start_ts = format_timestamp(seg["start"], "vtt")
            end_ts = format_timestamp(seg["end"], "vtt")
            text = _wrap_subtitle_text(seg["text"])
            seg_lang = seg.get("language", "")
            if seg_lang and seg_lang != primary_lang:
                text = f"<lang {seg_lang}>{text}</lang>"
            if subtitle_speakers and diarized and seg.get("speaker"):
                text = f"<v {seg['speaker']}>{text}</v>"
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    elif fmt == "json":
        output = {}
        if metadata:
            output["metadata"] = metadata
        output["segments"] = segments
        if visual_context:
            output["visual_context"] = visual_context
        if summary:
            output["summary"] = summary
        return json.dumps(output, indent=2, ensure_ascii=False)

    elif fmt == "tsv":
        header = "start\tend"
        if diarized:
            header += "\tspeaker"
        header += "\tlanguage\ttext"
        lines.append(header)
        for seg in segments:
            row = f"{seg['start']:.3f}\t{seg['end']:.3f}"
            if diarized:
                row += f"\t{seg.get('speaker', '')}"
            seg_text = seg['text'].replace('\t', '\\t').replace('\n', ' ')
            row += f"\t{seg.get('language', '')}\t{seg_text}"
            lines.append(row)
        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format: {fmt}")
