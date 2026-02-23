"""Output formatting: timestamps and multi-format transcript output."""

import json


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


def format_output(segments, fmt="txt", diarized=False, visual_context=None,
                  summary=None, metadata=None):
    """Format transcript segments into the requested output format."""
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
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp(seg["start"], "srt")
            end_ts = format_timestamp(seg["end"], "srt")
            speaker_prefix = f"[{seg['speaker']}] " if diarized and seg.get("speaker") else ""
            lines.append(f"{i}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"{speaker_prefix}{seg['text']}")
            lines.append("")
        return "\n".join(lines)

    elif fmt == "vtt":
        lines.append("WEBVTT\n")
        for seg in segments:
            start_ts = format_timestamp(seg["start"], "vtt")
            end_ts = format_timestamp(seg["end"], "vtt")
            speaker_prefix = f"<v {seg['speaker']}>" if diarized and seg.get("speaker") else ""
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"{speaker_prefix}{seg['text']}")
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
        header += "\ttext"
        lines.append(header)
        for seg in segments:
            row = f"{seg['start']:.3f}\t{seg['end']:.3f}"
            if diarized:
                row += f"\t{seg.get('speaker', '')}"
            row += f"\t{seg['text']}"
            lines.append(row)
        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format: {fmt}")
