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


def _split_long_segments(segments, max_chars=84, max_duration=4.0, min_chars=15):
    """Split segments that exceed subtitle limits using word-level timestamps.

    If a segment is too long (duration or character count), split at the best
    punctuation boundary.  Falls back to proportional splitting when no word
    timestamps are available.
    """
    result = []
    for seg in segments:
        text = seg["text"].strip()
        duration = seg["end"] - seg["start"]
        words = seg.get("words", [])

        if len(text) <= max_chars and duration <= max_duration:
            result.append(seg)
            continue

        if not words:
            # No word timestamps — split proportionally at sentence boundaries
            sentences = _split_at_punctuation(text)
            if len(sentences) <= 1:
                result.append(seg)
                continue
            total_chars = sum(len(s) for s in sentences)
            pos = seg["start"]
            for sent in sentences:
                frac = len(sent) / total_chars if total_chars > 0 else 1.0 / len(sentences)
                sent_dur = duration * frac
                new_seg = dict(seg)
                new_seg["start"] = pos
                new_seg["end"] = pos + sent_dur
                new_seg["text"] = sent
                new_seg["words"] = []
                result.append(new_seg)
                pos += sent_dur
            continue

        # Use word timestamps to find the best split point
        _split_segment_by_words(seg, words, max_chars, max_duration, result,
                                min_chars=min_chars)

    return result


def _split_at_punctuation(text):
    """Split text at sentence-ending punctuation, returning non-empty parts."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _split_segment_by_words(seg, words, max_chars, max_duration, result,
                            min_chars=15):
    """Recursively split a segment using word timestamps at the best break point."""
    text = seg["text"].strip()
    duration = seg["end"] - seg["start"]

    if len(text) <= max_chars and duration <= max_duration:
        result.append(seg)
        return

    if len(words) < 2:
        result.append(seg)
        return

    # Find the best split point — prefer punctuation boundaries near the middle
    mid = len(words) // 2
    best_idx = None
    best_score = -999

    for i in range(1, len(words)):
        # Check that both halves would meet minimum size
        left_text = "".join(w["word"] for w in words[:i]).strip()
        right_text = "".join(w["word"] for w in words[i:]).strip()
        if len(left_text) < min_chars or len(right_text) < min_chars:
            continue

        word_text = words[i - 1]["word"].rstrip()
        # Score: prefer sentence-ending punctuation, then commas, then midpoint
        score = 0
        if word_text and word_text[-1] in ".!?":
            score = 100
        elif word_text and word_text[-1] in ",;:":
            score = 50
        # Penalize distance from midpoint
        score -= abs(i - mid) * 2
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        # No valid split point that satisfies min_chars — keep as-is
        result.append(seg)
        return

    # Split into two sub-segments
    left_words = words[:best_idx]
    right_words = words[best_idx:]

    left_text = "".join(w["word"] for w in left_words).strip()
    right_text = "".join(w["word"] for w in right_words).strip()

    left_seg = dict(seg)
    left_seg["start"] = left_words[0]["start"]
    left_seg["end"] = left_words[-1]["end"]
    left_seg["text"] = left_text
    left_seg["words"] = left_words

    right_seg = dict(seg)
    right_seg["start"] = right_words[0]["start"]
    right_seg["end"] = right_words[-1]["end"]
    right_seg["text"] = right_text
    right_seg["words"] = right_words

    # Recurse in case sub-segments are still too long
    _split_segment_by_words(left_seg, left_words, max_chars, max_duration, result)
    _split_segment_by_words(right_seg, right_words, max_chars, max_duration, result)


def _cap_subtitle_duration(segments, max_duration=4.0, min_duration=1.0,
                           chars_per_second=15.0):
    """Cap subtitle display duration based on text length and max_duration.

    Prevents short lines from staying on screen for tens of seconds during
    music/credits/silence.  Display time is proportional to text length
    (at chars_per_second reading speed) but clamped to [min_duration, max_duration].
    """
    result = []
    for seg in segments:
        seg = dict(seg)
        duration = seg["end"] - seg["start"]
        if duration > max_duration:
            text_len = len(seg["text"].strip())
            ideal = max(min_duration, text_len / chars_per_second)
            seg["end"] = seg["start"] + min(ideal, max_duration)
        result.append(seg)
    return result


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
                  summary=None, metadata=None, subtitle_speakers=False,
                  subtitle_max_chars=84, subtitle_max_duration=4.0,
                  subtitle_max_gap=0.8, subtitle_min_chars=15):
    """Format transcript segments into the requested output format.

    Args:
        segments: List of transcript segment dicts
        fmt: Output format (txt, md, srt, vtt, json, tsv)
        diarized: Whether segments have speaker labels
        visual_context: Optional list of visual context entries
        summary: Optional meeting summary string
        metadata: Optional metadata dict
        subtitle_speakers: Show speaker labels in SRT/VTT subtitles (default: off)
        subtitle_max_chars: Max characters per subtitle cue
        subtitle_max_duration: Max seconds per subtitle cue
        subtitle_max_gap: Max gap in seconds to merge across
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
        # Sort → split long segments → cap durations → merge short ones
        sorted_segs = sorted(segments, key=lambda x: x["start"])
        split_segs = _split_long_segments(sorted_segs, max_chars=subtitle_max_chars,
                                          max_duration=subtitle_max_duration,
                                          min_chars=subtitle_min_chars)
        capped = _cap_subtitle_duration(split_segs, max_duration=subtitle_max_duration)
        merged = _merge_subtitle_segments(capped, max_chars=subtitle_max_chars,
                                          max_duration=subtitle_max_duration,
                                          max_gap=subtitle_max_gap)
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
        split_segs = _split_long_segments(sorted_segs, max_chars=subtitle_max_chars,
                                          max_duration=subtitle_max_duration,
                                          min_chars=subtitle_min_chars)
        capped = _cap_subtitle_duration(split_segs, max_duration=subtitle_max_duration)
        merged = _merge_subtitle_segments(capped, max_chars=subtitle_max_chars,
                                          max_duration=subtitle_max_duration,
                                          max_gap=subtitle_max_gap)
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
