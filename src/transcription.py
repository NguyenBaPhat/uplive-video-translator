from dataclasses import dataclass
from typing import List

import whisper


@dataclass
class SubtitleSegment:
    index: int
    start: float
    end: float
    text: str
    translated_text: str | None = None
    actual_start: float | None = None  # Actual timing after TTS synthesis (for sync)
    actual_end: float | None = None


def transcribe_audio(audio_path: str, model_size: str = "base") -> List[SubtitleSegment]:
    """
    Transcribe audio with Whisper and return timed subtitle segments.
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)

    segments: List[SubtitleSegment] = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        segments.append(
            SubtitleSegment(
                index=i,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
            )
        )
    return segments


def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: List[SubtitleSegment], path: str, use_translated: bool = False, use_actual_timing: bool = False) -> None:
    """
    Write segments to an SRT file.
    If use_actual_timing is True, use actual_start/actual_end (for synced subtitles).
    Otherwise, use original start/end from transcription.
    """
    lines: list[str] = []
    for seg in segments:
        text = seg.translated_text if use_translated and seg.translated_text else seg.text
        if not text:
            continue
        lines.append(str(seg.index))
        if use_actual_timing and seg.actual_start is not None and seg.actual_end is not None:
            lines.append(f"{format_timestamp(seg.actual_start)} --> {format_timestamp(seg.actual_end)}")
        else:
            lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        lines.append(text.strip())
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


