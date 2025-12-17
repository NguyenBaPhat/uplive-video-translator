from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

from gtts import gTTS
from pydub import AudioSegment

from .transcription import SubtitleSegment


def _synthesize_segment(
    seg: SubtitleSegment,
    language_code: str,
    output_dir: Path,
) -> Tuple[SubtitleSegment, AudioSegment]:
    """
    Synthesize a single subtitle segment into an AudioSegment.
    """
    tts = gTTS(text=seg.translated_text or "", lang=language_code)
    temp_file = output_dir / f".seg{seg.index}.mp3"
    tts.save(str(temp_file))
    piece = AudioSegment.from_file(str(temp_file))
    temp_file.unlink(missing_ok=True)
    return seg, piece


def _time_stretch_to_duration(piece: AudioSegment, target_ms: int) -> AudioSegment:
    """
    Time-stretch an AudioSegment so that its duration is closer to target_ms,
    while keeping the voice perception as one consistent speaker
    (only allow moderate speed changes).
    """
    if target_ms <= 0 or len(piece) == 0:
        return piece

    # Compute ratio between current and target duration.
    ratio = len(piece) / float(target_ms)

    # If very close already, keep as-is.
    if abs(ratio - 1.0) < 0.05:
        return piece

    # Limit stretching to avoid noticeable timbre changes.
    max_change = 0.25  # allow +/-25% speed change
    if ratio > 1.0 + max_change:
        ratio = 1.0 + max_change
    elif ratio < 1.0 - max_change:
        ratio = 1.0 - max_change

    new_frame_rate = int(piece.frame_rate * ratio)
    if new_frame_rate <= 0:
        return piece

    stretched = piece._spawn(piece.raw_data, overrides={"frame_rate": new_frame_rate})
    stretched = stretched.set_frame_rate(piece.frame_rate)
    return stretched


def synthesize_translated_audio(
    segments: Iterable[SubtitleSegment],
    language_code: str,
    output_path: str,
    total_duration_seconds: float,
) -> Tuple[str, float]:
    """
    Use gTTS to synthesize audio per translated subtitle, in parallel.
    Arrange segments sequentially (no overlap, no cutting).
    Returns (output_path, actual_total_duration_seconds) and updates
    each segment's actual_start/actual_end to match the synthesized audio timing.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output.parent

    ordered_segments: List[SubtitleSegment] = sorted(
        [s for s in segments if s.translated_text], key=lambda s: s.start
    )

    # Parallelize TTS HTTP calls to speed up processing.
    pieces: List[Tuple[SubtitleSegment, AudioSegment]] = []
    if ordered_segments:
        max_workers = min(8, len(ordered_segments))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_seg = {
                executor.submit(_synthesize_segment, seg, language_code, output_dir): seg
                for seg in ordered_segments
            }
            for future in as_completed(future_to_seg):
                seg, piece = future.result()
                pieces.append((seg, piece))

    # Arrange pieces sequentially: no overlap, no cutting.
    # Track actual timing for each segment to sync subtitles.
    pieces_sorted = sorted(pieces, key=lambda sp: sp[0].start)
    current_pos_ms = 0
    audio_segments: List[AudioSegment] = []

    for seg, piece in pieces_sorted:
        # Use original piece without cutting or aggressive stretching.
        # Only apply gentle time-stretch if needed (within ±25%).
        seg_duration_ms = int((seg.end - seg.start) * 1000)
        if seg_duration_ms > 0:
            # Gentle stretch towards original duration, but don't force it.
            adjusted = _time_stretch_to_duration(piece, seg_duration_ms)
        else:
            adjusted = piece

        # Record actual timing for this segment.
        actual_start_seconds = current_pos_ms / 1000.0
        actual_duration_seconds = len(adjusted) / 1000.0
        actual_end_seconds = actual_start_seconds + actual_duration_seconds

        seg.actual_start = actual_start_seconds
        seg.actual_end = actual_end_seconds

        # Append to combined audio (no overlay, sequential arrangement).
        audio_segments.append(adjusted)
        current_pos_ms += len(adjusted)

    # Combine all segments into one continuous audio track.
    if audio_segments:
        combined = sum(audio_segments)
    else:
        combined = AudioSegment.silent(duration=0)

    actual_total_duration_seconds = len(combined) / 1000.0
    combined.export(str(output), format=output.suffix.lstrip(".") or "mp3")
    return str(output), actual_total_duration_seconds


