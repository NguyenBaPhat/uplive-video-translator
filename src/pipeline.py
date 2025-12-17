from pathlib import Path
from typing import List
import subprocess

from moviepy.editor import AudioFileClip, VideoFileClip

from .config import AppConfig
from .gemini_client import GeminiClient
from .tts import synthesize_translated_audio
from .transcription import SubtitleSegment, transcribe_audio, write_srt


class VideoTranslationPipeline:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._gemini = GeminiClient(config)

    def run(
        self,
        input_video: str,
        output_dir: str,
        target_language: str | None = None,
    ) -> dict:
        """
        Orchestrate the full video translation:
        - extract and transcribe audio
        - translate subtitles with Gemini
        - synthesize translated audio
        - mux new audio with original video
        - export SRT files
        """
        print("[1/7] Preparing paths and configuration...")
        in_path = Path(input_video)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        normalized_lang = target_language or self._config.target_language
        language_code = normalized_lang.split("-", 1)[0]

        # Use language code in file names so multi-language batch runs do not overwrite each other.
        lang_suffix = f"_{language_code}" if normalized_lang else ""
        base_name = f"{in_path.stem}{lang_suffix}"
        audio_path = out_dir / f"{base_name}_audio.wav"

        # 1. Extract audio from video.
        print("[2/7] Extracting audio from input video...")
        clip = VideoFileClip(str(in_path))
        clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)

        # 2. Transcribe to timed segments.
        print("[3/7] Transcribing audio to timed segments with Whisper...")
        segments: List[SubtitleSegment] = transcribe_audio(str(audio_path))

        # 3. Translate each subtitle line with Gemini.
        print("[4/7] Translating subtitle lines with Gemini...")
        lines = [seg.text for seg in segments]
        translated_lines = self._gemini.translate_lines(lines)
        for seg, translated in zip(segments, translated_lines):
            seg.translated_text = translated

        # 4. Synthesize translated audio (sequential, no overlap, no cutting).
        # This will update each segment's actual_start/actual_end for accurate subtitle sync.
        print("[5/7] Synthesizing translated audio track (sequential, no overlap)...")
        translated_audio_path = out_dir / f"{base_name}_translated_audio.mp3"
        translated_audio_path_str, actual_audio_duration = synthesize_translated_audio(
            segments,
            language_code,
            str(translated_audio_path),
            total_duration_seconds=float(clip.duration),
        )

        # 5. Write original and translated SRT files.
        # For translated SRT, use actual timing to sync perfectly with synthesized audio.
        print("[6/7] Writing original and translated SRT files...")
        original_srt = out_dir / f"{base_name}_original.srt"
        translated_srt = out_dir / f"{base_name}_translated.srt"
        write_srt(segments, str(original_srt), use_translated=False, use_actual_timing=False)
        write_srt(segments, str(translated_srt), use_translated=True, use_actual_timing=True)

        # 6. Mux new audio with original video.
        # If audio is longer than video, extend video duration to match.
        print("[7/8] Muxing translated audio with video (no subtitles yet)...")
        temp_video_path = out_dir / f"{base_name}_translated_nosubs.mp4"
        translated_audio_clip = AudioFileClip(str(translated_audio_path))
        
        # Use the longer of video or audio duration to ensure nothing is cut.
        video_duration = float(clip.duration)
        audio_duration = float(translated_audio_clip.duration)
        final_duration = max(video_duration, audio_duration)
        
        # If audio is longer, loop the video frames to match audio length.
        if audio_duration > video_duration:
            # Loop video to match audio duration.
            loop_count = int(audio_duration / video_duration) + 1
            extended_video = clip.loop(duration=audio_duration)
            final_clip = extended_video.set_audio(translated_audio_clip)
        else:
            final_clip = clip.set_audio(translated_audio_clip)
        
        final_clip.write_videofile(
            str(temp_video_path),
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )

        clip.close()
        translated_audio_clip.close()
        final_clip.close()

        # 7. Burn translated subtitles into the video using ffmpeg.
        print("[8/8] Burning translated subtitles into final video...")
        translated_clip_path = out_dir / f"{base_name}_translated.mp4"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video_path),
            "-vf",
            f"subtitles={translated_srt}",
            "-c:a",
            "copy",
            str(translated_clip_path),
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # Optionally keep both the no-subs and with-subs versions; here we keep only the with-subs version.
        temp_video_path.unlink(missing_ok=True)

        return {
            "input_video": str(in_path),
            "translated_video": str(translated_clip_path),
            "original_srt": str(original_srt),
            "translated_srt": str(translated_srt),
            "translated_audio": str(translated_audio_path),
            "language": language_code,
        }


