"""
pipeline/extract.py
-------------------
Stage 1: Video segment extraction and clean audio extraction.

Responsibilities:
- Extract a time-bounded segment from a source video (lossless copy or re-encode).
- Export 16kHz mono WAV audio required by Whisper for stable transcription.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(command: list[str]) -> None:
    """Execute an ffmpeg command and surface stderr only on failure."""
    logger.debug("ffmpeg command: %s", " ".join(command))
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n"
            + result.stderr.decode("utf-8", errors="replace")
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_segment(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> str:
    """
    Extract a time-bounded clip from *input_path* and write it to *output_path*.

    Args:
        input_path:  Absolute path to the source video file.
        output_path: Destination path for the extracted clip.
        start_time:  Start position in seconds (float).
        end_time:    End position in seconds (float).

    Returns:
        Resolved path of the written clip.

    Notes:
        • Re-encodes with libx264/aac to ensure clean keyframe alignment.
        • For very long videos, consider stream-copy (-c copy) instead.
    """
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time}).")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[extract] Cutting segment  %.2fs → %.2fs  (%.2f s)", start_time, end_time, duration)

    _run_ffmpeg([
        "ffmpeg", "-y",
        "-ss", str(start_time),          # seek *before* -i for speed
        "-i", str(input_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",                    # visually lossless quality
        "-c:a", "aac",
        "-b:a", "192k",
        str(output),
    ])

    logger.info("[extract] Segment saved → %s", output)
    return str(output)


def extract_audio(video_path: str, audio_output: str) -> str:
    """
    Convert video audio track to 16 kHz mono PCM WAV required by Whisper.

    Args:
        video_path:   Path to the source video (or audio) file.
        audio_output: Destination path for the WAV file.

    Returns:
        Resolved path of the written WAV file.
    """
    output = Path(audio_output)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[extract] Extracting audio → 16kHz mono WAV from %s", video_path)

    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ar", "16000",      # 16 kHz — required for Whisper stability
        "-ac", "1",          # mono
        "-vn",               # drop video stream
        "-acodec", "pcm_s16le",   # uncompressed PCM for clean decoding
        str(output),
    ])

    logger.info("[extract] Audio saved → %s", output)
    return str(output)