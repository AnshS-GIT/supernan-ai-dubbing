import subprocess
import os
from pathlib import Path


def extract_segment(
    input_path: str,
    output_path: str,
    start_time: int,
    end_time: int,
) -> str:
    """
    Extract a video segment and normalize audio.

    Returns:
        Path to extracted clip.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if end_time <= start_time:
        raise ValueError("End time must be greater than start time.")

    duration = end_time - start_time

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-af",
        "loudnorm",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        str(output_path),
    ]

    print("Extracting segment...")
    print(f"Start: {start_time}s | End: {end_time}s | Duration: {duration}s")

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("FFmpeg extraction failed.")

    print(f"Segment saved to: {output_path}")

    return str(output_path)
