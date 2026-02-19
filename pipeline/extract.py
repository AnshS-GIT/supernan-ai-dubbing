import subprocess
from pathlib import Path


def extract_segment(input_path: str, output_path: str, start_time: int, end_time: int) -> str:
    duration = end_time - start_time
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        output_path,
    ]

    subprocess.run(command, check=True)
    print(f"[extract] Segment saved → {output_path}")
    return output_path


def extract_audio(video_path: str, audio_output: str) -> str:
    Path(audio_output).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        audio_output,
    ]

    subprocess.run(command, check=True)
    print(f"[extract] Audio extracted → {audio_output}")
    return audio_output