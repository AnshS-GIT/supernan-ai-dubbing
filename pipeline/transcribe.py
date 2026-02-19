import whisper
import json
from pathlib import Path


def transcribe_audio(video_path: str, output_json: str):
    model = whisper.load_model("small")

    print("Transcribing audio...")
    result = model.transcribe(video_path)

    segments = result["segments"]

    transcript_data = []

    for seg in segments:
        transcript_data.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    print(f"Transcript saved to {output_json}")

    return transcript_data
