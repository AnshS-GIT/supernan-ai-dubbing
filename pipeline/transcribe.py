import whisper
import json
from pathlib import Path


def transcribe_audio(video_path: str, output_json: str):
    print("Loading Whisper medium model...")
    model = whisper.load_model("medium")

    print("Transcribing Kannada audio...")
    result = model.transcribe(
        video_path,
        language="kn",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
        fp16=False
    )

    segments = result["segments"]

    transcript_data = {
        "language": "kn",
        "segments": []
    }

    for seg in segments:
        transcript_data["segments"].append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    print(f"Transcript saved to {output_json}")

    return transcript_data
