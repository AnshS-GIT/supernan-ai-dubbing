import whisper
import torch
import json
from pathlib import Path


def transcribe_audio(audio_path: str, output_json: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Whisper medium model...")
    model = whisper.load_model("medium").to(device)

    print("Transcribing Kannada audio...")

    result = model.transcribe(
        audio_path,
        language="kn",
        task="transcribe",
        beam_size=3,
        temperature=0.0,
        condition_on_previous_text=False
    )

    transcript_data = {
        "language": "kn",
        "segments": []
    }

    for seg in result["segments"]:
        transcript_data["segments"].append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    print("Transcript saved.")
    return transcript_data
