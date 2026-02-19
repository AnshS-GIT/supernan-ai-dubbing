import whisper
import torch
import json
from pathlib import Path


def transcribe_audio(audio_path: str, output_json: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Whisper large model...")
    model = whisper.load_model("large").to(device)

    print("Transcribing Kannada audio with improved segmentation...")

    result = model.transcribe(
        audio_path,
        language="kn",
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.3,
        compression_ratio_threshold=2.4
    )

    transcript_data = {
        "language": "kn",
        "segments": []
    }

    for seg in result["segments"]:
        transcript_data["segments"].append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    print("Transcript saved.")
    return transcript_data
