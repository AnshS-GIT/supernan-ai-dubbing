import json
import torch
import whisper
from pathlib import Path


def _is_valid_segment(text: str) -> bool:
    text = text.strip()
    if len(text) < 3:
        return False
    if len(set(text)) < 5:
        return False
    return True


def transcribe_audio(audio_path: str, output_json: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[transcribe] Device: {device}")

    print("[transcribe] Loading Whisper large model...")
    model = whisper.load_model("large").to(device)

    print("[transcribe] Transcribing...")
    result = model.transcribe(
        audio_path,
        language="kn",
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.3,
        compression_ratio_threshold=2.4,
    )

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not _is_valid_segment(text):
            print(f"[transcribe] Skipped corrupted segment: {repr(text)}")
            continue
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": text,
        })

    transcript_data = {
        "language": "kn",
        "segments": segments,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    print(f"[transcribe] {len(segments)} segments saved → {output_json}")
    return transcript_data
