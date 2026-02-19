import whisper
import json
from pathlib import Path


def transcribe_audio(video_path: str, output_json: str):

    print("Loading Whisper model...")
    model = whisper.load_model("small")

    print("Detecting language...")
    audio = whisper.load_audio(video_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    print(f"Detected language: {detected_language}")

    print("Transcribing audio...")
    result = model.transcribe(
        video_path,
        language=detected_language,
        task="transcribe"
    )

    segments = result["segments"]

    transcript_data = {
        "language": detected_language,
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
