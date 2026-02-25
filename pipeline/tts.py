import torch
import json
import soundfile as sf
import librosa
from pathlib import Path
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class XTTSGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id = "coqui/XTTS-v2"

        print("[tts] Loading XTTS-v2 model...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id
        ).to(self.device)

        self.model.eval()

    def generate(self, text: str, speaker_wav: str, output_path: str):
        print("[tts] Loading speaker reference audio...")

        speaker_audio, sr = librosa.load(speaker_wav, sr=24000)

        print("[tts] Preparing inputs...")

        inputs = self.processor(
            text=text,
            speaker_wav=speaker_audio,
            sampling_rate=24000,
            return_tensors="pt"
        ).to(self.device)

        print("[tts] Generating Hindi speech...")

        with torch.no_grad():
            speech = self.model.generate(**inputs)

        audio = speech.cpu().numpy().squeeze()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, 24000)

        print(f"[tts] Hindi TTS saved → {output_path}")


def generate_hindi_tts(input_json: str, output_audio: str, speaker_ref: str):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    hindi_text = " ".join([seg["text_hi"] for seg in data["segments"]])

    tts = XTTSGenerator()
    tts.generate(
        text=hindi_text,
        speaker_wav=speaker_ref,
        output_path=output_audio
    )