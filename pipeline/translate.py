from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
from pathlib import Path


class IndicTranslator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = "ai4bharat/indictrans2-indic-en-1B"

        print("Loading IndicTrans2 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def translate_kn_to_hi(self, text: str) -> str:
        input_text = f"<kn> {text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=256
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        return translated


def translate_transcript(input_json: str, output_json: str):
    translator = IndicTranslator()

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    translated_segments = []

    for seg in data["segments"]:
        hindi_text = translator.translate_kn_to_hi(seg["text"])

        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text_kn": seg["text"],
            "text_hi": hindi_text
        })

    final_output = {
        "source_language": "kn",
        "target_language": "hi",
        "segments": translated_segments
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print("Translation completed.")

    return final_output
