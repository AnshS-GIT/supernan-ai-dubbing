import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pipeline.refine import HindiRefiner


class NLLBTranslator:
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    SRC_LANG = "kan_Knda"
    TGT_LANG = "hin_Deva"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[translate] Device: {self.device}")

        print(f"[translate] Loading {self.MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME).to(self.device)
        )
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.TGT_LANG)

    def translate(self, text: str) -> str:
        self.tokenizer.src_lang = self.SRC_LANG

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=256,
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        return translated.strip()


def translate_transcript(input_json: str, output_json: str) -> dict:

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    translator = NLLBTranslator()
    refiner = HindiRefiner()

    translated_segments = []

    for seg in data["segments"]:
        text_kn = seg["text"].strip()

        rough_hindi = translator.translate(text_kn)

        refined_hindi = refiner.refine(rough_hindi)

        translated_segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text_kn": text_kn,
            "text_hi": refined_hindi,
        })

    final_output = {
        "source_language": "kn",
        "target_language": "hi",
        "segments": translated_segments,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"[translate] {len(translated_segments)} segments translated → {output_json}")
    return final_output

