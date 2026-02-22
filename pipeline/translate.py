import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class NLLBTranslator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = "facebook/nllb-200-distilled-600M"

        print("[translate] Loading NLLB-200 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256
            )

        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    def translate(self, text: str) -> str:
        # Step 1: Kannada → English
        english_text = self._translate(
            text,
            src_lang="kan_Knda",
            tgt_lang="eng_Latn"
        )

        # Step 2: English → Hindi
        hindi_text = self._translate(
            english_text,
            src_lang="eng_Latn",
            tgt_lang="hin_Deva"
        )

        return hindi_text


def translate_transcript(input_json: str, output_json: str) -> dict:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    translator = NLLBTranslator()
    translated_segments = []

    for seg in data["segments"]:
        text_kn = seg["text"].strip()

        translated_text = translator.translate(text_kn)

        translated_segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text_kn": text_kn,
            "text_hi": translated_text,
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