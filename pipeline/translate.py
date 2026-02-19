from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
from pathlib import Path


class NLLBTranslator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = "facebook/nllb-200-distilled-600M"

        print("Loading NLLB-200 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        # Language codes
        self.src_lang = "kan_Knda"
        self.tgt_lang = "hin_Deva"

    def translate_kn_to_hi(self, text: str) -> str:
        """
        Translate Kannada → Hindi using NLLB.
        """

        self.tokenizer.src_lang = self.src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                max_length=256
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        return translated


def translate_transcript(input_json: str, output_json: str):
    translator = NLLBTranslator()

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
