"""
pipeline/translate.py
---------------------
Stage 4: Kannada → Hindi translation using NLLB-200 (Meta).

Responsibilities:
- Load NLLB-200-distilled-600M once (no gated access, no HF token needed).
- Normalize each Kannada segment before translation.
- Translate Kannada → Hindi via English pivot for higher quality.
- Preserve start/end timestamps throughout.
- Serialize structured JSON output for the downstream TTS stage.

Translation strategy:
    Kannada (kan_Knda) → English (eng_Latn) → Hindi (hin_Deva)

    Pivot through English improves quality for low-resource pairs.
    Direct Kn→Hi is also possible by calling _translate() directly,
    but pivot yields noticeably better output with NLLB-200-distilled.

Output schema:
    {
        "source_language": "kn",
        "target_language": "hi",
        "segments": [
            {
                "start": <float>,
                "end": <float>,
                "text_kn_raw":   "<raw Kannada from ASR>",
                "text_kn_clean": "<normalized Kannada>",
                "text_hi":       "<translated Hindi>"
            }
        ]
    }

Model:
    facebook/nllb-200-distilled-600M
    — Free, ungated, no HuggingFace login required.
    — Works with transformers >= 4.39 / 5.x.

References:
    https://huggingface.co/facebook/nllb-200-distilled-600M
"""

import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from pipeline.normalize import normalize_kannada

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB-200 / Flores-200 language codes
_LANG_KN = "kan_Knda"   # Kannada in Kannada script
_LANG_EN = "eng_Latn"   # English in Latin script
_LANG_HI = "hin_Deva"   # Hindi in Devanagari script


# ---------------------------------------------------------------------------
# NLLB-200 Translator
# ---------------------------------------------------------------------------

class NLLBTranslator:
    """
    Wrapper around the NLLB-200-distilled-600M model for translation.

    Default strategy: pivot translation  Kannada → English → Hindi.
    Direct translation is also available via `_translate()`.

    Usage:
        translator = NLLBTranslator()
        hindi = translator.translate("ಹೇಗಿದ್ದೀರಿ?")
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[translate] Device: %s", self.device)

        logger.info("[translate] Loading NLLB-200 model: %s …", _MODEL_NAME)

        self.tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME,
            trust_remote_code=False,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            _MODEL_NAME,
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()

        logger.info("[translate] Model loaded on %s.", self.device)

    # -----------------------------------------------------------------------
    # Internal: generic single-pair translation
    # -----------------------------------------------------------------------

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate a single string between any NLLB-200 language pair.

        Args:
            text:     Source text string.
            src_lang: NLLB-200 source language code (e.g. "kan_Knda").
            tgt_lang: NLLB-200 target language code (e.g. "eng_Latn").

        Returns:
            Translated string.
        """
        if not text.strip():
            return ""

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=256,
                num_beams=5,
                early_stopping=True,
            )

        translated = self.tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
        )[0]

        return translated.strip()

    # -----------------------------------------------------------------------
    # Public: Kannada → Hindi (pivot through English)
    # -----------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """
        Translate Kannada text to Hindi using English as a pivot language.

        Pipeline:  Kannada → English → Hindi
        This yields better results than direct Kn→Hi for the distilled model.

        Args:
            text: Kannada text (should already be normalized).

        Returns:
            Translated Hindi string.
        """
        if not text.strip():
            return ""

        english = self._translate(text, _LANG_KN, _LANG_EN)
        logger.debug("[translate]   pivot_en: %r", english)

        hindi = self._translate(english, _LANG_EN, _LANG_HI)
        return hindi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_transcript(
    input_json: str,
    output_json: str,
) -> dict:
    """
    Translate a Whisper transcript JSON from Kannada to Hindi.

    Reads the transcript produced by ``transcribe_audio()``, normalizes
    each Kannada segment, translates it with NLLB-200 (pivot through
    English), and writes the enriched JSON to *output_json*.

    Args:
        input_json:  Path to the Whisper transcript JSON.
        output_json: Destination path for the translated JSON.

    Returns:
        Parsed output dict matching the schema defined at module top.
    """
    logger.info("[translate] Loading transcript: %s", input_json)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load model once — expensive, do not reload per segment.
    translator = NLLBTranslator()

    translated_segments: list[dict] = []
    total = len(data["segments"])

    for idx, seg in enumerate(data["segments"], start=1):
        text_kn_raw = seg["text"].strip()
        text_kn_clean = normalize_kannada(text_kn_raw)

        logger.info(
            "[translate] Segment %d/%d  (%.2fs – %.2fs)  raw=%r",
            idx, total, seg["start"], seg["end"], text_kn_raw[:50],
        )

        text_hi = translator.translate(text_kn_clean)

        logger.debug("[translate]   kn_raw:   %r", text_kn_raw)
        logger.debug("[translate]   kn_clean: %r", text_kn_clean)
        logger.debug("[translate]   hi:       %r", text_hi)

        translated_segments.append({
            "start":        round(seg["start"], 3),
            "end":          round(seg["end"], 3),
            "text_kn_raw":  text_kn_raw,
            "text_kn_clean": text_kn_clean,
            "text_hi":      text_hi,
        })

    final_output = {
        "source_language": "kn",
        "target_language": "hi",
        "segments": translated_segments,
    }

    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    logger.info(
        "[translate] %d segments translated → %s",
        len(translated_segments), output,
    )
    return final_output