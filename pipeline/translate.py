"""
pipeline/translate.py
---------------------
Stage 4: Kannada → Hindi translation using IndicTrans2 (ai4bharat).

Responsibilities:
- Load IndicTrans2 indic-indic model once.
- Normalize each Kannada segment before translation.
- Translate Kannada (kn) → Hindi (hi) directly (no pivot language).
- Preserve start/end timestamps throughout.
- Serialize structured JSON output for the downstream TTS stage.

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
    ai4bharat/indictrans2-indic-indic-1B   (Kannada → Hindi, direct)
    Requires HuggingFace token via:
        export HF_TOKEN=<your_token>   or   HUGGINGFACE_HUB_TOKEN=<your_token>

References:
    https://github.com/AI4Bharat/IndicTrans2
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from pipeline.normalize import normalize_kannada

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "ai4bharat/indictrans2-indic-indic-1B"

# IndicTrans2 language codes (Flores-200 / IndicTrans2 format)
_SRC_LANG = "kan_Knda"   # Kannada in Kannada script
_TGT_LANG = "hin_Deva"   # Hindi in Devanagari script


# ---------------------------------------------------------------------------
# IndicTrans2 wrapper
# ---------------------------------------------------------------------------

class IndicTrans2Translator:
    """
    Lazy-singleton wrapper around the IndicTrans2 indic-indic model.

    Usage:
        translator = IndicTrans2Translator()
        hindi = translator.translate("ಹೇಗಿದ್ದೀರಿ?")
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[translate] Device: %s", self.device)

        token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            logger.warning(
                "[translate] No HuggingFace token found. "
                "Set HF_TOKEN environment variable if the model is gated."
            )

        logger.info("[translate] Loading IndicTrans2 model: %s …", _MODEL_NAME)

        self.tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME,
            token=token,
            trust_remote_code=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            _MODEL_NAME,
            token=token,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        logger.info("[translate] Model loaded on %s.", self.device)

    def translate(self, text: str) -> str:
        """
        Translate a single Kannada string to Hindi.

        IndicTrans2 uses special language-tag tokens prepended to input text.
        The tokenizer handles this automatically via src_lang / tgt_lang.

        Args:
            text: Kannada text (should already be normalized).

        Returns:
            Translated Hindi string.
        """
        if not text.strip():
            return ""

        # IndicTrans2 requires the source language prefix injected by the tokenizer.
        self.tokenizer.src_lang = _SRC_LANG

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(_TGT_LANG)

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_transcript(
    input_json: str,
    output_json: str,
    hf_token: Optional[str] = None,
) -> dict:
    """
    Translate a Whisper transcript JSON from Kannada to Hindi.

    Reads the transcript produced by ``transcribe_audio()``, normalizes
    each Kannada segment, translates it with IndicTrans2, and writes
    the enriched JSON to *output_json*.

    Args:
        input_json:  Path to the Whisper transcript JSON.
        output_json: Destination path for the translated JSON.
        hf_token:    Optional HuggingFace token (falls back to env vars).

    Returns:
        Parsed output dict matching the schema defined at module top.
    """
    logger.info("[translate] Loading transcript: %s", input_json)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load model once — expensive, do not reload per segment.
    translator = IndicTrans2Translator(hf_token=hf_token)

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