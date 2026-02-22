"""
pipeline/transcribe.py
----------------------
Stage 2: Kannada audio transcription using OpenAI Whisper Large.

Responsibilities:
- Load Whisper Large on GPU (CUDA) when available, fall back to CPU.
- Transcribe Kannada (kn) audio with optimal parameters.
- Filter out hallucinated / corrupted ASR segments.
- Serialize clean transcript to JSON.

Output schema:
    {
        "language": "kn",
        "segments": [
            {"start": <float>, "end": <float>, "text": "<string>"}
        ]
    }
"""

import json
import logging
from pathlib import Path

import torch
import whisper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hallucination / noise filters
# ---------------------------------------------------------------------------

# Whisper sometimes produces these verbatim artifacts on silence or noise.
_KNOWN_HALLUCINATIONS: set[str] = {
    "...",
    "ಧನ್ಯವಾದ",      # "Thank you" repeated on silence
    "ಧನ್ಯವಾದಗಳು",
    "ಸಂಗೀತ",         # "[Music]"
    "[music]",
    "[silence]",
    "[applause]",
}


def _is_valid_segment(text: str, duration: float) -> bool:
    """
    Return True only if the segment passes all quality gates.

    Gates:
      1. Minimum duration   → 0.5 s
      2. Minimum text length → 3 characters (after strip)
      3. Minimum uniqueness  → at least 5 distinct characters
      4. Not a known hallucination string
    """
    stripped = text.strip()

    if duration < 0.5:
        logger.debug("[transcribe] Rejected (too short %.2fs): %r", duration, stripped)
        return False

    if len(stripped) < 3:
        logger.debug("[transcribe] Rejected (text too short): %r", stripped)
        return False

    if len(set(stripped)) < 5:
        logger.debug("[transcribe] Rejected (low uniqueness): %r", stripped)
        return False

    if stripped.lower() in {h.lower() for h in _KNOWN_HALLUCINATIONS}:
        logger.debug("[transcribe] Rejected (known hallucination): %r", stripped)
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, output_json: str) -> dict:
    """
    Transcribe a Kannada audio file with Whisper Large and save the result.

    Args:
        audio_path:  Path to the 16kHz mono WAV audio file.
        output_json: Destination path for the JSON transcript.

    Returns:
        Parsed transcript dict matching the output schema above.

    Notes:
        • word_timestamps=True enables fine-grained alignment useful for TTS sync.
        • condition_on_previous_text=False reduces error propagation across segments.
        • GPU is used automatically when torch.cuda.is_available().
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("[transcribe] Device: %s", device)

    logger.info("[transcribe] Loading Whisper 'large' model…")
    model = whisper.load_model("large", device=device)

    logger.info("[transcribe] Transcribing %s …", audio_path)
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

    segments: list[dict] = []
    rejected = 0

    for seg in result["segments"]:
        text = seg["text"].strip()
        duration = seg["end"] - seg["start"]

        if not _is_valid_segment(text, duration):
            rejected += 1
            continue

        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": text,
        })

    logger.info(
        "[transcribe] %d valid segments kept, %d rejected",
        len(segments), rejected,
    )

    transcript_data = {
        "language": "kn",
        "segments": segments,
    }

    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)

    logger.info("[transcribe] Transcript saved → %s", output)
    return transcript_data
