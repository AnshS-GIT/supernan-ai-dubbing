"""
dub_video.py
------------
Supernan AI Dubbing Pipeline — main orchestrator.

Usage:
    python dub_video.py \\
        --input  massage2.mp4 \\
        --start  0 \\
        --end    15

    # Full-length video (processes as a single segment):
    python dub_video.py --input massage2.mp4 --start 0 --end 3600

    # With explicit output directory:
    python dub_video.py --input massage2.mp4 --start 0 --end 15 --outdir outputs/run1

Stages:
    STEP 1 — Extract video segment (extract_segment)
    STEP 2 — Extract clean 16kHz mono audio (extract_audio)
    STEP 3 — Transcribe Kannada audio (Whisper Large)
    STEP 4 — Normalize + Translate Kannada → Hindi (IndicTrans2)

Future stages (placeholders ready):
    STEP 5 — TTS: synthesize Hindi audio from segments
    STEP 6 — Lip-sync: align TTS audio to video faces

Environment variables:
    HF_TOKEN or HUGGINGFACE_HUB_TOKEN — required if IndicTrans2 model is gated.
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.extract import extract_audio, extract_segment
from pipeline.transcribe import transcribe_audio
from pipeline.translate import translate_transcript

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step header helper
# ---------------------------------------------------------------------------

def _header(step: int, title: str) -> None:
    """Print a clearly visible stage separator to stdout."""
    print()
    print("=" * 60)
    print(f"  STEP {step}: {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    start_time: float,
    end_time: float,
    outdir: str = "outputs",
) -> dict:
    """
    Run the full dubbing pipeline for a single video segment.

    Args:
        input_path: Absolute or relative path to the source video.
        start_time: Segment start in seconds.
        end_time:   Segment end in seconds.
        outdir:     Directory to write all intermediate and final outputs.

    Returns:
        Dict with paths to all output files produced.

    Design note:
        The function is intentionally single-segment to keep memory bounded.
        For long video batch processing, call this function in a loop (or via
        concurrent.futures.ProcessPoolExecutor) over ordered 15-30 s windows.
    """
    base = Path(outdir)

    outputs = {
        "clip":        str(base / "clip.mp4"),
        "audio":       str(base / "audio.wav"),
        "transcript":  str(base / "transcript.json"),
        "translated":  str(base / "translated.json"),
    }

    # ------------------------------------------------------------------
    # STEP 1 — Video segment extraction
    # ------------------------------------------------------------------
    _header(1, "Extract Video Segment")
    extract_segment(
        input_path=input_path,
        output_path=outputs["clip"],
        start_time=start_time,
        end_time=end_time,
    )

    # ------------------------------------------------------------------
    # STEP 2 — Audio extraction (16kHz mono WAV)
    # ------------------------------------------------------------------
    _header(2, "Extract Clean Audio  (16 kHz mono WAV)")
    extract_audio(
        video_path=outputs["clip"],
        audio_output=outputs["audio"],
    )

    # ------------------------------------------------------------------
    # STEP 3 — Whisper Large transcription (Kannada)
    # ------------------------------------------------------------------
    _header(3, "Transcribe Kannada Audio  (Whisper Large)")
    transcribe_audio(
        audio_path=outputs["audio"],
        output_json=outputs["transcript"],
    )

    # ------------------------------------------------------------------
    # STEP 4 — Normalize Kannada + Translate → Hindi (IndicTrans2)
    # ------------------------------------------------------------------
    _header(4, "Normalize + Translate  (Kannada → Hindi via IndicTrans2)")
    translate_transcript(
        input_json=outputs["transcript"],
        output_json=outputs["translated"],
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Pipeline completed successfully.")
    print("=" * 60)
    print(f"  Segment clip   → {outputs['clip']}")
    print(f"  Audio WAV      → {outputs['audio']}")
    print(f"  Transcript     → {outputs['transcript']}")
    print(f"  Translation    → {outputs['translated']}")
    print()
    print("  Next stages (not yet implemented):")
    print("    STEP 5 — TTS: synthesize Hindi voice")
    print("    STEP 6 — Lip-sync: align audio to video")
    print("=" * 60)

    return outputs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supernan AI Dubbing Pipeline — Kannada → Hindi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the source video file (e.g. massage2.mp4)",
    )
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="Segment start time in seconds",
    )
    parser.add_argument(
        "--end", type=float, required=True,
        help="Segment end time in seconds",
    )
    parser.add_argument(
        "--outdir", default="outputs",
        help="Output directory for all generated files",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        input_path=args.input,
        start_time=args.start,
        end_time=args.end,
        outdir=args.outdir,
    )
