import argparse
import logging
import sys
from pathlib import Path

from pipeline.extract import extract_audio, extract_segment
from pipeline.transcribe import transcribe_audio
from pipeline.translate import translate_transcript
from pipeline.tts import generate_hindi_tts

from pydub import AudioSegment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _header(step: int, title: str) -> None:
    print()
    print("=" * 60)
    print(f"  STEP {step}: {title}")
    print("=" * 60)


def run_pipeline(
    input_path: str,
    start_time: float,
    end_time: float,
    outdir: str = "outputs",
) -> dict:

    base = Path(outdir)

    outputs = {
        "clip":        str(base / "clip.mp4"),
        "audio":       str(base / "audio.wav"),
        "transcript":  str(base / "transcript.json"),
        "translated":  str(base / "translated.json"),
        "speaker_ref": str(base / "speaker_ref.wav"),
        "tts_audio":   str(base / "hindi_tts.wav"),
    }

    # STEP 1 — Extract Video Segment
    _header(1, "Extract Video Segment")
    extract_segment(
        input_path=input_path,
        output_path=outputs["clip"],
        start_time=start_time,
        end_time=end_time,
    )

    # STEP 2 — Extract Clean Audio
    _header(2, "Extract Clean Audio  (16 kHz mono WAV)")
    extract_audio(
        video_path=outputs["clip"],
        audio_output=outputs["audio"],
    )

    # STEP 3 — Transcribe
    _header(3, "Transcribe Kannada Audio  (Whisper Large)")
    transcribe_audio(
        audio_path=outputs["audio"],
        output_json=outputs["transcript"],
    )

    # STEP 4 — Normalize + Translate
    _header(4, "Normalize + Translate  (Kannada → Hindi via NLLB-200)")
    translate_transcript(
        input_json=outputs["transcript"],
        output_json=outputs["translated"],
    )

    # STEP 5 — XTTS Voice Cloning
    _header(5, "Generate Hindi Voice  (XTTS v2 Voice Cloning)")

    print("[tts] Creating speaker reference clip...")
    audio = AudioSegment.from_wav(outputs["audio"])
    ref_audio = audio[:8000]  # first 8 seconds
    ref_audio.export(outputs["speaker_ref"], format="wav")

    generate_hindi_tts(
        input_json=outputs["translated"],
        output_audio=outputs["tts_audio"],
        speaker_ref=outputs["speaker_ref"],
    )

    print()
    print("=" * 60)
    print("  Pipeline completed successfully.")
    print("=" * 60)
    print(f"  Segment clip   → {outputs['clip']}")
    print(f"  Audio WAV      → {outputs['audio']}")
    print(f"  Transcript     → {outputs['transcript']}")
    print(f"  Translation    → {outputs['translated']}")
    print(f"  Hindi TTS      → {outputs['tts_audio']}")
    print("=" * 60)

    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supernan AI Dubbing Pipeline — Kannada → Hindi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", required=True,
        help="Path to the source video file",
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
        help="Output directory",
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