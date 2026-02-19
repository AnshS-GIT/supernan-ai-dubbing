import argparse
from pipeline.extract import extract_segment, extract_audio
from pipeline.transcribe import transcribe_audio
from pipeline.translate import translate_transcript

OUTPUTS = {
    "clip": "outputs/clip.mp4",
    "audio": "outputs/audio.wav",
    "transcript": "outputs/transcript.json",
    "translated": "outputs/translated.json",
}


def main():
    parser = argparse.ArgumentParser(description="Supernan AI Dubbing Pipeline")
    parser.add_argument("--input", required=True, help="Path to the source video file")
    parser.add_argument("--start", type=int, required=True, help="Segment start time (seconds)")
    parser.add_argument("--end", type=int, required=True, help="Segment end time (seconds)")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("STEP 1: Extract Video Segment")
    print("=" * 50)
    extract_segment(args.input, OUTPUTS["clip"], args.start, args.end)

    print("\n" + "=" * 50)
    print("STEP 2: Extract Clean Audio (16kHz mono)")
    print("=" * 50)
    extract_audio(OUTPUTS["clip"], OUTPUTS["audio"])

    print("\n" + "=" * 50)
    print("STEP 3: Transcribe Kannada Audio (Whisper)")
    print("=" * 50)
    transcribe_audio(OUTPUTS["audio"], OUTPUTS["transcript"])

    print("\n" + "=" * 50)
    print("STEP 4: Translate Kannada → Hindi (NLLB-200)")
    print("=" * 50)
    translate_transcript(OUTPUTS["transcript"], OUTPUTS["translated"])

    print("\n" + "=" * 50)
    print("Pipeline completed successfully.")
    print(f"Outputs saved in: outputs/")
    print("=" * 50)


if __name__ == "__main__":
    main()
