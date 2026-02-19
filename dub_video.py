import argparse
from pipeline.extract import extract_segment, extract_audio
from pipeline.transcribe import transcribe_audio
from pipeline.translate import translate_transcript


def main():
    parser = argparse.ArgumentParser(
        description="Supernan AI Dubbing Pipeline"
    )

    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--start", type=int, required=True, help="Start time in seconds")
    parser.add_argument("--end", type=int, required=True, help="End time in seconds")

    args = parser.parse_args()

    clip_path = "outputs/clip.mp4"
    audio_path = "outputs/audio.wav"
    transcript_path = "outputs/transcript.json"
    translated_path = "outputs/translated.json"

    print("\n===== STEP 1: Extract Video Segment =====")
    extract_segment(args.input, clip_path, args.start, args.end)

    print("\n===== STEP 2: Extract Clean Audio =====")
    extract_audio(clip_path, audio_path)

    print("\n===== STEP 3: Transcribe Kannada Audio =====")
    transcribe_audio(audio_path, transcript_path)

    print("\n===== STEP 4: Translate Kannada → Hindi =====")
    translate_transcript(transcript_path, translated_path)

    print("\nPipeline completed successfully.")
    print(f"Final outputs saved in: outputs/")


if __name__ == "__main__":
    main()
