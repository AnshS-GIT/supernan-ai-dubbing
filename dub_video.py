import argparse
from pipeline.extract import extract_segment, extract_audio
from pipeline.transcribe import transcribe_audio


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
    extract_segment(args.input, clip_path, args.start, args.end)

    extract_audio(clip_path, audio_path)

    transcribe_audio(audio_path, transcript_path)


if __name__ == "__main__":
    main()
