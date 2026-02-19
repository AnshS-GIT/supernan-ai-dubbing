import argparse
from pipeline.extract import extract_segment
from pipeline.transcribe import transcribe_audio


def main():
    parser = argparse.ArgumentParser(
        description="Supernan AI Dubbing Pipeline"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start time in seconds"
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End time in seconds"
    )

    args = parser.parse_args()

    output_clip = "outputs/clip.mp4"

    extract_segment(
        input_path=args.input,
        output_path=output_clip,
        start_time=args.start,
        end_time=args.end,
    )

    transcribe_audio(
        video_path=output_clip,
        output_json="outputs/transcript.json"
    )


if __name__ == "__main__":
    main()
