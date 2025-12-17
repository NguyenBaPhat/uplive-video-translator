import argparse
from pathlib import Path

from src.config import load_config
from src.pipeline import VideoTranslationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-powered video translation using Gemini (Uplive case study)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video file (e.g. data/input/sample_en_video.mp4).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory to store translated video, audio, and subtitles.",
    )
    parser.add_argument(
        "--target-lang",
        default=None,
        help="Target language code (e.g. vi, zh, es). Overrides TARGET_LANGUAGE env var if provided.",
    )
    parser.add_argument(
        "--target-langs",
        default=None,
        help=(
            "Comma-separated list of target language codes for multi-language batch "
            "(e.g. vi,es,fr). If set, this overrides --target-lang."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    pipeline = VideoTranslationPipeline(cfg)

    if args.target_langs:
        targets = [code.strip() for code in args.target_langs.split(",") if code.strip()]
    elif args.target_lang:
        targets = [args.target_lang]
    else:
        targets = [None]

    all_results = []
    for target in targets:
        results = pipeline.run(
            input_video=args.input,
            output_dir=args.output_dir,
            target_language=target,
        )
        all_results.append(results)

    print("Translation completed.")
    for res in all_results:
        lang = res.get("language") or "default"
        print(f"\nLanguage: {lang}")
        for key, value in res.items():
            if key == "language":
                continue
            print(f"- {key}: {Path(value)}")


if __name__ == "__main__":
    main()


