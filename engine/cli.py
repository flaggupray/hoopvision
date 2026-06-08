"""HoopVision CLI — basketball game analyzer.

Usage:
    python -m engine.cli run video.mp4 --output timeline.html
    python -m engine.cli run video.mp4 -o report.json --format json
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.pipeline import Pipeline, PipelineConfig
from engine.exporter import TimelineExporter, ExportConfig
from engine.schema import GameMetadata


def main():
    parser = argparse.ArgumentParser(
        prog="hoopvision",
        description="Basketball game AI analyzer — generate narrative game timelines from video",
    )
    parser.add_argument("--version", action="version", version="hoopvision 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser("run", help="Analyze a basketball video")
    run_parser.add_argument("video", help="Path to video file")
    run_parser.add_argument("--output", "-o", default="timeline.html", help="Output file path")
    run_parser.add_argument("--format", "-f", choices=["html", "json"], default="html")
    run_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    run_parser.add_argument("--sample-rate", type=int, default=6, help="Frames per second to analyze")
    run_parser.add_argument("--confidence", type=float, default=0.5)
    run_parser.add_argument("--no-ocr", action="store_true", help="Disable jersey number OCR")
    run_parser.add_argument("--no-classify", action="store_true", help="Disable action classification")
    run_parser.add_argument("--home", default="", help="Home team name")
    run_parser.add_argument("--away", default="", help="Away team name")
    run_parser.add_argument("--date", default="", help="Game date")
    run_parser.add_argument("--arena", default="", help="Arena name")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        _cmd_run(args)


def _cmd_run(args):
    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    print(f"HoopVision — analyzing: {os.path.basename(args.video)}")
    print(f"   Device: {args.device}  Sample rate: {args.sample_rate} fps")
    print(f"   OCR: {'on' if not args.no_ocr else 'off'}  "
          f"Classifier: {'on' if not args.no_classify else 'off'}")

    config = PipelineConfig(
        sample_rate=args.sample_rate,
        device=args.device,
        enable_ocr=not args.no_ocr,
        enable_classifier=not args.no_classify,
        confidence_threshold=args.confidence,
    )

    metadata = GameMetadata(
        home_team=args.home or "Home",
        away_team=args.away or "Away",
        date=args.date,
        arena=args.arena,
    )

    pipeline = Pipeline(config)
    print("   Analyzing...")
    timeline = pipeline.run(args.video, metadata)
    stats = pipeline.stats()

    print(f"   Frames analyzed: {stats['total_frames']}")
    print(f"   Events detected: {stats['total_events']}")
    print(f"   Time elapsed: {stats['elapsed_seconds']:.1f}s")

    exporter = TimelineExporter()

    if args.format == "html":
        exporter.export_html_file(timeline, args.output)
    elif args.format == "json":
        json_str = exporter.export_json(timeline)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)

    print(f"   Output: {os.path.abspath(args.output)}")
    print("Done!")


if __name__ == "__main__":
    main()
