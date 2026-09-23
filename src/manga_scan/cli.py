import argparse
import json
import logging
import platform
import sys
from pathlib import Path

from .config import Config
from .ingest import create_project
from .pipeline import edit, run
from .video import probe


def require_macos():
    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        raise RuntimeError("manga-scan supports Apple Silicon macOS only")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Local manga video → page images, PDF, and CBZ (no OCR / no cloud)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", help="Show video metadata")
    p.add_argument("video")
    for name in ("init", "scan"):
        p = sub.add_parser(
            name, help="Create a project" if name == "init" else "Create and process a project"
        )
        p.add_argument("video")
        p.add_argument("project")
        p.add_argument("--config")
        p.add_argument("--copy-source", action="store_true")
        if name == "scan":
            p.add_argument(
                "--roi",
                required=True,
                help='Normalized TL,TR,BR,BL JSON, e.g. "[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]"',
            )
    p = sub.add_parser("run", help="Process an initialized project (failed runs restart analysis)")
    p.add_argument("project")
    p.add_argument("--roi", help="Normalized four corners, JSON")
    p = sub.add_parser("export", help="Rebuild PDF and CBZ using saved review order and enabled pages")
    p.add_argument("project")
    p = sub.add_parser("add", help="Add a spread from a timestamp, in presentation seconds")
    p.add_argument("project")
    p.add_argument("time", type=float)
    p = sub.add_parser("ui", help="Start localhost-only review UI")
    p.add_argument("--projects", default="projects")
    p.add_argument("--config")
    p.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        require_macos()
        if args.command == "probe":
            print(json.dumps(probe(args.video), ensure_ascii=False, indent=2))
        elif args.command in ("init", "scan"):
            cfg = Config.load(args.config)
            roi = None
            if args.command == "scan":
                from .perspective import validate_roi

                roi = validate_roi(json.loads(args.roi)).tolist()
            create_project(args.video, args.project, cfg, args.copy_source)
            if args.command == "scan":
                run(args.project, roi)
                output = Path(args.project).resolve() / "output"
                print(output / "manga.pdf")
                print(output / "manga.cbz")
            else:
                print(Path(args.project).resolve())
        elif args.command == "run":
            run(args.project, json.loads(args.roi) if args.roi else None)
        elif args.command == "export":
            edit(args.project, "export")
        elif args.command == "add":
            edit(args.project, "add_frame", time=args.time)
        elif args.command == "ui":
            from .ui import serve

            serve(args.projects, Config.load(args.config), args.port)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
