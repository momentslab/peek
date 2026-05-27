from __future__ import annotations

import argparse
import json
from pathlib import Path

from peek.frames import (
    DEFAULT_FPS,
    DEFAULT_JPEG_QUALITY,
    default_worker_count,
    extract_frames_from_manifest,
    resolve_ffmpeg_executable,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--ffmpeg", type=str, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = extract_frames_from_manifest(
        manifest_path=args.manifest,
        output_root=args.output_root,
        fps=args.fps,
        workers=args.workers,
        ffmpeg_executable=resolve_ffmpeg_executable(args.ffmpeg),
        overwrite=args.overwrite,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
