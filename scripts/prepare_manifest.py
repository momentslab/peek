from __future__ import annotations

import argparse
import json
from pathlib import Path

from peek.data import load_activitynet_segments, sample_segments, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument(
        "--annotation-files",
        type=str,
        nargs="+",
        default=["train.json"],
        help="One or more annotation JSON filenames inside --annotations-root.",
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Random sample size; omit for the full split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    annotation_paths = [args.annotations_root / name for name in args.annotation_files]
    missing = [str(p) for p in annotation_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing annotation files: {missing}")

    segments, stats = load_activitynet_segments(
        annotation_paths=annotation_paths,
        videos_root=args.videos_root,
    )
    sampled = sample_segments(segments, sample_size=args.sample_size, seed=args.seed)
    write_manifest(sampled, args.output_manifest)
    print(
        json.dumps(
            {
                "annotation_files": args.annotation_files,
                "stats": stats,
                "sampled_segments": len(sampled),
                "output_manifest": str(args.output_manifest.resolve()),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
