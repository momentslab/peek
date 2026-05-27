from __future__ import annotations

import argparse
import json
from pathlib import Path

from peek.teacher import (
    DEFAULT_SIGLIP2_MODEL,
    DEFAULT_TEXT_MAX_LENGTH,
    compute_teacher_targets_from_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split-name",
        type=str,
        choices=("train", "val", "test", "custom"),
        default="custom",
    )
    parser.add_argument("--frames-root", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-max-length", type=int, default=DEFAULT_TEXT_MAX_LENGTH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip writing SigLIP2 image .pt files (targets only).",
    )
    args = parser.parse_args()

    summary = compute_teacher_targets_from_manifest(
        manifest_path=args.manifest,
        output_root=args.output_root,
        split_name=args.split_name,
        frames_root=args.frames_root,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        text_max_length=args.text_max_length,
        overwrite=args.overwrite,
        write_embeddings=not args.no_embeddings,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
