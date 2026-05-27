from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from peek.encoder import (
    DEFAULT_CROP_SIZE,
    DEFAULT_RESIZE_SIZE,
    DEFAULT_VARIANT,
    MOBILECLIP2_VARIANTS,
    precompute_embeddings_from_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variant",
        type=str,
        default=DEFAULT_VARIANT,
        choices=sorted(MOBILECLIP2_VARIANTS),
        help=f"MobileCLIP2 variant (default: {DEFAULT_VARIANT}).",
    )
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resize-size", type=int, default=DEFAULT_RESIZE_SIZE)
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument(
        "--num-load-workers",
        type=int,
        default=8,
        help="Threads for parallel frame loading per segment.",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=4,
        help="Segments to prefetch while the GPU is busy.",
    )
    args = parser.parse_args()

    summary = precompute_embeddings_from_manifest(
        manifest_path=args.manifest,
        frames_root=args.frames_root,
        output_root=args.output_root,
        variant=args.variant,
        encoder_batch_size=args.encoder_batch_size,
        device=args.device,
        overwrite=args.overwrite,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        num_load_workers=args.num_load_workers,
        prefetch=args.prefetch,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
