"""Run a pretrained PEEK selector on a single video.

Weights are downloaded from the Hugging Face Hub by default; pass --checkpoint
to use a local file instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from peek.encoder import DEFAULT_VARIANT, MOBILECLIP2_VARIANTS
from peek.frames import DEFAULT_FPS
from peek.inference import (
    DEFAULT_ENCODER_BATCH_SIZE,
    DEFAULT_HF_FILENAME,
    DEFAULT_HF_REPO,
    load_peek_pipeline,
    select_frames_from_video,
)


def _jsonify(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video_path", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a local PEEK checkpoint (.safetensors or .pt). "
            "If omitted, the pretrained weights are downloaded from the "
            f"Hugging Face Hub ({DEFAULT_HF_REPO})."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="Hugging Face Hub repo to download weights from when --checkpoint is unset.",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default=DEFAULT_HF_FILENAME,
        help="Weights filename within --hf-repo.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=DEFAULT_VARIANT,
        choices=sorted(MOBILECLIP2_VARIANTS),
        help="MobileCLIP2 variant the checkpoint was trained with.",
    )
    parser.add_argument("--k", type=int, default=1, help="Number of frames to pick.")
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="stratified_argmax",
        choices=["stratified_argmax", "topk", "uniform"],
        help="How to pick the final k frames from per-frame scores.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="FPS to extract frames at for scoring.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_ENCODER_BATCH_SIZE,
        help="MobileCLIP2 encoder batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--save-frames-dir",
        type=Path,
        default=None,
        help="If set, keep the extracted JPEG frames in this directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="If set, write the full inference report as JSON here.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    args = parser.parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if args.k <= 0:
        raise ValueError("--k must be positive")

    encoder, scorer, device = load_peek_pipeline(
        args.checkpoint,
        variant=args.variant,
        device=args.device,
        repo_id=args.hf_repo,
        filename=args.hf_filename,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output = select_frames_from_video(
        args.video_path,
        encoder=encoder,
        scorer=scorer,
        device=device,
        k=args.k,
        selection_mode=args.selection_mode,
        fps=args.fps,
        batch_size=args.batch_size,
        frames_dir=args.save_frames_dir,
        progress=not args.no_progress,
    )

    payload = {
        "video_path": str(args.video_path.resolve()),
        "checkpoint_path": (
            str(args.checkpoint.resolve())
            if args.checkpoint is not None
            else f"hf://{args.hf_repo}/{args.hf_filename}"
        ),
        "variant": args.variant,
        "selection_mode": args.selection_mode,
        "k": args.k,
        "fps": args.fps,
        "device": str(device),
        "frame_count": output.frame_count,
        "selected_frame_indices": output.selected_indices,
        "selected_timestamps_sec": output.selected_timestamps_sec,
        "selected_scores": output.selected_scores,
        "timings_sec": output.timings_sec,
        "gpu_peak_gb": (
            round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
            if device.type == "cuda"
            else None
        ),
        "frames_dir": (
            str(args.save_frames_dir.resolve()) if args.save_frames_dir else None
        ),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(_jsonify(payload), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(_jsonify(payload), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
