#!/usr/bin/env python3
"""Caption ANC or MSR-VTT with frames chosen by Uniform, Random, Oracle or PEEK.

Writes one JSONL row per (item, selector, k). Resumable: rows already in
``--output`` are skipped. Use ``--shard i/n`` to split the work across GPUs
or machines, then score the concatenated files with ``scripts/score_captions.py``.

Example:
    python scripts/eval_captioning.py --dataset msrvtt --vlm qwen25vl3b \
        --manifest data/msrvtt/test_manifest.jsonl --video-root data/msrvtt/videos \
        --output runs/msrvtt_qwen25vl3b.jsonl
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from peek.eval.captioner import Captioner
from peek.eval.data import anc_items, load_frames, msrvtt_items
from peek.eval.scoring import oracle_scores, peek_scores
from peek.eval.selectors import (
    linspace_cap,
    oracle_indices,
    peek_indices,
    random_bucketed_indices,
    random_seed,
    uniform_indices,
)

SELECTORS = ("uniform", "random", "oracle", "peek")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "eval_captioning.yaml"
SIGLIP2 = "google/siglip2-so400m-patch14-384"
VERSIONED_PACKAGES = (
    "torch", "transformers", "tokenizers", "accelerate", "qwen-vl-utils", "decord",
    "pillow", "numpy", "open_clip_torch", "imageio-ffmpeg", "pycocoevalcap",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=("anc", "msrvtt"), required=True)
    p.add_argument("--vlm", choices=("smolvlm2", "qwen25vl3b", "qwen25vl7b", "qwen35_4b"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--frames-root", type=Path, help="ANC: output of scripts/extract_frames.py")
    p.add_argument("--video-root", type=Path, help="MSR-VTT: folder with videoNNNN.mp4")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--selectors", nargs="+", choices=SELECTORS, default=list(SELECTORS))
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--checkpoint", default=None, help="PEEK weights; default: momentslab/peek on the Hub")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument(
        "--peek-max-frames",
        default="config",
        help="'config' (paper setting), 'none' (score every frame) or an integer",
    )
    p.add_argument("--shard", default="0/1", help="i/n: process items i, i+n, i+2n, ...")
    p.add_argument("--limit", type=int, default=None, help="only the first N items (smoke tests)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_settings(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(args.config.read_text())
    dataset = dict(config["datasets"][args.dataset])
    vlm = dict(dataset.pop("vlms")[args.vlm])
    vlm["prompt"] = config["prompts"][vlm["prompt"]]
    vlm["model_id"] = config["vlms"][args.vlm]["model_id"]
    if args.peek_max_frames == "none":
        dataset["peek_max_frames"] = None
    elif args.peek_max_frames != "config":
        dataset["peek_max_frames"] = int(args.peek_max_frames)
    return {"dataset": dataset, "vlm": vlm}


def main() -> int:
    args = parse_args()
    settings = resolve_settings(args)
    ds, vlm = settings["dataset"], settings["vlm"]
    device = torch.device(args.device)
    shard_index, shard_count = (int(x) for x in args.shard.split("/"))

    if args.dataset == "anc":
        if args.frames_root is None:
            sys.exit("--frames-root is required for ANC")
        items = list(anc_items(args.manifest, args.frames_root))
    else:
        if args.video_root is None:
            sys.exit("--video-root is required for MSR-VTT")
        items = list(msrvtt_items(args.manifest, args.video_root))
    if args.limit is not None:
        items = items[: args.limit]
    items = items[shard_index::shard_count]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, int]] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["key"], row["selector"], row["k"]))
    meta = {
        "argv": sys.argv,
        "settings": settings,
        "random_seed": args.random_seed,
        "versions": package_versions(),
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    captioner = Captioner(
        vlm["model_id"],
        device,
        image_max_pixels=vlm.get("image_max_pixels"),
        oom_retry_image_max_pixels=vlm.get("oom_retry_image_max_pixels"),
    )
    if "peek" in args.selectors:
        from peek.inference import load_peek_pipeline

        encoder, scorer, _ = load_peek_pipeline(args.checkpoint, device=device)
    if "oracle" in args.selectors:
        from peek.teacher import load_siglip2

        siglip_processor, siglip_model = load_siglip2(SIGLIP2, device)

    with args.output.open("a") as out:
        for position, item in enumerate(items, start=1):
            todo = [(s, k) for s in args.selectors for k in args.ks if (item.key, s, k) not in done]
            if not todo:
                continue
            started = time.perf_counter()
            try:
                frames = load_frames(item, ds["frame_source"], ds["fps"])
            except Exception as exc:  # unreadable video: record and move on
                for selector, k in todo:
                    out.write(json.dumps({"key": item.key, "selector": selector, "k": k,
                                          "status": "failed", "error": repr(exc)}) + "\n")
                continue
            n = len(frames)
            cache: dict[str, list] = {}
            for selector, k in todo:
                row = {"key": item.key, "video_id": item.video_id, "segment_id": item.segment_id,
                       "selector": selector, "k": k, "num_candidates": n}
                try:
                    if selector == "uniform":
                        indices = uniform_indices(n, k)
                    elif selector == "random":
                        seed = random_seed(ds["random_seed_scheme"], base_seed=args.random_seed,
                                           video_id=item.video_id, segment_id=item.segment_id, k=k)
                        indices = random_bucketed_indices(n, k, seed)
                    elif selector == "oracle":
                        if "oracle" not in cache:
                            cache["oracle"] = oracle_scores(frames, item.references, processor=siglip_processor,
                                                            model=siglip_model, device=device)
                        indices = oracle_indices(cache["oracle"], k, ds["oracle_selection"])
                    else:
                        if "peek" not in cache:
                            positions = linspace_cap(n, ds["peek_max_frames"])
                            cache["peek_positions"] = positions
                            cache["peek"] = peek_scores([frames[i] for i in positions], encoder=encoder,
                                                        scorer=scorer, device=device)
                        indices = peek_indices(cache["peek"], cache["peek_positions"], k)
                    caption, max_pixels = captioner.caption([frames[i] for i in sorted(indices)],
                                                            vlm["prompt"], vlm["max_new_tokens"])
                    row.update(status="success" if caption else "empty", frame_indices=sorted(indices),
                               caption=caption, image_max_pixels=max_pixels)
                except Exception as exc:  # the paper runs logged failures and scored the rest
                    row.update(status="failed", error=repr(exc))
                out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{position}/{len(items)}] {item.key} ({n} frames) {time.perf_counter() - started:.1f}s",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
