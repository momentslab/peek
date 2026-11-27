#!/usr/bin/env python3
"""Build the MSR-VTT test manifest (2,990 clips, 20 references each) used in the paper.

Annotations: ``train_val_test_videodatainfo.json`` from the HF dataset
``panzeyu2013/MSR-VTT``. Videos: ``MSRVTT_Videos.zip`` from the HF dataset
``friedrichor/MSR-VTT`` (3 fps, 298x224 h264 copies), unzipped so that
``--video-root`` contains ``video7010.mp4`` ... ``video9999.mp4``. The revisions
below are the ones the paper used.

Example:
    python scripts/prepare_msrvtt_manifest.py --video-root data/msrvtt/videos \
        --output data/msrvtt/test_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

ANNOTATION_REPO = "panzeyu2013/MSR-VTT"
ANNOTATION_REVISION = "74727689a5f5b8640c7f9a4d5fe7b856d943a3ef"
ANNOTATION_FILE = "train_val_test_videodatainfo.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--annotations", type=Path, help="local copy of " + ANNOTATION_FILE)
    args = p.parse_args()

    path = args.annotations or Path(
        hf_hub_download(ANNOTATION_REPO, ANNOTATION_FILE, repo_type="dataset", revision=ANNOTATION_REVISION)
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    captions: dict[str, list[dict]] = defaultdict(list)
    for sentence in payload["sentences"]:
        captions[str(sentence["video_id"])].append(sentence)

    test = sorted(
        (v for v in payload["videos"] if v["split"] == "test"),
        key=lambda v: int(str(v["video_id"]).removeprefix("video")),
    )
    if len(test) != 2990:
        raise ValueError(f"expected 2,990 test videos, found {len(test)}")

    missing = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for video in test:
            video_id = str(video["video_id"])
            refs = [s["caption"] for s in sorted(captions[video_id], key=lambda s: int(s.get("sen_id", 0)))]
            if len(refs) != 20:
                raise ValueError(f"{video_id} has {len(refs)} references, expected 20")
            if not (args.video_root / f"{video_id}.mp4").exists():
                missing.append(video_id)
            out.write(json.dumps({"video_id": video_id, "video_file": f"{video_id}.mp4", "captions": refs}) + "\n")
    if missing:
        print(f"warning: {len(missing)} test videos missing under {args.video_root}, e.g. {missing[:3]}")
    print(f"wrote {len(test)} clips to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
