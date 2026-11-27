"""Evaluation items and their 2 fps candidate frames for ANC and MSR-VTT."""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from peek.data import read_manifest


@dataclass(slots=True)
class EvalItem:
    key: str  # unique id of the scored item
    video_id: str
    segment_id: str
    references: list[str]
    source: str  # frames directory (ANC) or video file (MSR-VTT)


def anc_items(manifest: Path, frames_root: Path) -> Iterator[EvalItem]:
    """ANC val_1 segments from ``scripts/prepare_manifest.py``; one reference each."""
    for record in read_manifest(manifest):
        yield EvalItem(
            key=f"{record.video_id}/{record.segment_id}",
            video_id=record.video_id,
            segment_id=record.segment_id,
            references=[record.caption],
            source=str(frames_root / record.video_id / record.segment_id),
        )


def msrvtt_items(manifest: Path, video_root: Path) -> Iterator[EvalItem]:
    """MSR-VTT test clips from ``scripts/prepare_msrvtt_manifest.py``; 20 references each."""
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield EvalItem(
                key=row["video_id"],
                video_id=row["video_id"],
                segment_id=row["video_id"],
                references=list(row["captions"]),
                source=str(video_root / row["video_file"]),
            )


def load_jpeg_frames(segment_dir: str) -> list[Image.Image]:
    """Frames written by ``scripts/extract_frames.py`` (ffmpeg fps=2, -q:v 2)."""
    paths = sorted(Path(segment_dir).glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frames in {segment_dir}; run scripts/extract_frames.py first")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    return frames


def decode_frames_decord(video_path: str, sample_fps: float = 2.0) -> list[Image.Image]:
    """Decode frames at ``sample_fps`` exactly as the paper's MSR-VTT runs did.

    Frame indices are ``round(arange(0, total, native_fps / sample_fps))``
    (Python banker's rounding), clipped and de-duplicated. On the 3 fps MSR-VTT
    copy this keeps 2 of every 3 frames, not evenly spaced in time.
    """
    import decord

    reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(reader)
    native_fps = float(reader.get_avg_fps())
    if total <= 0:
        raise ValueError(f"Video has no decodable frames: {video_path}")
    if native_fps <= 0 or sample_fps >= native_fps:
        indices = list(range(total))
    else:
        raw = np.arange(0, total, native_fps / sample_fps, dtype=np.float64)
        indices = sorted({max(0, min(total - 1, int(round(i)))) for i in raw.tolist()}) or [0]
    batch = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in batch]


def load_frames(item: EvalItem, frame_source: str, fps: float) -> list[Image.Image]:
    if frame_source == "jpeg_segments":
        return load_jpeg_frames(item.source)
    if frame_source == "decord":
        return decode_frames_decord(item.source, fps)
    raise ValueError(f"Unknown frame source: {frame_source!r}")
