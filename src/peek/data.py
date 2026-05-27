from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi")


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    annotation_source: str
    video_id: str
    segment_index: int
    segment_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    caption: str
    video_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SegmentRecord":
        return cls(
            annotation_source=str(payload["annotation_source"]),
            video_id=str(payload["video_id"]),
            segment_index=int(payload["segment_index"]),
            segment_id=str(payload["segment_id"]),
            start_sec=float(payload["start_sec"]),
            end_sec=float(payload["end_sec"]),
            duration_sec=float(payload["duration_sec"]),
            caption=str(payload["caption"]),
            video_path=str(payload["video_path"]),
        )


def _annotation_tag(annotation_path: Path) -> str:
    return annotation_path.stem.replace("_", "")


def _clean_caption(caption: str) -> str:
    return " ".join(caption.strip().split())


def build_video_index(videos_root: Path) -> dict[str, Path]:
    """Map ``video_id`` (= file stem) to the actual video file on disk."""
    candidates: dict[str, list[Path]] = {}
    for ext in VIDEO_EXTENSIONS:
        for path in videos_root.rglob(f"*{ext}"):
            candidates.setdefault(path.stem, []).append(path)
    return {
        video_id: sorted(paths, key=lambda p: (len(p.parts), p.as_posix()))[0]
        for video_id, paths in candidates.items()
    }


def load_activitynet_segments(
    annotation_paths: Sequence[Path],
    videos_root: Path,
) -> tuple[list[SegmentRecord], dict[str, int]]:
    """Read ActivityNet Captions annotation JSONs and resolve video files.

    Annotation files are the standard ActivityNet Captions distribution
    (``train.json``, ``val_1.json``, ``val_2.json``). Each video entry has a
    list of ``timestamps`` and matching ``sentences``.
    """
    video_index = build_video_index(videos_root)
    records: list[SegmentRecord] = []
    stats: dict[str, int] = {
        "annotation_files": len(annotation_paths),
        "videos_in_annotations": 0,
        "videos_with_files": 0,
        "segments_total": 0,
        "segments_loaded": 0,
        "segments_skipped_invalid": 0,
        "videos_missing_files": 0,
    }

    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text())
        annotation_source = annotation_path.stem
        annotation_tag = _annotation_tag(annotation_path)
        for video_id, sample in payload.items():
            stats["videos_in_annotations"] += 1
            timestamps = sample["timestamps"]
            sentences = sample["sentences"]
            stats["segments_total"] += len(timestamps)
            video_path = video_index.get(video_id)
            if video_path is None:
                stats["videos_missing_files"] += 1
                continue
            stats["videos_with_files"] += 1
            duration_sec = float(sample["duration"])
            if len(timestamps) != len(sentences):
                raise ValueError(
                    f"{annotation_path} has mismatched timestamps/sentences for {video_id}"
                )
            for segment_index, (timestamp, caption) in enumerate(
                zip(timestamps, sentences, strict=True)
            ):
                start_sec = max(0.0, float(timestamp[0]))
                end_sec = min(duration_sec, float(timestamp[1]))
                if end_sec <= start_sec:
                    stats["segments_skipped_invalid"] += 1
                    continue
                records.append(
                    SegmentRecord(
                        annotation_source=annotation_source,
                        video_id=video_id,
                        segment_index=segment_index,
                        segment_id=f"{annotation_tag}_seg_{segment_index:05d}",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        duration_sec=duration_sec,
                        caption=_clean_caption(str(caption)),
                        video_path=str(video_path.resolve()),
                    )
                )
                stats["segments_loaded"] += 1
    return records, stats


def sample_segments(
    segments: Sequence[SegmentRecord],
    sample_size: int | None,
    seed: int,
) -> list[SegmentRecord]:
    if sample_size is None or sample_size >= len(segments):
        chosen = list(segments)
    else:
        rng = random.Random(seed)
        chosen = rng.sample(list(segments), k=sample_size)
    return sorted(
        chosen,
        key=lambda r: (r.video_id, r.annotation_source, r.segment_index),
    )


def write_manifest(records: Iterable[SegmentRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def read_manifest(manifest_path: Path) -> list[SegmentRecord]:
    records: list[SegmentRecord] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(SegmentRecord.from_dict(json.loads(line)))
    return records
