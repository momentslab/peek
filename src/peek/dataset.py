from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from peek.data import SegmentRecord, read_manifest


def segment_targets_path(targets_root: Path, record: SegmentRecord) -> Path:
    return targets_root / record.video_id / f"{record.segment_id}.json"


def segment_embeddings_path(embeddings_root: Path, record: SegmentRecord) -> Path:
    return embeddings_root / record.video_id / f"{record.segment_id}.pt"


def load_target_rows(targets_root: Path, record: SegmentRecord) -> list[dict[str, Any]]:
    payload = json.loads(
        segment_targets_path(targets_root, record).read_text(encoding="utf-8")
    )
    return list(payload["frame_targets"])


def load_embedding_payload(
    embeddings_root: Path, record: SegmentRecord
) -> dict[str, Any]:
    return torch.load(
        segment_embeddings_path(embeddings_root, record), map_location="cpu"
    )


@dataclass(slots=True)
class SegmentBatch:
    embeddings: torch.Tensor  # (B, T, D) float32
    targets: torch.Tensor  # (B, T) float32, in [0, 1]
    mask: torch.Tensor  # (B, T) bool, True for real frames
    frame_indices: torch.Tensor  # (B, T) long, -1 for padding
    records: list[SegmentRecord]


class PeekSegmentDataset(Dataset[dict[str, Any]]):
    """One sample = one captioned segment, with all of its candidate frames.

    Reads precomputed MobileCLIP2 embeddings and SigLIP2 targets. Optionally
    augments by dropping random frames and applying a temporal crop.

    Args:
        manifest_path: JSONL of ``SegmentRecord``s.
        embeddings_root: root containing ``{video_id}/{segment_id}.pt``.
        targets_root: root containing ``{video_id}/{segment_id}.json``.
        augment: enable temporal augmentation (only set ``True`` for training).
        crop_min_fraction: keep at least this fraction of the segment.
        frame_drop_min, frame_drop_max: per-frame dropout rate range.
        max_frames_per_segment: hard cap (subsamples uniformly after aug).
        min_frames_after_aug: never go below this number of frames.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        embeddings_root: Path,
        targets_root: Path,
        augment: bool,
        crop_min_fraction: float = 0.7,
        frame_drop_min: float = 0.05,
        frame_drop_max: float = 0.25,
        max_frames_per_segment: int | None = 32,
        min_frames_after_aug: int = 6,
    ) -> None:
        self.records = read_manifest(manifest_path)
        self.embeddings_root = embeddings_root
        self.targets_root = targets_root
        self.augment = augment
        self.crop_min_fraction = crop_min_fraction
        self.frame_drop_min = frame_drop_min
        self.frame_drop_max = frame_drop_max
        self.max_frames_per_segment = max_frames_per_segment
        self.min_frames_after_aug = max(1, int(min_frames_after_aug))
        self.available_records = [
            r
            for r in self.records
            if segment_embeddings_path(embeddings_root, r).exists()
            and segment_targets_path(targets_root, r).exists()
        ]
        if not self.available_records:
            raise FileNotFoundError(
                f"No cached embeddings/targets found for manifest {manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.available_records)

    def _subsample_indices(self, count: int) -> list[int]:
        if count <= 1:
            return list(range(count))
        floor = min(self.min_frames_after_aug, count) if self.augment else 1
        indices = list(range(count))

        if self.augment and self.crop_min_fraction < 1.0:
            min_length = max(floor, math.ceil(count * self.crop_min_fraction))
            min_length = min(min_length, count)
            length = random.randint(min_length, count)
            start = random.randint(0, count - length)
            indices = indices[start : start + length]

        if self.augment and self.frame_drop_max > 0.0 and len(indices) > 1:
            drop = random.uniform(self.frame_drop_min, self.frame_drop_max)
            kept = [idx for idx in indices if random.random() > drop]
            if len(kept) >= floor:
                indices = kept
            else:
                indices = sorted(random.sample(indices, min(floor, len(indices))))

        if (
            self.max_frames_per_segment is not None
            and len(indices) > self.max_frames_per_segment
        ):
            positions = np.linspace(
                0,
                len(indices) - 1,
                num=self.max_frames_per_segment,
                dtype=int,
            )
            indices = [indices[int(p)] for p in positions.tolist()]
        return indices

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.available_records[index]
        payload = load_embedding_payload(self.embeddings_root, record)
        frame_indices = payload["frame_indices"].to(dtype=torch.long)
        embeddings = payload["embeddings"].to(dtype=torch.float32)
        rows = load_target_rows(self.targets_root, record)
        target_by_idx = {
            int(r["frame_idx"]): float(r["clip_score_normalized"]) for r in rows
        }
        targets = torch.tensor(
            [target_by_idx[int(fi)] for fi in frame_indices.tolist()],
            dtype=torch.float32,
        )
        selected = self._subsample_indices(len(frame_indices))
        sel = torch.tensor(selected, dtype=torch.long)
        return {
            "record": record,
            "embeddings": embeddings.index_select(0, sel),
            "targets": targets.index_select(0, sel),
            "frame_indices": frame_indices.index_select(0, sel),
        }


def collate_batch(items: Sequence[dict[str, Any]]) -> SegmentBatch:
    """Pad a list of variable-length segments into a single batch."""
    max_length = max(int(item["embeddings"].shape[0]) for item in items)
    embedding_dim = int(items[0]["embeddings"].shape[1])
    batch_size = len(items)

    embeddings = torch.zeros(batch_size, max_length, embedding_dim, dtype=torch.float32)
    targets = torch.zeros(batch_size, max_length, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    frame_indices = torch.full(
        (batch_size, max_length), fill_value=-1, dtype=torch.long
    )
    records: list[SegmentRecord] = []
    for batch_index, item in enumerate(items):
        length = int(item["embeddings"].shape[0])
        embeddings[batch_index, :length] = item["embeddings"]
        targets[batch_index, :length] = item["targets"]
        mask[batch_index, :length] = True
        frame_indices[batch_index, :length] = item["frame_indices"]
        records.append(item["record"])
    return SegmentBatch(
        embeddings=embeddings,
        targets=targets,
        mask=mask,
        frame_indices=frame_indices,
        records=records,
    )
