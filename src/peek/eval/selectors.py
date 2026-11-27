"""Frame selectors compared in the paper: Uniform, Random, Oracle and PEEK.

All selectors return indices into the candidate frame list (2 fps frames of
the segment/clip), sorted in temporal order.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import numpy as np

from peek.selection import stratified_argmax, temporal_bucket_ranges, topk_indices, uniform_indices

__all__ = [
    "uniform_indices",
    "random_bucketed_indices",
    "random_seed",
    "linspace_cap",
    "peek_indices",
    "oracle_indices",
]


def random_seed(scheme: str, *, base_seed: int, video_id: str, segment_id: str, k: int) -> int:
    """Per-item seed of the paper's Random baseline.

    The two benchmarks were run by different scripts with different hashing
    schemes; both are kept so the Random rows reproduce. ANC's seed does not
    depend on k, MSR-VTT's does.
    """
    if scheme == "anc":
        raw = f"{base_seed}|random_bucketed|{video_id}|{segment_id}"
    elif scheme == "msrvtt":
        raw = f"{base_seed}|msrvtt_random_bucketed|{video_id}|{k}"
    else:
        raise ValueError(f"Unknown random seed scheme: {scheme!r}")
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big")


def random_bucketed_indices(n: int, k: int, seed: int) -> list[int]:
    """One uniformly random frame inside each of k equal temporal buckets."""
    rng = random.Random(seed)
    return [rng.randrange(start, end) for start, end in temporal_bucket_ranges(n, k)]


def linspace_cap(n: int, max_frames: int | None) -> list[int]:
    """Candidate positions PEEK scores: all n, or ``max_frames`` evenly spaced ones."""
    if max_frames is None or n <= max_frames:
        return list(range(n))
    return [int(p) for p in np.linspace(0, n - 1, num=max_frames, dtype=int).tolist()]


def peek_indices(scores: Sequence[float], positions: Sequence[int], k: int) -> list[int]:
    """Stratified argmax over the scored positions, mapped back to frame indices.

    ``scores[i]`` is PEEK's score for candidate frame ``positions[i]``. Buckets
    are formed over the scored list, as in the paper runs.
    """
    return [positions[i] for i in stratified_argmax(scores, k)]


def oracle_indices(scores: Sequence[float], k: int, mode: str) -> list[int]:
    if mode == "topk":
        return topk_indices(scores, k)
    if mode == "stratified_argmax":
        return stratified_argmax(scores, k)
    raise ValueError(f"Unknown oracle selection: {mode!r}")
