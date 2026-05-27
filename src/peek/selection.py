from __future__ import annotations

import math
from collections.abc import Sequence


def temporal_bucket_ranges(item_count: int, k: int) -> list[tuple[int, int]]:
    """Partition ``range(item_count)`` into ``k`` contiguous half-open buckets."""
    target = min(item_count, k)
    if target <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    for bucket_index in range(target):
        start = math.floor(bucket_index * item_count / target)
        end = math.floor((bucket_index + 1) * item_count / target)
        if end <= start:
            end = min(item_count, start + 1)
        ranges.append((start, end))
    return ranges


def stratified_argmax(scores: Sequence[float], k: int) -> list[int]:
    """Pick the highest-scoring frame inside each of ``k`` equal-width temporal
    buckets.

    This is the policy used in the PEEK paper. With ``k=1`` it reduces to a
    plain argmax over the whole video.

    Args:
        scores: per-frame scores in temporal order.
        k:      number of frames to pick.

    Returns:
        Selected frame indices, sorted in temporal order.
    """
    n = len(scores)
    selected: list[int] = []
    for start, end in temporal_bucket_ranges(n, k):
        bucket = range(start, end)
        # Highest score wins; ties broken by lowest frame index.
        best = min(bucket, key=lambda idx: (-float(scores[idx]), idx))
        selected.append(best)
    return selected


def topk_indices(scores: Sequence[float], k: int) -> list[int]:
    """Return the ``k`` highest-scoring frame indices, sorted temporally."""
    target = min(len(scores), k)
    ordered = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    return sorted(ordered[:target])


def uniform_indices(item_count: int, k: int) -> list[int]:
    """Return ``k`` evenly-spaced indices in ``[0, item_count)``."""
    target = min(item_count, k)
    if target <= 0:
        return []
    if target == 1:
        return [item_count // 2]
    picks: list[int] = []
    for i in range(target):
        raw = ((i + 0.5) * item_count / target) - 0.5
        idx = max(0, min(item_count - 1, round(raw)))
        if not picks or idx != picks[-1]:
            picks.append(idx)
    if len(picks) < target:
        for idx in range(item_count):
            if idx not in picks:
                picks.append(idx)
            if len(picks) == target:
                break
    return sorted(picks)
