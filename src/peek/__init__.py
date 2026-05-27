"""PEEK: query-free frame selection for video captioning."""

from peek.model import PeekScorer
from peek.selection import stratified_argmax, topk_indices, uniform_indices

__all__ = ["PeekScorer", "stratified_argmax", "topk_indices", "uniform_indices"]
__version__ = "0.1.0"
