from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("_cache", torch.empty(0), persistent=False)

    def _build(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        encoding = torch.zeros(length, self.dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.to(dtype=dtype).unsqueeze(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, length, _ = inputs.shape
        if (
            self._cache.numel() == 0
            or self._cache.shape[1] < length
            or self._cache.device != inputs.device
        ):
            self._cache = self._build(length, device=inputs.device, dtype=inputs.dtype)
        return inputs + self._cache[:, :length]


class PeekScorer(nn.Module):
    """Lightweight Transformer that predicts a per-frame relevance score.

    Inputs are frozen frame embeddings of shape (B, T, D). Output is (B, T)
    scores. With ``output_activation="identity"`` (the released setting), the
    outputs are unconstrained logits; ranking is recovered by sorting.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.15,
        output_activation: str = "identity",
    ) -> None:
        super().__init__()
        if output_activation not in {"sigmoid", "identity"}:
            raise ValueError(f"Unsupported output_activation: {output_activation}")
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.output_activation = output_activation

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Score every frame.

        Args:
            embeddings: ``(B, T, D)`` float tensor of frozen frame features.
            mask: ``(B, T)`` bool tensor - ``True`` for real frames, ``False``
                for padding.

        Returns:
            ``(B, T)`` predicted relevance scores.
        """
        embeddings = self.layer_norm(embeddings)
        hidden = self.input_projection(embeddings)
        hidden = self.position_encoding(hidden)
        hidden = self.transformer(hidden, src_key_padding_mask=~mask)
        logits = self.score_head(hidden).squeeze(-1)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits


def load_peek_state_dict(
    model: PeekScorer,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load a checkpoint into a ``PeekScorer`` with tolerant key matching."""
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    mismatched: list[str] = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            dropped.append(key)
            continue
        if hasattr(value, "shape") and value.shape != target.shape:
            mismatched.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(target.shape)}"
            )
            continue
        compatible[key] = value
    model.load_state_dict(compatible, strict=False)
    if dropped or mismatched:
        warnings.warn(
            "PeekScorer checkpoint loaded with compatibility filtering. "
            f"Dropped: {sorted(dropped)}; mismatched: {mismatched}.",
            stacklevel=2,
        )
