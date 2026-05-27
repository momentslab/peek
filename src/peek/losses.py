from __future__ import annotations

import torch


def listmle_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """ListMLE listwise ranking loss (Xia et al., 2008).

    Args:
        predictions: ``(B, T)`` predicted logits.
        targets:     ``(B, T)`` per-frame teacher scores (the higher the more
            relevant). Only the relative order of ``targets`` matters.
        mask:        ``(B, T)`` bool tensor — ``True`` for real frames.

    Returns:
        Scalar loss averaged across the batch (ignoring padded frames).
    """
    losses: list[torch.Tensor] = []
    for batch_index in range(predictions.shape[0]):
        valid = mask[batch_index]
        if not valid.any():
            continue
        pred = predictions[batch_index][valid].float()
        target = targets[batch_index][valid].float()
        _, indices = target.sort(dim=-1, descending=True)
        pred_sorted = pred.gather(0, indices)
        # Subtract max for numerical stability (does not change the loss).
        pred_sorted = pred_sorted - pred_sorted.max(dim=-1, keepdim=True).values
        log_cumsum = torch.logcumsumexp(pred_sorted.flip(-1), dim=-1).flip(-1)
        losses.append((log_cumsum - pred_sorted).mean())
    if not losses:
        return predictions.new_tensor(0.0)
    return torch.stack(losses).mean()
