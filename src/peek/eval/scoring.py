"""Per-frame scores for the content-aware selectors: PEEK and the SigLIP2 Oracle."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image

from peek.encoder import MobileCLIP2Encoder, transform_image
from peek.model import PeekScorer
from peek.teacher import DEFAULT_TEXT_MAX_LENGTH, _normalize


@torch.inference_mode()
def peek_scores(
    frames: Sequence[Image.Image],
    *,
    encoder: MobileCLIP2Encoder,
    scorer: PeekScorer,
    device: torch.device,
    batch_size: int = 64,
) -> list[float]:
    """fp32 MobileCLIP2-S0 embeddings, then one fp32 PEEK pass over the whole sequence.

    ponytail: the paper's ANC runs scored in bf16 inside padded batches. That
    rounding can't be recreated one segment at a time; fp32 reproduces more of
    the paper's ANC selections (237/240 vs 227/240 on a 60-segment check).
    """
    chunks = []
    for start in range(0, len(frames), batch_size):
        pixels = torch.stack([transform_image(f) for f in frames[start : start + batch_size]]).to(device)
        chunks.append(encoder(pixels).float())
    seq = torch.cat(chunks).unsqueeze(0)
    mask = torch.ones(seq.shape[:2], dtype=torch.bool, device=device)
    return scorer(seq, mask)[0].float().cpu().tolist()


@torch.inference_mode()
def oracle_scores(
    frames: Sequence[Image.Image],
    references: Sequence[str],
    *,
    processor,
    model,
    device: torch.device,
    batch_size: int = 32,
) -> list[float]:
    """SigLIP2 cosine between each frame and the references, averaged over references.

    With one reference (ANC) this is the teacher score PEEK was distilled from.
    """
    text = processor(
        text=list(references),
        padding="max_length",
        max_length=DEFAULT_TEXT_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    text = {k: v.to(device) for k, v in text.items() if isinstance(v, torch.Tensor)}
    text_emb = _normalize(model.text_model(**text).pooler_output.float())
    image_chunks = []
    for start in range(0, len(frames), batch_size):
        pixels = processor(images=list(frames[start : start + batch_size]), return_tensors="pt")["pixel_values"]
        out = model.vision_model(pixel_values=pixels.to(device=device, dtype=model.dtype))
        image_chunks.append(_normalize(out.pooler_output.float()))
    image_emb = torch.cat(image_chunks)
    return (image_emb @ text_emb.T).mean(dim=1).cpu().tolist()
