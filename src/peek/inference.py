from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from peek.encoder import (
    DEFAULT_CROP_SIZE,
    DEFAULT_RESIZE_SIZE,
    DEFAULT_VARIANT,
    MobileCLIP2Encoder,
    load_mobileclip_frame_tensor,
)
from peek.frames import DEFAULT_FPS, extract_frames_from_video
from peek.model import PeekScorer, load_peek_state_dict
from peek.selection import stratified_argmax, topk_indices, uniform_indices

DEFAULT_ENCODER_BATCH_SIZE = 16

# Pretrained weights live on the Hugging Face Hub (CC-BY-NC-SA-4.0), not in this
# Apache-2.0 code repository. See https://huggingface.co/momentslab/peek.
DEFAULT_HF_REPO = "momentslab/peek"
DEFAULT_HF_FILENAME = "peek_base.safetensors"


def resolve_checkpoint_path(
    checkpoint: Path | str | None = None,
    *,
    repo_id: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_HF_FILENAME,
) -> Path:
    """Resolve a checkpoint to a local file path.

    If ``checkpoint`` is an existing local file it is used as-is. Otherwise the
    weights are downloaded (and cached) from the Hugging Face Hub repo
    ``repo_id``. Pass a different ``repo_id`` to pull a non-default release.
    """
    if checkpoint is not None:
        local = Path(checkpoint)
        if local.exists():
            return local
        raise FileNotFoundError(f"Checkpoint not found: {local}")

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _load_state_dict(checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    """Load a state dict from a ``.safetensors`` or ``.pt`` checkpoint."""
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path))
    payload = torch.load(checkpoint_path, map_location="cpu")
    return (
        payload["model"]
        if isinstance(payload, dict) and "model" in payload
        else payload
    )


@dataclass(slots=True)
class InferenceOutput:
    selected_indices: list[int]
    selected_timestamps_sec: list[float]
    selected_scores: list[float]
    scores: list[float]
    frame_count: int
    timings_sec: dict[str, float]


def load_peek_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    ffn_dim: int = 1024,
    dropout: float = 0.15,
    output_activation: str = "identity",
) -> PeekScorer:
    """Build a ``PeekScorer`` and load weights from ``checkpoint_path``.

    Accepts the released ``peek_base.safetensors`` (model-only), a training
    ``.pt`` checkpoint storing ``{"model": state_dict, ...}``, or a raw state
    dict.
    """
    model = PeekScorer(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        output_activation=output_activation,
    )
    load_peek_state_dict(model, _load_state_dict(checkpoint_path))
    model.to(device).eval()
    return model


def score_frame_paths(
    frame_paths: list[Path],
    *,
    encoder: MobileCLIP2Encoder,
    scorer: PeekScorer,
    device: torch.device,
    batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
    progress: bool = True,
) -> tuple[list[float], dict[str, float]]:
    """Encode every frame with MobileCLIP2 and score with PEEK.

    Returns ``(per_frame_scores, timings)``.
    """
    pixel_chunks: list[torch.Tensor] = []
    encode_started = time.perf_counter()
    with torch.inference_mode():
        iterator = range(0, len(frame_paths), batch_size)
        if progress:
            iterator = tqdm(iterator, desc="MobileCLIP2 encoding")
        for start in iterator:
            batch_paths = frame_paths[start : start + batch_size]
            tensors = torch.stack(
                [
                    load_mobileclip_frame_tensor(
                        p,
                        resize_size=resize_size,
                        crop_size=crop_size,
                    )
                    for p in batch_paths
                ],
                dim=0,
            ).to(device, non_blocking=True)
            features = encoder(tensors).float().cpu()
            pixel_chunks.append(features)
    encode_sec = time.perf_counter() - encode_started

    embeddings = torch.cat(pixel_chunks, dim=0).unsqueeze(0).to(device)
    mask = torch.ones((1, embeddings.shape[1]), dtype=torch.bool, device=device)
    score_started = time.perf_counter()
    with torch.inference_mode():
        scores = scorer(embeddings, mask)[0].detach().float().cpu()
    score_sec = time.perf_counter() - score_started
    return scores.tolist(), {
        "mobileclip2_encode_sec": encode_sec,
        "peek_score_sec": score_sec,
    }


_SELECTORS: dict[str, Callable[[list[float], int], list[int]]] = {
    "stratified_argmax": stratified_argmax,
    "topk": topk_indices,
}


def select_indices(
    scores: list[float], k: int, mode: str = "stratified_argmax"
) -> list[int]:
    """Pick ``k`` frame indices from per-frame ``scores``."""
    if mode == "uniform":
        return uniform_indices(len(scores), k)
    fn = _SELECTORS.get(mode)
    if fn is None:
        raise ValueError(
            f"Unknown selection mode: {mode!r}. "
            f"Choose from {sorted(_SELECTORS) + ['uniform']}."
        )
    return fn(scores, k)


def select_frames_from_video(
    video_path: Path,
    *,
    encoder: MobileCLIP2Encoder,
    scorer: PeekScorer,
    device: torch.device,
    k: int,
    selection_mode: str = "stratified_argmax",
    fps: float = DEFAULT_FPS,
    batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    frames_dir: Path | None = None,
    progress: bool = True,
) -> InferenceOutput:
    """End-to-end: decode ``video_path`` -> encode → score -> pick ``k`` frames."""
    total_started = time.perf_counter()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if frames_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="peek_infer_")
        frames_dir = Path(temp_dir.name)
    try:
        extract_started = time.perf_counter()
        frame_paths = extract_frames_from_video(video_path, frames_dir, fps=fps)
        extract_sec = time.perf_counter() - extract_started

        scores, score_timings = score_frame_paths(
            frame_paths,
            encoder=encoder,
            scorer=scorer,
            device=device,
            batch_size=batch_size,
            progress=progress,
        )
        selected = select_indices(scores, k=k, mode=selection_mode)
        total_sec = time.perf_counter() - total_started
        return InferenceOutput(
            selected_indices=selected,
            selected_timestamps_sec=[round(idx / fps, 3) for idx in selected],
            selected_scores=[scores[idx] for idx in selected],
            scores=scores,
            frame_count=len(scores),
            timings_sec={
                "frame_extract_sec": extract_sec,
                **score_timings,
                "total_sec": total_sec,
            },
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def load_peek_pipeline(
    checkpoint_path: Path | str | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
    device: str | torch.device = "cpu",
    repo_id: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_HF_FILENAME,
) -> tuple[MobileCLIP2Encoder, PeekScorer, torch.device]:
    """Load both the MobileCLIP2 encoder and the PEEK scorer.

    With ``checkpoint_path=None`` (the default) the pretrained weights are
    downloaded from the Hugging Face Hub (``repo_id``). Pass a local path to use
    your own checkpoint instead.
    """
    torch_device = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )
    resolved = resolve_checkpoint_path(
        checkpoint_path, repo_id=repo_id, filename=filename
    )
    encoder = MobileCLIP2Encoder(variant=variant).to(torch_device).eval()
    scorer = load_peek_from_checkpoint(
        resolved,
        device=torch_device,
        embedding_dim=encoder.embedding_dim,
    )
    return encoder, scorer, torch_device
