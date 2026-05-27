from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from peek.data import SegmentRecord, read_manifest

DEFAULT_RESIZE_SIZE = 256
DEFAULT_CROP_SIZE = 256

# open_clip pretrained tag for the apple/MobileCLIP2-* HF checkpoints.
MOBILECLIP2_PRETRAINED_TAG = "dfndr2b"

MOBILECLIP2_VARIANTS = {
    "s0": "MobileCLIP2-S0",
    "s2": "MobileCLIP2-S2",
    "s3": "MobileCLIP2-S3",
    "s4": "MobileCLIP2-S4",
    "b": "MobileCLIP2-B",
    "l14": "MobileCLIP2-L-14",
}

DEFAULT_VARIANT = "s0"


def _require_open_clip():
    try:
        import open_clip
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "open_clip_torch is required. Install with: pip install open_clip_torch"
        ) from exc
    return open_clip


def _resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")
    scale = float(size) / float(min(width, height))
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )


def transform_image(
    image: Image.Image,
    *,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> torch.Tensor:
    """Resize-shorter-side + center-crop + [0, 1] scaling.

    Matches MobileCLIP2's open_clip preprocessing (mean=0, std=1; no ImageNet
    normalization).
    """
    image = image.convert("RGB")
    image = _resize_shorter_side(image, resize_size)
    width, height = image.size
    left = max(0, (width - crop_size) // 2)
    top = max(0, (height - crop_size) // 2)
    image = image.crop((left, top, left + crop_size, top + crop_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mobileclip_frame_tensor(
    image_path: Path,
    *,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> torch.Tensor:
    with Image.open(image_path) as image:
        return transform_image(image, resize_size=resize_size, crop_size=crop_size)


def _clone_inference_tensors(module: nn.Module) -> None:
    """Detach any ``is_inference()`` tensors loaded by open_clip so they can
    take gradients in case someone wants to fine-tune the encoder later.
    """

    def _is_inference(t: torch.Tensor) -> bool:
        is_inference = getattr(t, "is_inference", None)
        return bool(is_inference()) if is_inference is not None else False

    for child in module.children():
        _clone_inference_tensors(child)
    for name, param in list(module._parameters.items()):
        if param is not None and _is_inference(param):
            module._parameters[name] = nn.Parameter(
                param.detach().clone(), requires_grad=param.requires_grad
            )
    for name, buf in list(module._buffers.items()):
        if buf is not None and _is_inference(buf):
            module._buffers[name] = buf.detach().clone()


class MobileCLIP2Encoder(nn.Module):
    """Frozen MobileCLIP2 visual tower.

    ``forward(x)`` takes ``(N, 3, H, W)`` and returns ``(N, D)`` image
    features. ``D`` is read from the loaded checkpoint (``512`` for S0/S2).
    """

    def __init__(
        self,
        *,
        variant: str = DEFAULT_VARIANT,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        open_clip = _require_open_clip()
        key = variant.lower()
        if key not in MOBILECLIP2_VARIANTS:
            raise ValueError(
                f"Unsupported MobileCLIP2 variant: {variant!r}. "
                f"Choose from {sorted(MOBILECLIP2_VARIANTS)}."
            )
        model_name = MOBILECLIP2_VARIANTS[key]
        pretrained_tag = MOBILECLIP2_PRETRAINED_TAG if pretrained else None
        full_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_tag
        )
        self.visual = full_model.visual
        del full_model
        _clone_inference_tensors(self.visual)
        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()
        self.variant = key
        self.embedding_dim = self._infer_embed_dim()

    def _infer_embed_dim(self) -> int:
        device = next(self.visual.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(
                1, 3, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE, device=device
            )
            out = self.visual(dummy)
        if out.ndim != 2:
            raise RuntimeError(f"Unexpected feature shape {tuple(out.shape)}")
        return int(out.shape[-1])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values)


def _frame_paths_for_segment(
    frames_root: Path, video_id: str, segment_id: str
) -> list[Path]:
    segment_dir = frames_root / video_id / segment_id
    if not segment_dir.is_dir():
        return []
    return sorted(segment_dir.glob("frame_*.jpg"))


def _load_segment_tensors(
    frame_paths: list[Path],
    *,
    resize_size: int,
    crop_size: int,
    num_workers: int,
) -> torch.Tensor:
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        tensors = list(
            pool.map(
                lambda p: load_mobileclip_frame_tensor(
                    p,
                    resize_size=resize_size,
                    crop_size=crop_size,
                ),
                frame_paths,
            )
        )
    return torch.stack(tensors, dim=0)


def precompute_embeddings_from_manifest(
    *,
    manifest_path: Path,
    frames_root: Path,
    output_root: Path,
    variant: str = DEFAULT_VARIANT,
    encoder_batch_size: int = 128,
    device: str | torch.device | None = None,
    overwrite: bool = False,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
    num_load_workers: int = 8,
    prefetch: int = 4,
) -> dict[str, object]:
    """Run the frozen MobileCLIP2 encoder on every frame in the manifest.

    For each segment, writes a torch file with::

        {"video_id": ..., "segment_id": ...,
         "frame_indices": Tensor[N, int32],
         "embeddings":     Tensor[N, D, float16],
         "model_name":     "mobileclip2_<variant>"}

    These files are what ``peek.dataset.PeekSegmentDataset`` expects.
    """
    records = read_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    encoder = MobileCLIP2Encoder(variant=variant).to(torch_device).eval()
    model_name = f"mobileclip2_{variant.lower()}"

    n_done = n_skipped = n_failed = 0
    started_at = time.perf_counter()
    sentinel = object()
    prefetch_q: queue.Queue = queue.Queue(maxsize=prefetch)

    def _producer() -> None:
        for rec in records:
            out = output_root / rec.video_id / f"{rec.segment_id}.pt"
            if out.exists() and not overwrite:
                prefetch_q.put(("skip", rec, None, None))
                continue
            paths = _frame_paths_for_segment(frames_root, rec.video_id, rec.segment_id)
            if not paths:
                prefetch_q.put(("fail", rec, None, None))
                continue
            try:
                tensors = _load_segment_tensors(
                    paths,
                    resize_size=resize_size,
                    crop_size=crop_size,
                    num_workers=num_load_workers,
                )
                indices = torch.tensor(
                    [int(p.stem.split("_")[-1]) for p in paths],
                    dtype=torch.int32,
                )
                prefetch_q.put(("ok", rec, tensors, indices))
            except Exception as exc:
                prefetch_q.put(("err", rec, None, exc))
        prefetch_q.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()

    pbar = tqdm(total=len(records), desc=f"Encoding ({model_name})")
    while True:
        item = prefetch_q.get()
        if item is sentinel:
            break
        status, record, frame_tensors, extra = item
        pbar.update(1)
        if status == "skip":
            n_skipped += 1
            continue
        if status == "fail":
            n_failed += 1
            continue
        if status == "err":
            tqdm.write(f"FAILED {record.video_id}/{record.segment_id}: {extra}")
            n_failed += 1
            continue
        try:
            with torch.inference_mode():
                chunks: list[torch.Tensor] = []
                for start in range(0, int(frame_tensors.shape[0]), encoder_batch_size):
                    chunk = frame_tensors[start : start + encoder_batch_size].to(
                        torch_device, non_blocking=True
                    )
                    chunks.append(encoder(chunk).float().cpu())
                embeddings = torch.cat(chunks, dim=0).half()
            out_path = output_root / record.video_id / f"{record.segment_id}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "video_id": record.video_id,
                    "segment_id": record.segment_id,
                    "frame_indices": extra,
                    "embeddings": embeddings,
                    "model_name": model_name,
                },
                out_path,
            )
            n_done += 1
        except Exception as exc:
            tqdm.write(f"FAILED {record.video_id}/{record.segment_id}: {exc}")
            n_failed += 1
    pbar.close()

    summary = {
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root.resolve()),
        "model_name": model_name,
        "encoder_batch_size": encoder_batch_size,
        "device": str(torch_device),
        "records_total": len(records),
        "n_done": n_done,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "elapsed_sec": round(time.perf_counter() - started_at, 1),
    }
    (output_root / "precompute_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
