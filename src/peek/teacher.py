from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoProcessor, Siglip2Model

from peek.data import SegmentRecord, read_manifest

DEFAULT_SIGLIP2_MODEL = "google/siglip2-so400m-patch14-384"
DEFAULT_TEXT_MAX_LENGTH = 64


@dataclass(slots=True)
class TeacherResult:
    video_id: str
    segment_id: str
    status: str
    frame_count: int
    targets_path: str
    embeddings_path: str
    elapsed_sec: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def _frame_paths_for_segment(frames_root: Path, record: SegmentRecord) -> list[Path]:
    return sorted(
        (frames_root / record.video_id / record.segment_id).glob("frame_*.jpg")
    )


def _load_images(paths: Iterable[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for p in paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return images


def load_siglip2(
    model_name: str, device: torch.device
) -> tuple[AutoProcessor, torch.nn.Module]:
    processor = AutoProcessor.from_pretrained(model_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    config = AutoConfig.from_pretrained(model_name)
    # FixRes SigLIP2 checkpoints expose a SigLIP-compatible config.
    if getattr(config, "model_type", None) == "siglip":
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    else:
        model = Siglip2Model.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    return processor, model


def _encode_text(
    processor: AutoProcessor,
    model: torch.nn.Module,
    caption: str,
    device: torch.device,
    text_max_length: int,
) -> torch.Tensor:
    inputs = processor(
        text=[caption],
        padding="max_length",
        max_length=text_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tensor_inputs = {
        k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
    }
    text_out = model.text_model(**tensor_inputs)
    return _normalize(text_out.pooler_output.to(dtype=torch.float32))


def _encode_images(
    processor: AutoProcessor,
    model: torch.nn.Module,
    batch_paths: list[Path],
    device: torch.device,
) -> torch.Tensor:
    images = _load_images(batch_paths)
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)
    image_out = model.vision_model(pixel_values=pixel_values)
    return _normalize(image_out.pooler_output.to(dtype=torch.float32))


def compute_teacher_targets_for_record(
    record: SegmentRecord,
    *,
    frames_root: Path,
    targets_root: Path,
    embeddings_root: Path | None,
    processor: AutoProcessor,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    model_name: str,
    text_max_length: int = DEFAULT_TEXT_MAX_LENGTH,
    overwrite: bool = False,
) -> TeacherResult:
    started_at = time.perf_counter()
    targets_path = targets_root / record.video_id / f"{record.segment_id}.json"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path = (
        (embeddings_root / record.video_id / f"{record.segment_id}.pt")
        if embeddings_root is not None
        else Path("/dev/null")
    )
    if embeddings_root is not None:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        targets_path.exists()
        and (embeddings_root is None or embeddings_path.exists())
        and not overwrite
    ):
        existing = json.loads(targets_path.read_text(encoding="utf-8"))
        return TeacherResult(
            video_id=record.video_id,
            segment_id=record.segment_id,
            status="skipped",
            frame_count=len(existing.get("frame_targets", [])),
            targets_path=str(targets_path),
            embeddings_path=str(embeddings_path),
            elapsed_sec=time.perf_counter() - started_at,
        )

    frame_paths = _frame_paths_for_segment(frames_root, record)
    if not frame_paths:
        return TeacherResult(
            video_id=record.video_id,
            segment_id=record.segment_id,
            status="failed",
            frame_count=0,
            targets_path=str(targets_path),
            embeddings_path=str(embeddings_path),
            elapsed_sec=time.perf_counter() - started_at,
            error="No extracted frames found for segment",
        )

    try:
        with torch.inference_mode():
            text_features = _encode_text(
                processor,
                model,
                record.caption,
                device,
                text_max_length,
            )
            score_chunks: list[torch.Tensor] = []
            emb_chunks: list[torch.Tensor] = []
            for start in range(0, len(frame_paths), batch_size):
                batch_paths = frame_paths[start : start + batch_size]
                image_features = _encode_images(processor, model, batch_paths, device)
                scores = (image_features @ text_features.T).squeeze(-1)
                score_chunks.append(scores)
                if embeddings_root is not None:
                    emb_chunks.append(image_features)
            all_scores = torch.cat(score_chunks).cpu().tolist()
            embeddings = (
                torch.cat(emb_chunks).cpu() if embeddings_root is not None else None
            )

        frame_indices = [int(p.stem.split("_")[-1]) for p in frame_paths]
        normalized_scores = _min_max(all_scores)
        frame_targets = [
            {
                "frame_idx": fi,
                "clip_score_raw": float(rs),
                "clip_score_normalized": float(ns),
            }
            for fi, rs, ns in zip(
                frame_indices, all_scores, normalized_scores, strict=True
            )
        ]

        targets_path.write_text(
            json.dumps(
                {
                    **record.to_dict(),
                    "model_name": model_name,
                    "score_model": "siglip2",
                    "text_max_length": text_max_length,
                    "frame_count": len(frame_targets),
                    "score_min": min(all_scores) if all_scores else None,
                    "score_max": max(all_scores) if all_scores else None,
                    "frame_targets": frame_targets,
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if embeddings_root is not None and embeddings is not None:
            torch.save(
                {
                    "video_id": record.video_id,
                    "segment_id": record.segment_id,
                    "frame_indices": torch.tensor(frame_indices, dtype=torch.int32),
                    "embeddings": embeddings,
                    "model_name": model_name,
                },
                embeddings_path,
            )

        return TeacherResult(
            video_id=record.video_id,
            segment_id=record.segment_id,
            status="success",
            frame_count=len(frame_targets),
            targets_path=str(targets_path),
            embeddings_path=str(embeddings_path),
            elapsed_sec=time.perf_counter() - started_at,
        )
    except Exception as exc:
        return TeacherResult(
            video_id=record.video_id,
            segment_id=record.segment_id,
            status="failed",
            frame_count=0,
            targets_path=str(targets_path),
            embeddings_path=str(embeddings_path),
            elapsed_sec=time.perf_counter() - started_at,
            error=str(exc),
        )


def compute_teacher_targets_from_manifest(
    *,
    manifest_path: Path,
    output_root: Path,
    split_name: str,
    frames_root: Path | None = None,
    model_name: str = DEFAULT_SIGLIP2_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    text_max_length: int = DEFAULT_TEXT_MAX_LENGTH,
    overwrite: bool = False,
    write_embeddings: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records = read_manifest(manifest_path)
    if not records:
        raise ValueError(f"Manifest {manifest_path} is empty")

    frames_root = frames_root or (output_root / "frames")
    targets_root = output_root / f"{split_name}_targets"
    targets_root.mkdir(parents=True, exist_ok=True)
    embeddings_root = (
        output_root / f"{split_name}_embeddings" if write_embeddings else None
    )
    if embeddings_root is not None:
        embeddings_root.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    processor, model = load_siglip2(model_name, torch_device)

    status_counts: dict[str, int] = defaultdict(int)
    failures: list[dict[str, object]] = []
    total_frames = 0

    for record in tqdm(records, desc=f"SigLIP2 teacher ({split_name})"):
        result = compute_teacher_targets_for_record(
            record=record,
            frames_root=frames_root,
            targets_root=targets_root,
            embeddings_root=embeddings_root,
            processor=processor,
            model=model,
            device=torch_device,
            batch_size=batch_size,
            model_name=model_name,
            text_max_length=text_max_length,
            overwrite=overwrite,
        ).to_dict()
        status_counts[str(result["status"])] += 1
        total_frames += int(result["frame_count"])
        if result["status"] == "failed":
            failures.append(result)

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "targets_root": str(targets_root.resolve()),
        "embeddings_root": str(embeddings_root.resolve()) if embeddings_root else None,
        "model_name": model_name,
        "device": str(torch_device),
        "batch_size": batch_size,
        "text_max_length": text_max_length,
        "segments_requested": len(records),
        "total_frames": total_frames,
        "status_counts": dict(status_counts),
        "failures": failures,
    }
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    summary_path = reports_root / f"teacher_{split_name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return summary
