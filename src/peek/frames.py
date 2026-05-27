from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

from peek.data import SegmentRecord, read_manifest

DEFAULT_FPS = 2.0
DEFAULT_JPEG_QUALITY = 2


@dataclass(slots=True)
class ExtractionResult:
    video_id: str
    segment_id: str
    status: str
    frame_count: int
    output_dir: str
    elapsed_sec: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_ffmpeg_executable(explicit_path: str | None = None) -> str:
    return explicit_path or get_ffmpeg_exe()


def default_worker_count() -> int:
    return min(8, os.cpu_count() or 1)


def _count_frames(segment_dir: Path) -> int:
    return len(list(segment_dir.glob("frame_*.jpg")))


def _build_ffmpeg_command(
    *,
    ffmpeg_executable: str,
    record: SegmentRecord,
    duration_sec: float,
    fps: float | None,
    jpeg_quality: int,
    output_pattern: Path,
) -> list[str]:
    cmd = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-threads",
        "1",
        "-ss",
        f"{record.start_sec:.3f}",
        "-i",
        record.video_path,
        "-t",
        f"{duration_sec:.3f}",
    ]
    if fps is not None:
        cmd.extend(["-vf", f"fps={fps:g}"])
    cmd.extend(["-start_number", "0", "-q:v", str(jpeg_quality), str(output_pattern)])
    return cmd


def extract_segment_frames(
    record: SegmentRecord,
    frames_root: Path,
    ffmpeg_executable: str,
    fps: float,
    overwrite: bool = False,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> ExtractionResult:
    started_at = time.perf_counter()
    segment_dir = frames_root / record.video_id / record.segment_id
    metadata_path = segment_dir / "_segment.json"

    if not overwrite and metadata_path.exists():
        frame_count = _count_frames(segment_dir)
        if frame_count > 0:
            return ExtractionResult(
                video_id=record.video_id,
                segment_id=record.segment_id,
                status="skipped",
                frame_count=frame_count,
                output_dir=str(segment_dir),
                elapsed_sec=time.perf_counter() - started_at,
            )

    temp_dir = segment_dir.parent / f".{record.segment_id}.tmp"
    duration_sec = record.end_sec - record.start_sec

    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        completed = subprocess.run(
            _build_ffmpeg_command(
                ffmpeg_executable=ffmpeg_executable,
                record=record,
                duration_sec=duration_sec,
                fps=fps,
                jpeg_quality=jpeg_quality,
                output_pattern=temp_dir / "frame_%05d.jpg",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")

        frame_count = _count_frames(temp_dir)
        # Some clips are shorter than 1/fps seconds. Fall back to native fps.
        used_native_fps = False
        if frame_count == 0:
            for entry in temp_dir.iterdir():
                entry.unlink()
            completed = subprocess.run(
                _build_ffmpeg_command(
                    ffmpeg_executable=ffmpeg_executable,
                    record=record,
                    duration_sec=duration_sec,
                    fps=None,
                    jpeg_quality=jpeg_quality,
                    output_pattern=temp_dir / "frame_%05d.jpg",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")
            frame_count = _count_frames(temp_dir)
            used_native_fps = frame_count > 0
        if frame_count == 0:
            raise RuntimeError("ffmpeg completed but no frames were written")

        (temp_dir / "_segment.json").write_text(
            json.dumps(
                record.to_dict()
                | {
                    "requested_fps": fps,
                    "fallback_to_native_fps": used_native_fps,
                    "frame_count": frame_count,
                    "frame_pattern": "frame_%05d.jpg",
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if segment_dir.exists():
            shutil.rmtree(segment_dir)
        temp_dir.rename(segment_dir)

        return ExtractionResult(
            video_id=record.video_id,
            segment_id=record.segment_id,
            status="success",
            frame_count=frame_count,
            output_dir=str(segment_dir),
            elapsed_sec=time.perf_counter() - started_at,
        )
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return ExtractionResult(
            video_id=record.video_id,
            segment_id=record.segment_id,
            status="failed",
            frame_count=0,
            output_dir=str(segment_dir),
            elapsed_sec=time.perf_counter() - started_at,
            error=str(exc),
        )


def _extract_video_job(
    job: tuple[str, list[dict[str, object]], str, str, float, bool, int],
) -> list[dict[str, object]]:
    _, payloads, frames_root_str, ffmpeg_executable, fps, overwrite, jpeg_quality = job
    frames_root = Path(frames_root_str)
    return [
        extract_segment_frames(
            record=SegmentRecord.from_dict(p),
            frames_root=frames_root,
            ffmpeg_executable=ffmpeg_executable,
            fps=fps,
            overwrite=overwrite,
            jpeg_quality=jpeg_quality,
        ).to_dict()
        for p in payloads
    ]


def extract_frames_from_manifest(
    manifest_path: Path,
    output_root: Path,
    fps: float = DEFAULT_FPS,
    workers: int | None = None,
    ffmpeg_executable: str | None = None,
    overwrite: bool = False,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, object]:
    """Decode every segment in ``manifest_path`` into JPEG frames at ``fps``."""
    records = read_manifest(manifest_path)
    if not records:
        raise ValueError(f"Manifest {manifest_path} is empty")

    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    ffmpeg_executable = resolve_ffmpeg_executable(ffmpeg_executable)

    grouped: dict[str, list[SegmentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.video_id].append(record)
    jobs = [
        (
            video_id,
            [record.to_dict() for record in group],
            str(frames_root),
            ffmpeg_executable,
            fps,
            overwrite,
            jpeg_quality,
        )
        for video_id, group in sorted(grouped.items())
    ]

    worker_count = max(1, min(workers or default_worker_count(), len(jobs)))
    all_results: list[dict[str, object]] = []
    if worker_count == 1:
        for job in tqdm(jobs, total=len(jobs), desc="Extracting videos"):
            all_results.extend(_extract_video_job(job))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            iterator = pool.imap_unordered(_extract_video_job, jobs)
            for batch in tqdm(iterator, total=len(jobs), desc="Extracting videos"):
                all_results.extend(batch)

    status_counts: dict[str, int] = defaultdict(int)
    total_frames = 0
    failures: list[dict[str, object]] = []
    for result in all_results:
        status_counts[str(result["status"])] += 1
        total_frames += int(result["frame_count"])
        if result["status"] == "failed":
            failures.append(result)

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "frames_root": str(frames_root.resolve()),
        "fps": fps,
        "workers": worker_count,
        "segments_requested": len(records),
        "videos_requested": len(jobs),
        "total_frames_written": total_frames,
        "status_counts": dict(status_counts),
        "failures": failures,
    }
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "reports" / "extraction_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return summary


def extract_frames_from_video(
    video_path: Path,
    frames_dir: Path,
    *,
    fps: float = DEFAULT_FPS,
    ffmpeg_executable: str | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[Path]:
    """Decode an entire video file into JPEG frames; used by ``infer.py``."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_executable = resolve_ffmpeg_executable(ffmpeg_executable)
    cmd = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-threads",
        "1",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:g}",
        "-start_number",
        "0",
        "-q:v",
        str(jpeg_quality),
        str(frames_dir / "frame_%05d.jpg"),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError("ffmpeg produced no frames")
    return paths
