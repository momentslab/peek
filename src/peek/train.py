from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from peek.dataset import PeekSegmentDataset, SegmentBatch, collate_batch
from peek.losses import listmle_loss
from peek.model import PeekScorer

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "peek",
    "run_name": "peek_base",
    "runs_root": Path("runs"),
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_manifest": None,
    "val_manifest": None,
    "train_embeddings_root": None,
    "train_targets_root": None,
    "val_embeddings_root": None,
    "val_targets_root": None,
    # Encoder dim; MobileCLIP2-S0/S2 produce 512-d image features
    "embedding_dim": 512,
    # Scorer architecture
    "hidden_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "ffn_dim": 1024,
    "dropout": 0.15,
    "output_activation": "identity",
    # Optim
    "learning_rate": 2.0e-4,
    "weight_decay": 0.03,
    "batch_size": 1024,
    "num_workers": 4,
    "epochs": 25,
    "warmup_epochs": 2,
    "grad_clip_max_norm": 1.0,
    "checkpoint_every_epochs": 1,
    # Augmentation
    "random_frame_drop_min": 0.05,
    "random_frame_drop_max": 0.25,
    "random_crop_min_fraction": 0.7,
    "max_frames_per_segment": 32,
    "min_frames_after_aug": 6,
    # Evaluation
    "eval_ks": [1, 2, 4, 8],
    "best_metric_name": "val_spearman",
    "resume": False,
    "log_every_steps": 25,
}

_PATH_KEYS = {
    "runs_root",
    "train_manifest",
    "val_manifest",
    "train_embeddings_root",
    "train_targets_root",
    "val_embeddings_root",
    "val_targets_root",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if key in _PATH_KEYS and value is not None and not isinstance(value, Path):
            out[key] = Path(value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a YAML mapping: {path}")
    return _normalize_config({str(k): v for k, v in payload.items()})


def _spearman(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.size <= 1:
        return 1.0
    correlation = spearmanr(targets, predictions).statistic
    if correlation is None or np.isnan(correlation):
        return 0.0
    return float(correlation)


def _topk_recall(targets: np.ndarray, predictions: np.ndarray, k: int) -> float:
    effective_k = min(k, targets.shape[0])
    if effective_k <= 0:
        return 0.0
    oracle = set(np.argsort(-targets, kind="stable")[:effective_k].tolist())
    guess = set(np.argsort(-predictions, kind="stable")[:effective_k].tolist())
    return float(len(oracle & guess) / effective_k)


def _ndcg_at_k(targets: np.ndarray, predictions: np.ndarray, k: int) -> float:
    effective_k = min(k, targets.shape[0])
    if effective_k <= 0:
        return 0.0
    if effective_k == 1:
        oracle_idx = int(np.argmax(targets))
        guess_idx = int(np.argmax(predictions))
        return float(targets[guess_idx] / max(1e-12, targets[oracle_idx]))
    order = np.argsort(-predictions, kind="stable")[:effective_k]
    gains = targets[order]
    discounts = 1.0 / np.log2(np.arange(2, 2 + effective_k))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(targets)[::-1][:effective_k]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        min_lr_ratio = 0.05
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _save_checkpoint(
    path: Path,
    *,
    model: PeekScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": _jsonify(config),
        },
        path,
    )


def _run_validation(
    *,
    model: PeekScorer,
    dataloader: DataLoader,
    device: torch.device,
    eval_ks: Iterable[int],
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    rows: list[dict[str, Any]] = []
    amp_enabled = device.type == "cuda"
    eval_ks = list(eval_ks)

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            embeddings = batch.embeddings.to(device)
            targets = batch.targets.to(device)
            mask = batch.mask.to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
            ):
                predictions = model(embeddings, mask)
                loss = listmle_loss(predictions, targets, mask)
            total_loss += float(loss.item())
            total_batches += 1

            predictions_cpu = predictions.detach().float().cpu()
            targets_cpu = targets.detach().cpu()
            mask_cpu = mask.detach().cpu()
            for batch_index in range(predictions_cpu.shape[0]):
                length = int(mask_cpu[batch_index].sum().item())
                if length == 0:
                    continue
                pred_values = predictions_cpu[batch_index, :length].numpy()
                target_values = targets_cpu[batch_index, :length].numpy()
                row = {"spearman": _spearman(target_values, pred_values)}
                for k in eval_ks:
                    row[f"topk_recall@{k}"] = _topk_recall(
                        target_values, pred_values, k
                    )
                    row[f"ndcg@{k}"] = _ndcg_at_k(target_values, pred_values, k)
                rows.append(row)

    metrics: dict[str, float] = {
        "val_loss": total_loss / max(1, total_batches),
        "val_spearman": float(np.mean([r["spearman"] for r in rows])) if rows else 0.0,
    }
    for k in eval_ks:
        metrics[f"val_topk_recall@{k}"] = (
            float(np.mean([r[f"topk_recall@{k}"] for r in rows])) if rows else 0.0
        )
        metrics[f"val_ndcg@{k}"] = (
            float(np.mean([r[f"ndcg@{k}"] for r in rows])) if rows else 0.0
        )
    return metrics


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonify(payload), ensure_ascii=True) + "\n")


def _build_arg_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PEEK temporal frame scorer (Stage 2).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file. If not set, all config values must be provided on the CLI or will fall back to defaults.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=str(defaults["experiment_name"]),
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=str(defaults["run_name"]),
        help="Name of the run.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(defaults["runs_root"]),
        help="Root directory for storing experiment runs.",
    )
    parser.add_argument(
        "--seed", type=int, default=int(defaults["seed"]), help="Random seed."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(defaults["device"]),
        help="Device to train on.",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=defaults["train_manifest"],
        help="Path to the training manifest file.",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=defaults["val_manifest"],
        help="Path to the validation manifest file.",
    )
    parser.add_argument(
        "--train-embeddings-root",
        type=Path,
        default=defaults["train_embeddings_root"],
        help="Root directory for training embeddings.",
    )
    parser.add_argument(
        "--train-targets-root",
        type=Path,
        default=defaults["train_targets_root"],
        help="Root directory for training targets.",
    )
    parser.add_argument(
        "--val-embeddings-root",
        type=Path,
        default=defaults["val_embeddings_root"],
        help="Root directory for validation embeddings.",
    )
    parser.add_argument(
        "--val-targets-root",
        type=Path,
        default=defaults["val_targets_root"],
        help="Root directory for validation targets.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=int(defaults["embedding_dim"]),
        help="Dimension of the input embeddings.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(defaults["hidden_dim"]),
        help="Dimension of the hidden layers.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=int(defaults["num_heads"]),
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=int(defaults["num_layers"]),
        help="Number of transformer layers.",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=int(defaults["ffn_dim"]),
        help="Dimension of the feed-forward network.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=float(defaults["dropout"]),
        help="Dropout rate.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(defaults["learning_rate"]),
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(defaults["weight_decay"]),
        help="Weight decay.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(defaults["batch_size"]),
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(defaults["num_workers"]),
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(defaults["epochs"]),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=int(defaults["warmup_epochs"]),
        help="Number of warmup epochs.",
    )
    parser.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=float(defaults["grad_clip_max_norm"]),
        help="Maximum norm for gradient clipping.",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=int(defaults["checkpoint_every_epochs"]),
        help="Checkpoint every N epochs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=bool(defaults["resume"]),
        help="Whether to resume from a previous checkpoint. If set, the latest checkpoint in the run directory will be loaded if it exists.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=int(defaults["log_every_steps"]),
        help="Log every N steps.",
    )
    return parser


def _config_from_args(args: argparse.Namespace, base: dict[str, Any]) -> dict[str, Any]:
    config = dict(base)
    for key, value in vars(args).items():
        if key == "config":
            continue
        config[key.replace("-", "_")] = value
    return _normalize_config(config)


def main() -> int:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    base = dict(DEFAULT_CONFIG)
    if bootstrap_args.config is not None:
        base.update(_load_yaml(bootstrap_args.config))
    base = _normalize_config(base)

    parser = _build_arg_parser(base)
    args = parser.parse_args()
    config = _config_from_args(args, base)

    for required_key in (
        "train_manifest",
        "val_manifest",
        "train_embeddings_root",
        "train_targets_root",
        "val_embeddings_root",
        "val_targets_root",
    ):
        if config.get(required_key) is None:
            raise ValueError(
                f"{required_key} is required (set it in the YAML or on the CLI)."
            )

    _set_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = (
        Path(config["runs_root"])
        / str(config["experiment_name"])
        / str(config["run_name"])
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(_jsonify(config), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_last = run_dir / "checkpoints" / "checkpoint_last.pt"
    checkpoint_best = run_dir / "checkpoints" / "checkpoint_best.pt"

    device = torch.device(str(config["device"]))
    train_ds = PeekSegmentDataset(
        config["train_manifest"],
        embeddings_root=config["train_embeddings_root"],
        targets_root=config["train_targets_root"],
        augment=True,
        crop_min_fraction=float(config["random_crop_min_fraction"]),
        frame_drop_min=float(config["random_frame_drop_min"]),
        frame_drop_max=float(config["random_frame_drop_max"]),
        max_frames_per_segment=config["max_frames_per_segment"],
        min_frames_after_aug=int(config["min_frames_after_aug"]),
    )
    val_ds = PeekSegmentDataset(
        config["val_manifest"],
        embeddings_root=config["val_embeddings_root"],
        targets_root=config["val_targets_root"],
        augment=False,
        crop_min_fraction=1.0,
        frame_drop_min=0.0,
        frame_drop_max=0.0,
        max_frames_per_segment=config["max_frames_per_segment"],
    )

    nw = int(config["num_workers"])
    loader_kwargs: dict[str, Any] = {}
    if nw > 0 and device.type == "cuda":
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = PeekScorer(
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        ffn_dim=int(config["ffn_dim"]),
        dropout=float(config["dropout"]),
        output_activation=str(config["output_activation"]),
    ).to(device)
    print(
        f"PEEK scorer: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
        f"trainable parameters."
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_steps = len(train_loader) * int(config["epochs"])
    warmup_steps = len(train_loader) * int(config["warmup_epochs"])
    scheduler = _build_lr_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps
    )

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    best_epoch = -1

    if config["resume"] and checkpoint_last.exists():
        checkpoint = torch.load(checkpoint_last, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        print(f"Resumed from epoch {start_epoch} (best_metric={best_metric:.4f})")

    amp_enabled = device.type == "cuda"
    started_at = time.perf_counter()

    for epoch in range(start_epoch, int(config["epochs"])):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}"):
            embeddings = batch.embeddings.to(device)
            targets = batch.targets.to(device)
            mask = batch.mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
            ):
                predictions = model(embeddings, mask)
                loss = listmle_loss(predictions, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(config["grad_clip_max_norm"]),
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach().cpu().item()))
            if global_step % int(config["log_every_steps"]) == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "train/loss_step": losses[-1],
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                )

        val_metrics = _run_validation(
            model=model,
            dataloader=val_loader,
            device=device,
            eval_ks=list(config["eval_ks"]),
        )
        epoch_metrics = {
            "global_step": global_step,
            "epoch": epoch + 1,
            "train/loss_epoch": float(np.mean(losses)) if losses else 0.0,
            "epoch_time_sec": time.perf_counter() - epoch_started,
            **{f"validation/{k}": v for k, v in val_metrics.items()},
        }
        _append_jsonl(metrics_path, epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        metric_value = float(val_metrics[str(config["best_metric_name"])])
        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_best,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config=config,
            )
        _save_checkpoint(
            checkpoint_last,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            best_epoch=best_epoch,
            config=config,
        )
        if (epoch + 1) % int(config["checkpoint_every_epochs"]) == 0:
            _save_checkpoint(
                run_dir / "checkpoints" / f"checkpoint_epoch_{epoch + 1:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config=config,
            )

    summary = {
        "best_metric": best_metric,
        "best_metric_name": str(config["best_metric_name"]),
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "total_time_sec": time.perf_counter() - started_at,
        "checkpoint_best": str(checkpoint_best.resolve()),
        "checkpoint_last": str(checkpoint_last.resolve()),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
