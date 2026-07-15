from __future__ import annotations

import json
import math
import platform
import random
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from meteor_quant.marketlm.dataset import (
    MarketWindowDataset,
    PreparedMarketLMMetadata,
    load_prepared_metadata,
)
from meteor_quant.marketlm.fileio import atomic_file_write, atomic_json_write
from meteor_quant.marketlm.model import MarketLM, compute_losses
from meteor_quant.marketlm.schemas import MarketLMRunRequest

StatusUpdater = Callable[..., None]
_STOP_REQUESTED = False


def _handle_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _handle_stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_stop)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    atomic_file_write(path, lambda temporary: torch.save(payload, temporary))


def _select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_settings(
    device: torch.device,
    requested: str,
) -> tuple[bool, torch.dtype, bool]:
    if device.type != "cuda" or requested == "off":
        return False, torch.float32, False
    if requested == "bf16" or (
        requested == "auto" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    ):
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _autocast(
    device: torch.device, enabled: bool, dtype: torch.dtype
) -> AbstractContextManager[Any]:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _build_loader(
    dataset: MarketWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    drop_last: bool,
    pin_memory: bool,
) -> DataLoader[Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _infinite_batches(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


def _cosine_learning_rate(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    denominator = max(max_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("checkpoint_step_*.pt"))
    final = run_dir / "final.pt"
    if final.exists():
        candidates.append(final)
    return candidates[-1] if candidates else None


def _checkpoint_payload(
    *,
    model: MarketLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    best_validation: float,
    request: MarketLMRunRequest,
    metadata: PreparedMarketLMMetadata,
    prepared_dir: Path,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_validation": best_validation,
        "request": request.model_dump(mode="json"),
        "metadata": {
            "feature_dim": metadata.feature_dim,
            "feature_names": metadata.feature_names,
            "patch_size": metadata.patch_size,
            "context_patches": metadata.context_patches,
            "horizons_seconds": metadata.horizons_seconds,
            "target_mean": metadata.target_mean,
            "target_std": metadata.target_std,
            "timeframe_seconds": metadata.timeframe_seconds,
            "prepared_fingerprint": metadata.fingerprint,
        },
        "prepared_dir": str(prepared_dir),
    }


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    request: MarketLMRunRequest,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "autoregressive": 0.0, "forecast": 0.0, "direction": 0.0}
    correct = 0
    counted = 0
    batches = 0
    for inputs, next_patches, targets, directions, _endpoints in loader:
        inputs = inputs.to(device, non_blocking=True)
        next_patches = next_patches.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        directions = directions.to(device, non_blocking=True)
        with _autocast(device, amp_enabled, amp_dtype):
            output = model(inputs)
            losses = compute_losses(
                output,
                next_patch_targets=next_patches,
                forecast_targets=targets,
                direction_targets=directions,
                autoregressive_weight=request.training.autoregressive_loss_weight,
                forecast_weight=request.training.forecast_loss_weight,
                direction_weight=request.training.direction_loss_weight,
            )
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        predictions = output.direction_logits.argmax(dim=-1)
        mask = directions >= 0
        correct += int(((predictions == directions) & mask).sum().detach().cpu())
        counted += int(mask.sum().detach().cpu())
        batches += 1
    model.train()
    if batches == 0:
        raise RuntimeError("validation loader yielded no batches")
    metrics = {key: value / batches for key, value in totals.items()}
    metrics["direction_accuracy"] = correct / counted if counted else 0.0
    return metrics


def train_marketlm(
    request: MarketLMRunRequest,
    prepared_dir: Path,
    run_dir: Path,
    *,
    update_status: StatusUpdater,
) -> Path:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    _install_signal_handlers()
    training = request.training
    metadata = load_prepared_metadata(prepared_dir)
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)

    device = _select_device(training.device)
    amp_enabled, amp_dtype, needs_scaler = _amp_settings(device, training.amp)
    samples = training.max_steps * training.batch_size * training.gradient_accumulation_steps
    train_dataset = MarketWindowDataset(
        prepared_dir,
        "train",
        samples=samples,
        seed=training.seed,
        deterministic=False,
    )
    validation_dataset = MarketWindowDataset(
        prepared_dir,
        "validation",
        samples=training.validation_windows,
        seed=training.seed + 1,
        deterministic=True,
    )
    train_loader = _build_loader(
        train_dataset,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        prefetch_factor=training.prefetch_factor,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _build_loader(
        validation_dataset,
        batch_size=training.batch_size,
        num_workers=min(training.num_workers, 2),
        prefetch_factor=training.prefetch_factor,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    raw_model = MarketLM(
        feature_dim=metadata.feature_dim,
        patch_size=metadata.patch_size,
        horizons=len(metadata.horizons_seconds),
        config=request.model,
    ).to(device)
    model: torch.nn.Module = raw_model
    if training.compile == "on":
        if not hasattr(torch, "compile"):
            raise RuntimeError("this PyTorch build does not provide torch.compile")
        model = torch.compile(raw_model, mode=training.compile_mode)

    optimizer_kwargs: dict[str, Any] = {
        "lr": training.learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": training.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs)

    scaler: Any = None
    if device.type == "cuda" and needs_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_step = 0
    best_validation = float("inf")
    resume_checkpoint = _latest_checkpoint(run_dir)
    should_resume = training.resume == "on" or (
        training.resume == "auto" and resume_checkpoint is not None
    )
    if should_resume:
        if resume_checkpoint is None:
            raise FileNotFoundError("resume was requested but no checkpoint exists")
        checkpoint = cast(
            dict[str, Any],
            torch.load(resume_checkpoint, map_location=device, weights_only=False),
        )
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        best_validation = float(checkpoint.get("best_validation", best_validation))

    run_info = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "python_platform": platform.platform(),
        "amp": str(amp_dtype).replace("torch.", "") if amp_enabled else "off",
        "compile": training.compile,
        "parameters": raw_model.parameter_count(),
        "effective_batch_size": training.batch_size * training.gradient_accumulation_steps,
        "prepared_dir": str(prepared_dir),
    }
    atomic_json_write(run_info, run_dir / "run_info.json")
    update_status(
        state="training",
        progress=start_step / training.max_steps,
        step=start_step,
        max_steps=training.max_steps,
        message=(
            f"training {raw_model.parameter_count():,} parameters on {device}; "
            f"effective batch={run_info['effective_batch_size']}"
        ),
        prepared_dir=str(prepared_dir),
    )

    batches = _infinite_batches(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    recent_loss = 0.0
    recent_started = time.perf_counter()
    metrics_path = run_dir / "metrics.jsonl"

    completed_step = start_step
    try:
        for step in range(start_step, training.max_steps):
            if _STOP_REQUESTED:
                raise KeyboardInterrupt
            learning_rate = _cosine_learning_rate(
                step,
                warmup_steps=training.warmup_steps,
                max_steps=training.max_steps,
                peak_lr=training.learning_rate,
                min_lr=training.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            accumulated = {
                "total": 0.0,
                "autoregressive": 0.0,
                "forecast": 0.0,
                "direction": 0.0,
            }
            for _ in range(training.gradient_accumulation_steps):
                inputs, next_patches, targets, directions, _endpoints = next(batches)
                inputs = inputs.to(device, non_blocking=True)
                next_patches = next_patches.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                directions = directions.to(device, non_blocking=True)
                with _autocast(device, amp_enabled, amp_dtype):
                    output = model(inputs)
                    losses = compute_losses(
                        output,
                        next_patch_targets=next_patches,
                        forecast_targets=targets,
                        direction_targets=directions,
                        autoregressive_weight=training.autoregressive_loss_weight,
                        forecast_weight=training.forecast_loss_weight,
                        direction_weight=training.direction_loss_weight,
                    )
                    scaled_loss = losses["total"] / training.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                for key in accumulated:
                    accumulated[key] += float(losses[key].detach().cpu())

            if scaler is not None:
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                training.gradient_clip_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite gradient norm at step {step + 1}")
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            completed_step = step + 1
            recent_loss += accumulated["total"] / training.gradient_accumulation_steps
            if completed_step % training.log_interval == 0 or completed_step == training.max_steps:
                elapsed = max(time.perf_counter() - recent_started, 1e-9)
                patch_tokens = (
                    training.log_interval
                    * training.batch_size
                    * training.gradient_accumulation_steps
                    * metadata.context_patches
                )
                log_metrics = {
                    "loss": recent_loss / min(training.log_interval, completed_step),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "patch_tokens_per_second": patch_tokens / elapsed,
                }
                update_status(
                    state="training",
                    progress=completed_step / training.max_steps,
                    step=completed_step,
                    max_steps=training.max_steps,
                    message=f"step {completed_step:,}/{training.max_steps:,}",
                    metrics=log_metrics,
                )
                recent_loss = 0.0
                recent_started = time.perf_counter()

            if completed_step % training.validation_interval == 0:
                validation = _validate(
                    model,
                    validation_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    request=request,
                )
                record = {"step": completed_step, **validation}
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                update_status(
                    state="training",
                    progress=completed_step / training.max_steps,
                    step=completed_step,
                    max_steps=training.max_steps,
                    message=f"validation at step {completed_step:,}",
                    metrics=record,
                )
                if validation["total"] < best_validation:
                    best_validation = validation["total"]
                    _atomic_torch_save(
                        _checkpoint_payload(
                            model=raw_model,
                            optimizer=optimizer,
                            scaler=scaler,
                            step=completed_step,
                            best_validation=best_validation,
                            request=request,
                            metadata=metadata,
                            prepared_dir=prepared_dir,
                        ),
                        run_dir / "best.pt",
                    )

            if completed_step % training.checkpoint_interval == 0:
                _atomic_torch_save(
                    _checkpoint_payload(
                        model=raw_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=completed_step,
                        best_validation=best_validation,
                        request=request,
                        metadata=metadata,
                        prepared_dir=prepared_dir,
                    ),
                    run_dir / f"checkpoint_step_{completed_step:08d}.pt",
                )
    except KeyboardInterrupt:
        checkpoint_path: Path | None = None
        if completed_step > start_step:
            checkpoint_path = run_dir / f"checkpoint_step_{completed_step:08d}.pt"
            _atomic_torch_save(
                _checkpoint_payload(
                    model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=completed_step,
                    best_validation=best_validation,
                    request=request,
                    metadata=metadata,
                    prepared_dir=prepared_dir,
                ),
                checkpoint_path,
            )
        update_status(
            state="stopped",
            progress=completed_step / max(training.max_steps, 1),
            step=completed_step,
            max_steps=training.max_steps,
            message="training stopped by user; checkpoint saved",
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        )
        raise
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory. Reduce batch_size first and increase "
            "gradient_accumulation_steps to preserve effective batch size."
        ) from exc

    final_payload = _checkpoint_payload(
        model=raw_model,
        optimizer=optimizer,
        scaler=scaler,
        step=training.max_steps,
        best_validation=best_validation,
        request=request,
        metadata=metadata,
        prepared_dir=prepared_dir,
    )
    final_path = run_dir / "final.pt"
    _atomic_torch_save(final_payload, final_path)
    if not (run_dir / "best.pt").exists():
        _atomic_torch_save(final_payload, run_dir / "best.pt")
    update_status(
        state="completed",
        progress=1.0,
        step=training.max_steps,
        max_steps=training.max_steps,
        message="training completed",
        checkpoint_path=str(run_dir / "best.pt"),
    )
    return final_path
