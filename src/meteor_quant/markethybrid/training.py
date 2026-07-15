from __future__ import annotations

import copy
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
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from meteor_quant.markethybrid.dataset import MarketHybridWindowDataset
from meteor_quant.markethybrid.model import MarketHybrid, compute_hybrid_losses
from meteor_quant.markethybrid.schemas import (
    MarketHybridLossWeights,
    MarketHybridRunRequest,
    MarketHybridTrainableModules,
    MarketHybridTrainingStageConfig,
)
from meteor_quant.marketlm.dataset import (
    PreparedMarketLMMetadata,
    load_prepared_metadata,
)
from meteor_quant.marketlm.fileio import atomic_file_write, atomic_json_write

StatusUpdater = Callable[..., None]
_STOP_REQUESTED = False
_LOSS_KEYS = (
    "total",
    "autoregressive",
    "forecast",
    "direction",
    "actionable",
    "jepa",
    "policy_position",
    "policy_confidence",
    "policy_intent",
)
_MODULE_NAMES = (
    "online_encoder",
    "autoregressive_head",
    "forecast_head",
    "direction_head",
    "actionable_head",
    "jepa_predictor",
    "policy_position_head",
    "policy_confidence_head",
    "policy_intent_head",
)


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
        requested == "auto"
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ):
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _autocast(
    device: torch.device,
    enabled: bool,
    dtype: torch.dtype,
) -> AbstractContextManager[Any]:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _build_loader(
    dataset: MarketHybridWindowDataset,
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


def _ema_momentum(step: int, max_steps: int, start: float, end: float) -> float:
    progress = min(max(step / max(max_steps, 1), 0.0), 1.0)
    interpolation = 0.5 - 0.5 * math.cos(math.pi * progress)
    return start + (end - start) * interpolation


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("checkpoint_step_*.pt"))
    final = run_dir / "final.pt"
    if final.exists():
        candidates.append(final)
    return candidates[-1] if candidates else None


def _single_stage(request: MarketHybridRunRequest) -> MarketHybridTrainingStageConfig:
    training = request.training
    return MarketHybridTrainingStageConfig(
        name="single_stage",
        end_step=training.max_steps,
        learning_rate=training.learning_rate,
        min_learning_rate=training.min_learning_rate,
        warmup_steps=training.warmup_steps,
        loss_weights=MarketHybridLossWeights(
            autoregressive=training.autoregressive_loss_weight,
            forecast=training.forecast_loss_weight,
            direction=training.direction_loss_weight,
            actionable=training.actionable_loss_weight,
            jepa=training.jepa_loss_weight,
            policy_position=training.policy_position_loss_weight,
            policy_confidence=training.policy_confidence_loss_weight,
            policy_intent=training.policy_intent_loss_weight,
        ),
        trainable_modules=MarketHybridTrainableModules(),
    )


def _stages(request: MarketHybridRunRequest) -> list[MarketHybridTrainingStageConfig]:
    return request.training.stages if request.training.mode == "staged" else [_single_stage(request)]


def _stage_for_step(
    stages: list[MarketHybridTrainingStageConfig],
    step: int,
) -> tuple[int, int, MarketHybridTrainingStageConfig]:
    start = 0
    for index, stage in enumerate(stages):
        if step < stage.end_step:
            return index, start, stage
        start = stage.end_step
    return len(stages) - 1, stages[-2].end_step if len(stages) > 1 else 0, stages[-1]


def _model_modules(model: MarketHybrid) -> dict[str, torch.nn.Module | None]:
    return {
        "online_encoder": model.online_encoder,
        "autoregressive_head": model.next_patch_head,
        "forecast_head": model.forecast_head,
        "direction_head": model.direction_head,
        "actionable_head": model.actionable_head,
        "jepa_predictor": model.predictor,
        "policy_position_head": model.policy_position_head,
        "policy_confidence_head": model.policy_confidence_head,
        "policy_intent_head": model.execution_intent_head,
    }


def _configure_trainable_modules(
    model: MarketHybrid,
    stage: MarketHybridTrainingStageConfig,
) -> dict[str, int]:
    modules = _model_modules(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    flags = stage.trainable_modules.model_dump()
    counts: dict[str, int] = {}
    for name in _MODULE_NAMES:
        module = modules[name]
        enabled = bool(flags[name]) and module is not None
        if module is not None:
            module.requires_grad_(enabled)
            counts[name] = sum(parameter.numel() for parameter in module.parameters())
        else:
            counts[name] = 0
    if model.target_encoder is not None:
        model.target_encoder.requires_grad_(False)
    return counts


def _build_optimizer(
    model: MarketHybrid,
    stage: MarketHybridTrainingStageConfig,
    request: MarketHybridRunRequest,
    device: torch.device,
    previous: torch.optim.Optimizer | None = None,
) -> torch.optim.Optimizer:
    modules = _model_modules(model)
    multipliers = stage.learning_rate_multipliers.model_dump()
    parameter_groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name in _MODULE_NAMES:
        module = modules[name]
        if module is None:
            continue
        parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in seen
        ]
        if not parameters:
            continue
        seen.update(id(parameter) for parameter in parameters)
        multiplier = float(multipliers[name])
        parameter_groups.append(
            {
                "params": parameters,
                "lr": stage.learning_rate * multiplier,
                "lr_multiplier": multiplier,
                "name": name,
            }
        )
    if not parameter_groups:
        raise RuntimeError(f"stage {stage.name} has no trainable parameters")
    optimizer_kwargs: dict[str, Any] = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": request.training.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    if previous is not None:
        for parameter, state in previous.state.items():
            if parameter.requires_grad:
                optimizer.state[parameter] = copy.deepcopy(state)
    return optimizer


def _learning_rates(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        multiplier = float(group.get("lr_multiplier", 1.0))
        group["lr"] = base_learning_rate * multiplier
        result[str(group.get("name", "parameters"))] = float(group["lr"])
    return result


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    *,
    model: MarketHybrid,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    stage_index: int,
    stage_start: int,
    stage: MarketHybridTrainingStageConfig,
    best_loss: float,
    best_hybrid: float,
    best_by_stage: dict[str, dict[str, float]],
    request: MarketHybridRunRequest,
    metadata: PreparedMarketLMMetadata,
    prepared_dir: Path,
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "model_type": "markethybrid",
        "model": model.state_dict(),
        "deployment_model": model.deployment_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "global_step": step,
        "stage_name": stage.name,
        "stage_index": stage_index,
        "stage_local_step": step - stage_start,
        "schedule_hash": request.training.schedule_hash(),
        "best_validation": best_loss,
        "best_loss": best_loss,
        "best_hybrid": best_hybrid,
        "best_by_stage": best_by_stage,
        "rng_state": _rng_state(),
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
            "jepa_target_patch_offsets": request.jepa.target_patch_offsets,
        },
        "prepared_dir": str(prepared_dir),
    }


def _loss_weights(
    prepared_dir: Path,
    metadata: PreparedMarketLMMetadata,
    request: MarketHybridRunRequest,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    directions = np.load(prepared_dir / "directions.npy", mmap_mode="r")
    start, stop = metadata.split_endpoints["train"]
    horizons = len(metadata.horizons_seconds)
    counts = np.zeros((horizons, 3), dtype=np.int64)
    actionable = np.zeros((horizons, 2), dtype=np.int64)
    for offset in range(start, stop + 1, 1_000_000):
        chunk = np.asarray(
            directions[offset : min(offset + 1_000_000, stop + 1)],
            dtype=np.int64,
        )
        for horizon in range(horizons):
            valid = chunk[:, horizon]
            valid = valid[valid >= 0]
            if valid.size:
                counts[horizon] += np.bincount(valid, minlength=3)[:3]
                binary = (valid != 1).astype(np.int64, copy=False)
                actionable[horizon] += np.bincount(binary, minlength=2)[:2]
    class_weights = np.ones((horizons, 3), dtype=np.float32)
    for horizon in range(horizons):
        total = max(int(counts[horizon].sum()), 1)
        for class_index in range(3):
            count = int(counts[horizon, class_index])
            if count > 0:
                class_weights[horizon, class_index] = min(
                    total / (3.0 * count),
                    request.training.direction_class_weight_cap,
                )
    positive_weights = np.ones(horizons, dtype=np.float32)
    for horizon in range(horizons):
        negative = int(actionable[horizon, 0])
        positive = int(actionable[horizon, 1])
        if positive > 0:
            positive_weights[horizon] = min(
                negative / max(positive, 1),
                request.training.actionable_pos_weight_cap,
            )
    diagnostics = {
        "direction_counts": counts.tolist(),
        "actionable_counts": actionable.tolist(),
        "actionable_positive_rates": [
            float(row[1] / max(row.sum(), 1)) for row in actionable
        ],
    }
    return (
        torch.tensor(class_weights, device=device),
        torch.tensor(positive_weights, device=device),
        diagnostics,
    )


def _float_numpy(tensor: torch.Tensor) -> NDArray[np.float32]:
    """Convert floating model outputs to NumPy through float32.

    NumPy has no native bfloat16 dtype, so tensors emitted by CUDA/CPU
    autocast must be widened before crossing the PyTorch/NumPy boundary.
    """

    return cast(
        NDArray[np.float32],
        tensor.detach().to(device="cpu", dtype=torch.float32).numpy(),
    )


def _confusion(true: np.ndarray, predicted: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    valid = (true >= 0) & (true < classes) & (predicted >= 0) & (predicted < classes)
    np.add.at(matrix, (true[valid], predicted[valid]), 1)
    return matrix


def _classification_metrics(matrix: np.ndarray) -> tuple[float, float, float, float]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    recalls: list[float] = []
    f1s: list[float] = []
    for index in range(matrix.shape[0]):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - matrix[index, index])
        fn = float(matrix[index, :].sum() - matrix[index, index])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        recalls.append(recall)
        f1s.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    balanced = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    s = float(matrix.sum())
    c = float(np.trace(matrix))
    true_totals = matrix.sum(axis=1).astype(np.float64)
    predicted_totals = matrix.sum(axis=0).astype(np.float64)
    numerator = c * s - float(np.dot(true_totals, predicted_totals))
    denominator = math.sqrt(
        max(s * s - float(np.dot(predicted_totals, predicted_totals)), 0.0)
        * max(s * s - float(np.dot(true_totals, true_totals)), 0.0)
    )
    mcc = numerator / denominator if denominator > 0 else 0.0
    return accuracy, balanced, macro_f1, mcc


def _binary_pr_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(np.int64, copy=False)
    positives = int(target.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    tp = np.cumsum(sorted_target)
    fp = np.cumsum(1 - sorted_target)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    recall_before = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_before) * precision))


def _binary_roc_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(np.int64, copy=False)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    positive_rank_sum = float(ranks[target == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return 0.0
    x = left[valid].astype(np.float64, copy=False)
    y = right[valid].astype(np.float64, copy=False)
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return 0.0
    x = left[valid]
    y = right[valid]
    x_rank = np.empty(x.size, dtype=np.float64)
    y_rank = np.empty(y.size, dtype=np.float64)
    x_rank[np.argsort(x, kind="mergesort")] = np.arange(x.size)
    y_rank[np.argsort(y, kind="mergesort")] = np.arange(y.size)
    return _correlation(x_rank, y_rank)


def _hybrid_score(metrics: dict[str, float]) -> float:
    inverse_mae = min(max(1.0 - metrics["policy_position_mae"], 0.0), 1.0)
    rank_ic = min(max(metrics["forecast_rank_information_coefficient"], 0.0), 1.0)
    calibration_error = (
        abs(metrics["quantile_q10_coverage"] - 0.10)
        + abs(metrics["quantile_q50_coverage"] - 0.50)
        + abs(metrics["quantile_q90_coverage"] - 0.90)
    ) / 3.0
    calibration_quality = min(max(1.0 - calibration_error / 0.10, 0.0), 1.0)
    return (
        0.30 * metrics["actionable_pr_auc"]
        + 0.20 * metrics["direction_macro_f1"]
        + 0.15 * metrics["policy_intent_macro_f1"]
        + 0.10 * metrics["policy_position_sign_accuracy"]
        + 0.10 * inverse_mae
        + 0.10 * rank_ic
        + 0.05 * calibration_quality
    )


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    request: MarketHybridRunRequest,
    class_weights: torch.Tensor,
    actionable_pos_weights: torch.Tensor,
    loss_weights: MarketHybridLossWeights,
) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in _LOSS_KEYS}
    direction_true: list[np.ndarray] = []
    direction_predicted: list[np.ndarray] = []
    actionable_true: list[np.ndarray] = []
    actionable_score: list[np.ndarray] = []
    position_true: list[np.ndarray] = []
    position_predicted: list[np.ndarray] = []
    intent_true: list[np.ndarray] = []
    intent_predicted: list[np.ndarray] = []
    forecast_true: list[np.ndarray] = []
    forecast_quantiles: list[np.ndarray] = []
    batches = 0
    for (
        context,
        future,
        next_patch,
        targets,
        directions,
        policy_targets,
        _endpoints,
    ) in loader:
        context = context.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        next_patch = next_patch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        directions = directions.to(device, non_blocking=True)
        policy_targets = policy_targets.to(device, non_blocking=True)
        with _autocast(device, amp_enabled, amp_dtype):
            output = model(context, future)
            losses = compute_hybrid_losses(
                output,
                next_patch_targets=next_patch,
                forecast_targets=targets,
                direction_targets=directions,
                policy_targets=policy_targets,
                class_weights=class_weights,
                actionable_pos_weights=actionable_pos_weights,
                request=request,
                loss_weights=loss_weights,
            )
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        directions_np = directions.detach().cpu().numpy()
        valid = directions_np >= 0
        prediction_np = output.direction_logits.argmax(dim=-1).detach().cpu().numpy()
        direction_true.append(directions_np[valid])
        direction_predicted.append(prediction_np[valid])
        actionable_true.append((directions_np[valid] != 1).astype(np.int64))
        actionable_score.append(_float_numpy(output.actionable_logits.sigmoid())[valid])
        position_true.append(policy_targets[:, 0].detach().cpu().numpy())
        position_predicted.append(_float_numpy(output.target_position))
        intent_true.append(policy_targets[:, 2].long().detach().cpu().numpy())
        intent_predicted.append(
            output.execution_intent_logits.argmax(dim=-1).detach().cpu().numpy()
        )
        forecast_true.append(targets.detach().cpu().numpy().reshape(-1))
        forecast_quantiles.append(_float_numpy(output.quantiles).reshape(-1, 3))
        batches += 1
    model.train()
    if batches == 0:
        raise RuntimeError("validation loader yielded no batches")
    metrics = {key: value / batches for key, value in totals.items()}
    direction_true_np = np.concatenate(direction_true)
    direction_predicted_np = np.concatenate(direction_predicted)
    direction_matrix = _confusion(direction_true_np, direction_predicted_np, 3)
    (
        metrics["direction_accuracy"],
        metrics["direction_balanced_accuracy"],
        metrics["direction_macro_f1"],
        metrics["direction_mcc"],
    ) = _classification_metrics(direction_matrix)

    actionable_true_np = np.concatenate(actionable_true)
    actionable_score_np = np.concatenate(actionable_score)
    actionable_predicted_np = actionable_score_np >= request.training.actionable_threshold
    tp = int(((actionable_predicted_np == 1) & (actionable_true_np == 1)).sum())
    fp = int(((actionable_predicted_np == 1) & (actionable_true_np == 0)).sum())
    fn = int(((actionable_predicted_np == 0) & (actionable_true_np == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics["actionable_precision"] = precision
    metrics["actionable_recall"] = recall
    metrics["actionable_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
    metrics["actionable_pr_auc"] = _binary_pr_auc(actionable_true_np, actionable_score_np)
    metrics["actionable_roc_auc"] = _binary_roc_auc(actionable_true_np, actionable_score_np)
    metrics["actionable_positive_rate"] = float(actionable_true_np.mean())
    metrics["predicted_actionable_rate"] = float(actionable_predicted_np.mean())

    position_true_np = np.concatenate(position_true)
    position_predicted_np = np.concatenate(position_predicted)
    metrics["policy_position_mae"] = float(
        np.mean(np.abs(position_predicted_np - position_true_np))
    )
    metrics["policy_position_correlation"] = _correlation(
        position_predicted_np, position_true_np
    )
    nonzero = np.abs(position_true_np) >= request.policy.intent_deadband
    metrics["policy_position_sign_accuracy"] = (
        float(
            np.mean(
                np.sign(position_predicted_np[nonzero]) == np.sign(position_true_np[nonzero])
            )
        )
        if nonzero.any()
        else 0.0
    )
    metrics["policy_position_nonzero_mae"] = (
        float(np.mean(np.abs(position_predicted_np[nonzero] - position_true_np[nonzero])))
        if nonzero.any()
        else 0.0
    )

    intent_true_np = np.concatenate(intent_true)
    intent_predicted_np = np.concatenate(intent_predicted)
    intent_matrix = _confusion(intent_true_np, intent_predicted_np, 3)
    (
        metrics["policy_intent_accuracy"],
        metrics["policy_intent_balanced_accuracy"],
        metrics["policy_intent_macro_f1"],
        metrics["policy_intent_mcc"],
    ) = _classification_metrics(intent_matrix)

    forecast_true_np = np.concatenate(forecast_true)
    forecast_quantiles_np = np.concatenate(forecast_quantiles)
    median = forecast_quantiles_np[:, 1]
    metrics["forecast_information_coefficient"] = _correlation(median, forecast_true_np)
    metrics["forecast_rank_information_coefficient"] = _rank_correlation(
        median, forecast_true_np
    )
    metrics["quantile_q10_coverage"] = float(
        np.mean(forecast_true_np <= forecast_quantiles_np[:, 0])
    )
    metrics["quantile_q50_coverage"] = float(
        np.mean(forecast_true_np <= forecast_quantiles_np[:, 1])
    )
    metrics["quantile_q90_coverage"] = float(
        np.mean(forecast_true_np <= forecast_quantiles_np[:, 2])
    )
    metrics["quantile_crossing_rate"] = float(
        np.mean(
            (forecast_quantiles_np[:, 0] > forecast_quantiles_np[:, 1])
            | (forecast_quantiles_np[:, 1] > forecast_quantiles_np[:, 2])
        )
    )
    metrics["hybrid_score"] = _hybrid_score(metrics)
    return metrics


def _stage_payload(
    index: int,
    start: int,
    stage: MarketHybridTrainingStageConfig,
    step: int,
) -> dict[str, Any]:
    length = stage.end_step - start
    local = max(step - start, 0)
    return {
        "index": index,
        "name": stage.name,
        "local_step": local,
        "start_step": start,
        "end_step": stage.end_step,
        "progress": min(max(local / max(length, 1), 0.0), 1.0),
    }


def train_markethybrid(
    request: MarketHybridRunRequest,
    prepared_dir: Path,
    run_dir: Path,
    *,
    update_status: StatusUpdater,
) -> Path:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    _install_signal_handlers()
    training = request.training
    stages = _stages(request)
    metadata = load_prepared_metadata(prepared_dir)
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)

    device = _select_device(training.device)
    amp_enabled, amp_dtype, needs_scaler = _amp_settings(device, training.amp)
    samples = training.max_steps * training.batch_size * training.gradient_accumulation_steps
    train_dataset = MarketHybridWindowDataset(
        prepared_dir,
        request,
        "train",
        samples=samples,
        seed=training.seed,
        deterministic=False,
    )
    validation_dataset = MarketHybridWindowDataset(
        prepared_dir,
        request,
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
    raw_model = MarketHybrid(
        feature_dim=metadata.feature_dim,
        patch_size=metadata.patch_size,
        horizons=len(metadata.horizons_seconds),
        request=request,
        include_training_modules=True,
    ).to(device)
    model: torch.nn.Module = raw_model
    if training.compile == "on":
        if not hasattr(torch, "compile"):
            raise RuntimeError("this PyTorch build does not provide torch.compile")
        model = torch.compile(raw_model, mode=training.compile_mode)

    scaler: Any = None
    if device.type == "cuda" and needs_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    class_weights, actionable_pos_weights, label_diagnostics = _loss_weights(
        prepared_dir, metadata, request, device
    )

    start_step = 0
    best_loss = float("inf")
    best_hybrid = float("-inf")
    best_by_stage: dict[str, dict[str, float]] = {}
    resume_payload: dict[str, Any] | None = None
    resume_checkpoint = _latest_checkpoint(run_dir)
    should_resume = training.resume == "on" or (
        training.resume == "auto" and resume_checkpoint is not None
    )
    if should_resume:
        if resume_checkpoint is None:
            raise FileNotFoundError("resume was requested but no checkpoint exists")
        resume_payload = cast(
            dict[str, Any],
            torch.load(resume_checkpoint, map_location=device, weights_only=False),
        )
        checkpoint_schedule = resume_payload.get("schedule_hash")
        if (
            checkpoint_schedule is not None
            and checkpoint_schedule != training.schedule_hash()
            and not training.allow_schedule_change_on_resume
        ):
            raise RuntimeError(
                "MarketHybrid stage schedule differs from the checkpoint; set "
                "allow_schedule_change_on_resume=true only for an intentional migration"
            )
        raw_model.load_state_dict(resume_payload["model"])
        if scaler is not None and resume_payload.get("scaler") is not None:
            scaler.load_state_dict(resume_payload["scaler"])
        start_step = int(resume_payload.get("global_step", resume_payload["step"]))
        best_loss = float(
            resume_payload.get("best_loss", resume_payload.get("best_validation", best_loss))
        )
        best_hybrid = float(resume_payload.get("best_hybrid", best_hybrid))
        best_by_stage = dict(resume_payload.get("best_by_stage", {}))
        _restore_rng_state(resume_payload.get("rng_state"))

    stage_index, stage_start, stage = _stage_for_step(stages, min(start_step, training.max_steps - 1))
    module_counts = _configure_trainable_modules(raw_model, stage)
    optimizer = _build_optimizer(raw_model, stage, request, device)
    if resume_payload is not None and int(resume_payload.get("stage_index", stage_index)) == stage_index:
        try:
            optimizer.load_state_dict(resume_payload["optimizer"])
        except (ValueError, KeyError) as exc:
            if not training.allow_schedule_change_on_resume:
                raise RuntimeError(
                    "checkpoint optimizer is incompatible with the current stage"
                ) from exc

    counts = raw_model.parameter_counts()
    context_seconds = metadata.context_patches * metadata.patch_size * metadata.timeframe_seconds
    run_info = {
        "model_type": "markethybrid",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "python_platform": platform.platform(),
        "amp": str(amp_dtype).replace("torch.", "") if amp_enabled else "off",
        "compile": training.compile,
        "activation_checkpointing": request.model.activation_checkpointing,
        "parameter_counts": counts,
        "stage_trainable_parameter_counts": module_counts,
        "effective_batch_size": training.batch_size * training.gradient_accumulation_steps,
        "context_bars": metadata.context_patches * metadata.patch_size,
        "context_seconds": context_seconds,
        "primary_policy_horizon": max(
            request.policy.horizon_weights,
            key=request.policy.horizon_weights.__getitem__,
        ),
        "cost_threshold_bps": request.data.cost_threshold_bps,
        "label_diagnostics": label_diagnostics,
        "schedule_hash": training.schedule_hash(),
        "training_mode": training.mode,
        "stages": [item.model_dump(mode="json") for item in stages],
        "prepared_dir": str(prepared_dir),
    }
    atomic_json_write(run_info, run_dir / "run_info.json")
    update_status(
        state="training",
        progress=start_step / training.max_steps,
        step=start_step,
        max_steps=training.max_steps,
        stage=_stage_payload(stage_index, stage_start, stage, start_step),
        learning_rates={group["name"]: float(group["lr"]) for group in optimizer.param_groups},
        message=(
            f"training MarketHybrid stage {stage.name} on {device}; "
            f"trainable={sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}; "
            f"deployable={counts['deployable']:,}; effective batch="
            f"{run_info['effective_batch_size']}"
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
    stage_best_score = float("-inf")
    stage_no_improvement = 0
    early_stopped = False

    def save_payload(path: Path) -> None:
        _atomic_torch_save(
            _checkpoint_payload(
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                step=completed_step,
                stage_index=stage_index,
                stage_start=stage_start,
                stage=stage,
                best_loss=best_loss,
                best_hybrid=best_hybrid,
                best_by_stage=best_by_stage,
                request=request,
                metadata=metadata,
                prepared_dir=prepared_dir,
            ),
            path,
        )

    try:
        for step in range(start_step, training.max_steps):
            if _STOP_REQUESTED:
                raise KeyboardInterrupt
            next_stage_index, next_stage_start, next_stage = _stage_for_step(stages, step)
            if next_stage_index != stage_index:
                previous_name = stage.name
                previous_optimizer = optimizer
                stage_index, stage_start, stage = (
                    next_stage_index,
                    next_stage_start,
                    next_stage,
                )
                module_counts = _configure_trainable_modules(raw_model, stage)
                optimizer = _build_optimizer(
                    raw_model, stage, request, device, previous=previous_optimizer
                )
                optimizer.zero_grad(set_to_none=True)
                stage_best_score = float("-inf")
                stage_no_improvement = 0
                print(
                    "MarketHybrid stage transition:\n"
                    f"{previous_name} -> {stage.name}\n"
                    f"global step: {step}",
                    flush=True,
                )
                update_status(
                    state="training",
                    progress=step / training.max_steps,
                    step=step,
                    max_steps=training.max_steps,
                    stage=_stage_payload(stage_index, stage_start, stage, step),
                    message=(
                        f"MarketHybrid stage transition: {previous_name} -> {stage.name}; "
                        f"trainable={sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}"
                    ),
                )
            stage_length = stage.end_step - stage_start
            stage_local_step = step - stage_start
            base_learning_rate = _cosine_learning_rate(
                stage_local_step,
                warmup_steps=stage.warmup_steps,
                max_steps=stage_length,
                peak_lr=stage.learning_rate,
                min_lr=stage.min_learning_rate,
            )
            learning_rates = _learning_rates(optimizer, base_learning_rate)
            accumulated = {key: 0.0 for key in _LOSS_KEYS}
            for _ in range(training.gradient_accumulation_steps):
                (
                    context,
                    future,
                    next_patch,
                    targets,
                    directions,
                    policy_targets,
                    _endpoints,
                ) = next(batches)
                context = context.to(device, non_blocking=True)
                future = future.to(device, non_blocking=True)
                next_patch = next_patch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                directions = directions.to(device, non_blocking=True)
                policy_targets = policy_targets.to(device, non_blocking=True)
                with _autocast(device, amp_enabled, amp_dtype):
                    output = model(context, future)
                    losses = compute_hybrid_losses(
                        output,
                        next_patch_targets=next_patch,
                        forecast_targets=targets,
                        direction_targets=directions,
                        policy_targets=policy_targets,
                        class_weights=class_weights,
                        actionable_pos_weights=actionable_pos_weights,
                        request=request,
                        loss_weights=stage.loss_weights,
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
            trainable_parameters = [p for p in raw_model.parameters() if p.requires_grad]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
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
            momentum = _ema_momentum(
                completed_step,
                training.max_steps,
                request.jepa.ema_start,
                request.jepa.ema_end,
            )
            raw_model.update_target_encoder(momentum)
            recent_loss += accumulated["total"] / training.gradient_accumulation_steps

            if completed_step % training.log_interval == 0 or completed_step == training.max_steps:
                elapsed = max(time.perf_counter() - recent_started, 1e-9)
                interval_steps = min(training.log_interval, completed_step - start_step)
                patch_tokens = (
                    interval_steps
                    * training.batch_size
                    * training.gradient_accumulation_steps
                    * metadata.context_patches
                )
                log_metrics = {
                    "loss": recent_loss / max(interval_steps, 1),
                    "learning_rate": base_learning_rate,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "ema_momentum": momentum,
                    "patch_tokens_per_second": patch_tokens / elapsed,
                    "stage_name": stage.name,
                    "stage_index": stage_index,
                    "stage_local_step": completed_step - stage_start,
                }
                update_status(
                    state="training",
                    progress=completed_step / training.max_steps,
                    step=completed_step,
                    max_steps=training.max_steps,
                    stage=_stage_payload(stage_index, stage_start, stage, completed_step),
                    learning_rates=learning_rates,
                    message=(
                        f"MarketHybrid {stage.name} step "
                        f"{completed_step - stage_start:,}/{stage.end_step - stage_start:,}; "
                        f"global {completed_step:,}/{training.max_steps:,}"
                    ),
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
                    class_weights=class_weights,
                    actionable_pos_weights=actionable_pos_weights,
                    loss_weights=stage.loss_weights,
                )
                record: dict[str, Any] = {
                    "step": completed_step,
                    "global_step": completed_step,
                    "stage_name": stage.name,
                    "stage_index": stage_index,
                    "stage_local_step": completed_step - stage_start,
                    **validation,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                update_status(
                    state="training",
                    progress=completed_step / training.max_steps,
                    step=completed_step,
                    max_steps=training.max_steps,
                    stage=_stage_payload(stage_index, stage_start, stage, completed_step),
                    learning_rates=learning_rates,
                    message=f"MarketHybrid validation at step {completed_step:,}",
                    metrics=record,
                )
                stage_metrics = best_by_stage.setdefault(
                    stage.name, {"hybrid_score": float("-inf"), "total": float("inf")}
                )
                if validation["total"] < best_loss:
                    best_loss = validation["total"]
                    stage_metrics["total"] = min(stage_metrics["total"], best_loss)
                    save_payload(run_dir / "best_loss.pt")
                    if training.checkpoint_selection.metric == "total":
                        save_payload(run_dir / "best.pt")
                if validation["hybrid_score"] > best_hybrid:
                    best_hybrid = validation["hybrid_score"]
                    stage_metrics["hybrid_score"] = max(
                        stage_metrics["hybrid_score"], best_hybrid
                    )
                    save_payload(run_dir / "best_hybrid.pt")
                    if training.checkpoint_selection.metric == "hybrid_score":
                        save_payload(run_dir / "best.pt")
                minimum_delta = stage.early_stopping.minimum_delta
                if validation["hybrid_score"] > stage_best_score + minimum_delta:
                    stage_best_score = validation["hybrid_score"]
                    stage_no_improvement = 0
                else:
                    stage_no_improvement += 1
                if (
                    stage.early_stopping.enabled
                    and stage_no_improvement >= stage.early_stopping.patience_validations
                ):
                    early_stopped = True
                    print(
                        f"MarketHybrid early stopping in {stage.name} at step {completed_step}",
                        flush=True,
                    )
                    break

            if completed_step % training.checkpoint_interval == 0:
                save_payload(run_dir / f"checkpoint_step_{completed_step:08d}.pt")
    except KeyboardInterrupt:
        checkpoint_path: Path | None = None
        if completed_step > start_step:
            checkpoint_path = run_dir / f"checkpoint_step_{completed_step:08d}.pt"
            save_payload(checkpoint_path)
        update_status(
            state="stopped",
            progress=completed_step / max(training.max_steps, 1),
            step=completed_step,
            max_steps=training.max_steps,
            stage=_stage_payload(stage_index, stage_start, stage, completed_step),
            message="MarketHybrid training stopped; checkpoint saved",
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        )
        raise
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory. Reduce batch_size or enable activation checkpointing, "
            "then increase gradient_accumulation_steps to preserve effective batch size."
        ) from exc

    if early_stopped and stage.early_stopping.restore_best and (run_dir / "best_hybrid.pt").exists():
        best_payload = cast(
            dict[str, Any],
            torch.load(run_dir / "best_hybrid.pt", map_location=device, weights_only=False),
        )
        raw_model.load_state_dict(best_payload["model"])

    final_path = run_dir / "final.pt"
    save_payload(final_path)
    if not (run_dir / "best_loss.pt").exists():
        save_payload(run_dir / "best_loss.pt")
    if not (run_dir / "best_hybrid.pt").exists():
        save_payload(run_dir / "best_hybrid.pt")
    if not (run_dir / "best.pt").exists():
        selected = (
            run_dir / "best_hybrid.pt"
            if training.checkpoint_selection.metric == "hybrid_score"
            else run_dir / "best_loss.pt"
        )
        payload = cast(
            dict[str, Any], torch.load(selected, map_location="cpu", weights_only=False)
        )
        _atomic_torch_save(payload, run_dir / "best.pt")
    update_status(
        state="completed",
        progress=completed_step / training.max_steps,
        step=completed_step,
        max_steps=training.max_steps,
        stage=_stage_payload(stage_index, stage_start, stage, completed_step),
        message=(
            "MarketHybrid training completed"
            if not early_stopped
            else f"MarketHybrid policy fine-tuning early-stopped at step {completed_step:,}"
        ),
        checkpoint_path=str(run_dir / "best.pt"),
    )
    return final_path
