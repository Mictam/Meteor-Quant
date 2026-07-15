from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from meteor_quant.markethybrid.schemas import MarketHybridRunRequest
from meteor_quant.marketlm.dataset import (
    PreparedMarketLMMetadata,
    load_prepared_metadata,
)

SplitName = Literal["train", "validation", "test"]


class MarketHybridWindowDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    """Memory-mapped MarketHybrid windows with causal context and JEPA future patches."""

    def __init__(
        self,
        prepared_dir: str | Path,
        request: MarketHybridRunRequest,
        split: SplitName,
        *,
        samples: int,
        seed: int,
        deterministic: bool,
    ) -> None:
        self.prepared_dir = Path(prepared_dir)
        self.request = request
        self.metadata: PreparedMarketLMMetadata = load_prepared_metadata(self.prepared_dir)
        self.features = np.load(self.prepared_dir / "features.npy", mmap_mode="r")
        self.targets = np.load(self.prepared_dir / "targets.npy", mmap_mode="r")
        self.directions = np.load(self.prepared_dir / "directions.npy", mmap_mode="r")
        self.samples = int(samples)
        self.seed = int(seed)
        self.future_patches = max(request.jepa.target_patch_offsets)
        self.future_bars = self.future_patches * self.metadata.patch_size
        minimum, maximum = self.metadata.split_endpoints[split]
        self.endpoint_min = int(minimum)
        self.endpoint_max = min(
            int(maximum),
            len(self.features) - self.future_bars - 1,
        )
        if self.endpoint_min > self.endpoint_max:
            raise ValueError(
                f"{split} split has no endpoints with {self.future_patches} future patches"
            )
        if samples <= 0:
            raise ValueError("samples must be positive")
        if deterministic:
            count = min(self.samples, self.endpoint_max - self.endpoint_min + 1)
            self.fixed_endpoints = np.linspace(
                self.endpoint_min,
                self.endpoint_max,
                num=count,
                dtype=np.int64,
            )
        else:
            self.fixed_endpoints = None
        self._target_mean = torch.tensor(self.metadata.target_mean, dtype=torch.float32)
        self._target_std = torch.tensor(self.metadata.target_std, dtype=torch.float32)
        self._horizon_indexes = {
            horizon: self.metadata.horizons_seconds.index(horizon)
            for horizon in request.policy.horizon_weights
        }

    def __len__(self) -> int:
        return len(self.fixed_endpoints) if self.fixed_endpoints is not None else self.samples

    def _random_endpoint(self, index: int) -> int:
        value = (index + self.seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 31
        width = self.endpoint_max - self.endpoint_min + 1
        return self.endpoint_min + int(value % width)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        endpoint = (
            int(self.fixed_endpoints[index])
            if self.fixed_endpoints is not None
            else self._random_endpoint(index)
        )
        patch = self.metadata.patch_size
        context_count = self.metadata.context_patches
        start = endpoint - context_count * patch + 1
        stop = endpoint + self.future_bars + 1
        segment = np.array(self.features[start:stop], dtype=np.float32, copy=True)
        expected = (context_count + self.future_patches) * patch
        if segment.shape[0] != expected:
            raise RuntimeError(f"expected {expected} feature rows, got {segment.shape[0]}")
        patches = torch.from_numpy(segment).view(
            context_count + self.future_patches,
            patch,
            self.metadata.feature_dim,
        )
        context = patches[:context_count]
        future = patches[context_count:]
        next_patch_targets = torch.cat((context[1:], future[:1]), dim=0)
        forecast_targets = torch.from_numpy(
            np.array(self.targets[endpoint], dtype=np.float32, copy=True)
        )
        direction_targets = torch.from_numpy(
            np.array(self.directions[endpoint], dtype=np.int64, copy=True)
        )
        policy_targets = self._policy_targets(forecast_targets, direction_targets)
        return (
            context,
            future,
            next_patch_targets,
            forecast_targets,
            direction_targets,
            policy_targets,
            torch.tensor(endpoint, dtype=torch.int64),
        )

    def _policy_targets(
        self,
        normalized_returns: torch.Tensor,
        direction_targets: torch.Tensor,
    ) -> torch.Tensor:
        denormalized = normalized_returns * self._target_std + self._target_mean
        weighted_return = torch.zeros((), dtype=torch.float32)
        total_weight = 0.0
        for horizon, weight in self.request.policy.horizon_weights.items():
            weighted_return = (
                weighted_return + float(weight) * denormalized[self._horizon_indexes[horizon]]
            )
            total_weight += float(weight)
        weighted_return = weighted_return / max(total_weight, 1e-12)
        position = torch.clamp(
            weighted_return / self.request.policy.position_scale_bps,
            -1.0,
            1.0,
        )
        edge = weighted_return.abs() - self.request.data.cost_threshold_bps
        confidence = torch.sigmoid(edge / self.request.policy.confidence_temperature)
        intent = torch.tensor(1, dtype=torch.int64)
        if position < -self.request.policy.intent_deadband:
            intent = torch.tensor(0, dtype=torch.int64)
        elif position > self.request.policy.intent_deadband:
            intent = torch.tensor(2, dtype=torch.int64)
        if bool(torch.all(direction_targets < 0)):
            confidence = torch.zeros_like(confidence)
        return torch.stack((position, confidence, intent.float()))
