from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from meteor_quant.marketlm.schemas import MarketLMModelConfig


@dataclass(slots=True)
class MarketLMOutput:
    next_patch: torch.Tensor
    quantiles: torch.Tensor
    direction_logits: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs * torch.rsqrt(inputs.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized * self.weight


def _rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = query.shape[-2]
    dimension = query.shape[-1]
    if dimension % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    positions = torch.arange(sequence, device=query.device, dtype=torch.float32)
    frequencies = 1.0 / (
        base
        ** (
            torch.arange(0, dimension, 2, device=query.device, dtype=torch.float32)
            / dimension
        )
    )
    angles = torch.outer(positions, frequencies)
    embedding = torch.cat((angles, angles), dim=-1)
    cosine = embedding.cos().to(dtype=query.dtype)[None, None, :, :]
    sine = embedding.sin().to(dtype=query.dtype)[None, None, :, :]
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MarketLMModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dimension = config.d_model // config.n_heads
        self.rope_base = config.rope_base
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, dimension = inputs.shape
        qkv = self.qkv(inputs).view(
            batch,
            sequence,
            3,
            self.n_heads,
            self.head_dimension,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rope(query, key, self.rope_base)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, dimension)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate_and_value = nn.Linear(dimension, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_and_value(inputs).chunk(2, dim=-1)
        return self.dropout(self.output(F.silu(gate) * value))


class TransformerBlock(nn.Module):
    def __init__(self, config: MarketLMModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.mlp_hidden, config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.residual_dropout(self.attention(self.attention_norm(inputs)))
        return inputs + self.mlp(self.mlp_norm(inputs))


class MarketLM(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        patch_size: int,
        horizons: int,
        config: MarketLMModelConfig,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.horizons = horizons
        self.config = config
        patch_dimension = patch_size * feature_dim

        self.patch_norm = nn.LayerNorm(patch_dimension)
        self.patch_embedding = nn.Linear(patch_dimension, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.next_patch_head = nn.Linear(config.d_model, patch_dimension)
        self.forecast_head = nn.Linear(config.d_model, horizons * 3)
        self.direction_head = nn.Linear(config.d_model, horizons * 3)
        self.apply(self._initialize)

        residual_scale = (2.0 * config.n_layers) ** -0.5
        for name, parameter in self.named_parameters():
            if name.endswith("attention.output.weight") or name.endswith("mlp.output.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 * residual_scale)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, patches: torch.Tensor) -> MarketLMOutput:
        if patches.ndim != 4:
            raise ValueError("expected patches with shape [batch, sequence, patch, features]")
        batch, sequence, patch_size, feature_dim = patches.shape
        if patch_size != self.patch_size or feature_dim != self.feature_dim:
            raise ValueError(
                f"expected patch/features {self.patch_size}/{self.feature_dim}, "
                f"got {patch_size}/{feature_dim}"
            )
        flattened = patches.reshape(batch, sequence, patch_size * feature_dim)
        hidden = self.embedding_dropout(self.patch_embedding(self.patch_norm(flattened)))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)

        next_patch = self.next_patch_head(hidden).view(
            batch,
            sequence,
            self.patch_size,
            self.feature_dim,
        )
        raw_quantiles = self.forecast_head(hidden[:, -1]).view(batch, self.horizons, 3)
        median = raw_quantiles[..., 1]
        lower = median - F.softplus(raw_quantiles[..., 0])
        upper = median + F.softplus(raw_quantiles[..., 2])
        quantiles = torch.stack((lower, median, upper), dim=-1)
        direction_logits = self.direction_head(hidden[:, -1]).view(batch, self.horizons, 3)
        return MarketLMOutput(
            next_patch=next_patch,
            quantiles=quantiles,
            direction_logits=direction_logits,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    expanded_targets = targets.unsqueeze(-1)
    errors = expanded_targets - predictions
    levels = torch.tensor(
        quantile_levels,
        device=predictions.device,
        dtype=predictions.dtype,
    )
    return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()


def compute_losses(
    output: MarketLMOutput,
    *,
    next_patch_targets: torch.Tensor,
    forecast_targets: torch.Tensor,
    direction_targets: torch.Tensor,
    autoregressive_weight: float,
    forecast_weight: float,
    direction_weight: float,
) -> dict[str, torch.Tensor]:
    autoregressive = F.smooth_l1_loss(
        output.next_patch,
        next_patch_targets,
        beta=0.5,
    )
    forecast = pinball_loss(output.quantiles, forecast_targets)
    direction = F.cross_entropy(
        output.direction_logits.reshape(-1, 3),
        direction_targets.reshape(-1),
        ignore_index=-1,
    )
    total = (
        autoregressive_weight * autoregressive
        + forecast_weight * forecast
        + direction_weight * direction
    )
    return {
        "total": total,
        "autoregressive": autoregressive,
        "forecast": forecast,
        "direction": direction,
    }
