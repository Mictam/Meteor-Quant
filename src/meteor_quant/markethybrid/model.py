from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from meteor_quant.markethybrid.schemas import (
    MarketHybridLossWeights,
    MarketHybridPredictorConfig,
    MarketHybridRunRequest,
)
from meteor_quant.marketlm.model import RMSNorm, TransformerBlock, pinball_loss
from meteor_quant.marketlm.schemas import MarketLMModelConfig


class CausalMarketEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        patch_size: int,
        request: MarketHybridRunRequest,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.activation_checkpointing = request.model.activation_checkpointing
        patch_dimension = feature_dim * patch_size
        block_config = MarketLMModelConfig.model_validate(
            request.model.model_dump(exclude={"activation_checkpointing"})
        )
        self.patch_norm = nn.LayerNorm(patch_dimension)
        self.patch_embedding = nn.Linear(patch_dimension, request.model.d_model)
        self.embedding_dropout = nn.Dropout(request.model.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(block_config) for _ in range(request.model.n_layers)
        )
        self.final_norm = RMSNorm(request.model.d_model)
        self.apply(self._initialize)
        residual_scale = (2.0 * request.model.n_layers) ** -0.5
        for name, parameter in self.named_parameters():
            if name.endswith("attention.output.weight") or name.endswith("mlp.output.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 * residual_scale)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        patches: torch.Tensor,
        *,
        checkpoint_blocks: bool | None = None,
    ) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("expected [batch, sequence, patch, features]")
        batch, sequence, patch_size, feature_dim = patches.shape
        if patch_size != self.patch_size or feature_dim != self.feature_dim:
            raise ValueError(
                f"expected patch/features {self.patch_size}/{self.feature_dim}, "
                f"got {patch_size}/{feature_dim}"
            )
        flattened = patches.reshape(batch, sequence, patch_size * feature_dim)
        hidden = self.embedding_dropout(self.patch_embedding(self.patch_norm(flattened)))
        should_checkpoint = (
            self.activation_checkpointing if checkpoint_blocks is None else checkpoint_blocks
        )
        for block in self.blocks:
            if self.training and should_checkpoint and hidden.requires_grad:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.final_norm(hidden)


class PredictorBlock(nn.Module):
    def __init__(self, config: MarketHybridPredictorConfig) -> None:
        super().__init__()
        self.self_norm = RMSNorm(config.d_model)
        self.self_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_norm = RMSNorm(config.d_model)
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp_norm = RMSNorm(config.d_model)
        self.gate_and_value = nn.Linear(config.d_model, 2 * config.mlp_hidden, bias=False)
        self.output = nn.Linear(config.mlp_hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        queries = queries + self.dropout(attended)
        normalized = self.cross_norm(queries)
        attended, _ = self.cross_attention(
            normalized,
            context,
            context,
            need_weights=False,
        )
        queries = queries + self.dropout(attended)
        gate, value = self.gate_and_value(self.mlp_norm(queries)).chunk(2, dim=-1)
        return queries + self.dropout(self.output(F.silu(gate) * value))


class HorizonLatentPredictor(nn.Module):
    def __init__(
        self,
        encoder_dimension: int,
        target_count: int,
        config: MarketHybridPredictorConfig,
    ) -> None:
        super().__init__()
        self.context_projection = nn.Linear(
            encoder_dimension,
            config.d_model,
            bias=False,
        )
        self.queries = nn.Parameter(torch.empty(target_count, config.d_model))
        self.blocks = nn.ModuleList(PredictorBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.output_projection = nn.Linear(
            config.d_model,
            encoder_dimension,
            bias=False,
        )
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        projected = self.context_projection(context)
        queries = self.queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, projected)
        return self.output_projection(self.final_norm(queries))


@dataclass(slots=True)
class MarketHybridOutput:
    next_patch: torch.Tensor
    quantiles: torch.Tensor
    direction_logits: torch.Tensor
    actionable_logits: torch.Tensor
    target_position: torch.Tensor
    policy_confidence_logits: torch.Tensor
    execution_intent_logits: torch.Tensor
    predicted_latents: torch.Tensor | None = None
    target_latents: torch.Tensor | None = None


class MarketHybrid(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        patch_size: int,
        horizons: int,
        request: MarketHybridRunRequest,
        include_training_modules: bool = True,
    ) -> None:
        super().__init__()
        self.request = request
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.horizons = horizons
        dimension = request.model.d_model
        patch_dimension = feature_dim * patch_size
        self.online_encoder = CausalMarketEncoder(feature_dim, patch_size, request)
        self.next_patch_head = nn.Linear(dimension, patch_dimension)
        self.forecast_head = nn.Linear(dimension, horizons * 3)
        self.direction_head = nn.Linear(dimension, horizons * 3)
        self.actionable_head = nn.Linear(dimension, horizons)
        self.policy_position_head = nn.Linear(dimension, 1)
        self.policy_confidence_head = nn.Linear(dimension, 1)
        self.execution_intent_head = nn.Linear(dimension, 3)
        self.target_encoder: CausalMarketEncoder | None = None
        self.predictor: HorizonLatentPredictor | None = None
        if include_training_modules:
            self.target_encoder = copy.deepcopy(self.online_encoder)
            self.target_encoder.requires_grad_(False)
            self.target_encoder.activation_checkpointing = False
            self.predictor = HorizonLatentPredictor(
                dimension,
                len(request.jepa.target_patch_offsets),
                request.predictor,
            )
        self.apply(self._initialize_heads)

    @staticmethod
    def _initialize_heads(module: nn.Module) -> None:
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def train(self, mode: bool = True) -> MarketHybrid:
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self

    def forward(
        self,
        context: torch.Tensor,
        future: torch.Tensor | None = None,
    ) -> MarketHybridOutput:
        hidden = self.online_encoder(context)
        summary = hidden[:, -1]
        next_patch = self.next_patch_head(hidden).view(
            context.shape[0],
            context.shape[1],
            self.patch_size,
            self.feature_dim,
        )
        raw_quantiles = self.forecast_head(summary).view(
            context.shape[0],
            self.horizons,
            3,
        )
        median = raw_quantiles[..., 1]
        lower = median - F.softplus(raw_quantiles[..., 0])
        upper = median + F.softplus(raw_quantiles[..., 2])
        predicted_latents: torch.Tensor | None = None
        target_latents: torch.Tensor | None = None
        if future is not None:
            if self.predictor is None or self.target_encoder is None:
                raise RuntimeError(
                    "training modules are not available in this MarketHybrid instance"
                )
            predicted_latents = self.predictor(hidden)
            full = torch.cat((context, future), dim=1)
            with torch.no_grad():
                target_hidden = self.target_encoder(full, checkpoint_blocks=False)
                context_length = context.shape[1]
                indices = [
                    context_length + offset - 1 for offset in self.request.jepa.target_patch_offsets
                ]
                target_latents = target_hidden[:, indices]
        return MarketHybridOutput(
            next_patch=next_patch,
            quantiles=torch.stack((lower, median, upper), dim=-1),
            direction_logits=self.direction_head(summary).view(
                context.shape[0],
                self.horizons,
                3,
            ),
            actionable_logits=self.actionable_head(summary),
            target_position=torch.tanh(self.policy_position_head(summary)).squeeze(-1),
            policy_confidence_logits=self.policy_confidence_head(summary).squeeze(-1),
            execution_intent_logits=self.execution_intent_head(summary),
            predicted_latents=predicted_latents,
            target_latents=target_latents,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        if self.target_encoder is None:
            return
        for target, online in zip(
            self.target_encoder.parameters(),
            self.online_encoder.parameters(),
            strict=True,
        ):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for target, online in zip(
            self.target_encoder.buffers(),
            self.online_encoder.buffers(),
            strict=True,
        ):
            target.copy_(online)

    def deployment_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = (
            "online_encoder.",
            "next_patch_head.",
            "forecast_head.",
            "direction_head.",
            "actionable_head.",
            "policy_position_head.",
            "policy_confidence_head.",
            "execution_intent_head.",
        )
        return {
            name: value for name, value in self.state_dict().items() if name.startswith(prefixes)
        }

    def load_deployment_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=True)

    def parameter_counts(self) -> dict[str, int]:
        named = list(self.named_parameters())
        deployable_prefixes = (
            "online_encoder.",
            "next_patch_head.",
            "forecast_head.",
            "direction_head.",
            "actionable_head.",
            "policy_position_head.",
            "policy_confidence_head.",
            "execution_intent_head.",
        )
        return {
            "total_checkpoint": sum(parameter.numel() for _, parameter in named),
            "trainable": sum(
                parameter.numel() for _, parameter in named if parameter.requires_grad
            ),
            "deployable": sum(
                parameter.numel()
                for name, parameter in named
                if name.startswith(deployable_prefixes)
            ),
            "teacher": sum(
                parameter.numel() for name, parameter in named if name.startswith("target_encoder.")
            ),
            "predictor": sum(
                parameter.numel() for name, parameter in named if name.startswith("predictor.")
            ),
        }


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    dimension = matrix.shape[0]
    if dimension <= 1:
        return matrix.new_zeros(0)
    return matrix.flatten()[:-1].view(dimension - 1, dimension + 1)[:, 1:].flatten()


def jepa_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    latent_weight: float,
    variance_weight: float,
    covariance_weight: float,
) -> dict[str, torch.Tensor]:
    predicted_normalized = F.layer_norm(predicted, (predicted.shape[-1],))
    target_normalized = F.layer_norm(target.detach(), (target.shape[-1],))
    latent = F.smooth_l1_loss(predicted_normalized, target_normalized, beta=0.5)
    flat = predicted_normalized.reshape(-1, predicted_normalized.shape[-1])
    standard_deviation = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - standard_deviation).mean()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance_matrix = centered.T @ centered / max(1, flat.shape[0] - 1)
    off_diagonal = _off_diagonal(covariance_matrix)
    covariance = (
        off_diagonal.pow(2).sum() / max(1, flat.shape[1])
        if off_diagonal.numel()
        else flat.new_zeros(())
    )
    total = latent_weight * latent + variance_weight * variance + covariance_weight * covariance
    return {
        "total": total,
        "latent": latent,
        "variance": variance,
        "covariance": covariance,
    }


def weighted_direction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    losses = [
        F.cross_entropy(
            logits[:, horizon],
            targets[:, horizon],
            weight=class_weights[horizon],
            ignore_index=-1,
        )
        for horizon in range(logits.shape[1])
    ]
    return torch.stack(losses).mean()


def compute_hybrid_losses(
    output: MarketHybridOutput,
    *,
    next_patch_targets: torch.Tensor,
    forecast_targets: torch.Tensor,
    direction_targets: torch.Tensor,
    policy_targets: torch.Tensor,
    class_weights: torch.Tensor,
    actionable_pos_weights: torch.Tensor,
    request: MarketHybridRunRequest,
    loss_weights: MarketHybridLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    autoregressive = F.smooth_l1_loss(
        output.next_patch,
        next_patch_targets,
        beta=0.5,
    )
    forecast = pinball_loss(output.quantiles, forecast_targets)
    direction = weighted_direction_loss(
        output.direction_logits,
        direction_targets,
        class_weights,
    )
    actionable_targets = (direction_targets != 1).to(output.actionable_logits.dtype)
    valid = direction_targets >= 0
    actionable_raw = F.binary_cross_entropy_with_logits(
        output.actionable_logits,
        actionable_targets,
        pos_weight=actionable_pos_weights,
        reduction="none",
    )
    actionable = actionable_raw[valid].mean() if valid.any() else actionable_raw.mean()
    if output.predicted_latents is None or output.target_latents is None:
        jepa = forecast.new_zeros(())
    else:
        jepa = jepa_losses(
            output.predicted_latents,
            output.target_latents,
            latent_weight=request.jepa.latent_loss_weight,
            variance_weight=request.jepa.variance_loss_weight,
            covariance_weight=request.jepa.covariance_loss_weight,
        )["total"]
    position_target = policy_targets[:, 0]
    confidence_target = policy_targets[:, 1]
    intent_target = policy_targets[:, 2].long()
    policy_position = F.smooth_l1_loss(
        output.target_position,
        position_target,
        beta=0.2,
    )
    policy_confidence = F.binary_cross_entropy_with_logits(
        output.policy_confidence_logits,
        confidence_target,
    )
    policy_intent = F.cross_entropy(output.execution_intent_logits, intent_target)
    if loss_weights is None:
        loss_weights = MarketHybridLossWeights(
            autoregressive=request.training.autoregressive_loss_weight,
            forecast=request.training.forecast_loss_weight,
            direction=request.training.direction_loss_weight,
            actionable=request.training.actionable_loss_weight,
            jepa=request.training.jepa_loss_weight,
            policy_position=request.training.policy_position_loss_weight,
            policy_confidence=request.training.policy_confidence_loss_weight,
            policy_intent=request.training.policy_intent_loss_weight,
        )
    total = (
        loss_weights.autoregressive * autoregressive
        + loss_weights.forecast * forecast
        + loss_weights.direction * direction
        + loss_weights.actionable * actionable
        + loss_weights.jepa * jepa
        + loss_weights.policy_position * policy_position
        + loss_weights.policy_confidence * policy_confidence
        + loss_weights.policy_intent * policy_intent
    )
    return {
        "total": total,
        "autoregressive": autoregressive,
        "forecast": forecast,
        "direction": direction,
        "actionable": actionable,
        "jepa": jepa,
        "policy_position": policy_position,
        "policy_confidence": policy_confidence,
        "policy_intent": policy_intent,
    }
