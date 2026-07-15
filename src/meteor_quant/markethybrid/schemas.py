from __future__ import annotations

from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from meteor_quant.marketlm.schemas import (
    IndicatorSelection,
    MarketLMDataConfig,
    MarketLMModelConfig,
    MarketLMTrainingConfig,
)


def optimized_indicators() -> list[IndicatorSelection]:
    return [
        IndicatorSelection(
            key="wavetrend",
            parameters={"channel_length": 10, "average_length": 21, "signal_length": 4},
        ),
        IndicatorSelection(key="rsi", parameters={"period": 14}),
        IndicatorSelection(
            key="macd", parameters={"fast": 12, "slow": 26, "signal": 9}
        ),
        IndicatorSelection(key="atr", parameters={"period": 14}),
        IndicatorSelection(key="rolling_vwap", parameters={"period": 60}),
        IndicatorSelection(key="volume_zscore", parameters={"period": 100}),
    ]


class MarketHybridDataConfig(MarketLMDataConfig):
    timeframe_seconds: int = Field(default=15, ge=1, le=86_400)
    indicators: list[IndicatorSelection] = Field(default_factory=optimized_indicators)
    horizons_seconds: list[int] = Field(default_factory=lambda: [30, 60, 180, 300, 900])
    patch_size: int = Field(default=8, ge=1, le=256)
    context_patches: int = Field(default=256, ge=2, le=4_096)
    train_fraction: float = Field(default=0.80, gt=0.1, lt=0.95)
    validation_fraction: float = Field(default=0.10, gt=0.01, lt=0.4)
    purge_seconds: int | None = Field(default=900, ge=0)
    cost_threshold_bps: float = Field(default=30.0, ge=0, le=1_000)


class MarketHybridModelConfig(MarketLMModelConfig):
    activation_checkpointing: bool = True


class MarketHybridPredictorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d_model: int = Field(default=256, ge=32, le=4_096)
    n_layers: int = Field(default=4, ge=1, le=64)
    n_heads: int = Field(default=4, ge=1, le=64)
    mlp_hidden: int = Field(default=768, ge=64, le=32_768)
    dropout: float = Field(default=0.08, ge=0, lt=0.9)

    @model_validator(mode="after")
    def validate_attention(self) -> MarketHybridPredictorConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError("predictor d_model must be divisible by n_heads")
        return self


class MarketHybridJEPAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_patch_offsets: list[int] = Field(default_factory=lambda: [1, 2, 4, 7])
    ema_start: float = Field(default=0.996, ge=0, lt=1)
    ema_end: float = Field(default=0.9999, ge=0, lt=1)
    latent_loss_weight: float = Field(default=1.0, ge=0, le=100)
    variance_loss_weight: float = Field(default=0.05, ge=0, le=100)
    covariance_loss_weight: float = Field(default=0.005, ge=0, le=100)

    @model_validator(mode="after")
    def validate_offsets(self) -> MarketHybridJEPAConfig:
        normalized = sorted(set(self.target_patch_offsets))
        if normalized != self.target_patch_offsets or not normalized or normalized[0] < 1:
            raise ValueError("target_patch_offsets must be positive, unique and sorted")
        if self.ema_end < self.ema_start:
            raise ValueError("ema_end cannot be smaller than ema_start")
        return self


class MarketHybridPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon_weights: dict[int, float] = Field(
        default_factory=lambda: {30: 0.5, 60: 1.0, 180: 2.0, 300: 4.0, 900: 2.0}
    )
    position_scale_bps: float = Field(default=30.0, gt=0, le=10_000)
    confidence_temperature: float = Field(default=2.0, gt=0, le=10_000)
    intent_deadband: float = Field(default=0.25, ge=0, lt=1)


class MarketHybridLossWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    autoregressive: float = Field(default=0.25, ge=0, le=100)
    forecast: float = Field(default=2.0, ge=0, le=100)
    direction: float = Field(default=0.75, ge=0, le=100)
    actionable: float = Field(default=1.5, ge=0, le=100)
    jepa: float = Field(default=0.5, ge=0, le=100)
    policy_position: float = Field(default=0.75, ge=0, le=100)
    policy_confidence: float = Field(default=0.5, ge=0, le=100)
    policy_intent: float = Field(default=0.75, ge=0, le=100)


class MarketHybridTrainableModules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    online_encoder: bool = True
    autoregressive_head: bool = True
    forecast_head: bool = True
    direction_head: bool = True
    actionable_head: bool = True
    jepa_predictor: bool = True
    policy_position_head: bool = True
    policy_confidence_head: bool = True
    policy_intent_head: bool = True


class MarketHybridLearningRateMultipliers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    online_encoder: float = Field(default=1.0, ge=0, le=100)
    autoregressive_head: float = Field(default=1.0, ge=0, le=100)
    forecast_head: float = Field(default=1.0, ge=0, le=100)
    direction_head: float = Field(default=1.0, ge=0, le=100)
    actionable_head: float = Field(default=1.0, ge=0, le=100)
    jepa_predictor: float = Field(default=1.0, ge=0, le=100)
    policy_position_head: float = Field(default=1.0, ge=0, le=100)
    policy_confidence_head: float = Field(default=1.0, ge=0, le=100)
    policy_intent_head: float = Field(default=1.0, ge=0, le=100)


class MarketHybridEarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    patience_validations: int = Field(default=12, ge=1, le=10_000)
    minimum_delta: float = Field(default=0.001, ge=0, le=1_000)
    restore_best: bool = True


class MarketHybridTrainingStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    end_step: int = Field(ge=1)
    learning_rate: float = Field(gt=0, le=1)
    min_learning_rate: float = Field(ge=0, le=1)
    warmup_steps: int = Field(default=0, ge=0)
    loss_weights: MarketHybridLossWeights
    trainable_modules: MarketHybridTrainableModules
    learning_rate_multipliers: MarketHybridLearningRateMultipliers = Field(
        default_factory=MarketHybridLearningRateMultipliers
    )
    early_stopping: MarketHybridEarlyStoppingConfig = Field(
        default_factory=MarketHybridEarlyStoppingConfig
    )

    @model_validator(mode="after")
    def validate_stage(self) -> MarketHybridTrainingStageConfig:
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("stage min_learning_rate cannot exceed learning_rate")
        if self.warmup_steps >= self.end_step:
            raise ValueError("stage warmup_steps must be smaller than end_step")
        return self


def optimized_training_stages() -> list[MarketHybridTrainingStageConfig]:
    return [
        MarketHybridTrainingStageConfig(
            name="representation_pretraining",
            end_step=8_000,
            learning_rate=2e-4,
            min_learning_rate=2e-5,
            warmup_steps=1_500,
            loss_weights=MarketHybridLossWeights(
                autoregressive=0.5,
                forecast=2.0,
                direction=0.5,
                actionable=0.5,
                jepa=1.0,
                policy_position=0.0,
                policy_confidence=0.0,
                policy_intent=0.0,
            ),
            trainable_modules=MarketHybridTrainableModules(
                policy_position_head=False,
                policy_confidence_head=False,
                policy_intent_head=False,
            ),
        ),
        MarketHybridTrainingStageConfig(
            name="joint_training",
            end_step=24_000,
            learning_rate=1.5e-4,
            min_learning_rate=1.5e-5,
            warmup_steps=750,
            loss_weights=MarketHybridLossWeights(),
            trainable_modules=MarketHybridTrainableModules(),
        ),
        MarketHybridTrainingStageConfig(
            name="policy_finetuning",
            end_step=40_000,
            learning_rate=8e-5,
            min_learning_rate=8e-6,
            warmup_steps=500,
            loss_weights=MarketHybridLossWeights(
                autoregressive=0.1,
                forecast=1.5,
                direction=1.0,
                actionable=2.0,
                jepa=0.25,
                policy_position=1.0,
                policy_confidence=0.75,
                policy_intent=1.0,
            ),
            trainable_modules=MarketHybridTrainableModules(jepa_predictor=False),
            learning_rate_multipliers=MarketHybridLearningRateMultipliers(
                online_encoder=0.2,
                autoregressive_head=0.25,
                forecast_head=0.5,
                direction_head=0.75,
                actionable_head=1.0,
                jepa_predictor=0.0,
                policy_position_head=1.0,
                policy_confidence_head=1.0,
                policy_intent_head=1.0,
            ),
            early_stopping=MarketHybridEarlyStoppingConfig(enabled=True),
        ),
    ]


class MarketHybridCheckpointSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Literal["hybrid_score", "total"] = "hybrid_score"
    mode: Literal["max", "min"] = "max"


class MarketHybridTrainingConfig(MarketLMTrainingConfig):
    mode: Literal["single_stage", "staged"] = "staged"
    max_steps: int = Field(default=40_000, ge=1, le=10_000_000)
    batch_size: int = Field(default=8, ge=1, le=4_096)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=4_096)
    learning_rate: float = Field(default=1.5e-4, gt=0, le=1)
    min_learning_rate: float = Field(default=1.5e-5, ge=0, le=1)
    warmup_steps: int = Field(default=2_000, ge=0)
    validation_windows: int = Field(default=16_384, ge=32, le=1_000_000)
    actionable_loss_weight: float = Field(default=1.5, ge=0, le=100)
    jepa_loss_weight: float = Field(default=0.5, ge=0, le=100)
    policy_position_loss_weight: float = Field(default=0.75, ge=0, le=100)
    policy_confidence_loss_weight: float = Field(default=0.5, ge=0, le=100)
    policy_intent_loss_weight: float = Field(default=0.75, ge=0, le=100)
    direction_class_weight_cap: float = Field(default=8.0, ge=1, le=1_000)
    actionable_pos_weight_cap: float = Field(default=12.0, ge=1, le=1_000)
    actionable_threshold: float = Field(default=0.60, ge=0, le=1)
    stages: list[MarketHybridTrainingStageConfig] = Field(
        default_factory=optimized_training_stages
    )
    checkpoint_selection: MarketHybridCheckpointSelection = Field(
        default_factory=MarketHybridCheckpointSelection
    )
    allow_schedule_change_on_resume: bool = False

    @model_validator(mode="after")
    def validate_hybrid_training(self) -> MarketHybridTrainingConfig:
        legacy_fields = {
            "max_steps", "learning_rate", "min_learning_rate", "warmup_steps",
            "autoregressive_loss_weight", "forecast_loss_weight",
            "direction_loss_weight", "actionable_loss_weight", "jepa_loss_weight",
            "policy_position_loss_weight", "policy_confidence_loss_weight",
            "policy_intent_loss_weight",
        }
        if "mode" not in self.model_fields_set and self.model_fields_set.intersection(legacy_fields):
            self.mode = "single_stage"
        if self.mode == "staged":
            if not self.stages:
                raise ValueError("staged training requires at least one stage")
            ends = [stage.end_step for stage in self.stages]
            if ends != sorted(set(ends)):
                raise ValueError("stage end_step values must be unique and increasing")
            if ends[-1] != self.max_steps:
                raise ValueError("the final stage end_step must equal max_steps")
            start = 0
            for stage in self.stages:
                length = stage.end_step - start
                if length <= 0:
                    raise ValueError("every stage must contain at least one step")
                if stage.warmup_steps >= length:
                    raise ValueError(
                        f"stage {stage.name} warmup_steps must be smaller than its length"
                    )
                start = stage.end_step
        return self

    def schedule_hash(self) -> str:
        payload = self.model_dump_json(
            include={"mode", "max_steps", "stages", "checkpoint_selection"}
        )
        return sha256(payload.encode("utf-8")).hexdigest()


class MarketHybridRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="btc-markethybrid-15s-quality", min_length=1, max_length=120)
    data: MarketLMDataConfig = Field(default_factory=MarketHybridDataConfig)
    model: MarketHybridModelConfig = Field(default_factory=MarketHybridModelConfig)
    predictor: MarketHybridPredictorConfig = Field(default_factory=MarketHybridPredictorConfig)
    jepa: MarketHybridJEPAConfig = Field(default_factory=MarketHybridJEPAConfig)
    policy: MarketHybridPolicyConfig = Field(default_factory=MarketHybridPolicyConfig)
    training: MarketHybridTrainingConfig = Field(default_factory=MarketHybridTrainingConfig)
    prepare_only: bool = False

    @model_validator(mode="after")
    def validate_hybrid(self) -> MarketHybridRunRequest:
        largest_future_bars = max(self.jepa.target_patch_offsets) * self.data.patch_size
        if largest_future_bars > max(self.data.horizon_steps):
            raise ValueError(
                "largest JEPA target offset exceeds the largest forecast horizon; "
                "increase horizons or reduce target_patch_offsets"
            )
        if not self.policy.horizon_weights:
            self.policy.horizon_weights = {
                horizon: float(index + 1)
                for index, horizon in enumerate(self.data.horizons_seconds)
            }
        invalid = set(self.policy.horizon_weights).difference(self.data.horizons_seconds)
        if invalid and "policy" not in self.model_fields_set:
            self.policy.horizon_weights = {
                horizon: float(index + 1)
                for index, horizon in enumerate(self.data.horizons_seconds)
            }
            invalid = set()
        if invalid:
            raise ValueError(
                f"policy horizon weights reference missing horizons: {sorted(invalid)}"
            )
        if any(weight < 0 for weight in self.policy.horizon_weights.values()):
            raise ValueError("policy horizon weights cannot be negative")
        if sum(self.policy.horizon_weights.values()) <= 0:
            raise ValueError("policy horizon weights must have positive total weight")
        return self


class MarketHybridRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    checkpoint: Literal[
        "best", "best_hybrid", "best_loss", "final"
    ] = "best"
    primary_horizon_seconds: int | None = None


class MarketHybridIndicatorParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "indicator_only",
        "model_policy",
        "forecast_long_flat",
        "forecast_long_short",
        "median_sign_long_short",
    ] = "indicator_only"
    prediction_stride: int = Field(default=1, ge=1, le=100_000)
    batch_size: int = Field(default=128, ge=1, le=8_192)
    long_threshold_bps: float = Field(default=4.0, ge=-10_000, le=10_000)
    short_threshold_bps: float = Field(default=-4.0, ge=-10_000, le=10_000)
    direction_confidence_threshold: float = Field(default=0.55, ge=0, le=1)
    actionable_probability_threshold: float = Field(default=0.55, ge=0, le=1)
    policy_confidence_threshold: float = Field(default=0.55, ge=0, le=1)
    policy_position_deadband: float = Field(default=0.15, ge=0, le=1)
    position_multiplier: float = Field(default=1.0, ge=0, le=10)
    maximum_absolute_target: float = Field(default=1.0, ge=0, le=10)
    long_target: float = Field(default=1.0, ge=0, le=10)
    short_target: float = Field(default=-1.0, ge=-10, le=0)
    flat_target: float = Field(default=0.0, ge=-10, le=10)
    device: Literal["auto", "cuda", "cpu"] = "auto"


class MarketHybridMedianSignParameters(BaseModel):
    """Trade only from the sign of the selected MarketHybrid median forecast."""

    model_config = ConfigDict(extra="forbid")

    prediction_stride: int = Field(
        default=1,
        ge=1,
        le=100_000,
        description="Evaluate every N completed bars.",
    )
    batch_size: int = Field(
        default=128,
        ge=1,
        le=8_192,
        description="Inference batch size used during historical signal generation.",
    )
    median_deadband_bps: float = Field(
        default=0.0,
        ge=0,
        le=10_000,
        description=(
            "No sign change is acted on while the 300-second median remains inside "
            "this symmetric basis-point deadband. Zero implements the exact sign rule."
        ),
    )
    long_target: float = Field(
        default=1.0,
        ge=0,
        le=10,
        description="Target exposure when the 300-second median is positive.",
    )
    short_target: float = Field(
        default=-1.0,
        ge=-10,
        le=0,
        description="Target exposure when the 300-second median is negative.",
    )
    device: Literal["auto", "cuda", "cpu"] = "auto"


class MarketHybridJobStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    name: str
    kind: Literal["prepare", "train"]
    state: Literal[
        "queued", "preparing", "training", "completed", "failed", "stopping", "stopped"
    ]
    created_at: str
    updated_at: str
    pid: int | None = None
    progress: float = 0.0
    message: str = ""
    step: int = 0
    max_steps: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    stage: dict[str, Any] | None = None
    learning_rates: dict[str, float] = Field(default_factory=dict)
    prepared_dir: str | None = None
    checkpoint_path: str | None = None
    error: str | None = None
