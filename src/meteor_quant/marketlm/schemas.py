from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IndicatorSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    parameters: dict[str, float | int | bool | str] = Field(default_factory=dict)


class MarketLMDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str = "btcusdt_1s"
    timeframe_seconds: int = Field(default=1, ge=1, le=86_400)
    start_timestamp: int | None = None
    end_timestamp: int | None = None
    indicators: list[IndicatorSelection] = Field(default_factory=list)
    horizons_seconds: list[int] = Field(default_factory=lambda: [5, 15, 30, 60, 300])
    patch_size: int = Field(default=8, ge=1, le=256)
    context_patches: int = Field(default=384, ge=2, le=4_096)
    train_fraction: float = Field(default=0.80, gt=0.1, lt=0.95)
    validation_fraction: float = Field(default=0.10, gt=0.01, lt=0.4)
    purge_seconds: int | None = Field(default=None, ge=0)
    cost_threshold_bps: float = Field(default=2.0, ge=0, le=1_000)

    @model_validator(mode="after")
    def validate_data(self) -> MarketLMDataConfig:
        if (
            self.start_timestamp is not None
            and self.end_timestamp is not None
            and self.start_timestamp >= self.end_timestamp
        ):
            raise ValueError("start_timestamp must be smaller than end_timestamp")
        if self.train_fraction + self.validation_fraction >= 0.98:
            raise ValueError("train_fraction + validation_fraction must leave a test split")
        if not self.horizons_seconds:
            raise ValueError("at least one forecast horizon is required")
        unique = sorted(set(self.horizons_seconds))
        if unique != self.horizons_seconds:
            raise ValueError("horizons_seconds must be unique and sorted")
        invalid = [
            value
            for value in self.horizons_seconds
            if value <= 0 or value % self.timeframe_seconds != 0
        ]
        if invalid:
            raise ValueError(
                "every forecast horizon must be a positive multiple of timeframe_seconds: "
                f"{invalid}"
            )
        purge = self.purge_seconds
        if purge is not None and purge < max(self.horizons_seconds):
            raise ValueError("purge_seconds must be at least the maximum forecast horizon")
        return self

    @property
    def context_bars(self) -> int:
        return self.patch_size * self.context_patches

    @property
    def horizon_steps(self) -> list[int]:
        return [value // self.timeframe_seconds for value in self.horizons_seconds]

    @property
    def effective_purge_seconds(self) -> int:
        return max(self.purge_seconds or 0, max(self.horizons_seconds))


class MarketLMModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d_model: int = Field(default=384, ge=32, le=4_096)
    n_layers: int = Field(default=12, ge=1, le=128)
    n_heads: int = Field(default=6, ge=1, le=128)
    mlp_hidden: int = Field(default=1_152, ge=64, le=32_768)
    dropout: float = Field(default=0.08, ge=0, lt=0.9)
    rope_base: float = Field(default=10_000.0, gt=1)

    @model_validator(mode="after")
    def validate_attention(self) -> MarketLMModelConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        return self


class MarketLMTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=40_000, ge=1, le=10_000_000)
    batch_size: int = Field(default=32, ge=1, le=4_096)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=4_096)
    learning_rate: float = Field(default=2e-4, gt=0, le=1)
    min_learning_rate: float = Field(default=2e-5, ge=0, le=1)
    warmup_steps: int = Field(default=2_000, ge=0)
    weight_decay: float = Field(default=0.10, ge=0, le=10)
    gradient_clip_norm: float = Field(default=1.0, gt=0, le=1_000)
    validation_interval: int = Field(default=500, ge=1)
    checkpoint_interval: int = Field(default=2_000, ge=1)
    log_interval: int = Field(default=20, ge=1)
    validation_windows: int = Field(default=4_096, ge=32, le=1_000_000)
    num_workers: int = Field(default=0, ge=0, le=32)
    prefetch_factor: int = Field(default=2, ge=1, le=16)
    autoregressive_loss_weight: float = Field(default=0.5, ge=0, le=100)
    forecast_loss_weight: float = Field(default=2.0, ge=0, le=100)
    direction_loss_weight: float = Field(default=0.5, ge=0, le=100)
    amp: Literal["auto", "bf16", "fp16", "off"] = "auto"
    compile: Literal["off", "on"] = "off"
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    resume: Literal["auto", "on", "off"] = "auto"
    seed: int = 20260713
    device: Literal["auto", "cuda", "cpu"] = "auto"

    @model_validator(mode="after")
    def validate_training(self) -> MarketLMTrainingConfig:
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        return self


class MarketLMRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="marketlm-custom", min_length=1, max_length=120)
    data: MarketLMDataConfig = Field(default_factory=MarketLMDataConfig)
    model: MarketLMModelConfig = Field(default_factory=MarketLMModelConfig)
    training: MarketLMTrainingConfig = Field(default_factory=MarketLMTrainingConfig)
    prepare_only: bool = False


class MarketLMRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    checkpoint: Literal["best", "final"] = "best"
    primary_horizon_seconds: int | None = None


class MarketLMIndicatorParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["indicator_only", "long_flat", "long_short"] = "indicator_only"
    prediction_stride: int = Field(default=1, ge=1, le=100_000)
    batch_size: int = Field(default=256, ge=1, le=8_192)
    long_threshold_bps: float = Field(default=2.0, ge=-10_000, le=10_000)
    short_threshold_bps: float = Field(default=-2.0, ge=-10_000, le=10_000)
    confidence_threshold: float = Field(default=0.55, ge=0, le=1)
    long_target: float = Field(default=1.0, ge=0, le=10)
    short_target: float = Field(default=-1.0, ge=-10, le=0)
    flat_target: float = Field(default=0.0, ge=-10, le=10)
    device: Literal["auto", "cuda", "cpu"] = "auto"


class MarketLMJobStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    name: str
    kind: Literal["prepare", "train"]
    state: Literal["queued", "preparing", "training", "completed", "failed", "stopping", "stopped"]
    created_at: str
    updated_at: str
    pid: int | None = None
    progress: float = 0.0
    message: str = ""
    step: int = 0
    max_steps: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    prepared_dir: str | None = None
    checkpoint_path: str | None = None
    error: str | None = None
