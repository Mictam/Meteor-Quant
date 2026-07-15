from __future__ import annotations

import math
from typing import Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from meteor_quant.domain import Bar, IndicatorSpec, StrategyDecision
from meteor_quant.strategies.sdk import SignalPlan, StrategyContext, StrategyPlugin
from meteor_quant.timesfm.runtime import (
    DEFAULT_MODEL_ID,
    ForecastRuntime,
    RuntimeConfig,
    get_timesfm_runtime,
    installed_timesfm_version,
    model_source_identity,
)

_PLOT_KEYS = (
    "timesfm_forecast_price",
    "timesfm_q10_price",
    "timesfm_q90_price",
    "timesfm_return_bps",
    "timesfm_q10_return_bps",
    "timesfm_q90_return_bps",
    "timesfm_uncertainty_bps",
)


class TimesFMParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["indicator_only", "long_flat", "long_short"] = "indicator_only"
    model_id_or_path: str = Field(default=DEFAULT_MODEL_ID, min_length=1, max_length=1_024)
    context_length: int = Field(default=1_024, ge=32, le=16_000)
    forecast_horizon_seconds: int = Field(default=300, ge=1, le=86_400)
    prediction_stride: int = Field(
        default=20,
        ge=1,
        le=1_000_000,
        description="Bars between TimesFM forecasts; values are forward-filled between forecasts.",
    )
    max_predictions: int = Field(
        default=20_000,
        ge=1,
        le=1_000_000,
        description="Safety cap for model calls in one historical backtest.",
    )
    auto_increase_stride: bool = Field(
        default=True,
        description="Increase prediction_stride automatically when the selected range exceeds max_predictions.",
    )
    maximum_output_rows: int = Field(
        default=25_000_000,
        ge=10_000,
        le=500_000_000,
        description="Reject excessively large result frames before allocating forecast columns.",
    )
    batch_size: int = Field(default=16, ge=1, le=1_024)
    require_cuda: bool = True
    torch_compile: bool = False
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True
    local_files_only: bool = False
    huggingface_cache_dir: str = Field(default="", max_length=1_024)

    long_threshold_bps: float = Field(default=10.0, ge=-100_000, le=100_000)
    short_threshold_bps: float = Field(default=-10.0, ge=-100_000, le=100_000)
    require_quantile_confirmation: bool = False
    long_q10_floor_bps: float = Field(default=0.0, ge=-100_000, le=100_000)
    short_q90_ceiling_bps: float = Field(default=0.0, ge=-100_000, le=100_000)
    maximum_uncertainty_bps: float = Field(
        default=0.0,
        ge=0,
        le=1_000_000,
        description="Zero disables the q90-q10 uncertainty filter.",
    )
    long_target: float = Field(default=1.0, ge=0, le=10)
    short_target: float = Field(default=-1.0, ge=-10, le=0)
    flat_target: float = Field(default=0.0, ge=-10, le=10)

    @model_validator(mode="after")
    def validate_parameters(self) -> TimesFMParameters:
        if self.short_threshold_bps >= self.long_threshold_bps:
            raise ValueError("short_threshold_bps must be smaller than long_threshold_bps")
        return self


class TimesFMStrategy(StrategyPlugin):
    key = "timesfm_2_5"
    name = "TimesFM 2.5"
    description = (
        "Google TimesFM 2.5 zero-shot close-price forecast with median and q10/q90 bands. "
        "Use indicator-only mode or convert forecasts into target positions."
    )
    parameter_model = TimesFMParameters
    minimum_bars = 32
    indicator_specs = (
        IndicatorSpec("timesfm_forecast_price", "TimesFM median forecast", "price", "price"),
        IndicatorSpec("timesfm_q10_price", "TimesFM q10", "price", "price", 1),
        IndicatorSpec("timesfm_q90_price", "TimesFM q90", "price", "price", 1),
        IndicatorSpec("timesfm_return_bps", "TimesFM median return (bps)", "indicator", "number"),
        IndicatorSpec("timesfm_q10_return_bps", "TimesFM q10 return", "indicator", "number", 1),
        IndicatorSpec("timesfm_q90_return_bps", "TimesFM q90 return", "indicator", "number", 1),
        IndicatorSpec("timesfm_uncertainty_bps", "TimesFM uncertainty", "indicator", "number", 1),
    )

    def __init__(self, parameters: dict[str, object] | None = None) -> None:
        super().__init__(parameters)
        self._last_live_forecast_timestamp: int | None = None
        self._last_live_plots: dict[str, float | None] = self._empty_plots()

    @property
    def typed_parameters(self) -> TimesFMParameters:
        return cast(TimesFMParameters, self.parameters)

    def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
        p = self.typed_parameters
        row_count = int(frame.select(pl.len()).collect(engine="streaming").item())
        if row_count > p.maximum_output_rows:
            raise ValueError(
                f"TimesFM selected {row_count:,} rows, above maximum_output_rows="
                f"{p.maximum_output_rows:,}. Use a larger timeframe or a smaller date range."
            )
        base = frame.select("timestamp", "close").collect(engine="streaming")
        if base.height < p.context_length:
            return self._empty_signal_plan(frame)

        timestamps = base.get_column("timestamp").to_numpy().astype(np.int64, copy=False)
        closes = base.get_column("close").to_numpy().astype(np.float32, copy=False)
        timeframe_seconds = self._infer_timeframe(timestamps)
        horizon_steps = self._horizon_steps(timeframe_seconds)
        self._validate_model_limits(p.context_length, horizon_steps)
        stride = self._effective_stride(base.height, p.context_length)
        endpoints: NDArray[np.int64] = np.arange(
            p.context_length - 1, base.height, stride, dtype=np.int64
        )

        forecast_price = np.full(base.height, np.nan, dtype=np.float32)
        q10_price = np.full(base.height, np.nan, dtype=np.float32)
        q90_price = np.full(base.height, np.nan, dtype=np.float32)
        runtime = self._runtime(horizon_steps)

        for offset in range(0, endpoints.size, p.batch_size):
            endpoint_batch = endpoints[offset : offset + p.batch_size]
            windows = [
                np.ascontiguousarray(
                    closes[int(endpoint) - p.context_length + 1 : int(endpoint) + 1],
                    dtype=np.float32,
                )
                for endpoint in endpoint_batch
            ]
            point, quantiles = runtime.forecast(windows, horizon_steps)
            self._validate_forecast_shapes(point, quantiles, len(windows), horizon_steps)
            index = horizon_steps - 1
            forecast_price[endpoint_batch] = point[:, index]
            q10_price[endpoint_batch] = quantiles[:, index, 1]
            q90_price[endpoint_batch] = quantiles[:, index, 9]

        current = closes.astype(np.float64, copy=False)
        forecast_return = self._returns_bps(forecast_price, current)
        q10_return = self._returns_bps(q10_price, current)
        q90_return = self._returns_bps(q90_price, current)
        uncertainty = q90_return - q10_return

        predictions = pl.DataFrame(
            {
                "timestamp": timestamps,
                "timesfm_forecast_price": forecast_price,
                "timesfm_q10_price": q10_price,
                "timesfm_q90_price": q90_price,
                "timesfm_return_bps": forecast_return,
                "timesfm_q10_return_bps": q10_return,
                "timesfm_q90_return_bps": q90_return,
                "timesfm_uncertainty_bps": uncertainty,
            }
        ).with_columns(
            pl.col(name).fill_nan(None).forward_fill() for name in _PLOT_KEYS
        )
        predictions = self._add_targets(predictions)
        enriched = frame.join(predictions.lazy(), on="timestamp", how="left")
        return SignalPlan(enriched, plot_columns=_PLOT_KEYS)

    def on_start(self, context: StrategyContext) -> None:
        del context
        if self.typed_parameters.context_length > 5_000:
            raise ValueError(
                "TimesFM live paper mode stores at most 5,000 bars; set context_length <= 5000"
            )

    def on_live_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
        p = self.typed_parameters
        if len(context.bars) < p.context_length:
            return StrategyDecision(plots=self._empty_plots())
        timestamps: NDArray[np.int64] = np.asarray(
            [item.timestamp for item in context.bars[-min(len(context.bars), 64) :]],
            dtype=np.int64,
        )
        timeframe_seconds = self._infer_timeframe(timestamps)
        horizon_steps = self._horizon_steps(timeframe_seconds)
        self._validate_model_limits(p.context_length, horizon_steps)
        if (
            self._last_live_forecast_timestamp is not None
            and bar.timestamp - self._last_live_forecast_timestamp
            < p.prediction_stride * timeframe_seconds
        ):
            return StrategyDecision(plots=dict(self._last_live_plots))

        closes = np.asarray(
            [item.close for item in context.bars[-p.context_length :]],
            dtype=np.float32,
        )
        runtime = self._runtime(horizon_steps)
        point, quantiles = runtime.forecast([closes], horizon_steps)
        self._validate_forecast_shapes(point, quantiles, 1, horizon_steps)
        index = horizon_steps - 1
        current = float(closes[-1])
        point_price = float(point[0, index])
        q10 = float(quantiles[0, index, 1])
        q90 = float(quantiles[0, index, 9])
        median_bps = self._single_return_bps(point_price, current)
        q10_bps = self._single_return_bps(q10, current)
        q90_bps = self._single_return_bps(q90, current)
        plots: dict[str, float | None] = {
            "timesfm_forecast_price": point_price,
            "timesfm_q10_price": q10,
            "timesfm_q90_price": q90,
            "timesfm_return_bps": median_bps,
            "timesfm_q10_return_bps": q10_bps,
            "timesfm_q90_return_bps": q90_bps,
            "timesfm_uncertainty_bps": q90_bps - q10_bps,
        }
        target, reason = self._target_for_values(median_bps, q10_bps, q90_bps)
        self._last_live_forecast_timestamp = bar.timestamp
        self._last_live_plots = plots
        return StrategyDecision(target, reason, plots)

    def signal_cache_identity(self) -> dict[str, object]:
        p = self.typed_parameters
        return {
            "integration_schema": 1,
            "timesfm_version": installed_timesfm_version(),
            "model": model_source_identity(p.model_id_or_path),
        }

    def _runtime(self, horizon_steps: int) -> ForecastRuntime:
        p = self.typed_parameters
        return get_timesfm_runtime(
            RuntimeConfig(
                model_id_or_path=p.model_id_or_path,
                cache_dir=p.huggingface_cache_dir or None,
                local_files_only=p.local_files_only,
                max_context=p.context_length,
                max_horizon=horizon_steps,
                batch_size=p.batch_size,
                torch_compile=p.torch_compile,
                normalize_inputs=p.normalize_inputs,
                use_continuous_quantile_head=p.use_continuous_quantile_head,
                force_flip_invariance=p.force_flip_invariance,
                infer_is_positive=p.infer_is_positive,
                fix_quantile_crossing=p.fix_quantile_crossing,
                require_cuda=p.require_cuda,
            )
        )

    def _add_targets(self, predictions: pl.DataFrame) -> pl.DataFrame:
        p = self.typed_parameters
        available = pl.col("timesfm_return_bps").is_not_null()
        uncertainty_ok = (
            pl.lit(True)
            if p.maximum_uncertainty_bps <= 0
            else pl.col("timesfm_uncertainty_bps") <= p.maximum_uncertainty_bps
        )
        long_ok = available & (pl.col("timesfm_return_bps") >= p.long_threshold_bps) & uncertainty_ok
        short_ok = available & (pl.col("timesfm_return_bps") <= p.short_threshold_bps) & uncertainty_ok
        if p.require_quantile_confirmation:
            long_ok = long_ok & (pl.col("timesfm_q10_return_bps") >= p.long_q10_floor_bps)
            short_ok = short_ok & (pl.col("timesfm_q90_return_bps") <= p.short_q90_ceiling_bps)

        if p.mode == "indicator_only":
            target = pl.lit(p.flat_target)
            reason = pl.lit("timesfm_indicator_only")
        elif p.mode == "long_flat":
            target = pl.when(long_ok).then(pl.lit(p.long_target)).otherwise(pl.lit(p.flat_target))
            reason = pl.when(long_ok).then(pl.lit("timesfm_long")).otherwise(pl.lit("timesfm_flat"))
        else:
            target = (
                pl.when(long_ok)
                .then(pl.lit(p.long_target))
                .when(short_ok)
                .then(pl.lit(p.short_target))
                .otherwise(pl.lit(p.flat_target))
            )
            reason = (
                pl.when(long_ok)
                .then(pl.lit("timesfm_long"))
                .when(short_ok)
                .then(pl.lit("timesfm_short"))
                .otherwise(pl.lit("timesfm_flat"))
            )
        return predictions.with_columns(
            target.cast(pl.Float64).alias("target_fraction"),
            reason.alias("signal_reason"),
        )

    def _target_for_values(
        self,
        median_bps: float,
        q10_bps: float,
        q90_bps: float,
    ) -> tuple[float | None, str]:
        p = self.typed_parameters
        if p.mode == "indicator_only":
            return None, "timesfm_indicator_only"
        uncertainty_ok = (
            p.maximum_uncertainty_bps <= 0
            or q90_bps - q10_bps <= p.maximum_uncertainty_bps
        )
        long_ok = median_bps >= p.long_threshold_bps and uncertainty_ok
        short_ok = median_bps <= p.short_threshold_bps and uncertainty_ok
        if p.require_quantile_confirmation:
            long_ok = long_ok and q10_bps >= p.long_q10_floor_bps
            short_ok = short_ok and q90_bps <= p.short_q90_ceiling_bps
        if long_ok:
            return p.long_target, "timesfm_long"
        if p.mode == "long_short" and short_ok:
            return p.short_target, "timesfm_short"
        return p.flat_target, "timesfm_flat"

    def _effective_stride(self, row_count: int, context_length: int) -> int:
        p = self.typed_parameters
        candidate_rows = max(row_count - context_length + 1, 0)
        requested_count = math.ceil(candidate_rows / p.prediction_stride)
        if requested_count <= p.max_predictions:
            return p.prediction_stride
        minimum_stride = max(1, math.ceil(candidate_rows / p.max_predictions))
        if not p.auto_increase_stride:
            raise ValueError(
                f"TimesFM would run {requested_count:,} forecasts, above max_predictions="
                f"{p.max_predictions:,}. Set prediction_stride >= {minimum_stride}, select a "
                "smaller range, or enable auto_increase_stride."
            )
        return max(p.prediction_stride, minimum_stride)

    def _horizon_steps(self, timeframe_seconds: int) -> int:
        horizon = self.typed_parameters.forecast_horizon_seconds
        if horizon % timeframe_seconds != 0:
            raise ValueError(
                f"forecast_horizon_seconds={horizon} must be a multiple of the observed "
                f"{timeframe_seconds}-second timeframe"
            )
        return horizon // timeframe_seconds

    @staticmethod
    def _validate_model_limits(context_length: int, horizon_steps: int) -> None:
        if horizon_steps <= 0:
            raise ValueError("TimesFM horizon must contain at least one bar")
        compiled_context = math.ceil(context_length / 32) * 32
        compiled_horizon = math.ceil(horizon_steps / 128) * 128
        if compiled_horizon > 1_024:
            raise ValueError(
                "TimesFM 2.5 continuous quantile forecasts support at most 1,024 "
                "compiled horizon bars"
            )
        if compiled_context + compiled_horizon > 16_384:
            raise ValueError(
                "TimesFM 2.5 requires rounded context plus rounded horizon <= 16,384 "
                f"(got {compiled_context} + {compiled_horizon})"
            )

    @staticmethod
    def _infer_timeframe(timestamps: NDArray[np.int64]) -> int:
        if timestamps.size < 2:
            raise ValueError("TimesFM requires at least two timestamps")
        deltas = np.diff(timestamps)
        positive = deltas[deltas > 0]
        if positive.size == 0:
            raise ValueError("TimesFM timestamps are not strictly increasing")
        observed = int(np.median(positive))
        if observed <= 0:
            raise ValueError("could not infer a positive timeframe")
        return observed

    @staticmethod
    def _validate_forecast_shapes(
        point: NDArray[np.float32],
        quantiles: NDArray[np.float32],
        batch_size: int,
        horizon_steps: int,
    ) -> None:
        if point.shape != (batch_size, horizon_steps):
            raise RuntimeError(
                f"TimesFM point forecast shape {point.shape} does not match "
                f"({batch_size}, {horizon_steps})"
            )
        if quantiles.ndim != 3 or quantiles.shape[:2] != (batch_size, horizon_steps):
            raise RuntimeError(
                f"TimesFM quantile forecast shape {quantiles.shape} is invalid"
            )
        if quantiles.shape[2] < 10:
            raise RuntimeError("TimesFM quantile forecast must contain mean and q10..q90")

    @staticmethod
    def _returns_bps(
        forecast: NDArray[np.float32],
        current: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (forecast.astype(np.float64) / current - 1.0) * 10_000.0
        result[~np.isfinite(result)] = np.nan
        return result.astype(np.float32)

    @staticmethod
    def _single_return_bps(forecast: float, current: float) -> float:
        if not math.isfinite(forecast) or not math.isfinite(current) or current <= 0:
            return math.nan
        return (forecast / current - 1.0) * 10_000.0

    def _empty_signal_plan(self, frame: pl.LazyFrame) -> SignalPlan:
        enriched = frame.with_columns(
            *(pl.lit(None, dtype=pl.Float64).alias(name) for name in _PLOT_KEYS),
            pl.lit(self.typed_parameters.flat_target, dtype=pl.Float64).alias("target_fraction"),
            pl.lit("timesfm_warmup").alias("signal_reason"),
        )
        return SignalPlan(enriched, plot_columns=_PLOT_KEYS)

    @staticmethod
    def _empty_plots() -> dict[str, float | None]:
        return {name: None for name in _PLOT_KEYS}
