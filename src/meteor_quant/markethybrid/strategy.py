from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
from pydantic import BaseModel, Field, create_model

from meteor_quant.domain import Bar, StrategyDecision
from meteor_quant.markethybrid.inference import MarketHybridIndicator
from meteor_quant.markethybrid.schemas import (
    MarketHybridIndicatorParameters,
    MarketHybridMedianSignParameters,
)
from meteor_quant.strategies.sdk import SignalPlan, StrategyContext, StrategyPlugin


def registered_markethybrid_strategies(
    registered_dir: Path,
) -> list[type[StrategyPlugin]]:
    classes: list[type[StrategyPlugin]] = []
    if not registered_dir.exists():
        return classes
    for path in sorted(registered_dir.glob("*.json")):
        try:
            registration = cast(
                dict[str, Any],
                json.loads(path.read_text(encoding="utf-8")),
            )
            indicator = MarketHybridIndicator(registration)
            classes.append(_strategy_class(registration, indicator))
            if 300 in indicator.metadata.horizons_seconds:
                classes.append(_median_sign_strategy_class(registration, indicator))
        except Exception:
            continue
    return classes


def _strategy_class(
    registration: dict[str, Any],
    prototype: MarketHybridIndicator,
) -> type[StrategyPlugin]:
    model_id = str(registration["model_id"])
    display_name = str(registration.get("display_name") or model_id)
    description = str(
        registration.get("description")
        or "Registered MarketHybrid model with JEPA and learned policy heads."
    )
    strategy_key = f"markethybrid_{model_id}".replace("-", "_")
    specs = prototype.indicator_specs()
    minimum_bars = prototype.minimum_bars
    required_timeframe_seconds = prototype.metadata.timeframe_seconds
    recommended_stride = max(1, round(60 / required_timeframe_seconds))
    dynamic_parameter_model = cast(
        type[BaseModel],
        create_model(
            f"MarketHybridParameters_{model_id.replace('-', '_')}",
            __base__=MarketHybridIndicatorParameters,
            mode=(
                Literal[
                    "indicator_only",
                    "model_policy",
                    "forecast_long_flat",
                    "forecast_long_short",
                ],
                Field(default="indicator_only"),
            ),
            prediction_stride=(
                int,
                Field(default=recommended_stride, ge=1, le=100_000),
            ),
        ),
    )

    class RegisteredMarketHybridStrategy(StrategyPlugin):
        parameter_model = dynamic_parameter_model
        indicator_specs = specs

        def __init__(self, parameters: dict[str, Any] | None = None) -> None:
            super().__init__(parameters)
            self._indicator = MarketHybridIndicator(registration)

        def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
            parameters = cast(MarketHybridIndicatorParameters, self.parameters)
            return self._indicator.build_signal_plan(frame, parameters)

        def on_live_bar(
            self,
            context: StrategyContext,
            bar: Bar,
        ) -> StrategyDecision:
            del bar
            parameters = cast(MarketHybridIndicatorParameters, self.parameters)
            return self._indicator.predict_live(list(context.bars), parameters)

        def signal_cache_identity(self) -> dict[str, Any]:
            checkpoint = self._indicator.checkpoint_path.stat()
            return {
                "model_type": "markethybrid",
                "model_id": model_id,
                "checkpoint_path": str(self._indicator.checkpoint_path.resolve()),
                "checkpoint_size": checkpoint.st_size,
                "checkpoint_mtime_ns": checkpoint.st_mtime_ns,
                "prepared_fingerprint": self._indicator.metadata.fingerprint,
                "primary_horizon_seconds": self._indicator.primary_horizon_seconds,
            }

    RegisteredMarketHybridStrategy.key = strategy_key
    RegisteredMarketHybridStrategy.name = f"MarketHybrid · {display_name}"
    RegisteredMarketHybridStrategy.description = description
    RegisteredMarketHybridStrategy.minimum_bars = minimum_bars
    RegisteredMarketHybridStrategy.required_timeframe_seconds = required_timeframe_seconds
    RegisteredMarketHybridStrategy.__name__ = f"RegisteredMarketHybrid_{model_id.replace('-', '_')}"
    return RegisteredMarketHybridStrategy


def _median_sign_indicator_parameters(
    parameters: MarketHybridMedianSignParameters,
) -> MarketHybridIndicatorParameters:
    return MarketHybridIndicatorParameters(
        mode="median_sign_long_short",
        prediction_stride=parameters.prediction_stride,
        batch_size=parameters.batch_size,
        long_threshold_bps=parameters.median_deadband_bps,
        short_threshold_bps=-parameters.median_deadband_bps,
        long_target=parameters.long_target,
        short_target=parameters.short_target,
        flat_target=0.0,
        device=parameters.device,
    )


def _median_sign_strategy_class(
    registration: dict[str, Any],
    prototype: MarketHybridIndicator,
) -> type[StrategyPlugin]:
    model_id = str(registration["model_id"])
    display_name = str(registration.get("display_name") or model_id)
    strategy_key = f"markethybrid_{model_id}_median_sign_300s".replace("-", "_")
    horizon_seconds = 300
    specs = prototype.indicator_specs(horizon_seconds)
    minimum_bars = prototype.minimum_bars
    required_timeframe_seconds = prototype.metadata.timeframe_seconds

    class RegisteredMarketHybridMedianSignStrategy(StrategyPlugin):
        parameter_model = MarketHybridMedianSignParameters
        indicator_specs = specs

        def __init__(self, parameters: dict[str, Any] | None = None) -> None:
            super().__init__(parameters)
            self._indicator = MarketHybridIndicator(registration)

        def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
            parameters = cast(MarketHybridMedianSignParameters, self.parameters)
            return self._indicator.build_signal_plan(
                frame,
                _median_sign_indicator_parameters(parameters),
                horizon_seconds=horizon_seconds,
            )

        def on_live_bar(
            self,
            context: StrategyContext,
            bar: Bar,
        ) -> StrategyDecision:
            del bar
            parameters = cast(MarketHybridMedianSignParameters, self.parameters)
            return self._indicator.predict_live(
                list(context.bars),
                _median_sign_indicator_parameters(parameters),
                horizon_seconds=horizon_seconds,
            )

        def signal_cache_identity(self) -> dict[str, Any]:
            checkpoint = self._indicator.checkpoint_path.stat()
            return {
                "model_type": "markethybrid",
                "strategy_type": "median_sign_300s",
                "model_id": model_id,
                "checkpoint_path": str(self._indicator.checkpoint_path.resolve()),
                "checkpoint_size": checkpoint.st_size,
                "checkpoint_mtime_ns": checkpoint.st_mtime_ns,
                "prepared_fingerprint": self._indicator.metadata.fingerprint,
                "forecast_horizon_seconds": horizon_seconds,
            }

    RegisteredMarketHybridMedianSignStrategy.key = strategy_key
    RegisteredMarketHybridMedianSignStrategy.name = (
        f"MarketHybrid 300s Median Sign · {display_name}"
    )
    RegisteredMarketHybridMedianSignStrategy.description = (
        "Targets long whenever the 300-second median forecast is positive and short "
        "whenever it is negative. A same-side target is held without sending duplicate "
        "paper orders; no actionability, direction-confidence, or policy-confidence "
        "filter is applied."
    )
    RegisteredMarketHybridMedianSignStrategy.minimum_bars = minimum_bars
    RegisteredMarketHybridMedianSignStrategy.required_timeframe_seconds = (
        required_timeframe_seconds
    )
    RegisteredMarketHybridMedianSignStrategy.__name__ = (
        f"RegisteredMarketHybridMedianSign_{model_id.replace('-', '_')}"
    )
    return RegisteredMarketHybridMedianSignStrategy
