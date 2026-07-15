from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import polars as pl
from pydantic import BaseModel, Field, create_model

from meteor_quant.domain import Bar, StrategyDecision
from meteor_quant.marketlm.inference import MarketLMIndicator
from meteor_quant.marketlm.schemas import MarketLMIndicatorParameters
from meteor_quant.strategies.sdk import SignalPlan, StrategyContext, StrategyPlugin


def registered_marketlm_strategies(
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
            indicator = MarketLMIndicator(registration)
            classes.append(_strategy_class(registration, indicator))
        except Exception:
            continue
    return classes


def _strategy_class(
    registration: dict[str, Any],
    prototype: MarketLMIndicator,
) -> type[StrategyPlugin]:
    model_id = str(registration["model_id"])
    display_name = str(registration.get("display_name") or model_id)
    description = str(
        registration.get("description")
        or "Registered MarketLM model exposed as a forecast indicator."
    )
    strategy_key = f"marketlm_{model_id}".replace("-", "_")
    specs = prototype.indicator_specs()
    minimum_bars = prototype.minimum_bars
    required_timeframe_seconds = prototype.metadata.timeframe_seconds
    recommended_stride = max(1, round(60 / required_timeframe_seconds))
    dynamic_parameter_model = cast(
        type[BaseModel],
        create_model(
            f"MarketLMParameters_{model_id.replace('-', '_')}",
            __base__=MarketLMIndicatorParameters,
            prediction_stride=(
                int,
                Field(default=recommended_stride, ge=1, le=100_000),
            ),
        ),
    )

    class RegisteredMarketLMStrategy(StrategyPlugin):
        parameter_model = dynamic_parameter_model
        indicator_specs = specs

        def __init__(self, parameters: dict[str, Any] | None = None) -> None:
            super().__init__(parameters)
            self._indicator = MarketLMIndicator(registration)

        def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
            parameters = cast(MarketLMIndicatorParameters, self.parameters)
            return self._indicator.build_signal_plan(frame, parameters)

        def on_live_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
            del bar
            parameters = cast(MarketLMIndicatorParameters, self.parameters)
            return self._indicator.predict_live(list(context.bars), parameters)

        def signal_cache_identity(self) -> dict[str, Any]:
            checkpoint = self._indicator.checkpoint_path.stat()
            return {
                "model_id": model_id,
                "checkpoint_path": str(self._indicator.checkpoint_path.resolve()),
                "checkpoint_size": checkpoint.st_size,
                "checkpoint_mtime_ns": checkpoint.st_mtime_ns,
                "prepared_fingerprint": self._indicator.metadata.fingerprint,
                "primary_horizon_seconds": self._indicator.primary_horizon_seconds,
            }

    RegisteredMarketLMStrategy.key = strategy_key
    RegisteredMarketLMStrategy.name = f"MarketLM · {display_name}"
    RegisteredMarketLMStrategy.description = description
    RegisteredMarketLMStrategy.minimum_bars = minimum_bars
    RegisteredMarketLMStrategy.required_timeframe_seconds = required_timeframe_seconds
    RegisteredMarketLMStrategy.__name__ = f"RegisteredMarketLM_{model_id.replace('-', '_')}"
    return RegisteredMarketLMStrategy
