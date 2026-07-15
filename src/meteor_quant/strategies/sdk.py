from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeVar

import polars as pl
from pydantic import BaseModel, ConfigDict

from meteor_quant.domain import AccountSnapshot, Bar, IndicatorSpec, StrategyDecision

ParameterT = TypeVar("ParameterT", bound=BaseModel)


class EmptyParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(slots=True, frozen=True)
class StrategyContext:
    symbol: str
    bars: Sequence[Bar]
    account: AccountSnapshot
    bar_index: int

    @property
    def closes(self) -> list[float]:
        return [bar.close for bar in self.bars]

    @property
    def highs(self) -> list[float]:
        return [bar.high for bar in self.bars]

    @property
    def lows(self) -> list[float]:
        return [bar.low for bar in self.bars]

    @property
    def volumes(self) -> list[float]:
        return [bar.volume for bar in self.bars]


@dataclass(slots=True, frozen=True)
class SignalPlan:
    frame: pl.LazyFrame
    target_column: str = "target_fraction"
    reason_column: str | None = "signal_reason"
    plot_columns: tuple[str, ...] = ()


class StrategyPlugin(ABC):
    """One strategy definition shared by vectorized backtests and live paper trading."""

    key: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    parameter_model: ClassVar[type[BaseModel]] = EmptyParameters
    indicator_specs: ClassVar[tuple[IndicatorSpec, ...]] = ()
    minimum_bars: ClassVar[int] = 1
    required_timeframe_seconds: ClassVar[int | None] = None
    execution_mode: ClassVar[Literal["target_fraction"]] = "target_fraction"

    def __init__(self, parameters: Mapping[str, Any] | None = None) -> None:
        self.parameters: BaseModel = self.parameter_model.model_validate(parameters or {})

    @abstractmethod
    def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
        """Return a lazy frame containing target_fraction and optional plot columns."""
        raise NotImplementedError

    @abstractmethod
    def on_live_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
        """Evaluate one completed live bar using bounded history."""
        raise NotImplementedError

    def on_start(self, context: StrategyContext) -> None:
        del context

    def on_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
        return self.on_live_bar(context, bar)

    def signal_cache_identity(self) -> dict[str, Any]:
        """Return immutable inputs that determine historical signal output."""
        return {"strategy_key": self.key}

    @classmethod
    def metadata(cls, source: str) -> dict[str, Any]:
        return {
            "key": cls.key,
            "name": cls.name,
            "description": cls.description,
            "minimum_bars": cls.minimum_bars,
            "required_timeframe_seconds": cls.required_timeframe_seconds,
            "execution_mode": cls.execution_mode,
            "parameter_schema": cls.parameter_model.model_json_schema(),
            "indicator_specs": [item.to_dict() for item in cls.indicator_specs],
            "source": source,
        }
