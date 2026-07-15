"""Example strategy loaded dynamically by Meteor Quant."""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from meteor_quant.domain import Bar, IndicatorSpec, StrategyDecision
from meteor_quant.strategies.sdk import SignalPlan, StrategyContext, StrategyPlugin


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fast: int = Field(default=12, ge=2, le=100_000)
    slow: int = Field(default=48, ge=3, le=500_000)
    threshold_bps: float = Field(default=1.0, ge=0, le=1_000)
    target: float = Field(default=1.0, ge=0, le=10)

    @model_validator(mode="after")
    def validate_periods(self) -> Parameters:
        if self.fast >= self.slow:
            raise ValueError("fast must be smaller than slow")
        return self


class EmaImpulseStrategy(StrategyPlugin):
    key = "example_ema_impulse"
    name = "Example EMA impulse"
    description = (
        "Python plugin: enter when the fast EMA exceeds the slow EMA by a configurable threshold."
    )
    parameter_model = Parameters
    minimum_bars = 48
    indicator_specs = (
        IndicatorSpec("ema_fast", "Fast EMA", "price", "price"),
        IndicatorSpec("ema_slow", "Slow EMA", "price", "price"),
    )

    def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
        p = self.parameters
        enriched = (
            frame.with_columns(
                pl.col("close").ewm_mean(span=p.fast, adjust=False).alias("ema_fast"),
                pl.col("close").ewm_mean(span=p.slow, adjust=False).alias("ema_slow"),
            )
            .with_columns(
                ((pl.col("ema_fast") / pl.col("ema_slow") - 1.0) * 10_000.0).alias("ema_spread_bps")
            )
            .with_columns(
                pl.when(pl.col("ema_spread_bps") > p.threshold_bps)
                .then(pl.lit(p.target))
                .otherwise(pl.lit(0.0))
                .alias("target_fraction"),
                pl.when(pl.col("ema_spread_bps") > p.threshold_bps)
                .then(pl.lit("ema_impulse_long"))
                .otherwise(pl.lit("ema_impulse_flat"))
                .alias("signal_reason"),
            )
        )
        return SignalPlan(enriched, plot_columns=("ema_fast", "ema_slow"))

    def on_live_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
        p = self.parameters
        closes = context.closes
        if len(closes) < p.slow:
            return StrategyDecision(plots={"ema_fast": None, "ema_slow": None})

        def ema(values: list[float], period: int) -> float:
            alpha = 2.0 / (period + 1.0)
            value = values[0]
            for item in values[1:]:
                value = alpha * item + (1.0 - alpha) * value
            return value

        fast = ema(closes[-p.slow :], p.fast)
        slow = ema(closes[-p.slow :], p.slow)
        spread_bps = (fast / slow - 1.0) * 10_000.0
        target = p.target if spread_bps > p.threshold_bps else 0.0
        reason = "ema_impulse_long" if target else "ema_impulse_flat"
        return StrategyDecision(target, reason, {"ema_fast": fast, "ema_slow": slow})


STRATEGY = EmaImpulseStrategy
