from __future__ import annotations

from typing import Literal, cast

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from meteor_quant.domain import Bar, IndicatorSpec, StrategyDecision
from meteor_quant.strategies.sdk import SignalPlan, StrategyContext, StrategyPlugin


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    value = sum(values[:period]) / period
    for item in values[period:]:
        value = alpha * item + (1.0 - alpha) * value
    return value


def _rsi(values: list[float], period: int) -> float | None:
    if len(values) <= period:
        return None
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    gains = [max(delta, 0.0) for delta in deltas[-period:]]
    losses = [max(-delta, 0.0) for delta in deltas[-period:]]
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period
    if average_loss == 0:
        return 100.0
    rs = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + rs)


class SmaCrossParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fast: int = Field(default=20, ge=2, le=100_000)
    slow: int = Field(default=100, ge=3, le=500_000)
    long_target: float = Field(default=1.0, ge=0, le=10)
    short_target: float = Field(default=0.0, ge=-10, le=0)

    @model_validator(mode="after")
    def validate_periods(self) -> SmaCrossParameters:
        if self.fast >= self.slow:
            raise ValueError("fast must be smaller than slow")
        return self


class SmaCrossStrategy(StrategyPlugin):
    key = "sma_cross"
    name = "SMA crossover"
    description = "Long above the slow average and flat/short below it."
    parameter_model = SmaCrossParameters
    minimum_bars = 100
    indicator_specs = (
        IndicatorSpec("sma_fast", "Fast SMA", "price", "price"),
        IndicatorSpec("sma_slow", "Slow SMA", "price", "price"),
    )

    def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
        p = cast(SmaCrossParameters, self.parameters)
        enriched = frame.with_columns(
            pl.col("close").rolling_mean(window_size=p.fast).alias("sma_fast"),
            pl.col("close").rolling_mean(window_size=p.slow).alias("sma_slow"),
        ).with_columns(
            pl.when(pl.col("sma_fast").is_null() | pl.col("sma_slow").is_null())
            .then(None)
            .when(pl.col("sma_fast") > pl.col("sma_slow"))
            .then(pl.lit(p.long_target))
            .otherwise(pl.lit(p.short_target))
            .cast(pl.Float64)
            .alias("target_fraction"),
            pl.when(pl.col("sma_fast") > pl.col("sma_slow"))
            .then(pl.lit("fast_above_slow"))
            .otherwise(pl.lit("fast_below_slow"))
            .alias("signal_reason"),
        )
        return SignalPlan(enriched, plot_columns=("sma_fast", "sma_slow"))

    def on_live_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
        p = cast(SmaCrossParameters, self.parameters)
        fast = _sma(context.closes, p.fast)
        slow = _sma(context.closes, p.slow)
        if fast is None or slow is None:
            return StrategyDecision(plots={"sma_fast": fast, "sma_slow": slow})
        target = p.long_target if fast > slow else p.short_target
        reason = "fast_above_slow" if fast > slow else "fast_below_slow"
        return StrategyDecision(target, reason, {"sma_fast": fast, "sma_slow": slow})


class RsiParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period: int = Field(default=14, ge=2, le=10_000)
    oversold: float = Field(default=30.0, ge=0, le=50)
    overbought: float = Field(default=70.0, ge=50, le=100)
    long_target: float = Field(default=1.0, ge=0, le=10)
    exit_target: float = Field(default=0.0, ge=-10, le=10)


class RsiMeanReversionStrategy(StrategyPlugin):
    key = "rsi_mean_reversion"
    name = "RSI mean reversion"
    description = "Enter when RSI is oversold and exit when it is overbought."
    parameter_model = RsiParameters
    minimum_bars = 15
    indicator_specs = (IndicatorSpec("rsi", "RSI", "indicator", "number"),)

    def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
        p = cast(RsiParameters, self.parameters)
        delta = pl.col("close").diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
        enriched = (
            frame.with_columns(
                gain.ewm_mean(alpha=1.0 / p.period, adjust=False, min_samples=p.period).alias(
                    "avg_gain"
                ),
                loss.ewm_mean(alpha=1.0 / p.period, adjust=False, min_samples=p.period).alias(
                    "avg_loss"
                ),
            )
            .with_columns(
                (100.0 - 100.0 / (1.0 + pl.col("avg_gain") / pl.col("avg_loss")))
                .fill_nan(100.0)
                .alias("rsi")
            )
            .with_columns(
                pl.when(pl.col("rsi").is_null())
                .then(None)
                .when(pl.col("rsi") <= p.oversold)
                .then(pl.lit(p.long_target))
                .when(pl.col("rsi") >= p.overbought)
                .then(pl.lit(p.exit_target))
                .otherwise(None)
                .cast(pl.Float64)
                .forward_fill()
                .fill_null(p.exit_target)
                .alias("target_fraction"),
                pl.when(pl.col("rsi") <= p.oversold)
                .then(pl.lit("rsi_oversold"))
                .when(pl.col("rsi") >= p.overbought)
                .then(pl.lit("rsi_overbought"))
                .otherwise(pl.lit("hold"))
                .alias("signal_reason"),
            )
        )
        return SignalPlan(
            enriched.drop("avg_gain", "avg_loss"),
            plot_columns=("rsi",),
        )

    def on_live_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
        p = cast(RsiParameters, self.parameters)
        value = _rsi(context.closes, p.period)
        if value is None:
            return StrategyDecision(plots={"rsi": None})
        if value <= p.oversold:
            return StrategyDecision(p.long_target, "rsi_oversold", {"rsi": value})
        if value >= p.overbought:
            return StrategyDecision(p.exit_target, "rsi_overbought", {"rsi": value})
        return StrategyDecision(plots={"rsi": value})


class WaveTrendParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel_length: int = Field(default=10, ge=2, le=10_000)
    average_length: int = Field(default=21, ge=2, le=10_000)
    signal_length: int = Field(default=4, ge=2, le=1_000)
    oversold: float = Field(default=-53.0, le=0)
    overbought: float = Field(default=53.0, ge=0)
    long_target: float = Field(default=1.0, ge=0, le=10)
    exit_target: float = Field(default=0.0, ge=-10, le=10)
    entry_mode: Literal["cross", "cross_in_zone"] = "cross_in_zone"


class WaveTrendStrategy(StrategyPlugin):
    key = "wavetrend"
    name = "WaveTrend crossover"
    description = "LazyBear-style WaveTrend cross with optional overbought/oversold gating."
    parameter_model = WaveTrendParameters
    minimum_bars = 40
    indicator_specs = (
        IndicatorSpec("wt1", "WT1", "indicator", "number"),
        IndicatorSpec("wt2", "WT2", "indicator", "number"),
    )

    def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
        p = cast(WaveTrendParameters, self.parameters)
        ap = ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("ap")
        enriched = (
            frame.with_columns(ap)
            .with_columns(pl.col("ap").ewm_mean(span=p.channel_length, adjust=False).alias("esa"))
            .with_columns(
                (pl.col("ap") - pl.col("esa"))
                .abs()
                .ewm_mean(span=p.channel_length, adjust=False)
                .alias("d")
            )
            .with_columns(
                pl.when(pl.col("d") > 1e-12)
                .then((pl.col("ap") - pl.col("esa")) / (0.015 * pl.col("d")))
                .otherwise(0.0)
                .alias("ci")
            )
            .with_columns(pl.col("ci").ewm_mean(span=p.average_length, adjust=False).alias("wt1"))
            .with_columns(pl.col("wt1").rolling_mean(window_size=p.signal_length).alias("wt2"))
        )
        cross_up = (pl.col("wt1") > pl.col("wt2")) & (
            pl.col("wt1").shift(1) <= pl.col("wt2").shift(1)
        )
        cross_down = (pl.col("wt1") < pl.col("wt2")) & (
            pl.col("wt1").shift(1) >= pl.col("wt2").shift(1)
        )
        if p.entry_mode == "cross_in_zone":
            cross_up = cross_up & (pl.col("wt1") <= p.oversold)
            cross_down = cross_down & (pl.col("wt1") >= p.overbought)
        enriched = enriched.with_columns(
            pl.when(cross_up)
            .then(pl.lit(p.long_target))
            .when(cross_down)
            .then(pl.lit(p.exit_target))
            .otherwise(None)
            .cast(pl.Float64)
            .forward_fill()
            .fill_null(p.exit_target)
            .alias("target_fraction"),
            pl.when(cross_up)
            .then(pl.lit("wavetrend_cross_up"))
            .when(cross_down)
            .then(pl.lit("wavetrend_cross_down"))
            .otherwise(pl.lit("hold"))
            .alias("signal_reason"),
        )
        return SignalPlan(
            enriched.drop("ap", "esa", "d", "ci"),
            plot_columns=("wt1", "wt2"),
        )

    def on_live_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
        p = cast(WaveTrendParameters, self.parameters)
        minimum = max(p.channel_length * 3, p.average_length * 3, p.signal_length + 2)
        if len(context.bars) < minimum:
            return StrategyDecision(plots={"wt1": None, "wt2": None})
        ap = [(item.high + item.low + item.close) / 3.0 for item in context.bars]
        esa_values: list[float] = []
        alpha_esa = 2.0 / (p.channel_length + 1.0)
        esa = ap[0]
        for value in ap:
            esa = alpha_esa * value + (1.0 - alpha_esa) * esa
            esa_values.append(esa)
        deviations = [abs(value - mean) for value, mean in zip(ap, esa_values, strict=True)]
        d_values: list[float] = []
        d = deviations[0]
        for value in deviations:
            d = alpha_esa * value + (1.0 - alpha_esa) * d
            d_values.append(d)
        ci = [
            (value - mean) / (0.015 * dev) if dev else 0.0
            for value, mean, dev in zip(ap, esa_values, d_values, strict=True)
        ]
        alpha_tci = 2.0 / (p.average_length + 1.0)
        wt1_values: list[float] = []
        wt1 = ci[0]
        for value in ci:
            wt1 = alpha_tci * value + (1.0 - alpha_tci) * wt1
            wt1_values.append(wt1)
        wt2_values = [
            sum(wt1_values[max(0, index - p.signal_length + 1) : index + 1])
            / len(wt1_values[max(0, index - p.signal_length + 1) : index + 1])
            for index in range(len(wt1_values))
        ]
        current1, previous1 = wt1_values[-1], wt1_values[-2]
        current2, previous2 = wt2_values[-1], wt2_values[-2]
        up = current1 > current2 and previous1 <= previous2
        down = current1 < current2 and previous1 >= previous2
        if p.entry_mode == "cross_in_zone":
            up = up and current1 <= p.oversold
            down = down and current1 >= p.overbought
        plots: dict[str, float | None] = {"wt1": current1, "wt2": current2}
        if up:
            return StrategyDecision(p.long_target, "wavetrend_cross_up", plots)
        if down:
            return StrategyDecision(p.exit_target, "wavetrend_cross_down", plots)
        return StrategyDecision(plots=plots)


BUILTIN_STRATEGIES = (SmaCrossStrategy, RsiMeanReversionStrategy, WaveTrendStrategy)
