from __future__ import annotations

import math
from typing import Any

import polars as pl

from meteor_quant.marketlm.schemas import IndicatorSelection, MarketLMDataConfig

INDICATOR_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "key": "sma",
        "name": "Simple moving average",
        "description": "Close distance from a causal rolling mean.",
        "parameters": {"period": {"type": "integer", "default": 50, "minimum": 2, "maximum": 500_000}},
        "outputs": ["sma_distance_bps"],
    },
    {
        "key": "ema",
        "name": "Exponential moving average",
        "description": "Close distance from a causal exponential mean.",
        "parameters": {"period": {"type": "integer", "default": 50, "minimum": 2, "maximum": 500_000}},
        "outputs": ["ema_distance_bps"],
    },
    {
        "key": "rsi",
        "name": "RSI",
        "description": "Wilder-style relative strength index.",
        "parameters": {"period": {"type": "integer", "default": 14, "minimum": 2, "maximum": 100_000}},
        "outputs": ["rsi"],
    },
    {
        "key": "atr",
        "name": "ATR",
        "description": "Average true range normalized to basis points.",
        "parameters": {"period": {"type": "integer", "default": 14, "minimum": 2, "maximum": 100_000}},
        "outputs": ["atr_bps"],
    },
    {
        "key": "bollinger",
        "name": "Bollinger position",
        "description": "Rolling z-score and band width.",
        "parameters": {
            "period": {"type": "integer", "default": 20, "minimum": 2, "maximum": 100_000},
            "stddev": {"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 20.0},
        },
        "outputs": ["bollinger_z", "bollinger_width_bps"],
    },
    {
        "key": "macd",
        "name": "MACD",
        "description": "Fast/slow EMA spread, signal and histogram in basis points.",
        "parameters": {
            "fast": {"type": "integer", "default": 12, "minimum": 2, "maximum": 100_000},
            "slow": {"type": "integer", "default": 26, "minimum": 3, "maximum": 200_000},
            "signal": {"type": "integer", "default": 9, "minimum": 2, "maximum": 100_000},
        },
        "outputs": ["macd_bps", "macd_signal_bps", "macd_histogram_bps"],
    },
    {
        "key": "wavetrend",
        "name": "WaveTrend",
        "description": "LazyBear-style WT1/WT2 oscillator with zero-deviation protection.",
        "parameters": {
            "channel_length": {"type": "integer", "default": 10, "minimum": 2, "maximum": 100_000},
            "average_length": {"type": "integer", "default": 21, "minimum": 2, "maximum": 100_000},
            "signal_length": {"type": "integer", "default": 4, "minimum": 2, "maximum": 10_000},
        },
        "outputs": ["wavetrend_wt1", "wavetrend_wt2", "wavetrend_diff"],
    },
    {
        "key": "vwap",
        "name": "Rolling VWAP",
        "description": "Close distance from rolling volume-weighted typical price.",
        "parameters": {"period": {"type": "integer", "default": 60, "minimum": 2, "maximum": 500_000}},
        "outputs": ["vwap_distance_bps"],
    },
    {
        "key": "volume_zscore",
        "name": "Volume z-score",
        "description": "Causal rolling standard score of base volume.",
        "parameters": {"period": {"type": "integer", "default": 100, "minimum": 2, "maximum": 500_000}},
        "outputs": ["volume_zscore"],
    },
)

_CATALOG_BY_KEY = {item["key"]: item for item in INDICATOR_CATALOG}

BASE_FEATURE_NAMES: tuple[str, ...] = (
    "log_return_bps",
    "open_gap_bps",
    "high_from_close_bps",
    "low_from_close_bps",
    "candle_body_bps",
    "range_bps",
    "log_volume",
    "log_quote_volume",
    "log_trade_count",
    "taker_buy_imbalance",
    "time_second_sin",
    "time_second_cos",
    "time_day_sin",
    "time_day_cos",
    "time_weekday_sin",
    "time_weekday_cos",
)


def indicator_capabilities() -> list[dict[str, Any]]:
    return [dict(item) for item in INDICATOR_CATALOG]


def default_indicators() -> list[IndicatorSelection]:
    return [
        IndicatorSelection(key="rsi", parameters={"period": 14}),
        IndicatorSelection(
            key="wavetrend",
            parameters={"channel_length": 10, "average_length": 21, "signal_length": 4},
        ),
        IndicatorSelection(
            key="macd", parameters={"fast": 12, "slow": 26, "signal": 9}
        ),
    ]


def _number(
    selection: IndicatorSelection,
    key: str,
    *,
    integer: bool = False,
) -> float | int:
    definition = _CATALOG_BY_KEY[selection.key]
    parameter = definition["parameters"][key]
    value = selection.parameters.get(key, parameter["default"])
    parsed: float | int = int(value) if integer else float(value)
    minimum = parameter.get("minimum")
    maximum = parameter.get("maximum")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{selection.key}.{key} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{selection.key}.{key} must be <= {maximum}")
    return parsed


def validate_indicators(selections: list[IndicatorSelection]) -> None:
    seen: set[str] = set()
    for selection in selections:
        if selection.key not in _CATALOG_BY_KEY:
            raise ValueError(f"unknown MarketLM indicator: {selection.key}")
        if selection.key in seen:
            raise ValueError(f"indicator {selection.key} was selected more than once")
        seen.add(selection.key)
        definition = _CATALOG_BY_KEY[selection.key]
        unknown = set(selection.parameters) - set(definition["parameters"])
        if unknown:
            raise ValueError(f"unknown {selection.key} parameters: {sorted(unknown)}")
        for key, parameter in definition["parameters"].items():
            _number(selection, key, integer=parameter["type"] == "integer")
        if selection.key == "macd":
            fast = int(_number(selection, "fast", integer=True))
            slow = int(_number(selection, "slow", integer=True))
            if fast >= slow:
                raise ValueError("macd.fast must be smaller than macd.slow")


def feature_names_for(selections: list[IndicatorSelection]) -> list[str]:
    validate_indicators(selections)
    names = list(BASE_FEATURE_NAMES)
    for selection in selections:
        names.extend(_CATALOG_BY_KEY[selection.key]["outputs"])
    return names


def indicator_warmup_bars(selections: list[IndicatorSelection]) -> int:
    warmup = 2
    for selection in selections:
        if selection.key in {"sma", "ema", "rsi", "atr", "vwap", "volume_zscore"} or selection.key == "bollinger":
            warmup = max(warmup, int(_number(selection, "period", integer=True)) + 2)
        elif selection.key == "macd":
            slow = int(_number(selection, "slow", integer=True))
            signal = int(_number(selection, "signal", integer=True))
            warmup = max(warmup, slow + signal + 4)
        elif selection.key == "wavetrend":
            channel = int(_number(selection, "channel_length", integer=True))
            average = int(_number(selection, "average_length", integer=True))
            signal = int(_number(selection, "signal_length", integer=True))
            warmup = max(warmup, channel + average + signal + 4)
    return warmup


def _safe_log_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when((numerator > 0) & (denominator > 0))
        .then((numerator / denominator).log() * 10_000.0)
        .otherwise(None)
    )


def _base_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    previous_close = pl.col("close").shift(1)
    timestamp_dt = pl.from_epoch("timestamp", time_unit="s")
    seconds_of_minute = timestamp_dt.dt.second().cast(pl.Float64)
    seconds_of_day = (
        timestamp_dt.dt.hour().cast(pl.Float64) * 3_600.0
        + timestamp_dt.dt.minute().cast(pl.Float64) * 60.0
        + seconds_of_minute
    )
    weekday = timestamp_dt.dt.weekday().cast(pl.Float64)
    two_pi = 2.0 * math.pi
    return frame.with_columns(
        _safe_log_ratio(pl.col("close"), previous_close).alias("log_return_bps"),
        _safe_log_ratio(pl.col("open"), previous_close).alias("open_gap_bps"),
        _safe_log_ratio(pl.col("high"), pl.col("close")).alias("high_from_close_bps"),
        _safe_log_ratio(pl.col("low"), pl.col("close")).alias("low_from_close_bps"),
        _safe_log_ratio(pl.col("close"), pl.col("open")).alias("candle_body_bps"),
        _safe_log_ratio(pl.col("high"), pl.col("low")).alias("range_bps"),
        pl.col("volume").clip(lower_bound=0).log1p().alias("log_volume"),
        pl.col("quote_asset_volume").clip(lower_bound=0).log1p().alias("log_quote_volume"),
        pl.col("number_of_trades").cast(pl.Float64).clip(lower_bound=0).log1p().alias(
            "log_trade_count"
        ),
        pl.when(pl.col("volume") > 1e-12)
        .then(pl.col("taker_buy_base_volume") / pl.col("volume") * 2.0 - 1.0)
        .otherwise(0.0)
        .clip(-1.0, 1.0)
        .alias("taker_buy_imbalance"),
        (seconds_of_minute * two_pi / 60.0).sin().alias("time_second_sin"),
        (seconds_of_minute * two_pi / 60.0).cos().alias("time_second_cos"),
        (seconds_of_day * two_pi / 86_400.0).sin().alias("time_day_sin"),
        (seconds_of_day * two_pi / 86_400.0).cos().alias("time_day_cos"),
        (weekday * two_pi / 7.0).sin().alias("time_weekday_sin"),
        (weekday * two_pi / 7.0).cos().alias("time_weekday_cos"),
    )


def _add_sma(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    period = int(_number(selection, "period", integer=True))
    return frame.with_columns(
        ((pl.col("close") / pl.col("close").rolling_mean(period) - 1.0) * 10_000.0).alias(
            "sma_distance_bps"
        )
    )


def _add_ema(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    period = int(_number(selection, "period", integer=True))
    ema = pl.col("close").ewm_mean(span=period, adjust=False, min_samples=period)
    return frame.with_columns(((pl.col("close") / ema - 1.0) * 10_000.0).alias("ema_distance_bps"))


def _add_rsi(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    period = int(_number(selection, "period", integer=True))
    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    return (
        frame.with_columns(
            gain.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period).alias("__rsi_gain"),
            loss.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period).alias("__rsi_loss"),
        )
        .with_columns(
            pl.when(pl.col("__rsi_loss") <= 1e-18)
            .then(100.0)
            .otherwise(
                100.0
                - 100.0 / (1.0 + pl.col("__rsi_gain") / pl.col("__rsi_loss"))
            )
            .alias("rsi")
        )
        .drop("__rsi_gain", "__rsi_loss")
    )


def _add_atr(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    period = int(_number(selection, "period", integer=True))
    previous_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )
    atr = true_range.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)
    return frame.with_columns((atr / pl.col("close") * 10_000.0).alias("atr_bps"))


def _add_bollinger(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    period = int(_number(selection, "period", integer=True))
    stddev = float(_number(selection, "stddev"))
    mean = pl.col("close").rolling_mean(period)
    deviation = pl.col("close").rolling_std(period, ddof=0)
    return frame.with_columns(
        pl.when(deviation > 1e-18)
        .then((pl.col("close") - mean) / deviation)
        .otherwise(0.0)
        .alias("bollinger_z"),
        (2.0 * stddev * deviation / mean * 10_000.0).alias("bollinger_width_bps"),
    )


def _add_macd(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    fast = int(_number(selection, "fast", integer=True))
    slow = int(_number(selection, "slow", integer=True))
    signal = int(_number(selection, "signal", integer=True))
    fast_ema = pl.col("close").ewm_mean(span=fast, adjust=False, min_samples=fast)
    slow_ema = pl.col("close").ewm_mean(span=slow, adjust=False, min_samples=slow)
    return (
        frame.with_columns(((fast_ema - slow_ema) / pl.col("close") * 10_000.0).alias("macd_bps"))
        .with_columns(
            pl.col("macd_bps")
            .ewm_mean(span=signal, adjust=False, min_samples=signal)
            .alias("macd_signal_bps")
        )
        .with_columns(
            (pl.col("macd_bps") - pl.col("macd_signal_bps")).alias("macd_histogram_bps")
        )
    )


def _add_wavetrend(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    channel = int(_number(selection, "channel_length", integer=True))
    average = int(_number(selection, "average_length", integer=True))
    signal = int(_number(selection, "signal_length", integer=True))
    return (
        frame.with_columns(((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("__wt_ap"))
        .with_columns(
            pl.col("__wt_ap")
            .ewm_mean(span=channel, adjust=False, min_samples=channel)
            .alias("__wt_esa")
        )
        .with_columns(
            (pl.col("__wt_ap") - pl.col("__wt_esa"))
            .abs()
            .ewm_mean(span=channel, adjust=False, min_samples=channel)
            .alias("__wt_d")
        )
        .with_columns(
            pl.when(pl.col("__wt_d") > 1e-12)
            .then((pl.col("__wt_ap") - pl.col("__wt_esa")) / (0.015 * pl.col("__wt_d")))
            .otherwise(0.0)
            .alias("__wt_ci")
        )
        .with_columns(
            pl.col("__wt_ci")
            .ewm_mean(span=average, adjust=False, min_samples=average)
            .alias("wavetrend_wt1")
        )
        .with_columns(
            pl.col("wavetrend_wt1").rolling_mean(signal).alias("wavetrend_wt2")
        )
        .with_columns(
            (pl.col("wavetrend_wt1") - pl.col("wavetrend_wt2")).alias("wavetrend_diff")
        )
        .drop("__wt_ap", "__wt_esa", "__wt_d", "__wt_ci")
    )


def _add_vwap(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    period = int(_number(selection, "period", integer=True))
    typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    numerator = (typical * pl.col("volume")).rolling_sum(period)
    denominator = pl.col("volume").rolling_sum(period)
    vwap = pl.when(denominator > 1e-18).then(numerator / denominator).otherwise(None)
    return frame.with_columns(((pl.col("close") / vwap - 1.0) * 10_000.0).alias("vwap_distance_bps"))


def _add_volume_zscore(frame: pl.LazyFrame, selection: IndicatorSelection) -> pl.LazyFrame:
    period = int(_number(selection, "period", integer=True))
    mean = pl.col("volume").rolling_mean(period)
    deviation = pl.col("volume").rolling_std(period, ddof=0)
    return frame.with_columns(
        pl.when(deviation > 1e-18)
        .then((pl.col("volume") - mean) / deviation)
        .otherwise(0.0)
        .alias("volume_zscore")
    )


_ADDERS = {
    "sma": _add_sma,
    "ema": _add_ema,
    "rsi": _add_rsi,
    "atr": _add_atr,
    "bollinger": _add_bollinger,
    "macd": _add_macd,
    "wavetrend": _add_wavetrend,
    "vwap": _add_vwap,
    "volume_zscore": _add_volume_zscore,
}


def build_feature_frame(
    frame: pl.LazyFrame,
    config: MarketLMDataConfig,
    *,
    include_targets: bool,
) -> tuple[pl.LazyFrame, list[str], list[str]]:
    """Build causal features and optional future-return targets.

    Feature expressions only reference the current or earlier rows. Negative shifts are
    created exclusively for supervised targets.
    """

    validate_indicators(config.indicators)
    enriched = _base_features(frame)
    for selection in config.indicators:
        enriched = _ADDERS[selection.key](enriched, selection)
    feature_names = feature_names_for(config.indicators)
    target_names: list[str] = []
    if include_targets:
        expressions: list[pl.Expr] = []
        for seconds, steps in zip(config.horizons_seconds, config.horizon_steps, strict=True):
            name = f"target_return_h{seconds}_bps"
            target_names.append(name)
            future_timestamp = pl.col("timestamp").shift(-steps)
            future_close = pl.col("close").shift(-steps)
            expressions.append(
                pl.when(future_timestamp - pl.col("timestamp") == seconds)
                .then(_safe_log_ratio(future_close, pl.col("close")))
                .otherwise(None)
                .alias(name)
            )
        enriched = enriched.with_columns(expressions)
    return enriched, feature_names, target_names


def canonical_indicator_payload(selections: list[IndicatorSelection]) -> list[dict[str, Any]]:
    validate_indicators(selections)
    return [selection.model_dump(mode="json") for selection in selections]


def parameters_with_defaults(selection: IndicatorSelection) -> dict[str, float | int]:
    if selection.key not in _CATALOG_BY_KEY:
        raise ValueError(f"unknown MarketLM indicator: {selection.key}")
    result: dict[str, float | int] = {}
    for key, definition in _CATALOG_BY_KEY[selection.key]["parameters"].items():
        result[key] = _number(selection, key, integer=definition["type"] == "integer")
    return result


def normalized_selection(selection: IndicatorSelection) -> dict[str, Any]:
    return {"key": selection.key, "parameters": parameters_with_defaults(selection)}


def normalized_indicators(selections: list[IndicatorSelection]) -> list[dict[str, Any]]:
    return [normalized_selection(selection) for selection in selections]
