from __future__ import annotations

import math
from typing import cast

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from meteor_quant.domain import AccountSnapshot, Bar
from meteor_quant.strategies.sdk import StrategyContext
from meteor_quant.timesfm.runtime import ForecastRuntime
from meteor_quant.timesfm.strategy import TimesFMStrategy


class FakeTimesFMRuntime:
    device = "cuda:0"

    def __init__(self) -> None:
        self.forecast_count = 0

    def forecast(
        self,
        inputs: list[NDArray[np.float32]],
        horizon: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        self.forecast_count += len(inputs)
        point = np.empty((len(inputs), horizon), dtype=np.float32)
        quantiles = np.empty((len(inputs), horizon, 10), dtype=np.float32)
        for row, values in enumerate(inputs):
            last = float(values[-1])
            path = last * (1.0 + np.arange(1, horizon + 1, dtype=np.float32) * 0.0005)
            point[row] = path
            quantiles[row, :, 0] = path
            for index in range(1, 10):
                quantiles[row, :, index] = path * (0.998 + index * 0.0004)
        return point, quantiles


def _frame(row_count: int = 2_000, timeframe_seconds: int = 15) -> pl.LazyFrame:
    close = [100.0 + index * 0.01 + math.sin(index / 20.0) for index in range(row_count)]
    return pl.DataFrame(
        {
            "timestamp": [index * timeframe_seconds for index in range(row_count)],
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [1.0] * row_count,
        }
    ).lazy()


def test_timesfm_indicator_builds_causal_forecasts_and_targets(monkeypatch) -> None:
    fake = FakeTimesFMRuntime()
    monkeypatch.setattr(
        "meteor_quant.timesfm.strategy.get_timesfm_runtime",
        lambda _config: cast(ForecastRuntime, fake),
    )
    strategy = TimesFMStrategy(
        {
            "mode": "long_flat",
            "context_length": 64,
            "forecast_horizon_seconds": 60,
            "prediction_stride": 10,
            "max_predictions": 1_000,
            "batch_size": 8,
            "long_threshold_bps": 1.0,
        }
    )

    result = strategy.build_signals(_frame()).frame.collect()

    assert result.height == 2_000
    assert result.get_column("timesfm_forecast_price").is_not_null().sum() > 1_900
    assert result.get_column("timesfm_return_bps").is_not_null().sum() > 1_900
    assert result.get_column("timesfm_uncertainty_bps").is_not_null().sum() > 1_900
    assert result.get_column("target_fraction").max() == 1.0
    assert fake.forecast_count == math.ceil((2_000 - 64 + 1) / 10)


def test_timesfm_auto_stride_honors_prediction_cap(monkeypatch) -> None:
    fake = FakeTimesFMRuntime()
    monkeypatch.setattr(
        "meteor_quant.timesfm.strategy.get_timesfm_runtime",
        lambda _config: cast(ForecastRuntime, fake),
    )
    strategy = TimesFMStrategy(
        {
            "context_length": 64,
            "forecast_horizon_seconds": 60,
            "prediction_stride": 1,
            "max_predictions": 25,
            "auto_increase_stride": True,
        }
    )

    strategy.build_signals(_frame()).frame.collect()

    assert fake.forecast_count <= 25


def test_timesfm_live_indicator_reuses_forecast_until_stride(monkeypatch) -> None:
    fake = FakeTimesFMRuntime()
    monkeypatch.setattr(
        "meteor_quant.timesfm.strategy.get_timesfm_runtime",
        lambda _config: cast(ForecastRuntime, fake),
    )
    strategy = TimesFMStrategy(
        {
            "mode": "indicator_only",
            "context_length": 32,
            "forecast_horizon_seconds": 60,
            "prediction_stride": 4,
        }
    )
    bars = [
        Bar(index * 15, 100 + index, 101 + index, 99 + index, 100.5 + index, 1.0)
        for index in range(40)
    ]
    account = AccountSnapshot(0, 10_000, 0, 0, 100, 10_000, 0, 0, 0, 0)
    first = strategy.on_live_bar(
        StrategyContext("BTC/USD", tuple(bars[:32]), account, 31),
        bars[31],
    )
    second = strategy.on_live_bar(
        StrategyContext("BTC/USD", tuple(bars[:33]), account, 32),
        bars[32],
    )

    assert first.plots["timesfm_return_bps"] is not None
    assert second.plots == first.plots
    assert fake.forecast_count == 1


def test_timesfm_validates_compiled_context_and_horizon_limit() -> None:
    strategy = TimesFMStrategy(
        {
            "context_length": 16_000,
            "forecast_horizon_seconds": 500,
        }
    )

    with pytest.raises(ValueError, match="rounded context"):
        strategy._validate_model_limits(16_000, 500)


def test_timesfm_rejects_large_result_before_loading_close_arrays(monkeypatch) -> None:
    monkeypatch.setattr(
        pl.LazyFrame,
        "collect",
        _guard_timesfm_collect(pl.LazyFrame.collect),
    )
    strategy = TimesFMStrategy(
        {
            "context_length": 64,
            "forecast_horizon_seconds": 60,
            "maximum_output_rows": 10_000,
        }
    )

    with pytest.raises(ValueError, match="maximum_output_rows"):
        strategy.build_signals(_frame(row_count=10_001))


def _guard_timesfm_collect(original_collect):
    call_count = 0

    def guarded(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise AssertionError("close arrays were collected after the row cap was exceeded")
        return original_collect(self, *args, **kwargs)

    return guarded
