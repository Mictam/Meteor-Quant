from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from meteor_quant.strategies.builtin import WaveTrendStrategy


def _oscillating_frame(row_count: int = 4_000) -> pl.LazyFrame:
    close = [100.0 + 8.0 * math.sin(index / 18.0) for index in range(row_count)]
    return pl.DataFrame(
        {
            "timestamp": list(range(row_count)),
            "open": close,
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [1.0] * row_count,
        }
    ).lazy()


def test_wavetrend_vectorized_pipeline_does_not_propagate_initial_nan() -> None:
    strategy = WaveTrendStrategy({"entry_mode": "cross"})
    result = strategy.build_signals(_oscillating_frame()).frame.collect()

    assert result.select(pl.col("wt1").is_finite().sum()).item() > 3_900
    assert result.select(pl.col("wt2").is_finite().sum()).item() > 3_800
    assert result.select((pl.col("target_fraction") > 0).sum()).item() > 0
    assert result.select((pl.col("signal_reason") == "wavetrend_cross_up").sum()).item() > 0
    assert result.select((pl.col("signal_reason") == "wavetrend_cross_down").sum()).item() > 0


def test_wavetrend_default_zone_gating_still_emits_signals() -> None:
    strategy = WaveTrendStrategy({})
    result = strategy.build_signals(_oscillating_frame()).frame.collect()

    event_count = result.select(
        pl.col("signal_reason").is_in(["wavetrend_cross_up", "wavetrend_cross_down"]).sum()
    ).item()
    assert event_count > 0


def test_wavetrend_backtest_produces_orders(data_dir: Path) -> None:
    from meteor_quant.datasets import DatasetCatalog
    from meteor_quant.engine import BacktestConfig, HybridBacktestService

    catalog = DatasetCatalog(data_dir)
    service = HybridBacktestService(catalog, data_dir / "cache", data_dir / "results")
    strategy = WaveTrendStrategy({})
    prepared = service.prepare_signals(
        dataset_key="btcusdt_1s",
        strategy=strategy,
        timeframe_seconds=1,
        start_timestamp=None,
        end_timestamp=None,
    )
    result = service.run(
        prepared=prepared,
        strategy_name=strategy.name,
        config=BacktestConfig(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0),
        engine="python",
    )

    assert result["metrics"]["fill_count"] > 0
