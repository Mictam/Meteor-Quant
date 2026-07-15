from __future__ import annotations

from pathlib import Path

from meteor_quant.datasets import DatasetCatalog
from meteor_quant.engine import BacktestConfig, HybridBacktestService
from meteor_quant.strategies.builtin import SmaCrossStrategy


def test_signal_cache_and_streaming_backtest(data_dir: Path) -> None:
    catalog = DatasetCatalog(data_dir)
    service = HybridBacktestService(catalog, data_dir / "cache", data_dir / "results")
    strategy = SmaCrossStrategy({"fast": 10, "slow": 30})

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
        config=BacktestConfig(max_equity_points=500),
        engine="python",
    )

    assert result["engine"] == "python-arrow"
    assert result["metrics"]["bar_count"] == 1_200
    assert result["metrics"]["fill_count"] > 0
    assert len(result["equity"]) <= 502
    assert Path(result["result_path"]).exists()

    chart = service.chart_payload(prepared.cache_key, 300)
    assert chart["source_row_count"] == 1_200
    assert len(chart["bars"]) <= 302
    assert chart["plot_columns"] == ["sma_fast", "sma_slow"]


def test_fill_is_at_next_bar_open(data_dir: Path) -> None:
    catalog = DatasetCatalog(data_dir)
    service = HybridBacktestService(catalog, data_dir / "cache", data_dir / "results")
    strategy = SmaCrossStrategy({"fast": 2, "slow": 3})
    prepared = service.prepare_signals(
        dataset_key="btcusdt_1s",
        strategy=strategy,
        timeframe_seconds=1,
        start_timestamp=1_600_000_000,
        end_timestamp=1_600_000_020,
    )
    result = service.run(
        prepared=prepared,
        strategy_name=strategy.name,
        config=BacktestConfig(fee_bps=0, slippage_bps=0, spread_bps=0),
        engine="python",
    )
    first_fill = result["fills"][0]
    assert first_fill["timestamp"] >= 1_600_000_003
