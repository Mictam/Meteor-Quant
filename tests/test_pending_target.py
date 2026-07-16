from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from aegis_quant_hybrid.engine import BacktestConfig, PreparedSignals, PythonArrowEngine


@pytest.mark.parametrize(
    "targets",
    [
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, None, None, None, None],
    ],
    ids=["forward-filled", "sparse"],
)
def test_unchanged_target_is_executed_only_once(
    tmp_path: Path, targets: list[float | None]
) -> None:
    timestamps = [1, 2, 3, 4, 5]
    prices = [100.0, 200.0, 50.0, 300.0, 80.0]
    signal_path = tmp_path / "constant-target.parquet"
    pq.write_table(
        pa.table(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [1.0] * len(prices),
                "target_fraction": pa.array(targets, type=pa.float64()),
            }
        ),
        signal_path,
    )
    prepared = PreparedSignals(
        path=signal_path,
        cache_key="constant-target",
        strategy_key="constant-target",
        parameters={},
        dataset_key="fixture",
        timeframe_seconds=1,
        start_timestamp=None,
        end_timestamp=None,
        plot_columns=(),
        row_count=len(prices),
        first_timestamp=timestamps[0],
        last_timestamp=timestamps[-1],
    )

    result = PythonArrowEngine().run(
        prepared,
        BacktestConfig(
            fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            minimum_order_notional=1.0,
        ),
    )

    assert result["metrics"]["fill_count"] == 1
    assert result["metrics"]["fills_returned"] == 1
    assert result["fills"][0]["timestamp"] == 2
