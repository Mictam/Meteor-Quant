from __future__ import annotations

from pathlib import Path

from meteor_quant.strategies.registry import StrategyRegistry


def test_builtin_and_user_plugins_load() -> None:
    plugin_dir = Path(__file__).resolve().parents[1] / "user_strategies"
    registry = StrategyRegistry(plugin_dir)
    metadata = registry.list_metadata()
    keys = {item["key"] for item in metadata["strategies"]}
    assert {"sma_cross", "rsi_mean_reversion", "wavetrend", "timesfm_2_5", "example_ema_impulse"} <= keys
    assert metadata["errors"] == []
