from __future__ import annotations

from typing import Any

from meteor_quant.strategies.sdk import SignalPlan, StrategyContext, StrategyPlugin

__all__ = ["SignalPlan", "StrategyContext", "StrategyPlugin", "StrategyRegistry"]


def __getattr__(name: str) -> Any:
    if name == "StrategyRegistry":
        from meteor_quant.strategies.registry import StrategyRegistry

        return StrategyRegistry
    raise AttributeError(name)
