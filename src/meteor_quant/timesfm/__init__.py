"""TimesFM 2.5 zero-shot forecast indicator integration."""

from __future__ import annotations

from typing import Any

from meteor_quant.timesfm.runtime import timesfm_capabilities

__all__ = ["TimesFMStrategy", "timesfm_capabilities"]


def __getattr__(name: str) -> Any:
    if name == "TimesFMStrategy":
        from meteor_quant.timesfm.strategy import TimesFMStrategy

        return TimesFMStrategy
    raise AttributeError(name)
