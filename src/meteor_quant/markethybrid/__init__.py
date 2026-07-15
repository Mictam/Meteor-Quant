"""MarketHybrid training, inference, and registration support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meteor_quant.markethybrid.model import MarketHybrid

__all__ = ["MarketHybrid"]


def __getattr__(name: str) -> Any:
    if name == "MarketHybrid":
        from meteor_quant.markethybrid.model import MarketHybrid

        return MarketHybrid
    raise AttributeError(name)
