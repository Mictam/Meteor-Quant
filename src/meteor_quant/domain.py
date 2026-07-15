from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class SessionStatus(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class Bar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_asset_volume: float = 0.0
    number_of_trades: int = 0
    taker_buy_base_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0
    symbol: str = "BTC/USD"
    timeframe_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class Quote:
    timestamp_ms: int
    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(slots=True, frozen=True)
class MarketTrade:
    timestamp_ms: int
    price: float
    quantity: float
    side: Side
    trade_id: int | None = None


@dataclass(slots=True, frozen=True)
class IndicatorSpec:
    key: str
    label: str
    pane: Literal["price", "indicator", "equity"] = "price"
    format: Literal["price", "number", "percent"] = "number"
    line_width: int = 2
    time_offset_seconds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StrategyDecision:
    target_fraction: float | None = None
    reason: str = ""
    plots: dict[str, float | None] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AccountSnapshot:
    timestamp: int
    cash: float
    quantity: float
    avg_entry_price: float
    mark_price: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    exposure_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class Fill:
    timestamp: int
    side: Side
    quantity: float
    price: float
    fee: float
    slippage_bps: float
    reason: str
    strategy_key: str
    position_after: float
    cash_after: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class EquityPoint:
    timestamp: int
    equity: float
    drawdown_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class PlotPoint:
    timestamp: int
    values: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "values": self.values}


@dataclass(slots=True)
class BacktestResult:
    id: str
    strategy_key: str
    strategy_name: str
    parameters: dict[str, Any]
    created_at: str
    bars: list[Bar]
    fills: list[Fill]
    equity: list[EquityPoint]
    plots: list[PlotPoint]
    indicator_specs: list[IndicatorSpec]
    metrics: dict[str, float | int | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "strategy_key": self.strategy_key,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "bars": [item.to_dict() for item in self.bars],
            "fills": [item.to_dict() for item in self.fills],
            "equity": [item.to_dict() for item in self.equity],
            "plots": [item.to_dict() for item in self.plots],
            "indicator_specs": [item.to_dict() for item in self.indicator_specs],
            "metrics": self.metrics,
        }

    @classmethod
    def now_iso(cls) -> str:
        return datetime.now(UTC).isoformat()
