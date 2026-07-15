from __future__ import annotations

import math
from dataclasses import dataclass

from meteor_quant.domain import AccountSnapshot, Fill, Quote, Side


@dataclass(slots=True, frozen=True)
class BrokerConfig:
    initial_equity: float = 10_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 1.5
    minimum_order_notional: float = 10.0
    allow_short: bool = False
    max_leverage: float = 1.0

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fees and slippage cannot be negative")
        if self.minimum_order_notional < 0:
            raise ValueError("minimum_order_notional cannot be negative")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")


class PaperBroker:
    """Single-symbol deterministic paper broker using signed position accounting."""

    def __init__(self, config: BrokerConfig) -> None:
        self.config = config
        self.cash = config.initial_equity
        self.quantity = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0

    def snapshot(self, timestamp: int, mark_price: float) -> AccountSnapshot:
        if mark_price <= 0:
            raise ValueError("mark_price must be positive")
        equity = self.cash + self.quantity * mark_price
        unrealized = self._unrealized(mark_price)
        exposure = (self.quantity * mark_price / equity) if equity > 0 else 0.0
        return AccountSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            quantity=self.quantity,
            avg_entry_price=self.avg_entry_price,
            mark_price=mark_price,
            equity=equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            fees_paid=self.fees_paid,
            exposure_fraction=exposure,
        )

    def rebalance_to_fraction(
        self,
        *,
        target_fraction: float,
        quote: Quote,
        reason: str,
        strategy_key: str,
        timestamp: int,
    ) -> Fill | None:
        if not math.isfinite(target_fraction):
            raise ValueError("target_fraction must be finite")
        lower = -self.config.max_leverage if self.config.allow_short else 0.0
        target_fraction = min(self.config.max_leverage, max(lower, target_fraction))
        mark = quote.mid
        snapshot = self.snapshot(timestamp, mark)
        if snapshot.equity <= 0:
            return None
        target_quantity = target_fraction * snapshot.equity / mark
        delta = target_quantity - self.quantity
        if abs(delta * mark) < self.config.minimum_order_notional:
            return None
        side = Side.BUY if delta > 0 else Side.SELL
        slip = self.config.slippage_bps / 10_000.0
        execution_price = quote.ask * (1.0 + slip) if delta > 0 else quote.bid * (1.0 - slip)
        fee_rate = self.config.fee_bps / 10_000.0
        if delta > 0 and not self.config.allow_short and self.config.max_leverage <= 1.0:
            affordable = max(self.cash, 0.0) / (execution_price * (1.0 + fee_rate))
            delta = min(delta, affordable)
            if abs(delta * mark) < self.config.minimum_order_notional:
                return None
        fee = abs(delta * execution_price) * fee_rate
        previous_quantity = self.quantity
        previous_average = self.avg_entry_price

        self.cash -= delta * execution_price + fee
        self.quantity += delta
        self.fees_paid += fee
        self.realized_pnl += self._realized_on_fill(
            previous_quantity, previous_average, delta, execution_price
        )
        self.avg_entry_price = self._new_average(
            previous_quantity, previous_average, delta, execution_price
        )
        if abs(self.quantity) < 1e-12:
            self.quantity = 0.0
            self.avg_entry_price = 0.0

        return Fill(
            timestamp=timestamp,
            side=side,
            quantity=abs(delta),
            price=execution_price,
            fee=fee,
            slippage_bps=self.config.slippage_bps,
            reason=reason,
            strategy_key=strategy_key,
            position_after=self.quantity,
            cash_after=self.cash,
        )

    @staticmethod
    def _realized_on_fill(old_qty: float, old_average: float, delta: float, price: float) -> float:
        if old_qty == 0 or old_qty * delta >= 0:
            return 0.0
        closed_quantity = min(abs(old_qty), abs(delta))
        direction = 1.0 if old_qty > 0 else -1.0
        return closed_quantity * (price - old_average) * direction

    @staticmethod
    def _new_average(old_qty: float, old_average: float, delta: float, price: float) -> float:
        new_qty = old_qty + delta
        if new_qty == 0:
            return 0.0
        if old_qty == 0 or old_qty * delta > 0:
            return (abs(old_qty) * old_average + abs(delta) * price) / abs(new_qty)
        if old_qty * new_qty > 0:
            return old_average
        return price

    def _unrealized(self, mark_price: float) -> float:
        if self.quantity == 0:
            return 0.0
        direction = 1.0 if self.quantity > 0 else -1.0
        return abs(self.quantity) * (mark_price - self.avg_entry_price) * direction
