from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from meteor_quant.broker import BrokerConfig, PaperBroker
from meteor_quant.domain import AccountSnapshot, Bar, Quote, SessionStatus
from meteor_quant.events import EventHub
from meteor_quant.market import (
    CandleAggregation,
    CandleAggregator,
    KrakenMarketStream,
    KrakenRestClient,
    aggregate_trades_to_bars,
)
from meteor_quant.storage import Database
from meteor_quant.strategies.sdk import StrategyContext, StrategyPlugin

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PaperSessionConfig:
    strategy_key: str
    parameters: dict[str, Any]
    symbol: str
    rest_pair: str
    timeframe_seconds: int = 60
    initial_equity: float = 10_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 1.5
    minimum_order_notional: float = 10.0
    allow_short: bool = False
    max_leverage: float = 1.0
    history_size: int = 5_000
    bootstrap_bars: int = 500
    bootstrap_max_trade_pages: int = 120
    bootstrap_trade_page_delay_seconds: float = 0.15


class PaperSession:
    def __init__(
        self,
        *,
        config: PaperSessionConfig,
        strategy: StrategyPlugin,
        rest_client: KrakenRestClient,
        stream: KrakenMarketStream,
        database: Database,
        hub: EventHub,
    ) -> None:
        self.id = uuid4().hex
        self.config = config
        self.strategy = strategy
        self.rest_client = rest_client
        self.stream = stream
        self.database = database
        self.hub = hub
        self.broker = PaperBroker(
            BrokerConfig(
                initial_equity=config.initial_equity,
                fee_bps=config.fee_bps,
                slippage_bps=config.slippage_bps,
                minimum_order_notional=config.minimum_order_notional,
                allow_short=config.allow_short,
                max_leverage=config.max_leverage,
            )
        )
        self.history: deque[Bar] = deque(maxlen=config.history_size)
        self.aggregator = CandleAggregator(config.symbol, config.timeframe_seconds)
        self.last_quote: Quote | None = None
        self.status = SessionStatus.STOPPED
        self.status_message = ""
        self.last_account: AccountSnapshot | None = None
        self._last_target_fraction: float | None = None
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._last_trade_id: int | None = None
        self._last_trade_timestamp_ms: int | None = None
        self._bootstrap_source = ""
        self._bootstrap_pages = 0
        self._bootstrap_bar_count = 0

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            raise RuntimeError("paper session is already running")
        self.status = SessionStatus.STARTING
        self.status_message = "bootstrapping market history"
        self._stop_event.clear()
        try:
            await self._bootstrap()
        except Exception as exc:
            self.status = SessionStatus.ERROR
            self.status_message = f"{type(exc).__name__}: {exc}"
            raise
        self._task = asyncio.create_task(self._run(), name=f"paper-session-{self.id}")

    async def stop(self) -> None:
        if self._task is None:
            self.status = SessionStatus.STOPPED
            return
        self.status = SessionStatus.STOPPING
        self._stop_event.set()
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self.status = SessionStatus.STOPPED
        await self.hub.publish("session", self.state())

    async def _bootstrap(self) -> None:
        if self.config.timeframe_seconds < 60:
            completed_bars, current_bar = await self._bootstrap_subminute()
        else:
            completed_bars, current_bar = await self._bootstrap_ohlc()
        minimum_required = max(2, self.strategy.minimum_bars)
        if len(completed_bars) < minimum_required:
            raise RuntimeError(
                f"Kraken bootstrap produced {len(completed_bars):,} completed "
                f"{self.config.timeframe_seconds}-second bars, but strategy {self.strategy.name} "
                f"requires {minimum_required:,}. Run a longer collection session, increase "
                "METEOR_KRAKEN_TRADE_BOOTSTRAP_MAX_PAGES, or choose a shorter-context strategy."
            )
        completed_bars = completed_bars[-max(minimum_required, self.config.bootstrap_bars) :]
        self._bootstrap_bar_count = len(completed_bars)
        self.database.upsert_bars(completed_bars)
        self.aggregator.seed_current(current_bar)
        self.last_quote = _quote_from_price(int(time.time() * 1000), current_bar.close)
        self.last_account = self.broker.snapshot(current_bar.timestamp, current_bar.close)
        self.strategy.on_start(
            StrategyContext(
                self.config.symbol,
                tuple(),
                self.last_account,
                -1,
            )
        )
        bootstrap_plots: list[dict[str, Any]] = []
        for index, bar in enumerate(completed_bars):
            self.history.append(bar)
            account = self.broker.snapshot(bar.timestamp, bar.close)
            decision = self.strategy.on_bar(
                StrategyContext(self.config.symbol, tuple(self.history), account, index),
                bar,
            )
            bootstrap_plots.append({"timestamp": bar.timestamp, "values": decision.plots})
        self.status_message = f"ready: {self._bootstrap_source}"
        await self.hub.publish(
            "bootstrap",
            {
                "session": self.state(),
                "bars": [bar.to_dict() for bar in completed_bars] + [current_bar.to_dict()],
                "plots": bootstrap_plots,
                "indicator_specs": [item.to_dict() for item in self.strategy.indicator_specs],
            },
        )

    async def _bootstrap_ohlc(self) -> tuple[list[Bar], Bar]:
        if self.config.timeframe_seconds % 60 != 0:
            raise ValueError("whole-minute Kraken timeframe must be divisible by 60 seconds")
        interval_minutes = self.config.timeframe_seconds // 60
        bars = await self.rest_client.fetch_ohlc(
            pair=self.config.rest_pair,
            symbol=self.config.symbol,
            interval_minutes=interval_minutes,
            include_incomplete=True,
        )
        if len(bars) < 2:
            raise RuntimeError("Kraken returned insufficient OHLC bootstrap bars")
        self._bootstrap_source = "Kraken OHLC"
        return bars[:-1], bars[-1]

    async def _bootstrap_subminute(self) -> tuple[list[Bar], Bar]:
        timeframe = self.config.timeframe_seconds
        required = max(2, self.strategy.minimum_bars, self.config.bootstrap_bars)
        now = int(time.time())
        lookback_seconds = (required + 4) * timeframe
        stored = self.database.load_bars(
            symbol=self.config.symbol,
            timeframe_seconds=timeframe,
            limit=required + 8,
        )
        recent_stored = bool(stored) and now - stored[-1].timestamp <= lookback_seconds * 2
        baseline_start = now - lookback_seconds
        fetch_start = baseline_start
        if recent_stored:
            fetch_start = max(baseline_start, stored[-1].timestamp - timeframe)
        fetch_start -= fetch_start % timeframe
        history = await self.rest_client.fetch_trade_history(
            pair=self.config.rest_pair,
            start_timestamp=fetch_start,
            end_timestamp=now,
            max_pages=self.config.bootstrap_max_trade_pages,
            page_delay_seconds=self.config.bootstrap_trade_page_delay_seconds,
            max_staleness_seconds=max(5, timeframe * 2),
        )
        self._bootstrap_pages = history.pages
        trade_ids = [trade.trade_id for trade in history.trades if trade.trade_id is not None]
        self._last_trade_id = max(trade_ids) if trade_ids else None
        self._last_trade_timestamp_ms = history.last_timestamp_ms
        if not history.trades:
            if len(stored) < required:
                raise RuntimeError("Kraken recent-trades bootstrap returned no usable trades")
            price = stored[-1].close
            current_timestamp = now - (now % timeframe)
            current = Bar(
                timestamp=current_timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,
                symbol=self.config.symbol,
                timeframe_seconds=timeframe,
            )
            self._bootstrap_source = "local Kraken bar cache"
            return stored[-required:], current
        fetched_completed, current = aggregate_trades_to_bars(
            history.trades,
            symbol=self.config.symbol,
            timeframe_seconds=timeframe,
            end_timestamp_ms=now * 1000,
        )
        merged = {bar.timestamp: bar for bar in stored if bar.timestamp < current.timestamp}
        merged.update({bar.timestamp: bar for bar in fetched_completed if bar.timestamp < current.timestamp})
        completed = sorted(merged.values(), key=lambda item: item.timestamp)
        if not history.complete:
            latest = history.last_timestamp_ms or 0
            age_seconds = max(0.0, now - latest / 1000)
            raise RuntimeError(
                "Kraken trade bootstrap reached its page limit before catching up to live data "
                f"(pages={history.pages}, latest trade age={age_seconds:.1f}s). Increase "
                "METEOR_KRAKEN_TRADE_BOOTSTRAP_MAX_PAGES or reuse accumulated local Kraken bars."
            )
        self._bootstrap_source = (
            f"Kraken recent trades ({history.pages} REST page"
            f"{'s' if history.pages != 1 else ''})"
        )
        return completed[-required:], current

    async def _run(self) -> None:
        self.status = SessionStatus.RUNNING
        await self.hub.publish("session", self.state())
        try:
            async for event in self.stream.events(self._stop_event):
                if event.kind == "quote" and event.quote is not None:
                    self.last_quote = event.quote
                    await self._advance_market_clock(event.quote.timestamp_ms)
                    await self.hub.publish(
                        "quote",
                        {
                            "timestamp_ms": event.quote.timestamp_ms,
                            "bid": event.quote.bid,
                            "ask": event.quote.ask,
                            "last": event.quote.last,
                        },
                    )
                elif event.kind == "trade" and event.trade is not None:
                    trade = event.trade
                    if (
                        trade.trade_id is not None
                        and self._last_trade_id is not None
                        and trade.trade_id <= self._last_trade_id
                    ):
                        continue
                    if trade.trade_id is not None:
                        self._last_trade_id = trade.trade_id
                    self._last_trade_timestamp_ms = trade.timestamp_ms
                    await self._handle_aggregation(self.aggregator.add_trade(trade))
                elif event.kind == "clock" and event.timestamp_ms is not None:
                    await self._advance_market_clock(event.timestamp_ms)
                elif event.kind == "status":
                    self.status_message = event.message or ""
                    if self.status_message == "connected" and self.config.timeframe_seconds < 60:
                        await self._catch_up_subminute_trades()
                    await self.hub.publish("market_status", {"message": self.status_message})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("paper session failed")
            self.status = SessionStatus.ERROR
            self.status_message = f"{type(exc).__name__}: {exc}"
            await self.hub.publish("session", self.state())

    async def _catch_up_subminute_trades(self) -> None:
        if self._last_trade_timestamp_ms is None:
            return
        now = int(time.time())
        timeframe = self.config.timeframe_seconds
        start = max(0, self._last_trade_timestamp_ms // 1000 - timeframe * 2)
        history = await self.rest_client.fetch_trade_history(
            pair=self.config.rest_pair,
            start_timestamp=start,
            end_timestamp=now,
            max_pages=min(self.config.bootstrap_max_trade_pages, 20),
            page_delay_seconds=self.config.bootstrap_trade_page_delay_seconds,
            max_staleness_seconds=max(5, timeframe * 2),
        )
        for trade in history.trades:
            if (
                trade.trade_id is not None
                and self._last_trade_id is not None
                and trade.trade_id <= self._last_trade_id
            ):
                continue
            if trade.trade_id is not None:
                self._last_trade_id = trade.trade_id
            self._last_trade_timestamp_ms = trade.timestamp_ms
            await self._handle_aggregation(self.aggregator.add_trade(trade))

    async def _advance_market_clock(self, timestamp_ms: int) -> None:
        aggregation = self.aggregator.advance_to(timestamp_ms)
        if aggregation is not None:
            await self._handle_aggregation(aggregation)

    async def _handle_aggregation(self, aggregation: CandleAggregation) -> None:
        for closed in aggregation.closed_bars:
            await self._on_closed_bar(closed)
        await self.hub.publish("bar_update", aggregation.current.to_dict())

    async def _on_closed_bar(self, bar: Bar) -> None:
        self.database.upsert_bars([bar])
        self.history.append(bar)
        mark = self.last_quote.mid if self.last_quote is not None else bar.close
        account = self.broker.snapshot(bar.timestamp, mark)
        decision = self.strategy.on_bar(
            StrategyContext(
                self.config.symbol, tuple(self.history), account, len(self.history) - 1
            ),
            bar,
        )
        await self.hub.publish("bar_closed", bar.to_dict())
        await self.hub.publish("plots", {"timestamp": bar.timestamp, "values": decision.plots})
        target_changed = decision.target_fraction is not None and (
            self._last_target_fraction is None
            or abs(decision.target_fraction - self._last_target_fraction) > 1e-12
        )
        if target_changed and decision.target_fraction is not None:
            self._last_target_fraction = decision.target_fraction
            quote = self.last_quote or _quote_from_price(bar.timestamp * 1000, bar.close)
            fill = self.broker.rebalance_to_fraction(
                target_fraction=decision.target_fraction,
                quote=quote,
                reason=decision.reason,
                strategy_key=self.strategy.key,
                timestamp=max(
                    bar.timestamp + self.config.timeframe_seconds, quote.timestamp_ms // 1000
                ),
            )
            if fill is not None:
                self.database.save_fill(self.id, fill)
                await self.hub.publish("fill", fill.to_dict())
        current_mark = self.last_quote.mid if self.last_quote is not None else bar.close
        self.last_account = self.broker.snapshot(int(time.time()), current_mark)
        self.database.save_equity(self.id, self.last_account)
        await self.hub.publish("account", self.last_account.to_dict())

    def state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "status_message": self.status_message,
            "strategy_key": self.strategy.key,
            "strategy_name": self.strategy.name,
            "parameters": self.strategy.parameters.model_dump(mode="json"),
            "symbol": self.config.symbol,
            "timeframe_seconds": self.config.timeframe_seconds,
            "bootstrap_source": self._bootstrap_source or None,
            "bootstrap_bars": self._bootstrap_bar_count,
            "bootstrap_trade_pages": self._bootstrap_pages,
            "account": self.last_account.to_dict() if self.last_account else None,
            "indicator_specs": [item.to_dict() for item in self.strategy.indicator_specs],
        }


def _quote_from_price(timestamp_ms: int, price: float, spread_bps: float = 1.0) -> Quote:
    half = spread_bps / 20_000.0
    return Quote(timestamp_ms, price * (1.0 - half), price * (1.0 + half), price)


class PaperSessionManager:
    def __init__(self) -> None:
        self.current: PaperSession | None = None
        self._lock = asyncio.Lock()

    async def replace(self, session: PaperSession) -> None:
        async with self._lock:
            if self.current is not None:
                await self.current.stop()
            self.current = None
            await session.start()
            self.current = session

    async def stop(self) -> None:
        async with self._lock:
            if self.current is not None:
                await self.current.stop()

    def state(self) -> dict[str, Any]:
        if self.current is None:
            return {"status": SessionStatus.STOPPED.value}
        return self.current.state()
