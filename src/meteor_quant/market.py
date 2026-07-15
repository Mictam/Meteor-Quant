from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import websockets

from meteor_quant.domain import Bar, MarketTrade, Quote, Side

LOGGER = logging.getLogger(__name__)


class KrakenApiError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class KrakenTradePage:
    trades: tuple[MarketTrade, ...]
    next_cursor: str


@dataclass(slots=True, frozen=True)
class KrakenTradeHistory:
    trades: tuple[MarketTrade, ...]
    pages: int
    complete: bool
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None


class KrakenRestClient:
    def __init__(self, base_url: str, timeout_seconds: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def fetch_ohlc(
        self,
        *,
        pair: str,
        symbol: str,
        interval_minutes: int,
        since: int | None = None,
        include_incomplete: bool = False,
    ) -> list[Bar]:
        params: dict[str, str | int] = {
            "pair": pair,
            "interval": interval_minutes,
            "assetVersion": 1,
        }
        if since is not None:
            params["since"] = since
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout_seconds
        ) as client:
            response = await client.get("/0/public/OHLC", params=params)
            response.raise_for_status()
            payload = response.json()
        errors = payload.get("error", [])
        if errors:
            raise KrakenApiError("; ".join(str(item) for item in errors))
        result = payload.get("result", {})
        rows: list[list[Any]] | None = None
        if isinstance(result, dict):
            for key, value in result.items():
                if key != "last" and isinstance(value, list):
                    rows = value
                    break
        if rows is None:
            raise KrakenApiError("OHLC response did not contain a candle series")
        selected = rows if include_incomplete else rows[:-1]
        return [
            Bar(
                timestamp=int(float(row[0])),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[6]),
                quote_asset_volume=float(row[5]) * float(row[6]),
                number_of_trades=int(row[7]) if len(row) > 7 else 0,
                symbol=symbol,
                timeframe_seconds=interval_minutes * 60,
            )
            for row in selected
        ]

    async def fetch_recent_trades(
        self,
        *,
        pair: str,
        since: str | None = None,
        count: int = 1_000,
    ) -> KrakenTradePage:
        if count < 1 or count > 1_000:
            raise ValueError("Kraken trade count must be between 1 and 1000")
        params: dict[str, str | int] = {
            "pair": pair,
            "assetVersion": 1,
            "count": count,
        }
        if since is not None:
            params["since"] = since
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout_seconds
        ) as client:
            response = await client.get("/0/public/Trades", params=params)
            response.raise_for_status()
            payload = response.json()
        errors = payload.get("error", [])
        if errors:
            raise KrakenApiError("; ".join(str(item) for item in errors))
        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise KrakenApiError("Trades response did not contain an object result")
        rows: list[Any] | None = None
        for key, value in result.items():
            if key != "last" and isinstance(value, list):
                rows = value
                break
        if rows is None:
            raise KrakenApiError("Trades response did not contain a trade series")
        trades = tuple(
            trade
            for row in rows
            if isinstance(row, list) and (trade := _trade_from_rest_row(row)) is not None
        )
        return KrakenTradePage(trades, str(result.get("last", since or "")))

    async def fetch_trade_history(
        self,
        *,
        pair: str,
        start_timestamp: int,
        end_timestamp: int,
        max_pages: int,
        page_delay_seconds: float,
        max_staleness_seconds: int,
    ) -> KrakenTradeHistory:
        if start_timestamp >= end_timestamp:
            raise ValueError("trade-history start must be before end")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        cursor = str(start_timestamp * 1_000_000_000)
        trades_by_key: dict[tuple[object, ...], MarketTrade] = {}
        pages = 0
        last_cursor = ""
        target_ms = end_timestamp * 1000
        complete = False
        for page_index in range(max_pages):
            page: KrakenTradePage | None = None
            for attempt in range(4):
                try:
                    page = await self.fetch_recent_trades(
                        pair=pair, since=cursor, count=1_000
                    )
                    break
                except (KrakenApiError, httpx.HTTPStatusError) as exc:
                    if attempt >= 3 or not _is_retryable_market_error(exc):
                        raise
                    await asyncio.sleep(max(1.0, page_delay_seconds) * (2**attempt))
            if page is None:
                raise RuntimeError("Kraken trade page retry loop completed without a result")
            pages += 1
            for trade in page.trades:
                if trade.timestamp_ms > target_ms:
                    continue
                key: tuple[object, ...]
                if trade.trade_id is not None:
                    key = ("id", trade.trade_id)
                else:
                    key = (
                        "values",
                        trade.timestamp_ms,
                        trade.price,
                        trade.quantity,
                        trade.side.value,
                    )
                trades_by_key[key] = trade
            latest_ms = max((trade.timestamp_ms for trade in trades_by_key.values()), default=0)
            if latest_ms >= target_ms - max_staleness_seconds * 1000:
                complete = True
                break
            if not page.trades or not page.next_cursor or page.next_cursor in {cursor, last_cursor}:
                break
            last_cursor = cursor
            cursor = page.next_cursor
            if page_index + 1 < max_pages and page_delay_seconds > 0:
                await asyncio.sleep(page_delay_seconds)
        trades = tuple(sorted(trades_by_key.values(), key=_trade_sort_key))
        return KrakenTradeHistory(
            trades=trades,
            pages=pages,
            complete=complete,
            first_timestamp_ms=trades[0].timestamp_ms if trades else None,
            last_timestamp_ms=trades[-1].timestamp_ms if trades else None,
        )


@dataclass(slots=True, frozen=True)
class MarketEvent:
    kind: str
    quote: Quote | None = None
    trade: MarketTrade | None = None
    message: str | None = None
    timestamp_ms: int | None = None


class KrakenMarketStream:
    def __init__(
        self,
        websocket_url: str,
        symbol: str,
        *,
        trade_snapshot: bool = False,
        clock_interval_seconds: float = 1.0,
    ) -> None:
        self.websocket_url = websocket_url
        self.symbol = symbol
        self.trade_snapshot = trade_snapshot
        self.clock_interval_seconds = max(0.1, clock_interval_seconds)

    async def events(self, stop_event: asyncio.Event) -> AsyncIterator[MarketEvent]:
        delay = 1.0
        while not stop_event.is_set():
            try:
                async with websockets.connect(
                    self.websocket_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=4096,
                ) as websocket:
                    await websocket.send(
                        json.dumps(
                            {
                                "method": "subscribe",
                                "params": {
                                    "channel": "ticker",
                                    "symbol": [self.symbol],
                                    "event_trigger": "bbo",
                                    "snapshot": True,
                                },
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "method": "subscribe",
                                "params": {
                                    "channel": "trade",
                                    "symbol": [self.symbol],
                                    "snapshot": self.trade_snapshot,
                                },
                            }
                        )
                    )
                    delay = 1.0
                    yield MarketEvent("status", message="connected")
                    while not stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(
                                websocket.recv(), timeout=self.clock_interval_seconds
                            )
                        except TimeoutError:
                            yield MarketEvent("clock", timestamp_ms=int(time.time() * 1000))
                            continue
                        payload = json.loads(raw)
                        if isinstance(payload, dict):
                            async for event in self._parse(payload):
                                yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("Kraken stream disconnected: %s", exc)
                yield MarketEvent("status", message=f"reconnecting: {type(exc).__name__}: {exc}")
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                delay = min(delay * 2.0, 30.0)

    async def _parse(self, payload: dict[str, Any]) -> AsyncIterator[MarketEvent]:
        channel = payload.get("channel")
        data = payload.get("data")
        if channel == "ticker" and isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                bid = _first_number(item.get("bid"))
                ask = _first_number(item.get("ask"))
                last = _first_number(item.get("last"))
                if bid and ask:
                    timestamp_ms = _timestamp_ms(item.get("timestamp"))
                    yield MarketEvent(
                        "quote",
                        quote=Quote(
                            timestamp_ms=timestamp_ms,
                            bid=bid,
                            ask=ask,
                            last=last or (bid + ask) / 2.0,
                        ),
                        timestamp_ms=timestamp_ms,
                    )
        elif channel == "trade" and isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                price = _first_number(item.get("price"))
                quantity = _first_number(item.get("qty"))
                side_text = str(item.get("side", "buy")).lower()
                trade_id = _optional_int(item.get("trade_id"))
                if price and quantity:
                    timestamp_ms = _timestamp_ms(item.get("timestamp"))
                    yield MarketEvent(
                        "trade",
                        trade=MarketTrade(
                            timestamp_ms=timestamp_ms,
                            price=price,
                            quantity=quantity,
                            side=Side.SELL if side_text == "sell" else Side.BUY,
                            trade_id=trade_id,
                        ),
                        timestamp_ms=timestamp_ms,
                    )
        elif payload.get("success") is False:
            yield MarketEvent(
                "status", message=f"subscription error: {payload.get('error', payload)}"
            )



def _is_retryable_market_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    message = str(exc).lower()
    return "rate limit" in message or "throttle" in message or "service unavailable" in message

def _trade_from_rest_row(row: list[Any]) -> MarketTrade | None:
    if len(row) < 4:
        return None
    price = _first_number(row[0])
    quantity = _first_number(row[1])
    if price <= 0 or quantity <= 0:
        return None
    try:
        timestamp_ms = int(float(row[2]) * 1000)
    except (TypeError, ValueError):
        return None
    side_text = str(row[3]).lower()
    trade_id = _optional_int(row[6]) if len(row) > 6 else None
    return MarketTrade(
        timestamp_ms=timestamp_ms,
        price=price,
        quantity=quantity,
        side=Side.SELL if side_text in {"s", "sell"} else Side.BUY,
        trade_id=trade_id,
    )


def _trade_sort_key(trade: MarketTrade) -> tuple[int, int]:
    return trade.timestamp_ms, trade.trade_id if trade.trade_id is not None else -1


def _first_number(value: Any) -> float:
    if isinstance(value, list):
        value = value[0] if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _timestamp_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number if number > 10_000_000_000 else number * 1000)
    if isinstance(value, str):
        try:
            normalized = value.replace("Z", "+00:00")
            return int(datetime.fromisoformat(normalized).timestamp() * 1000)
        except ValueError:
            try:
                number = float(value)
                return int(number if number > 10_000_000_000 else number * 1000)
            except ValueError:
                pass
    return int(datetime.now().timestamp() * 1000)


@dataclass(slots=True, frozen=True)
class CandleAggregation:
    current: Bar
    closed_bars: tuple[Bar, ...] = ()

    @property
    def closed(self) -> Bar | None:
        return self.closed_bars[-1] if self.closed_bars else None


class CandleAggregator:
    def __init__(self, symbol: str, timeframe_seconds: int) -> None:
        if timeframe_seconds <= 0:
            raise ValueError("timeframe_seconds must be positive")
        self.symbol = symbol
        self.timeframe_seconds = timeframe_seconds
        self.current: Bar | None = None
        self.previous_close: float | None = None

    def seed_previous_close(self, price: float) -> None:
        if price <= 0:
            raise ValueError("previous close must be positive")
        self.previous_close = price

    def seed_current(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            raise ValueError("seed bar symbol does not match aggregator symbol")
        if bar.timeframe_seconds != self.timeframe_seconds:
            raise ValueError("seed bar timeframe does not match aggregator timeframe")
        self.current = bar
        self.previous_close = bar.close

    def advance_to(self, timestamp_ms: int) -> CandleAggregation | None:
        if self.current is None:
            return None
        target_timestamp = timestamp_ms // 1000
        target_bucket = target_timestamp - (target_timestamp % self.timeframe_seconds)
        if target_bucket <= self.current.timestamp:
            return CandleAggregation(self.current)
        closed = self._advance_to_bucket(target_bucket)
        return CandleAggregation(self.current, tuple(closed))

    def add_trade(self, trade: MarketTrade) -> CandleAggregation:
        timestamp = trade.timestamp_ms // 1000
        bucket = timestamp - (timestamp % self.timeframe_seconds)
        closed: list[Bar] = []
        if self.current is None:
            open_price = self.previous_close if self.previous_close is not None else trade.price
            self.current = Bar(
                timestamp=bucket,
                open=open_price,
                high=max(open_price, trade.price),
                low=min(open_price, trade.price),
                close=trade.price,
                volume=trade.quantity,
                quote_asset_volume=trade.price * trade.quantity,
                number_of_trades=1,
                taker_buy_base_volume=(
                    trade.quantity if trade.side == Side.BUY else 0.0
                ),
                taker_buy_quote_volume=(
                    trade.price * trade.quantity if trade.side == Side.BUY else 0.0
                ),
                symbol=self.symbol,
                timeframe_seconds=self.timeframe_seconds,
            )
            self.previous_close = trade.price
            return CandleAggregation(self.current)
        if bucket < self.current.timestamp:
            return CandleAggregation(self.current)
        if bucket > self.current.timestamp:
            closed.extend(self._advance_to_bucket(bucket))
        self.current = Bar(
            timestamp=self.current.timestamp,
            open=self.current.open,
            high=max(self.current.high, trade.price),
            low=min(self.current.low, trade.price),
            close=trade.price,
            volume=self.current.volume + trade.quantity,
            quote_asset_volume=(
                self.current.quote_asset_volume + trade.price * trade.quantity
            ),
            number_of_trades=self.current.number_of_trades + 1,
            taker_buy_base_volume=(
                self.current.taker_buy_base_volume
                + (trade.quantity if trade.side == Side.BUY else 0.0)
            ),
            taker_buy_quote_volume=(
                self.current.taker_buy_quote_volume
                + (trade.price * trade.quantity if trade.side == Side.BUY else 0.0)
            ),
            symbol=self.symbol,
            timeframe_seconds=self.timeframe_seconds,
        )
        self.previous_close = trade.price
        return CandleAggregation(self.current, tuple(closed))

    def _advance_to_bucket(self, target_bucket: int) -> list[Bar]:
        if self.current is None:
            return []
        closed: list[Bar] = []
        while self.current.timestamp < target_bucket:
            closed.append(self.current)
            next_timestamp = self.current.timestamp + self.timeframe_seconds
            price = self.current.close
            self.current = Bar(
                timestamp=next_timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,
                symbol=self.symbol,
                timeframe_seconds=self.timeframe_seconds,
            )
            self.previous_close = price
        return closed


def aggregate_trades_to_bars(
    trades: Iterable[MarketTrade],
    *,
    symbol: str,
    timeframe_seconds: int,
    end_timestamp_ms: int,
) -> tuple[list[Bar], Bar]:
    ordered = sorted(trades, key=_trade_sort_key)
    if not ordered:
        raise ValueError("cannot build sub-minute bars without trades")
    aggregator = CandleAggregator(symbol, timeframe_seconds)
    completed: list[Bar] = []
    for trade in ordered:
        aggregation = aggregator.add_trade(trade)
        completed.extend(aggregation.closed_bars)
    final = aggregator.advance_to(end_timestamp_ms)
    if final is not None:
        completed.extend(final.closed_bars)
    if aggregator.current is None:
        raise RuntimeError("trade aggregation did not produce a current bar")
    deduplicated = {bar.timestamp: bar for bar in completed}
    return sorted(deduplicated.values(), key=lambda item: item.timestamp), aggregator.current
