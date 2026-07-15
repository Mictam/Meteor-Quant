from __future__ import annotations

import time
from pathlib import Path

import pytest

from meteor_quant.api import KRAKEN_PAPER_TIMEFRAMES_SECONDS, PaperStartRequest
from meteor_quant.domain import MarketTrade, Side
from meteor_quant.events import EventHub
from meteor_quant.live import PaperSession, PaperSessionConfig
from meteor_quant.market import (
    CandleAggregator,
    KrakenMarketStream,
    KrakenRestClient,
    KrakenTradeHistory,
    KrakenTradePage,
    aggregate_trades_to_bars,
)
from meteor_quant.storage import Database
from meteor_quant.strategies.builtin import SmaCrossStrategy


def _trade(timestamp_ms: int, price: float, trade_id: int) -> MarketTrade:
    return MarketTrade(timestamp_ms, price, 0.01, Side.BUY, trade_id)


def test_candle_aggregator_fills_empty_one_second_bars() -> None:
    aggregator = CandleAggregator("BTC/USD", 1)
    first = aggregator.add_trade(_trade(100_200, 100.0, 1))
    assert first.current.timestamp == 100
    advanced = aggregator.advance_to(103_000)
    assert advanced is not None
    assert [bar.timestamp for bar in advanced.closed_bars] == [100, 101, 102]
    assert advanced.closed_bars[1].volume == 0.0
    assert advanced.closed_bars[1].close == 100.0
    assert advanced.current.timestamp == 103


def test_trade_jump_closes_all_five_second_buckets() -> None:
    aggregator = CandleAggregator("BTC/USD", 5)
    aggregator.add_trade(_trade(100_100, 100.0, 1))
    result = aggregator.add_trade(_trade(115_100, 103.0, 2))
    assert [bar.timestamp for bar in result.closed_bars] == [100, 105, 110]
    assert [bar.volume for bar in result.closed_bars] == [0.01, 0.0, 0.0]
    assert result.current.timestamp == 115
    assert result.current.open == 100.0
    assert result.current.close == 103.0
    assert result.current.number_of_trades == 1
    assert result.current.quote_asset_volume == pytest.approx(1.03)
    assert result.current.taker_buy_base_volume == pytest.approx(0.01)


def test_aggregate_recent_trades_builds_current_subminute_bar() -> None:
    trades = [_trade(100_100, 100.0, 1), _trade(105_100, 101.0, 2)]
    completed, current = aggregate_trades_to_bars(
        trades,
        symbol="BTC/USD",
        timeframe_seconds=5,
        end_timestamp_ms=112_000,
    )
    assert [bar.timestamp for bar in completed] == [100, 105]
    assert current.timestamp == 110
    assert current.volume == 0.0
    assert current.close == 101.0


def test_paper_request_accepts_requested_subminute_intervals() -> None:
    for interval in (1, 5, 15, 30):
        request = PaperStartRequest(strategy_key="sma_cross", timeframe_seconds=interval)
        assert request.timeframe_seconds == interval
        assert interval in KRAKEN_PAPER_TIMEFRAMES_SECONDS


class _FakeRecentTradeClient:
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
        del pair, max_pages, page_delay_seconds, max_staleness_seconds
        trades = tuple(
            _trade(second * 1000 + 100, 100.0 + index * 0.01, index + 1)
            for index, second in enumerate(range(start_timestamp, end_timestamp + 1))
        )
        return KrakenTradeHistory(
            trades=trades,
            pages=2,
            complete=True,
            first_timestamp_ms=trades[0].timestamp_ms,
            last_timestamp_ms=trades[-1].timestamp_ms,
        )


@pytest.mark.asyncio
async def test_subminute_bootstrap_uses_recent_trades_and_produces_strategy_history(
    tmp_path: Path,
) -> None:
    session = PaperSession(
        config=PaperSessionConfig(
            strategy_key="sma_cross",
            parameters={"fast": 5, "slow": 20},
            symbol="BTC/USD",
            rest_pair="XBTUSD",
            timeframe_seconds=1,
            bootstrap_bars=100,
            bootstrap_max_trade_pages=10,
        ),
        strategy=SmaCrossStrategy({"fast": 5, "slow": 20}),
        rest_client=_FakeRecentTradeClient(),  # type: ignore[arg-type]
        stream=KrakenMarketStream("wss://example.invalid", "BTC/USD", trade_snapshot=True),
        database=Database(tmp_path / "paper.sqlite3"),
        hub=EventHub(),
    )
    completed, current = await session._bootstrap_subminute()
    assert len(completed) == 100
    assert current.timeframe_seconds == 1
    assert current.timestamp >= int(time.time()) - 2
    assert session._last_trade_id is not None
    assert session._bootstrap_pages == 2


def test_database_persists_subminute_trade_flow_fields(tmp_path: Path) -> None:
    from meteor_quant.domain import Bar

    database = Database(tmp_path / "flow.sqlite3")
    bar = Bar(
        timestamp=100,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=2.0,
        quote_asset_volume=201.0,
        number_of_trades=7,
        taker_buy_base_volume=1.25,
        taker_buy_quote_volume=125.5,
        symbol="BTC/USD",
        timeframe_seconds=5,
    )
    database.upsert_bars([bar])
    loaded = database.load_bars(symbol="BTC/USD", timeframe_seconds=5)
    assert loaded == [bar]


@pytest.mark.asyncio
async def test_kraken_websocket_trade_snapshot_uses_taker_side_and_trade_id() -> None:
    stream = KrakenMarketStream("wss://example.invalid", "BTC/USD", trade_snapshot=True)
    payload = {
        "channel": "trade",
        "type": "snapshot",
        "data": [
            {
                "symbol": "BTC/USD",
                "side": "buy",
                "price": 64000.0,
                "qty": 0.25,
                "trade_id": 12345,
                "timestamp": "2026-07-13T12:00:00.100000Z",
            }
        ],
    }
    events = [event async for event in stream._parse(payload)]
    assert len(events) == 1
    trade = events[0].trade
    assert trade is not None
    assert trade.side == Side.BUY
    assert trade.trade_id == 12345
    assert trade.price == 64000.0


@pytest.mark.asyncio
async def test_recent_trade_history_paginates_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = KrakenRestClient("https://example.invalid")
    pages = [
        KrakenTradePage(
            (_trade(100_100, 100.0, 1), _trade(101_100, 101.0, 2)),
            "cursor-1",
        ),
        KrakenTradePage(
            (_trade(101_100, 101.0, 2), _trade(109_100, 109.0, 3)),
            "cursor-2",
        ),
    ]

    async def fake_page(*, pair: str, since: str | None, count: int) -> KrakenTradePage:
        del pair, since, count
        return pages.pop(0)

    monkeypatch.setattr(client, "fetch_recent_trades", fake_page)
    history = await client.fetch_trade_history(
        pair="XBTUSD",
        start_timestamp=100,
        end_timestamp=110,
        max_pages=5,
        page_delay_seconds=0.0,
        max_staleness_seconds=2,
    )
    assert history.complete is True
    assert history.pages == 2
    assert [trade.trade_id for trade in history.trades] == [1, 2, 3]
