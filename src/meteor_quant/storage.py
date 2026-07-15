from __future__ import annotations

import csv
import sqlite3
import threading
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from meteor_quant.domain import AccountSnapshot, Bar, Fill


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT NOT NULL,
                    timeframe_seconds INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    quote_asset_volume REAL NOT NULL DEFAULT 0,
                    number_of_trades INTEGER NOT NULL DEFAULT 0,
                    taker_buy_base_volume REAL NOT NULL DEFAULT 0,
                    taker_buy_quote_volume REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (symbol, timeframe_seconds, timestamp)
                );
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    slippage_bps REAL NOT NULL,
                    reason TEXT NOT NULL,
                    strategy_key TEXT NOT NULL,
                    position_after REAL NOT NULL,
                    cash_after REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS equity (
                    session_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    quantity REAL NOT NULL,
                    mark_price REAL NOT NULL,
                    PRIMARY KEY (session_id, timestamp)
                );
                """
            )
            existing_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(bars)").fetchall()
            }
            required_columns = {
                "quote_asset_volume": "REAL NOT NULL DEFAULT 0",
                "number_of_trades": "INTEGER NOT NULL DEFAULT 0",
                "taker_buy_base_volume": "REAL NOT NULL DEFAULT 0",
                "taker_buy_quote_volume": "REAL NOT NULL DEFAULT 0",
            }
            for name, definition in required_columns.items():
                if name not in existing_columns:
                    connection.execute(f"ALTER TABLE bars ADD COLUMN {name} {definition}")

    def upsert_bars(self, bars: list[Bar]) -> int:
        if not bars:
            return 0
        rows = [
            (
                bar.symbol,
                bar.timeframe_seconds,
                bar.timestamp,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.quote_asset_volume,
                bar.number_of_trades,
                bar.taker_buy_base_volume,
                bar.taker_buy_quote_volume,
            )
            for bar in bars
        ]
        with self._lock, self._connect() as connection:
            connection.executemany(
                """INSERT INTO bars(
                    symbol, timeframe_seconds, timestamp, open, high, low, close, volume,
                    quote_asset_volume, number_of_trades, taker_buy_base_volume,
                    taker_buy_quote_volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe_seconds, timestamp) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume,
                    quote_asset_volume=excluded.quote_asset_volume,
                    number_of_trades=excluded.number_of_trades,
                    taker_buy_base_volume=excluded.taker_buy_base_volume,
                    taker_buy_quote_volume=excluded.taker_buy_quote_volume""",
                rows,
            )
        return len(rows)

    def load_bars(
        self,
        *,
        symbol: str,
        timeframe_seconds: int,
        limit: int = 10_000,
        start: int | None = None,
        end: int | None = None,
    ) -> list[Bar]:
        clauses = ["symbol = ?", "timeframe_seconds = ?"]
        params: list[Any] = [symbol, timeframe_seconds]
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)
        params.append(limit)
        query = f"""SELECT * FROM (
            SELECT * FROM bars WHERE {" AND ".join(clauses)}
            ORDER BY timestamp DESC LIMIT ?
        ) ORDER BY timestamp ASC"""
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            Bar(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                quote_asset_volume=row["quote_asset_volume"],
                number_of_trades=row["number_of_trades"],
                taker_buy_base_volume=row["taker_buy_base_volume"],
                taker_buy_quote_volume=row["taker_buy_quote_volume"],
                symbol=row["symbol"],
                timeframe_seconds=row["timeframe_seconds"],
            )
            for row in rows
        ]

    def save_fill(self, session_id: str, fill: Fill) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO fills(session_id, timestamp, side, quantity, price, fee, slippage_bps,
                reason, strategy_key, position_after, cash_after) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    fill.timestamp,
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    fill.fee,
                    fill.slippage_bps,
                    fill.reason,
                    fill.strategy_key,
                    fill.position_after,
                    fill.cash_after,
                ),
            )

    def save_equity(self, session_id: str, snapshot: AccountSnapshot) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO equity(session_id, timestamp, equity, cash, quantity, mark_price)
                VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(session_id, timestamp) DO UPDATE SET
                equity=excluded.equity, cash=excluded.cash, quantity=excluded.quantity, mark_price=excluded.mark_price""",
                (
                    session_id,
                    snapshot.timestamp,
                    snapshot.equity,
                    snapshot.cash,
                    snapshot.quantity,
                    snapshot.mark_price,
                ),
            )


def parse_csv_bars(content: bytes, symbol: str, timeframe_seconds: int) -> list[Bar]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header")
    aliases = {name.lower().strip(): name for name in reader.fieldnames}
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in aliases]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    bars: list[Bar] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            timestamp = _parse_timestamp(str(row[aliases["timestamp"]]))
            bars.append(
                Bar(
                    timestamp=timestamp,
                    open=float(row[aliases["open"]]),
                    high=float(row[aliases["high"]]),
                    low=float(row[aliases["low"]]),
                    close=float(row[aliases["close"]]),
                    volume=float(row[aliases["volume"]]),
                    symbol=symbol,
                    timeframe_seconds=timeframe_seconds,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid CSV row {line_number}: {exc}") from exc
    deduplicated = {bar.timestamp: bar for bar in bars}
    result = sorted(deduplicated.values(), key=lambda item: item.timestamp)
    if len(result) < 2:
        raise ValueError("CSV must contain at least two valid bars")
    return result


def _parse_timestamp(value: str) -> int:
    value = value.strip()
    try:
        number = float(value)
        return int(number / 1000 if number > 10_000_000_000 else number)
    except ValueError:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp())
