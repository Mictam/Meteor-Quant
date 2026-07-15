# Architecture

## 1. Boundaries

### React dashboard

The dashboard is a control and visualization client only. It never calculates P&L and never accesses source CSVs directly.

### FastAPI control plane

FastAPI owns:

- request validation;
- dataset and strategy discovery;
- cache orchestration;
- Rust process invocation;
- fallback-engine selection;
- persisted result lookup;
- Kraken paper-session lifecycle;
- WebSocket event fan-out;
- serving the prebuilt React bundle.

It delegates CPU-heavy operations to worker threads so the event loop remains available.

### Polars research layer

Polars owns columnar transformations:

- source header normalization;
- timestamp conversion;
- timeframe resampling;
- rolling and exponentially weighted indicators;
- vectorized target generation;
- prepared signal Parquet output;
- bounded chart aggregation.

The canonical dataset schema is:

```text
timestamp                 Int64 Unix seconds
open                      Float64
high                      Float64
low                       Float64
close                     Float64
volume                    Float64
quote_asset_volume        Float64
number_of_trades          Int64
taker_buy_base_volume     Float64
taker_buy_quote_volume    Float64
```

### Rust event-driven engine

The Rust binary reads prepared Parquet in Arrow record batches. It owns:

- target-change detection;
- next-bar execution;
- signed position quantity;
- cash and mark-to-market equity;
- long-only and short/leverage constraints;
- fee, spread, and slippage application;
- streaming drawdown and return statistics;
- bounded fill and equity output.

The Rust process accepts a Parquet file path and writes one JSON result. This process boundary avoids per-row Python/Rust calls and keeps the engine independently testable.

### Python Arrow fallback

The fallback reads the same prepared Parquet schema in PyArrow batches and implements the same state transitions. It is deliberately kept separate from Polars signal generation so it tests the engine contract rather than reusing vectorized portfolio shortcuts.

## 2. Dataset lifecycle

```text
raw CSVs
  -> source signature(path, bytes, mtime)
  -> canonical Parquet
  -> metadata JSON
```

The source signature prevents stale canonical data after either CSV changes.

## 3. Signal-cache lifecycle

The signal cache key includes:

- canonical dataset update timestamp;
- dataset key;
- strategy key;
- validated parameter JSON;
- timeframe;
- start and end timestamp;
- signal schema version.

This permits repeated cost-model experiments without recalculating indicators. Fees, spread, slippage, leverage, and initial equity are intentionally not part of the signal key because they affect execution, not signal generation.

## 4. Strategy contract

A strategy cannot mutate broker state. It emits a desired signed exposure fraction.

```text
0.0   flat
1.0   100% long
-1.0  100% short when shorting is enabled
```

The execution engine clamps targets to configured policy.

Historical strategies operate on a `LazyFrame`. Live strategies receive immutable bar history and an account snapshot.

## 5. Causality

Prepared row `N` contains a target calculated from data available through the close of row `N`. The engine stores that target as pending and applies it at row `N+1` open.

Targets are executed only when their value changes. This avoids accidental fee churn from forward-filled vectorized positions while retaining deterministic state transitions.

## 6. Visualization

The browser never receives tens of millions of rows. The chart endpoint dynamically calculates a UTC-aligned aggregation bucket from the selected duration and point limit. It returns:

- aggregated OHLCV;
- last target in each bucket;
- last indicator value in each bucket;
- original signal-row count and chosen bucket size.

Fills and sampled equity remain tied to original timestamps.

## 7. Live paper trading

The live path uses:

- Kraken public REST OHLC for bootstrap;
- Kraken WebSocket v2 ticker/trade feeds;
- a candle aggregator;
- the strategy's `on_live_bar` method;
- the deterministic Python paper broker;
- SQLite only for live bars/fills/equity;
- WebSocket events to the browser.

No private Kraken credentials are read.

## 8. Extension points

- new Python strategies in `user_strategies/`;
- new native indicators in Polars expressions or future Rust kernels;
- alternative dataset catalogs behind `DatasetCatalog`;
- future multi-symbol engine input through a versioned prepared schema;
- future batch-research workers that reuse the signal and result contracts.
