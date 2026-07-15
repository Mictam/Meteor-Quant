# Python Strategy API

## Required class metadata

```python
class MyStrategy(StrategyPlugin):
    key = "my_strategy"
    name = "My strategy"
    description = "What the strategy does."
    parameter_model = Parameters
    minimum_bars = 100
    indicator_specs = (...)
```

Keys must be unique across built-ins and user plugins.

## Parameters

Use a Pydantic model with `extra="forbid"` so misspelled parameters fail instead of being ignored.

```python
class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period: int = Field(default=20, ge=2, le=100_000)
```

The React form is generated from `model_json_schema()`.

## Historical path

```python
def build_signals(self, frame: pl.LazyFrame) -> SignalPlan:
    enriched = frame.with_columns(
        pl.col("close").rolling_mean(self.parameters.period).alias("average")
    ).with_columns(
        pl.when(pl.col("close") > pl.col("average"))
        .then(1.0)
        .otherwise(0.0)
        .alias("target_fraction")
    )
    return SignalPlan(enriched, plot_columns=("average",))
```

Rules:

1. Keep the input lazy.
2. Prefer Polars expressions over Python loops or row UDFs.
3. Produce `target_fraction` as `Float64` or a castable numeric type.
4. A null target means no new target at that row.
5. A forward-filled target is valid; the engine executes only value changes.
6. Plot columns must exist in the returned frame.
7. Do not calculate fills, fees, cash, P&L, or leverage in strategy code.

Available base columns:

```text
timestamp
open
high
low
close
volume
quote_asset_volume
number_of_trades
taker_buy_base_volume
taker_buy_quote_volume
```

## Live path

```python
def on_live_bar(self, context: StrategyContext, bar: Bar) -> StrategyDecision:
    if len(context.bars) < self.minimum_bars:
        return StrategyDecision()
    target = 1.0 if condition else 0.0
    return StrategyDecision(
        target_fraction=target,
        reason="condition_name",
        plots={"average": calculated_average},
    )
```

The live method is called only for completed candles. Keep its working history bounded and deterministic.

## Indicators

```python
indicator_specs = (
    IndicatorSpec("average", "Average", "price", "price"),
    IndicatorSpec("oscillator", "Oscillator", "indicator", "number"),
)
```

Pane options:

- `price`: overlay on candles;
- `indicator`: lower pane;
- `equity`: reserved for account plots.

## Loading

Export one class:

```python
STRATEGY = MyStrategy
```

or multiple classes:

```python
STRATEGY = (StrategyA, StrategyB)
```

Press **Reload plugins** or call:

```text
POST /api/strategies/reload
```

Load failures are isolated and displayed without preventing valid strategies from starting.
