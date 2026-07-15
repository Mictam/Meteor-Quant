# Train MarketLM

Meteor Quant includes a **Train MarketLM** tab immediately after **Python plugins**. It trains a causal, patch-based transformer on the same prepared Binance data used by the backtester and registers the resulting checkpoint as a normal Meteor Quant indicator/strategy.

## Data flow

```text
Binance CSV -> canonical Parquet -> Polars causal features
            -> chronological split + purge gap
            -> train-only normalization
            -> memory-mapped tensors
            -> MarketLM worker process
            -> best/final checkpoint
            -> registered MarketLM indicator
            -> normal Meteor Quant backtest signal cache
            -> Rust/Python next-bar execution engine
```

Training never runs inside the FastAPI process. A separate worker owns data preparation and GPU training, writes atomic status/checkpoint files, and can be stopped or resumed without blocking the dashboard.

## Configure a run

The tab exposes:

- source dataset and date range;
- bar interval: 1s, 5s, 15s, 1m, 5m, 15m, or 1h;
- forecast horizons, each an exact multiple of the selected interval;
- patch size and context patches;
- chronological train/validation/test fractions and purge gap;
- OHLCV/trade-derived base features;
- selectable SMA, EMA, RSI, ATR, Bollinger, MACD, WaveTrend, rolling VWAP, and volume z-score features;
- transformer dimensions, layers, heads, MLP width, dropout, and RoPE base;
- batch size, gradient accumulation, learning rates, warmup, validation/checkpoint cadence, AMP, device, and `torch.compile`.

### Causality

Every model input feature references only the current or an earlier bar. Negative shifts are used only to create supervised future-return labels. Normalization is fitted only on the chronological training split. Validation and test endpoints are separated from earlier splits by a purge gap at least as long as the maximum forecast horizon.

## Model outputs

For each selected horizon the model learns:

1. autoregressive reconstruction of the next normalized feature patch;
2. q10, median, and q90 future log-return forecasts;
3. down/flat/up direction probabilities using the configured cost threshold.

The registered indicator exposes the chosen primary horizon as:

```text
marketlm_return_bps
marketlm_lower_bps
marketlm_upper_bps
marketlm_prob_up
marketlm_prob_down
```

## Register and use a checkpoint

After a training run completes, press **Register as indicator**. Reloaded strategy metadata will contain:

```text
MarketLM · <run name>
```

Select it in the regular backtest tab. Its parameters support:

- `indicator_only`: plot forecasts without changing the portfolio;
- `long_flat`: enter long on a bullish forecast and move flat on a bearish forecast;
- `long_short`: choose long or short targets from forecast and confidence thresholds;
- prediction stride and inference batch size;
- CPU, CUDA, or automatic device selection.

A model trained on 1-second bars must be backtested on 1-second bars. The API rejects mismatched intervals instead of silently resampling model inputs.

For multi-year 1-second inference, start with a prediction stride of 60 or higher. The model prediction is forward-filled between evaluated endpoints, while the execution engine still processes every bar.

## Use MarketLM inside a Python plugin

Registered models are also available as a library-level indicator:

```python
from pathlib import Path

from meteor_quant.marketlm.inference import MarketLMIndicator
from meteor_quant.marketlm.schemas import MarketLMIndicatorParameters

indicator = MarketLMIndicator.from_registered(
    "20260713T120000-abcd1234",
    Path("data/marketlm/registered"),
)

enriched = indicator.predict_frame(
    frame,
    MarketLMIndicatorParameters(
        mode="indicator_only",
        prediction_stride=60,
        batch_size=256,
        device="auto",
    ),
)

# `enriched` now contains the MarketLM columns. A plugin can combine them
# with WaveTrend, RSI, order-flow, or any other causal Polars expression.
combined = enriched.lazy().with_columns(
    (
        (pl.col("marketlm_return_bps") > 3.0)
        & (pl.col("marketlm_prob_up") > 58.0)
        & (pl.col("wavetrend_wt1") > pl.col("wavetrend_wt2"))
    ).cast(pl.Float64).alias("target_fraction")
)
```

The WaveTrend columns are present directly when WaveTrend was selected as a training feature. Otherwise calculate WaveTrend in the plugin before applying the composite rule.

## Live-paper constraints

Kraken paper trading currently bootstraps whole-minute OHLC bars and receives at most 719 completed historical bars from the public endpoint. A registered MarketLM strategy can start live only when:

- its training interval exactly matches the paper interval;
- its required context plus indicator warmup does not exceed 719 bars.

A 1-second model or a long-context model remains fully usable for historical backtests but is rejected for Kraken paper sessions with a precise explanation. Train a 60-second model with a shorter context for live paper trading.

## Storage

```text
data/marketlm/prepared/<fingerprint>/
  features.npy       normalized float16 input features
  targets.npy        normalized return targets
  directions.npy     down/flat/up labels
  timestamps.npy
  close.npy
  metadata.json

data/marketlm/runs/<run-id>/
  spec.json
  status.json
  worker.log
  metrics.jsonl
  checkpoint_step_*.pt
  best.pt
  final.pt
  run_info.json

data/marketlm/registered/<model-id>.json
```

Prepared tensors are fingerprinted by source data version, range, interval, indicators, horizons, context, split, and cost threshold. An unchanged configuration reuses the same tensors.

## Windows notes

Python 3.11.5 is supported. Install everything with:

```powershell
.\install.ps1
```

Or add MarketLM to an existing environment:

```powershell
.\setup-marketlm.ps1
```

On Windows, keep data workers at `0` if multiprocessing is unstable. If CUDA runs out of memory, reduce `batch_size` and increase `gradient_accumulation_steps` to preserve the effective batch size.

## Atomic file safety

MarketLM writes status, prepared tensors, and published datasets through unique sibling paths and atomic replacement. Memory maps are explicitly closed before publication, completed tensor directories are immutable after validation, and concurrent jobs for the same fingerprint cannot overwrite each other. These guarantees apply across Windows, Linux, and macOS; no platform-specific repair script is required.
