# TimesFM 2.5 indicator

Meteor Quant includes TimesFM 2.5 as a built-in zero-shot forecast indicator and optional target-position strategy.

## Scope

The integration uses Google's official `google/timesfm-2.5-200m-pytorch` checkpoint through the official `timesfm` Python package. It forecasts the future BTC close-price path from a causal window of historical closes.

This is distinct from **Train MarketLM**:

- **TimesFM 2.5** is pretrained and runs zero-shot. It does not fit itself to your CSV or consume WaveTrend/RSI features in this release.
- **MarketLM** is trained on your Binance history and can learn selected OHLCV and indicator features.

Use TimesFM for a strong pretrained baseline and uncertainty bands. Use MarketLM when you need dataset-specific feature learning.

## Installation

First verify that the project virtual environment contains CUDA PyTorch:

```powershell
cd <project-directory>
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected:

```text
2.x.x+cu... 12.x True NVIDIA GeForce RTX 4060 Ti
```

Then install TimesFM without allowing pip to replace the CUDA wheel:

```powershell
.\setup-timesfm.ps1 -DownloadModel
```

Without pre-downloading the checkpoint:

```powershell
.\setup-timesfm.ps1
```

Download later:

```powershell
.\download-timesfm.ps1
```

Restart Meteor Quant and press **Reload plugins/strategies** if it was already running.

## Indicator outputs

The chart receives these columns:

```text
timesfm_forecast_price       median/point forecast at the selected horizon
timesfm_q10_price            lower forecast band
timesfm_q90_price            upper forecast band
timesfm_return_bps           median return from current close
timesfm_q10_return_bps       q10 return
timesfm_q90_return_bps       q90 return
timesfm_uncertainty_bps      q90 minus q10 return width
```

The three price forecasts render over the candle chart. Return and uncertainty values render in indicator panes.

## Execution modes

### `indicator_only`

Produces plots but no portfolio target. This is the recommended first mode for evaluating forecast behavior.

### `long_flat`

Uses `long_threshold_bps` to move to `long_target`; otherwise moves to `flat_target`.

### `long_short`

Uses the long and short thresholds to choose `long_target`, `short_target`, or `flat_target`.

All targets pass through the normal event-driven broker. A forecast generated at completed bar `N` can fill only at bar `N+1` open.

## Core parameters

| Parameter | Meaning |
|---|---|
| `model_id_or_path` | Hugging Face model ID or local checkpoint directory |
| `context_length` | Number of historical bars supplied per forecast |
| `forecast_horizon_seconds` | Forecast endpoint measured in seconds |
| `prediction_stride` | Number of bars between expensive model evaluations |
| `max_predictions` | Maximum forecasts allowed for one backtest |
| `auto_increase_stride` | Automatically increase stride when the range exceeds the cap |
| `maximum_output_rows` | Safety cap before allocating result columns |
| `batch_size` | Rolling windows evaluated in one TimesFM batch |
| `require_cuda` | Fail instead of silently using CPU |
| `local_files_only` | Prohibit network downloads and use an existing cache/local model |
| `torch_compile` | Enable the official runtime's Torch compilation option |

## Quantile and trading filters

| Parameter | Meaning |
|---|---|
| `long_threshold_bps` | Minimum median projected return for a long target |
| `short_threshold_bps` | Maximum median projected return for a short target |
| `require_quantile_confirmation` | Require the uncertainty band to confirm the direction |
| `long_q10_floor_bps` | q10 must be at least this value for a long signal |
| `short_q90_ceiling_bps` | q90 must be at most this value for a short signal |
| `maximum_uncertainty_bps` | Reject signals whose q90-q10 width is too large; zero disables |

A conservative long condition is:

```text
median return >= long threshold
q10 return >= 0
uncertainty <= configured maximum
```

This requires even the lower forecast band to remain non-negative.

## Recommended 15-second configuration

Balanced quality/performance baseline:

```text
mode                       indicator_only
context_length             4096
forecast_horizon_seconds   300
prediction_stride          20
max_predictions            20000
batch_size                 16
require_cuda               true
normalize_inputs           true
use_continuous_quantile_head true
force_flip_invariance      true
fix_quantile_crossing      true
```

Interpretation:

```text
4096 × 15 seconds = 17 hours 4 minutes of context
300 seconds       = 20 bars forecast horizon
20-bar stride     = one new forecast every 5 minutes
```

Higher-resolution profile for a short date range:

```text
context_length             8192
forecast_horizon_seconds   300
prediction_stride          4
batch_size                 16
max_predictions            50000
```

Use this only on a bounded range. Evaluating every minute across several years is expensive even on a GPU.

## Model limits

Meteor Quant validates the official model's compiled limits before loading it:

- compiled context is rounded to a multiple of 32;
- compiled horizon is rounded to a multiple of 128;
- compiled horizon must not exceed 1,024 bars for continuous quantiles;
- rounded context plus rounded horizon must not exceed 16,384.

The horizon in seconds must also be an exact multiple of the selected backtest timeframe.

Examples on 15-second bars:

```text
300 seconds  -> 20 raw horizon bars -> 128 compiled horizon slots
900 seconds  -> 60 raw horizon bars -> 128 compiled horizon slots
3600 seconds -> 240 raw horizon bars -> 256 compiled horizon slots
```

## Large historical ranges

The model is not evaluated on every row by default. Meteor Quant calculates the requested number of rolling forecasts and, when necessary, increases `prediction_stride` so it does not exceed `max_predictions`. Each forecast is then forward-filled until the next evaluated endpoint.

This keeps the authoritative engine at the original bar resolution while bounding 200M-parameter inference work.

For the complete multi-year dataset:

- prefer 15-second or slower bars;
- leave `auto_increase_stride=true`;
- start with `max_predictions=20000`;
- use a smaller date range when comparing fine prediction strides.

The default `maximum_output_rows=25,000,000` rejects an excessively large result frame before allocating seven forecast columns. Full multi-year 1-second inference should be split into ranges or run on a coarser timeframe.

## Causality

For endpoint row `N`, Meteor Quant constructs an input window ending at row `N`. No row after `N` is sent to TimesFM. The forecast is attached to row `N`, forward-filled only into subsequent rows, and any resulting target fills at row `N+1` open.

Therefore the integration does not use future bars to create the signal attached to the current row.

## Local and offline models

To use an explicitly downloaded directory:

```text
model_id_or_path = C:\models\timesfm-2.5-200m-pytorch
local_files_only = true
```

Alternatively, keep the official model ID and set `local_files_only=true` after it has been cached by `download-timesfm.ps1`.

The model source identity and installed TimesFM package version participate in the signal-cache key. Replacing a local checkpoint or package invalidates old cached signals.

## API diagnostics

Check the integration from PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/timesfm/capabilities | ConvertTo-Json -Depth 10
```

Expected GPU fields include:

```json
{
  "installed": true,
  "torch_installed": true,
  "cuda_available": true,
  "device": "NVIDIA GeForce RTX 4060 Ti",
  "model_id": "google/timesfm-2.5-200m-pytorch"
}
```

## Current limitations

- The integration is univariate and uses close prices only.
- TimesFM covariate/XReg support is not wired into the Meteor Quant indicator yet.
- The model is not fine-tuned from the dashboard.
- Live Kraken inference is constrained by the number and interval of bars available to the paper session.
- The first use may pause while the approximately 1 GB checkpoint is downloaded and loaded.
