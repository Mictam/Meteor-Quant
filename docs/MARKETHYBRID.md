# Train MarketHybrid

Meteor Quant includes a dedicated **Train MarketHybrid** tab. MarketHybrid keeps the causal MarketLM forecasting objectives and adds JEPA future-latent prediction, actionability classification, and deployable policy heads.

## Architecture

```text
causal feature patches
        |
        v
online market encoder --------------------+
        |                                  |
        +-> next-patch reconstruction      +-> return quantiles
        +-> direction logits               +-> actionable probability
        +-> policy position/confidence/intent
        |
        +-> JEPA predictor -> future latent targets
                              ^
                              |
                    EMA target encoder
                    (training only)
```

The EMA target encoder and JEPA predictor are used only while training. Registered checkpoints load the smaller deployment graph: online encoder plus forecast, actionability, and policy heads.

## Objectives

For every batch, MarketHybrid jointly optimizes:

- next-feature-patch reconstruction;
- q10/median/q90 future-return forecasts for every selected horizon;
- down/flat/up direction classification;
- actionable-after-cost classification;
- JEPA future-latent prediction at configurable patch offsets;
- latent variance and covariance regularization;
- target position regression;
- policy confidence regression;
- execution-intent classification.

All labels are created from future bars, but inputs remain strictly causal. Splits are chronological, separated by the configured purge gap, and normalization is fitted only on the training split.

## Data reuse

MarketHybrid uses the same feature fingerprint and memory-mapped tensors as MarketLM. If dataset, range, timeframe, indicators, horizons, patch size, context, splits, and cost threshold are unchanged, no second multi-year preparation is required.

## Recommended RTX 4060 Ti configuration

For 15-second BTCUSDT research:

```text
horizons:                 30, 60, 180, 300, 900 seconds
patch size:               8
context patches:          256
encoder:                  384 x 12 layers x 6 heads
predictor:                256 x 4 layers x 4 heads
JEPA offsets:             1, 2, 4, 7 patches
EMA teacher:              0.996 -> 0.9999
batch size:               8-16 initially
activation checkpointing: enabled
gradient accumulation:    2-4
AMP:                      auto
workers on Windows:       0
```

The largest JEPA target offset multiplied by patch size must fit inside the largest forecast horizon measured in bars.

## Registered indicator outputs

A completed checkpoint can be registered as `MarketHybrid · <name>`. It exposes:

```text
markethybrid_return_bps
markethybrid_lower_bps
markethybrid_upper_bps
markethybrid_prob_up
markethybrid_prob_down
markethybrid_actionable
markethybrid_policy_position
markethybrid_policy_confidence
markethybrid_policy_intent
```

Available modes:

- `indicator_only`: plot forecasts and policy outputs without trading;
- `model_policy`: use the learned position, confidence, and actionability heads;
- `forecast_long_flat`: derive long/flat targets from forecast thresholds;
- `forecast_long_short`: derive long/flat/short targets from forecasts.

All targets still pass through the authoritative broker simulator with next-bar-open fills, fees, spread, slippage, leverage limits, and shorting policy.

## Console training

Create a JSON configuration and submit it through the running API, or use the CLI:

```powershell
meteor start-markethybrid --config .\markethybrid-15s.json
```

Monitor it with:

```powershell
meteor markethybrid-status --run-id <RUN_ID>
Get-Content .\data\markethybrid\runs\<RUN_ID>\worker.log -Wait -Tail 50
```

Stop cleanly:

```powershell
meteor stop-markethybrid --run-id <RUN_ID>
```

Register the best checkpoint:

```powershell
meteor register-markethybrid --run-id <RUN_ID> --checkpoint best --primary-horizon 300
```

## Storage

```text
data/marketlm/prepared/<fingerprint>/   shared prepared tensors
data/markethybrid/runs/<run-id>/        logs, metrics, checkpoints, status
data/markethybrid/registered/<id>.json  registered deployment models
```

## CUDA

Run `setup-markethybrid.ps1` after installing a CUDA-enabled PyTorch wheel in the project `.venv`. The script refuses CPU-only PyTorch so a 50M-parameter hybrid job cannot silently start on the CPU.


## 300-second median-sign strategy

Every registered MarketHybrid model that contains a `300`-second forecast horizon now
adds a second strategy to the backtest and Kraken paper strategy lists:

```text
MarketHybrid 300s Median Sign · <registered model name>
```

The strategy intentionally ignores actionability, direction-confidence, and learned-policy
filters. On each completed source bar it evaluates the model's 300-second median return:

- median greater than the configured deadband: target `long_target`;
- median lower than the negative deadband: target `short_target`;
- median inside the deadband, including exactly zero: retain the previous target.

The default deadband is `0.0` basis points, which implements the exact positive/negative
sign rule. A repeated median with the same sign does not submit another paper fill because
the authoritative execution layer sees an unchanged target. A sign reversal produces one
rebalance order that closes and, when `Allow short` is enabled, reverses the position. When
shorting is disabled, a negative target can only close a long position and remain flat.

Recommended paper settings for the 15-second model:

```text
Prediction stride:      1
Median deadband (bps):  0
Long target:            1
Short target:          -1
Live timeframe:         15 seconds
Bootstrap bars:         at least the model minimum (normally about 2,148+)
Allow short:            enabled for true long/short reversal
```
