# Shared BTC/USD forecast overlay

Meteor Quant plots MarketLM and MarketHybrid forecasts directly on the BTC/USD candle chart.

## Price conversion

The trained targets are log returns in basis points. For a prediction made at close price `P_t`, the displayed target price is:

```text
forecast_price = P_t * exp(predicted_log_return_bps / 10,000)
```

The median, q10, and q90 outputs are converted independently.

## Axis alignment

- The candle series and forecast-price lines use the same Lightweight Charts `right` price scale.
- Forecast points are plotted at `prediction timestamp + primary horizon`, so their x-coordinate matches the candle they predict.
- Return-bps, probabilities, actionability, and policy values remain in the lower indicator pane because they are not price-valued quantities.
- For sub-minute sessions, second labels are enabled automatically.

Existing trained checkpoints remain compatible. No retraining or prepared-data rebuild is required.
