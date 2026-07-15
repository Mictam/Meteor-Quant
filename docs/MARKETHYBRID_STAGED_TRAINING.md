# MarketHybrid staged training

Meteor Quant separates representation learning from deployable policy optimization.

## Stages

### Representation pretraining — steps 0–8,000

Policy heads are frozen. The encoder learns causal market representations through next-patch reconstruction, return quantiles, direction/actionability and JEPA future-latent prediction.

### Joint training — steps 8,001–24,000

All deployable heads are unfrozen. Forecasting, actionability, policy position, policy confidence and execution intent jointly shape the encoder.

### Policy fine-tuning — steps 24,001–40,000

The JEPA predictor is frozen. The encoder remains trainable at 20% of the stage learning rate, while policy/actionability heads receive full learning rate. Hybrid-score early stopping is enabled.

## Checkpoints

- `best_loss.pt`: lowest weighted validation loss.
- `best_hybrid.pt`: highest transparent hybrid validation score.
- `best.pt`: configured selection winner; defaults to `best_hybrid.pt` semantics.
- `final.pt`: terminal model, or restored best-hybrid model after policy-stage early stopping.

The hybrid score combines actionable PR-AUC, direction macro F1, policy-intent macro F1, policy-position sign accuracy, inverse position MAE, forecast rank IC and quantile calibration.

## Resume safety

Every checkpoint records global step, stage, stage-local step, schedule hash, optimizer, AMP scaler, EMA teacher and RNG states. A changed schedule fails closed unless `allow_schedule_change_on_resume` is explicitly enabled.

## Cache behavior

Prepared tensor fingerprints include data- and label-affecting inputs such as `cost_threshold_bps`. Training stages, loss weights, learning rates, batch size and optimizer settings do not alter the prepared-data fingerprint.
