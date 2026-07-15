from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray

from meteor_quant.domain import Bar, IndicatorSpec, StrategyDecision
from meteor_quant.markethybrid.model import MarketHybrid
from meteor_quant.markethybrid.schemas import (
    MarketHybridIndicatorParameters,
    MarketHybridRunRequest,
)
from meteor_quant.marketlm.dataset import (
    PreparedMarketLMMetadata,
    load_prepared_metadata,
)
from meteor_quant.marketlm.features import build_feature_frame
from meteor_quant.strategies.sdk import SignalPlan


class MarketHybridIndicator:
    """Deployable MarketHybrid forecast and learned-policy indicator."""

    def __init__(self, registration: dict[str, Any]) -> None:
        self.registration = registration
        self.model_id = str(registration["model_id"])
        self.checkpoint_path = Path(str(registration["checkpoint_path"]))
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"MarketHybrid checkpoint not found: {self.checkpoint_path}")
        self.prepared_dir = Path(str(registration["prepared_dir"]))
        self.metadata: PreparedMarketLMMetadata = load_prepared_metadata(self.prepared_dir)
        self.primary_horizon_seconds = int(registration["primary_horizon_seconds"])
        self._checkpoint: dict[str, Any] | None = None
        self._request: MarketHybridRunRequest | None = None
        self._models: dict[str, MarketHybrid] = {}

    @classmethod
    def from_registered(
        cls,
        model_id: str,
        registered_dir: str | Path,
    ) -> MarketHybridIndicator:
        path = Path(registered_dir) / f"{model_id}.json"
        if not path.exists():
            raise KeyError(f"registered MarketHybrid model not found: {model_id}")
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        return cls(payload)

    @property
    def request(self) -> MarketHybridRunRequest:
        self._load_checkpoint()
        assert self._request is not None
        return self._request

    @property
    def minimum_bars(self) -> int:
        return self.metadata.context_bars + self.metadata.indicator_warmup_bars

    def indicator_specs(
        self,
        horizon_seconds: int | None = None,
    ) -> tuple[IndicatorSpec, ...]:
        horizon = self._resolve_horizon(horizon_seconds)
        return (
            IndicatorSpec(
                "markethybrid_forecast_price",
                f"MarketHybrid median {horizon}s price",
                "price",
                "price",
                2,
                horizon,
            ),
            IndicatorSpec(
                "markethybrid_lower_price",
                f"MarketHybrid q10 {horizon}s price",
                "price",
                "price",
                1,
                horizon,
            ),
            IndicatorSpec(
                "markethybrid_upper_price",
                f"MarketHybrid q90 {horizon}s price",
                "price",
                "price",
                1,
                horizon,
            ),
            IndicatorSpec(
                "markethybrid_return_bps",
                f"MarketHybrid median {horizon}s (bps)",
                "indicator",
                "number",
            ),
            IndicatorSpec(
                "markethybrid_lower_bps",
                f"MarketHybrid q10 {horizon}s",
                "indicator",
                "number",
                1,
            ),
            IndicatorSpec(
                "markethybrid_upper_bps",
                f"MarketHybrid q90 {horizon}s",
                "indicator",
                "number",
                1,
            ),
            IndicatorSpec(
                "markethybrid_prob_up",
                "MarketHybrid P(up)",
                "indicator",
                "percent",
                1,
            ),
            IndicatorSpec(
                "markethybrid_prob_down",
                "MarketHybrid P(down)",
                "indicator",
                "percent",
                1,
            ),
            IndicatorSpec(
                "markethybrid_actionable",
                "MarketHybrid actionable",
                "indicator",
                "percent",
                1,
            ),
            IndicatorSpec(
                "markethybrid_policy_position",
                "MarketHybrid policy position",
                "indicator",
                "number",
                1,
            ),
            IndicatorSpec(
                "markethybrid_policy_confidence",
                "MarketHybrid policy confidence",
                "indicator",
                "percent",
                1,
            ),
            IndicatorSpec(
                "markethybrid_policy_intent",
                "MarketHybrid intent (-1/0/+1)",
                "indicator",
                "number",
                1,
            ),
        )

    def build_signal_plan(
        self,
        frame: pl.LazyFrame,
        parameters: MarketHybridIndicatorParameters,
        *,
        horizon_seconds: int | None = None,
    ) -> SignalPlan:
        selected_horizon = self._resolve_horizon(horizon_seconds)
        enriched = self.predict_frame(
            frame,
            parameters,
            horizon_seconds=selected_horizon,
        )
        return SignalPlan(
            enriched.lazy(),
            target_column="target_fraction",
            reason_column="signal_reason",
            plot_columns=tuple(
                spec.key for spec in self.indicator_specs(selected_horizon)
            ),
        )

    def predict_frame(
        self,
        frame: pl.LazyFrame,
        parameters: MarketHybridIndicatorParameters,
        *,
        horizon_seconds: int | None = None,
    ) -> pl.DataFrame:
        selected_horizon = self._resolve_horizon(horizon_seconds)
        request = self.request
        feature_frame, feature_names, _targets = build_feature_frame(
            frame,
            request.data,
            include_targets=False,
        )
        canonical = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        schema_names = set(feature_frame.collect_schema().names())
        selected_canonical = [name for name in canonical if name in schema_names]
        collected = feature_frame.select([*selected_canonical, *feature_names]).collect(
            engine="streaming"
        )
        if collected.height < self.metadata.context_bars:
            return self._empty_predictions(collected, parameters)
        timestamps = (
            collected.get_column("timestamp")
            .to_numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )
        if len(timestamps) > 2:
            positive_deltas = np.diff(timestamps)
            positive_deltas = positive_deltas[positive_deltas > 0]
            if positive_deltas.size:
                observed = int(np.median(positive_deltas))
                if observed != self.metadata.timeframe_seconds:
                    raise ValueError(
                        f"MarketHybrid expects {self.metadata.timeframe_seconds}s bars, "
                        f"but the selected frame is approximately {observed}s"
                    )

        matrix = (
            collected.select(feature_names)
            .to_numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )
        finite_rows = np.isfinite(matrix).all(axis=1)
        mean = np.asarray(self.metadata.feature_mean, dtype=np.float32)
        std = np.asarray(self.metadata.feature_std, dtype=np.float32)
        normalized = np.clip((matrix - mean) / std, -12.0, 12.0)
        normalized[~np.isfinite(normalized)] = 0.0

        row_count = collected.height
        outputs = {
            "median": np.full(row_count, np.nan, dtype=np.float64),
            "lower": np.full(row_count, np.nan, dtype=np.float64),
            "upper": np.full(row_count, np.nan, dtype=np.float64),
            "probability_up": np.full(row_count, np.nan, dtype=np.float64),
            "probability_down": np.full(row_count, np.nan, dtype=np.float64),
            "actionable": np.full(row_count, np.nan, dtype=np.float64),
            "policy_position": np.full(row_count, np.nan, dtype=np.float64),
            "policy_confidence": np.full(row_count, np.nan, dtype=np.float64),
            "policy_intent": np.full(row_count, np.nan, dtype=np.float64),
        }
        context_bars = self.metadata.context_bars
        invalid = (~finite_rows).astype(np.int64)
        prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(invalid)))
        candidates = np.arange(
            context_bars - 1,
            row_count,
            parameters.prediction_stride,
        )
        valid_endpoints = [
            int(endpoint)
            for endpoint in candidates
            if prefix[endpoint + 1] - prefix[endpoint - context_bars + 1] == 0
        ]
        model, device = self._model(parameters.device)
        target_mean = np.asarray(self.metadata.target_mean, dtype=np.float32)
        target_std = np.asarray(self.metadata.target_std, dtype=np.float32)
        primary_index = self.metadata.horizons_seconds.index(selected_horizon)

        for offset in range(0, len(valid_endpoints), parameters.batch_size):
            endpoint_batch = valid_endpoints[offset : offset + parameters.batch_size]
            if not endpoint_batch:
                continue
            windows = np.stack(
                [
                    normalized[endpoint - context_bars + 1 : endpoint + 1]
                    for endpoint in endpoint_batch
                ],
                axis=0,
            ).reshape(
                len(endpoint_batch),
                self.metadata.context_patches,
                self.metadata.patch_size,
                self.metadata.feature_dim,
            )
            tensor = torch.from_numpy(windows).to(device, non_blocking=True)
            with torch.inference_mode():
                prediction = model(tensor)
                quantiles = prediction.quantiles.float().cpu().numpy()
                probabilities = prediction.direction_logits.float().softmax(dim=-1).cpu().numpy()
                actionable = prediction.actionable_logits.float().sigmoid().cpu().numpy()
                policy_position = prediction.target_position.float().cpu().numpy()
                policy_confidence = (
                    prediction.policy_confidence_logits.float().sigmoid().cpu().numpy()
                )
                policy_intent = (
                    prediction.execution_intent_logits.float().argmax(dim=-1).cpu().numpy() - 1
                )
            quantiles = quantiles * target_std[None, :, None] + target_mean[None, :, None]
            for batch_index, endpoint in enumerate(endpoint_batch):
                outputs["lower"][endpoint] = quantiles[batch_index, primary_index, 0]
                outputs["median"][endpoint] = quantiles[batch_index, primary_index, 1]
                outputs["upper"][endpoint] = quantiles[batch_index, primary_index, 2]
                outputs["probability_down"][endpoint] = probabilities[
                    batch_index,
                    primary_index,
                    0,
                ]
                outputs["probability_up"][endpoint] = probabilities[
                    batch_index,
                    primary_index,
                    2,
                ]
                outputs["actionable"][endpoint] = actionable[
                    batch_index,
                    primary_index,
                ]
                outputs["policy_position"][endpoint] = policy_position[batch_index]
                outputs["policy_confidence"][endpoint] = policy_confidence[batch_index]
                outputs["policy_intent"][endpoint] = policy_intent[batch_index]

        close_values = collected.get_column("close").to_numpy().astype(np.float64, copy=False)
        forecast_price = self._forward_fill(
            self._price_from_log_return_bps(close_values, outputs["median"])
        )
        lower_price = self._forward_fill(
            self._price_from_log_return_bps(close_values, outputs["lower"])
        )
        upper_price = self._forward_fill(
            self._price_from_log_return_bps(close_values, outputs["upper"])
        )
        for key, values in outputs.items():
            outputs[key] = self._forward_fill(values)
        targets, reasons = self._targets(outputs, parameters)
        return collected.with_columns(
            pl.Series("markethybrid_forecast_price", forecast_price),
            pl.Series("markethybrid_lower_price", lower_price),
            pl.Series("markethybrid_upper_price", upper_price),
            pl.Series("markethybrid_return_bps", outputs["median"]),
            pl.Series("markethybrid_lower_bps", outputs["lower"]),
            pl.Series("markethybrid_upper_bps", outputs["upper"]),
            pl.Series("markethybrid_prob_up", outputs["probability_up"] * 100.0),
            pl.Series("markethybrid_prob_down", outputs["probability_down"] * 100.0),
            pl.Series("markethybrid_actionable", outputs["actionable"] * 100.0),
            pl.Series(
                "markethybrid_policy_position",
                outputs["policy_position"],
            ),
            pl.Series(
                "markethybrid_policy_confidence",
                outputs["policy_confidence"] * 100.0,
            ),
            pl.Series("markethybrid_policy_intent", outputs["policy_intent"]),
            pl.Series("target_fraction", targets),
            pl.Series("signal_reason", reasons),
        )

    def predict_live(
        self,
        bars: list[Bar],
        parameters: MarketHybridIndicatorParameters,
        *,
        horizon_seconds: int | None = None,
    ) -> StrategyDecision:
        selected_horizon = self._resolve_horizon(horizon_seconds)
        if len(bars) < self.minimum_bars:
            return StrategyDecision(
                plots={
                    spec.key: None
                    for spec in self.indicator_specs(selected_horizon)
                }
            )
        frame = pl.DataFrame(
            {
                "timestamp": [bar.timestamp for bar in bars],
                "open": [bar.open for bar in bars],
                "high": [bar.high for bar in bars],
                "low": [bar.low for bar in bars],
                "close": [bar.close for bar in bars],
                "volume": [bar.volume for bar in bars],
                "quote_asset_volume": [bar.quote_asset_volume for bar in bars],
                "number_of_trades": [bar.number_of_trades for bar in bars],
                "taker_buy_base_volume": [bar.taker_buy_base_volume for bar in bars],
                "taker_buy_quote_volume": [bar.taker_buy_quote_volume for bar in bars],
            }
        )
        enriched = self.predict_frame(
            frame.lazy(),
            parameters.model_copy(update={"prediction_stride": 1, "batch_size": 1}),
            horizon_seconds=selected_horizon,
        )
        last = enriched.row(-1, named=True)
        plots = {
            spec.key: (
                float(last[spec.key])
                if last.get(spec.key) is not None and np.isfinite(float(last[spec.key]))
                else None
            )
            for spec in self.indicator_specs(selected_horizon)
        }
        target_value = last.get("target_fraction")
        target = float(target_value) if target_value is not None else None
        return StrategyDecision(target, str(last.get("signal_reason") or ""), plots)

    def _resolve_horizon(self, horizon_seconds: int | None) -> int:
        selected = self.primary_horizon_seconds if horizon_seconds is None else int(horizon_seconds)
        if selected not in self.metadata.horizons_seconds:
            raise ValueError(
                f"MarketHybrid model {self.model_id} does not provide a {selected}-second "
                f"forecast; available horizons are {self.metadata.horizons_seconds}"
            )
        return selected

    def _load_checkpoint(self) -> None:
        if self._checkpoint is not None:
            return
        checkpoint = cast(
            dict[str, Any],
            torch.load(self.checkpoint_path, map_location="cpu", weights_only=False),
        )
        request = MarketHybridRunRequest.model_validate(checkpoint["request"])
        self._checkpoint = checkpoint
        self._request = request

    def _model(self, requested_device: str) -> tuple[MarketHybrid, torch.device]:
        self._load_checkpoint()
        assert self._checkpoint is not None
        request = self.request
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
            device = torch.device("cuda")
        elif requested_device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        key = str(device)
        model = self._models.get(key)
        if model is None:
            model = MarketHybrid(
                feature_dim=self.metadata.feature_dim,
                patch_size=self.metadata.patch_size,
                horizons=len(self.metadata.horizons_seconds),
                request=request,
                include_training_modules=False,
            )
            deployment = self._checkpoint.get("deployment_model")
            if not isinstance(deployment, dict):
                raise ValueError("checkpoint does not contain a deployable MarketHybrid model")
            model.load_deployment_state_dict(deployment)
            model.to(device)
            model.eval()
            self._models[key] = model
        return model, device

    def _empty_predictions(
        self,
        frame: pl.DataFrame,
        parameters: MarketHybridIndicatorParameters,
    ) -> pl.DataFrame:
        row_count = frame.height
        nulls = np.full(row_count, np.nan)
        return frame.with_columns(
            *[
                pl.Series(name, nulls)
                for name in (
                    "markethybrid_forecast_price",
                    "markethybrid_lower_price",
                    "markethybrid_upper_price",
                    "markethybrid_return_bps",
                    "markethybrid_lower_bps",
                    "markethybrid_upper_bps",
                    "markethybrid_prob_up",
                    "markethybrid_prob_down",
                    "markethybrid_actionable",
                    "markethybrid_policy_position",
                    "markethybrid_policy_confidence",
                    "markethybrid_policy_intent",
                )
            ],
            pl.Series(
                "target_fraction",
                np.full(row_count, parameters.flat_target, dtype=np.float64),
            ),
            pl.Series("signal_reason", ["markethybrid_warmup"] * row_count),
        )

    @staticmethod
    def _price_from_log_return_bps(
        close: np.ndarray,
        return_bps: np.ndarray,
    ) -> np.ndarray:
        output = np.full(len(close), np.nan, dtype=np.float64)
        valid = np.isfinite(close) & (close > 0.0) & np.isfinite(return_bps)
        output[valid] = close[valid] * np.exp(return_bps[valid] / 10_000.0)
        return output

    @staticmethod
    def _forward_fill(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not finite.any():
            return values
        indices = np.where(finite, np.arange(len(values)), -1)
        np.maximum.accumulate(indices, out=indices)
        output = values.copy()
        valid = indices >= 0
        output[valid] = values[indices[valid]]
        return output

    @staticmethod
    def _targets(
        outputs: dict[str, np.ndarray],
        parameters: MarketHybridIndicatorParameters,
    ) -> tuple[NDArray[np.float64], list[str]]:
        row_count = len(outputs["median"])
        targets: NDArray[np.float64] = np.full(
            row_count,
            parameters.flat_target,
            dtype=np.float64,
        )
        reasons = ["markethybrid_warmup"] * row_count
        current = parameters.flat_target
        for index in range(row_count):
            values = {key: array[index] for key, array in outputs.items()}
            if not all(np.isfinite(value) for value in values.values()):
                targets[index] = current
                continue
            if parameters.mode == "indicator_only":
                current = parameters.flat_target
                reasons[index] = "markethybrid_indicator_only"
            elif parameters.mode == "median_sign_long_short":
                if values["median"] > parameters.long_threshold_bps:
                    current = parameters.long_target
                    reasons[index] = "markethybrid_300s_median_positive"
                elif values["median"] < parameters.short_threshold_bps:
                    current = parameters.short_target
                    reasons[index] = "markethybrid_300s_median_negative"
                else:
                    reasons[index] = "markethybrid_300s_median_hold"
            elif parameters.mode == "model_policy":
                qualified = (
                    values["actionable"] >= parameters.actionable_probability_threshold
                    and values["policy_confidence"] >= parameters.policy_confidence_threshold
                    and abs(values["policy_position"]) >= parameters.policy_position_deadband
                )
                if qualified:
                    current = float(
                        np.clip(
                            values["policy_position"] * parameters.position_multiplier,
                            -parameters.maximum_absolute_target,
                            parameters.maximum_absolute_target,
                        )
                    )
                    reasons[index] = "markethybrid_policy"
                else:
                    current = parameters.flat_target
                    reasons[index] = "markethybrid_policy_filtered"
            else:
                bullish = (
                    values["median"] >= parameters.long_threshold_bps
                    and values["probability_up"] >= parameters.direction_confidence_threshold
                    and values["actionable"] >= parameters.actionable_probability_threshold
                )
                bearish = (
                    values["median"] <= parameters.short_threshold_bps
                    and values["probability_down"] >= parameters.direction_confidence_threshold
                    and values["actionable"] >= parameters.actionable_probability_threshold
                )
                if bullish:
                    current = parameters.long_target
                    reasons[index] = "markethybrid_bullish"
                elif bearish:
                    current = (
                        parameters.short_target
                        if parameters.mode == "forecast_long_short"
                        else parameters.flat_target
                    )
                    reasons[index] = "markethybrid_bearish"
                else:
                    current = parameters.flat_target
                    reasons[index] = "markethybrid_no_edge"
            targets[index] = current
        return targets, reasons
