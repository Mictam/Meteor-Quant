from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray

from meteor_quant.domain import Bar, IndicatorSpec, StrategyDecision
from meteor_quant.marketlm.dataset import (
    PreparedMarketLMMetadata,
    load_prepared_metadata,
)
from meteor_quant.marketlm.features import build_feature_frame
from meteor_quant.marketlm.model import MarketLM
from meteor_quant.marketlm.schemas import (
    MarketLMIndicatorParameters,
    MarketLMRunRequest,
)
from meteor_quant.strategies.sdk import SignalPlan


class MarketLMIndicator:
    """Reusable trained MarketLM indicator for backtests and Python plugins."""

    def __init__(self, registration: dict[str, Any]) -> None:
        self.registration = registration
        self.model_id = str(registration["model_id"])
        self.checkpoint_path = Path(str(registration["checkpoint_path"]))
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"MarketLM checkpoint not found: {self.checkpoint_path}")
        self.prepared_dir = Path(str(registration["prepared_dir"]))
        self.metadata: PreparedMarketLMMetadata = load_prepared_metadata(self.prepared_dir)
        self.primary_horizon_seconds = int(registration["primary_horizon_seconds"])
        self._checkpoint: dict[str, Any] | None = None
        self._request: MarketLMRunRequest | None = None
        self._models: dict[str, MarketLM] = {}

    @classmethod
    def from_registered(cls, model_id: str, registered_dir: str | Path) -> MarketLMIndicator:
        path = Path(registered_dir) / f"{model_id}.json"
        if not path.exists():
            raise KeyError(f"registered MarketLM model not found: {model_id}")
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        return cls(payload)

    @property
    def request(self) -> MarketLMRunRequest:
        self._load_checkpoint()
        assert self._request is not None
        return self._request

    @property
    def minimum_bars(self) -> int:
        return self.metadata.context_bars + self.metadata.indicator_warmup_bars

    def indicator_specs(self) -> tuple[IndicatorSpec, ...]:
        horizon = self.primary_horizon_seconds
        return (
            IndicatorSpec(
                "marketlm_forecast_price",
                f"MarketLM median {horizon}s price",
                "price",
                "price",
                2,
                horizon,
            ),
            IndicatorSpec(
                "marketlm_lower_price",
                f"MarketLM q10 {horizon}s price",
                "price",
                "price",
                1,
                horizon,
            ),
            IndicatorSpec(
                "marketlm_upper_price",
                f"MarketLM q90 {horizon}s price",
                "price",
                "price",
                1,
                horizon,
            ),
            IndicatorSpec(
                "marketlm_return_bps",
                f"MarketLM median {horizon}s (bps)",
                "indicator",
                "number",
            ),
            IndicatorSpec(
                "marketlm_lower_bps",
                f"MarketLM q10 {horizon}s",
                "indicator",
                "number",
                1,
            ),
            IndicatorSpec(
                "marketlm_upper_bps",
                f"MarketLM q90 {horizon}s",
                "indicator",
                "number",
                1,
            ),
            IndicatorSpec(
                "marketlm_prob_up",
                "MarketLM P(up)",
                "indicator",
                "percent",
                1,
            ),
            IndicatorSpec(
                "marketlm_prob_down",
                "MarketLM P(down)",
                "indicator",
                "percent",
                1,
            ),
        )

    def build_signal_plan(
        self,
        frame: pl.LazyFrame,
        parameters: MarketLMIndicatorParameters,
    ) -> SignalPlan:
        enriched = self.predict_frame(frame, parameters)
        return SignalPlan(
            enriched.lazy(),
            target_column="target_fraction",
            reason_column="signal_reason",
            plot_columns=tuple(spec.key for spec in self.indicator_specs()),
        )

    def predict_frame(
        self,
        frame: pl.LazyFrame,
        parameters: MarketLMIndicatorParameters,
    ) -> pl.DataFrame:
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
        timestamps = collected.get_column("timestamp").to_numpy().astype(np.int64, copy=False)
        if len(timestamps) > 2:
            positive_deltas = np.diff(timestamps)
            positive_deltas = positive_deltas[positive_deltas > 0]
            if positive_deltas.size:
                observed = int(np.median(positive_deltas))
                if observed != self.metadata.timeframe_seconds:
                    raise ValueError(
                        f"MarketLM model expects {self.metadata.timeframe_seconds}s bars, "
                        f"but the selected backtest frame is approximately {observed}s"
                    )

        matrix = collected.select(feature_names).to_numpy().astype(np.float32, copy=False)
        finite_rows = np.isfinite(matrix).all(axis=1)
        mean = np.asarray(self.metadata.feature_mean, dtype=np.float32)
        std = np.asarray(self.metadata.feature_std, dtype=np.float32)
        normalized = np.clip((matrix - mean) / std, -12.0, 12.0)
        normalized[~np.isfinite(normalized)] = 0.0

        row_count = collected.height
        median = np.full(row_count, np.nan, dtype=np.float64)
        lower = np.full(row_count, np.nan, dtype=np.float64)
        upper = np.full(row_count, np.nan, dtype=np.float64)
        probability_up = np.full(row_count, np.nan, dtype=np.float64)
        probability_down = np.full(row_count, np.nan, dtype=np.float64)

        context_bars = self.metadata.context_bars
        invalid = (~finite_rows).astype(np.int64)
        prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(invalid)))
        candidates = np.arange(context_bars - 1, row_count, parameters.prediction_stride)
        valid_endpoints = [
            int(endpoint)
            for endpoint in candidates
            if prefix[endpoint + 1] - prefix[endpoint - context_bars + 1] == 0
        ]
        model, device = self._model(parameters.device)
        target_mean = np.asarray(self.metadata.target_mean, dtype=np.float32)
        target_std = np.asarray(self.metadata.target_std, dtype=np.float32)
        patch_size = self.metadata.patch_size
        context_patches = self.metadata.context_patches
        primary_index = self.metadata.horizons_seconds.index(self.primary_horizon_seconds)

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
                context_patches,
                patch_size,
                self.metadata.feature_dim,
            )
            tensor = torch.from_numpy(windows).to(device, non_blocking=True)
            with torch.inference_mode():
                prediction = model(tensor)
                quantiles = prediction.quantiles.float().cpu().numpy()
                probabilities = prediction.direction_logits.float().softmax(dim=-1).cpu().numpy()
            quantiles = quantiles * target_std[None, :, None] + target_mean[None, :, None]
            for batch_index, endpoint in enumerate(endpoint_batch):
                lower[endpoint] = quantiles[batch_index, primary_index, 0]
                median[endpoint] = quantiles[batch_index, primary_index, 1]
                upper[endpoint] = quantiles[batch_index, primary_index, 2]
                probability_down[endpoint] = probabilities[batch_index, primary_index, 0]
                probability_up[endpoint] = probabilities[batch_index, primary_index, 2]

        close_values = collected.get_column("close").to_numpy().astype(np.float64, copy=False)
        forecast_price = self._forward_fill(self._price_from_log_return_bps(close_values, median))
        lower_price = self._forward_fill(self._price_from_log_return_bps(close_values, lower))
        upper_price = self._forward_fill(self._price_from_log_return_bps(close_values, upper))
        median = self._forward_fill(median)
        lower = self._forward_fill(lower)
        upper = self._forward_fill(upper)
        probability_up = self._forward_fill(probability_up)
        probability_down = self._forward_fill(probability_down)
        targets, reasons = self._targets(
            median,
            probability_up,
            probability_down,
            parameters,
        )
        return collected.with_columns(
            pl.Series("marketlm_forecast_price", forecast_price),
            pl.Series("marketlm_lower_price", lower_price),
            pl.Series("marketlm_upper_price", upper_price),
            pl.Series("marketlm_return_bps", median),
            pl.Series("marketlm_lower_bps", lower),
            pl.Series("marketlm_upper_bps", upper),
            pl.Series("marketlm_prob_up", probability_up * 100.0),
            pl.Series("marketlm_prob_down", probability_down * 100.0),
            pl.Series("target_fraction", targets),
            pl.Series("signal_reason", reasons),
        )

    def predict_live(
        self,
        bars: list[Bar],
        parameters: MarketLMIndicatorParameters,
    ) -> StrategyDecision:
        if len(bars) < self.minimum_bars:
            return StrategyDecision(plots={spec.key: None for spec in self.indicator_specs()})
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
        live_parameters = parameters.model_copy(update={"prediction_stride": 1, "batch_size": 1})
        enriched = self.predict_frame(frame.lazy(), live_parameters)
        last = enriched.row(-1, named=True)
        plots = {
            spec.key: (
                float(last[spec.key])
                if last.get(spec.key) is not None and np.isfinite(float(last[spec.key]))
                else None
            )
            for spec in self.indicator_specs()
        }
        target_value = last.get("target_fraction")
        target = float(target_value) if target_value is not None else None
        return StrategyDecision(target, str(last.get("signal_reason") or ""), plots)

    def _load_checkpoint(self) -> None:
        if self._checkpoint is not None:
            return
        checkpoint = cast(
            dict[str, Any],
            torch.load(self.checkpoint_path, map_location="cpu", weights_only=False),
        )
        request = MarketLMRunRequest.model_validate(checkpoint["request"])
        self._checkpoint = checkpoint
        self._request = request

    def _model(self, requested_device: str) -> tuple[MarketLM, torch.device]:
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
            model = MarketLM(
                feature_dim=self.metadata.feature_dim,
                patch_size=self.metadata.patch_size,
                horizons=len(self.metadata.horizons_seconds),
                config=request.model,
            )
            model.load_state_dict(self._checkpoint["model"])
            model.to(device)
            model.eval()
            self._models[key] = model
        return model, device

    def _empty_predictions(
        self,
        frame: pl.DataFrame,
        parameters: MarketLMIndicatorParameters,
    ) -> pl.DataFrame:
        row_count = frame.height
        targets = np.full(row_count, parameters.flat_target, dtype=np.float64)
        reasons = ["marketlm_warmup"] * row_count
        nulls = np.full(row_count, np.nan)
        return frame.with_columns(
            pl.Series("marketlm_forecast_price", nulls),
            pl.Series("marketlm_lower_price", nulls),
            pl.Series("marketlm_upper_price", nulls),
            pl.Series("marketlm_return_bps", nulls),
            pl.Series("marketlm_lower_bps", nulls),
            pl.Series("marketlm_upper_bps", nulls),
            pl.Series("marketlm_prob_up", nulls),
            pl.Series("marketlm_prob_down", nulls),
            pl.Series("target_fraction", targets),
            pl.Series("signal_reason", reasons),
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
        median: np.ndarray,
        probability_up: np.ndarray,
        probability_down: np.ndarray,
        parameters: MarketLMIndicatorParameters,
    ) -> tuple[np.ndarray, list[str]]:
        targets: NDArray[np.float64] = np.full(
            len(median), parameters.flat_target, dtype=np.float64
        )
        reasons = ["marketlm_warmup"] * len(median)
        current = parameters.flat_target
        for index in range(len(median)):
            if not (
                np.isfinite(median[index])
                and np.isfinite(probability_up[index])
                and np.isfinite(probability_down[index])
            ):
                targets[index] = current
                continue
            if parameters.mode == "indicator_only":
                current = parameters.flat_target
                reasons[index] = "marketlm_indicator_only"
            else:
                bullish = (
                    median[index] >= parameters.long_threshold_bps
                    and probability_up[index] >= parameters.confidence_threshold
                )
                bearish = (
                    median[index] <= parameters.short_threshold_bps
                    and probability_down[index] >= parameters.confidence_threshold
                )
                if bullish:
                    current = parameters.long_target
                    reasons[index] = "marketlm_bullish"
                elif bearish:
                    current = (
                        parameters.short_target
                        if parameters.mode == "long_short"
                        else parameters.flat_target
                    )
                    reasons[index] = "marketlm_bearish"
                else:
                    reasons[index] = "marketlm_hold"
            targets[index] = current
        return targets, reasons
