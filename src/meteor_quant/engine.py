from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from aegis_quant_hybrid import __version__
from aegis_quant_hybrid.datasets import DatasetCatalog, stable_hash
from aegis_quant_hybrid.strategies.sdk import StrategyPlugin


@dataclass(slots=True, frozen=True)
class BacktestConfig:
    initial_equity: float = 10_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 1.0
    spread_bps: float = 1.0
    minimum_order_notional: float = 10.0
    allow_short: bool = False
    max_leverage: float = 1.0
    max_equity_points: int = 10_000
    max_fills_returned: int = 100_000

    def validate(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if min(self.fee_bps, self.slippage_bps, self.spread_bps) < 0:
            raise ValueError("fee, slippage, and spread cannot be negative")
        if self.minimum_order_notional < 0:
            raise ValueError("minimum_order_notional cannot be negative")
        if self.max_leverage <= 0 or self.max_leverage > 10:
            raise ValueError("max_leverage must be in (0, 10]")
        if self.max_equity_points < 100:
            raise ValueError("max_equity_points must be at least 100")


@dataclass(slots=True, frozen=True)
class PreparedSignals:
    path: Path
    cache_key: str
    strategy_key: str
    parameters: dict[str, Any]
    dataset_key: str
    timeframe_seconds: int
    start_timestamp: int | None
    end_timestamp: int | None
    plot_columns: tuple[str, ...]
    row_count: int
    first_timestamp: int
    last_timestamp: int


class HybridBacktestService:
    def __init__(
        self,
        catalog: DatasetCatalog,
        cache_dir: Path,
        result_dir: Path,
        rust_engine: Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.signal_dir = cache_dir / "signals"
        self.result_dir = result_dir
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.rust_engine = rust_engine

    def signal_cache_key(
        self,
        *,
        dataset_key: str,
        strategy: StrategyPlugin,
        timeframe_seconds: int,
        start_timestamp: int | None,
        end_timestamp: int | None,
    ) -> str:
        descriptor = self.catalog.prepare(dataset_key)
        return stable_hash(
            {
                "dataset": dataset_key,
                "dataset_updated_at": descriptor.updated_at,
                "strategy": strategy.key,
                "strategy_identity": strategy.signal_cache_identity(),
                "parameters": strategy.parameters.model_dump(mode="json"),
                "timeframe_seconds": timeframe_seconds,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "signal_schema_version": 4,
            }
        )

    def signal_cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        return (
            self.signal_dir / f"{cache_key}.parquet",
            self.signal_dir / f"{cache_key}.json",
        )

    def prepare_signals(
        self,
        *,
        dataset_key: str,
        strategy: StrategyPlugin,
        timeframe_seconds: int,
        start_timestamp: int | None,
        end_timestamp: int | None,
        force: bool = False,
    ) -> PreparedSignals:
        cache_key = self.signal_cache_key(
            dataset_key=dataset_key,
            strategy=strategy,
            timeframe_seconds=timeframe_seconds,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        signal_path, metadata_path = self.signal_cache_paths(cache_key)
        if signal_path.exists() and metadata_path.exists() and not force:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return PreparedSignals(path=signal_path, **metadata)

        base = self.catalog.scan(
            dataset_key,
            timeframe_seconds=timeframe_seconds,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        plan = strategy.build_signals(base)
        required = ["timestamp", "open", "high", "low", "close", "volume", plan.target_column]
        optional = [name for name in plan.plot_columns if name not in required]
        if plan.reason_column is not None:
            optional.append(plan.reason_column)
        schema_names = set(plan.frame.collect_schema().names())
        missing = set(required) - schema_names
        if missing:
            raise ValueError(
                f"strategy {strategy.key} did not produce required columns: {sorted(missing)}"
            )
        selected = plan.frame.select(required + [name for name in optional if name in schema_names])
        selected = selected.with_columns(
            pl.col(plan.target_column).cast(pl.Float64, strict=False).alias("target_fraction")
        )
        if plan.target_column != "target_fraction":
            selected = selected.drop(plan.target_column)
        temp_path = signal_path.with_suffix(".parquet.tmp")
        temp_path.unlink(missing_ok=True)
        selected.sink_parquet(
            temp_path,
            compression="zstd",
            compression_level=3,
            statistics=True,
            maintain_order=True,
        )
        temp_path.replace(signal_path)
        stats = (
            pl.scan_parquet(signal_path)
            .select(
                pl.len().alias("row_count"),
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("timestamp").max().alias("last_timestamp"),
            )
            .collect()
            .row(0, named=True)
        )
        if not stats["row_count"] or stats["row_count"] < 2:
            signal_path.unlink(missing_ok=True)
            raise ValueError("selected backtest range contains fewer than two bars")
        metadata = {
            "cache_key": cache_key,
            "strategy_key": strategy.key,
            "parameters": strategy.parameters.model_dump(mode="json"),
            "dataset_key": dataset_key,
            "timeframe_seconds": timeframe_seconds,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "plot_columns": list(plan.plot_columns),
            "row_count": int(stats["row_count"]),
            "first_timestamp": int(stats["first_timestamp"]),
            "last_timestamp": int(stats["last_timestamp"]),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return PreparedSignals(path=signal_path, **metadata)

    def run(
        self,
        *,
        prepared: PreparedSignals,
        strategy_name: str,
        config: BacktestConfig,
        engine: Literal["auto", "rust", "python"] = "auto",
    ) -> dict[str, Any]:
        config.validate()
        result_id = uuid4().hex
        result_path = self.result_dir / f"{result_id}.json"
        binary = self._find_rust_engine()
        started = time.perf_counter()
        if engine == "rust" and binary is None:
            raise RuntimeError("Rust engine is not built. Run .\\build-rust.ps1 or ./build-rust.sh")
        if engine != "python" and binary is not None:
            payload = self._run_rust(binary, prepared, config, result_path)
        else:
            payload = PythonArrowEngine().run(prepared, config)
            result_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        payload.update(
            {
                "id": result_id,
                "strategy_key": prepared.strategy_key,
                "strategy_name": strategy_name,
                "parameters": prepared.parameters,
                "dataset_key": prepared.dataset_key,
                "timeframe_seconds": prepared.timeframe_seconds,
                "start_timestamp": prepared.first_timestamp,
                "end_timestamp": prepared.last_timestamp,
                "signal_cache_key": prepared.cache_key,
                "result_path": str(result_path),
                "wall_time_seconds": time.perf_counter() - started,
            }
        )
        result_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return payload

    def load_result(self, result_id: str) -> dict[str, Any]:
        path = self.result_dir / f"{result_id}.json"
        if not path.exists():
            raise KeyError("backtest result not found")
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    def chart_payload(self, cache_key: str, max_points: int = 5_000) -> dict[str, Any]:
        max_points = min(max(max_points, 200), 20_000)
        signal_path = self.signal_dir / f"{cache_key}.parquet"
        metadata_path = self.signal_dir / f"{cache_key}.json"
        if not signal_path.exists() or not metadata_path.exists():
            raise KeyError("signal cache not found")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        first_timestamp = int(metadata["first_timestamp"])
        last_timestamp = int(metadata["last_timestamp"])
        duration = max(last_timestamp - first_timestamp + 1, 1)
        bucket_seconds = max(math.ceil(duration / max_points), int(metadata["timeframe_seconds"]))
        every = f"{bucket_seconds}s"
        plot_columns = [name for name in metadata.get("plot_columns", []) if name]
        frame = pl.scan_parquet(signal_path).with_columns(
            pl.from_epoch("timestamp", time_unit="s").alias("datetime")
        )
        aggregations: list[pl.Expr] = [
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("target_fraction").last().alias("target_fraction"),
        ]
        aggregations.extend(pl.col(name).last().alias(name) for name in plot_columns)
        sampled = (
            frame.group_by_dynamic(
                "datetime", every=every, period=every, label="left", closed="left"
            )
            .agg(*aggregations)
            .drop("datetime")
            .filter(pl.col("open").is_not_null())
            .collect(engine="streaming")
        )
        bars = sampled.select("timestamp", "open", "high", "low", "close", "volume").to_dicts()
        plots = [
            {
                "timestamp": row["timestamp"],
                "values": {name: row.get(name) for name in plot_columns},
            }
            for row in sampled.select(["timestamp", *plot_columns]).to_dicts()
        ]
        return {
            "bars": bars,
            "plots": plots,
            "plot_columns": plot_columns,
            "bucket_seconds": bucket_seconds,
            "source_row_count": metadata["row_count"],
        }

    def _run_rust(
        self, binary: Path, prepared: PreparedSignals, config: BacktestConfig, result_path: Path
    ) -> dict[str, Any]:
        command = [
            str(binary),
            "backtest",
            "--input",
            str(prepared.path),
            "--output",
            str(result_path),
            "--timeframe-seconds",
            str(prepared.timeframe_seconds),
            "--initial-equity",
            str(config.initial_equity),
            "--fee-bps",
            str(config.fee_bps),
            "--slippage-bps",
            str(config.slippage_bps),
            "--spread-bps",
            str(config.spread_bps),
            "--minimum-order-notional",
            str(config.minimum_order_notional),
            "--max-leverage",
            str(config.max_leverage),
            "--max-equity-points",
            str(config.max_equity_points),
            "--max-fills-returned",
            str(config.max_fills_returned),
        ]
        if config.allow_short:
            command.append("--allow-short")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Rust engine failed ({completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
            )
        return cast(dict[str, Any], json.loads(result_path.read_text(encoding="utf-8")))

    def _find_rust_engine(self) -> Path | None:
        candidates: list[Path] = []
        if self.rust_engine is not None:
            candidates.append(self.rust_engine)
        project_root = self.catalog.data_dir.parent
        executable = "aegis-engine.exe" if sys.platform == "win32" else "aegis-engine"
        candidates.extend(
            [
                project_root / "rust" / "aegis-engine" / "target" / "release" / executable,
                project_root / "bin" / executable,
            ]
        )
        which = shutil.which("aegis-engine")
        if which:
            candidates.append(Path(which))
        return next((candidate.resolve() for candidate in candidates if candidate.exists()), None)


class PythonArrowEngine:
    """Streaming reference engine used when the Rust binary has not been built yet."""

    def run(self, prepared: PreparedSignals, config: BacktestConfig) -> dict[str, Any]:
        return self.run_range(prepared, config, start_timestamp=None, end_timestamp=None)

    def run_range(
        self,
        prepared: PreparedSignals,
        config: BacktestConfig,
        *,
        start_timestamp: int | None,
        end_timestamp: int | None,
    ) -> dict[str, Any]:
        parquet = pq.ParquetFile(prepared.path)
        total_rows = parquet.metadata.num_rows
        stride = max(1, math.ceil(total_rows / config.max_equity_points))
        cash = config.initial_equity
        quantity = 0.0
        fees_paid = 0.0
        pending_target: float | None = None
        last_signal_target: float | None = None
        peak_equity = config.initial_equity
        max_drawdown = 0.0
        equity_points: list[dict[str, float | int]] = []
        fills: list[dict[str, Any]] = []
        fill_count = 0
        invested_count = 0
        return_count = 0
        return_mean = 0.0
        return_m2 = 0.0
        previous_equity: float | None = None
        first_close: float | None = None
        last_close = 0.0
        first_timestamp = 0
        last_timestamp = 0
        row_index = 0

        columns = ["timestamp", "open", "high", "low", "close", "volume", "target_fraction"]
        for batch in parquet.iter_batches(batch_size=262_144, columns=columns, use_threads=True):
            values = self._batch_columns(batch)
            for offset in range(batch.num_rows):
                timestamp = int(values["timestamp"][offset].as_py())
                if start_timestamp is not None and timestamp < start_timestamp:
                    continue
                if end_timestamp is not None and timestamp > end_timestamp:
                    continue
                open_price = float(values["open"][offset].as_py())
                close_price = float(values["close"][offset].as_py())
                if row_index == 0:
                    first_close = close_price
                    first_timestamp = timestamp
                if pending_target is not None:
                    target_to_execute = pending_target
                    pending_target = None
                    cash, quantity, fee, fill = self._rebalance(
                        cash=cash,
                        quantity=quantity,
                        target=target_to_execute,
                        open_price=open_price,
                        timestamp=timestamp,
                        config=config,
                    )
                    if fill is not None:
                        fees_paid += fee
                        fill_count += 1
                        if len(fills) < config.max_fills_returned:
                            fills.append(fill)
                raw_target = values["target_fraction"][offset].as_py()
                if raw_target is not None and math.isfinite(float(raw_target)):
                    target_value = float(raw_target)
                    if last_signal_target is None or abs(target_value - last_signal_target) > 1e-12:
                        pending_target = target_value
                        last_signal_target = target_value
                equity = cash + quantity * close_price
                if abs(quantity) > 1e-15:
                    invested_count += 1
                peak_equity = max(peak_equity, equity)
                drawdown = ((equity / peak_equity) - 1.0) * 100.0 if peak_equity > 0 else -100.0
                max_drawdown = min(max_drawdown, drawdown)
                if previous_equity is not None and previous_equity > 0:
                    value = equity / previous_equity - 1.0
                    return_count += 1
                    delta = value - return_mean
                    return_mean += delta / return_count
                    return_m2 += delta * (value - return_mean)
                previous_equity = equity
                if row_index % stride == 0 or row_index == total_rows - 1:
                    equity_points.append(
                        {"timestamp": timestamp, "equity": equity, "drawdown_pct": drawdown}
                    )
                last_close = close_price
                last_timestamp = timestamp
                row_index += 1

        if row_index < 2 or first_close is None or previous_equity is None:
            raise ValueError("backtest requires at least two rows")
        periods_per_year = 365.0 * 24.0 * 3600.0 / prepared.timeframe_seconds
        sharpe: float | None = None
        volatility: float | None = None
        if return_count > 1:
            variance = return_m2 / (return_count - 1)
            deviation = math.sqrt(max(variance, 0.0))
            if deviation > 0:
                sharpe = return_mean / deviation * math.sqrt(periods_per_year)
                volatility = deviation * math.sqrt(periods_per_year) * 100.0
        total_return = (previous_equity / config.initial_equity - 1.0) * 100.0
        metrics: dict[str, Any] = {
            "initial_equity": config.initial_equity,
            "final_equity": previous_equity,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "annualized_volatility_pct": volatility,
            "buy_and_hold_return_pct": (last_close / first_close - 1.0) * 100.0,
            "fill_count": fill_count,
            "fills_returned": len(fills),
            "fees_paid": fees_paid,
            "bar_count": row_index,
            "invested_bar_fraction": invested_count / row_index,
        }
        return {
            "engine": "python-arrow",
            "engine_version": __version__,
            "metrics": metrics,
            "fills": fills,
            "equity": equity_points,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
        }

    @staticmethod
    def _batch_columns(batch: pa.RecordBatch) -> dict[str, pa.Array]:
        return {
            name: batch.column(batch.schema.get_field_index(name)) for name in batch.schema.names
        }

    @staticmethod
    def _rebalance(
        *,
        cash: float,
        quantity: float,
        target: float,
        open_price: float,
        timestamp: int,
        config: BacktestConfig,
    ) -> tuple[float, float, float, dict[str, Any] | None]:
        lower = -config.max_leverage if config.allow_short else 0.0
        target = min(config.max_leverage, max(lower, target))
        equity = cash + quantity * open_price
        if equity <= 0:
            return cash, quantity, 0.0, None
        desired_quantity = target * equity / open_price
        delta = desired_quantity - quantity
        if abs(delta * open_price) < config.minimum_order_notional:
            return cash, quantity, 0.0, None
        buy = delta > 0
        friction = (config.spread_bps / 2.0 + config.slippage_bps) / 10_000.0
        execution_price = open_price * (1.0 + friction if buy else 1.0 - friction)
        fee_rate = config.fee_bps / 10_000.0
        if buy and not config.allow_short and config.max_leverage <= 1.0:
            affordable = max(cash, 0.0) / (execution_price * (1.0 + fee_rate))
            delta = min(delta, affordable)
            if abs(delta * open_price) < config.minimum_order_notional:
                return cash, quantity, 0.0, None
        fee = abs(delta * execution_price) * fee_rate
        cash_after = cash - delta * execution_price - fee
        quantity_after = quantity + delta
        fill = {
            "timestamp": timestamp,
            "side": "buy" if buy else "sell",
            "quantity": abs(delta),
            "price": execution_price,
            "fee": fee,
            "position_after": quantity_after,
            "cash_after": cash_after,
            "reason": "target_rebalance",
        }
        return cash_after, quantity_after, fee, fill
