from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch.utils.data import Dataset

from meteor_quant.datasets import DatasetCatalog, stable_hash
from meteor_quant.marketlm.features import (
    build_feature_frame,
    indicator_warmup_bars,
    normalized_indicators,
)
from meteor_quant.marketlm.fileio import (
    atomic_json_write,
    interprocess_file_lock,
    retry_file_operation,
)
from meteor_quant.marketlm.schemas import MarketLMRunRequest

FEATURE_SCHEMA_VERSION = 1
StatusCallback = Callable[[float, str], None]
SplitName = Literal["train", "validation", "test"]


@dataclass(slots=True, frozen=True)
class PreparedMarketLMMetadata:
    version: int
    fingerprint: str
    dataset_key: str
    dataset_updated_at: str | None
    symbol: str
    rows: int
    feature_dim: int
    feature_names: list[str]
    timeframe_seconds: int
    horizons_seconds: list[int]
    target_names: list[str]
    indicators: list[dict[str, Any]]
    patch_size: int
    context_patches: int
    context_bars: int
    indicator_warmup_bars: int
    cost_threshold_bps: float
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: list[float]
    target_std: list[float]
    train_boundary: int
    validation_boundary: int
    split_endpoints: dict[str, list[int]]
    first_timestamp: int
    last_timestamp: int
    source_start_timestamp: int | None
    source_end_timestamp: int | None
    semantics: dict[str, str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreparedMarketLMMetadata:
        return cls(**payload)


def load_prepared_metadata(path: str | Path) -> PreparedMarketLMMetadata:
    root = Path(path)
    payload = cast(
        dict[str, Any],
        json.loads((root / "metadata.json").read_text(encoding="utf-8")),
    )
    metadata = PreparedMarketLMMetadata.from_dict(payload)
    required = (
        root / "features.npy",
        root / "targets.npy",
        root / "directions.npy",
        root / "timestamps.npy",
        root / "close.npy",
    )
    if not all(item.exists() for item in required):
        raise FileNotFoundError(f"prepared MarketLM dataset is incomplete: {root}")
    return metadata


def _retry_file_operation(
    operation: Callable[[], None],
    *,
    path: Path,
    attempts: int = 10,
) -> None:
    """Retry Windows-sensitive file cleanup after native readers are closed."""

    delay_seconds = 0.05
    for attempt in range(attempts):
        try:
            operation()
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            if attempt + 1 >= attempts:
                raise PermissionError(
                    f"could not release temporary MarketLM file after {attempts} "
                    f"attempts: {path}"
                ) from exc
            gc.collect()
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 0.5)


def _unlink_with_retry(path: Path) -> None:
    _retry_file_operation(
        lambda: path.unlink(missing_ok=True),
        path=path,
    )


def _rmtree_with_retry(path: Path, *, ignore_errors: bool = False) -> None:
    if not path.exists():
        return
    try:
        retry_file_operation(
            lambda: shutil.rmtree(path),
            path=path,
            attempts=60,
        )
    except PermissionError:
        if not ignore_errors:
            raise


def _close_memmap(array: np.memmap[Any, Any]) -> None:
    """Flush and close a NumPy memmap deterministically on Windows."""

    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def _publish_prepared_directory(temporary: Path, destination: Path) -> None:
    """Publish a complete immutable tensor directory without open-handle races.

    The normal path is a directory rename. If Windows still denies that rename
    (for example because an antivirus scanner acquired a transient handle), copy
    into a second closed-handle directory, validate it, and rename that directory.
    """

    try:
        retry_file_operation(
            lambda: os.replace(temporary, destination),
            path=destination,
            attempts=60,
        )
        return
    except PermissionError:
        publish = destination.parent / (
            f".{destination.name}.publish-{os.getpid()}-{uuid4().hex}"
        )
        try:
            shutil.copytree(temporary, publish)
            load_prepared_metadata(publish)
            retry_file_operation(
                lambda: os.replace(publish, destination),
                path=destination,
                attempts=60,
            )
        finally:
            _rmtree_with_retry(publish, ignore_errors=True)
            _rmtree_with_retry(temporary, ignore_errors=True)


def _safe_mean_std(values: np.ndarray, axis: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axis)
    std = np.nanstd(values, axis=axis)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def preparation_fingerprint(catalog: DatasetCatalog, request: MarketLMRunRequest) -> str:
    descriptor = catalog.prepare(request.data.dataset_key)
    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_key": descriptor.key,
        "dataset_updated_at": descriptor.updated_at,
        "timeframe_seconds": request.data.timeframe_seconds,
        "start_timestamp": request.data.start_timestamp,
        "end_timestamp": request.data.end_timestamp,
        "indicators": normalized_indicators(request.data.indicators),
        "horizons_seconds": request.data.horizons_seconds,
        "patch_size": request.data.patch_size,
        "context_patches": request.data.context_patches,
        "train_fraction": request.data.train_fraction,
        "validation_fraction": request.data.validation_fraction,
        "purge_seconds": request.data.effective_purge_seconds,
        "cost_threshold_bps": request.data.cost_threshold_bps,
    }
    return str(stable_hash(payload))


def _lazy_mean_std(
    frame: pl.LazyFrame,
    names: list[str],
    *,
    offset: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        raise ValueError("normalization slice must contain at least one row")
    expressions: list[pl.Expr] = []
    for index, name in enumerate(names):
        expressions.extend(
            (
                pl.col(name).mean().alias(f"__mean_{index}"),
                pl.col(name).std(ddof=0).alias(f"__std_{index}"),
            )
        )
    result = frame.slice(offset, length).select(expressions).collect().row(0, named=True)
    mean = np.asarray([result[f"__mean_{index}"] for index in range(len(names))], dtype=np.float32)
    std = np.asarray([result[f"__std_{index}"] for index in range(len(names))], dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("normalization statistics contain non-finite values")
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    return mean, std


def _batch_matrix(
    batch: pa.RecordBatch,
    names: list[str],
    dtype: np.dtype[Any],
) -> np.ndarray:
    arrays = [
        batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
        for name in names
    ]
    return np.column_stack(arrays).astype(dtype, copy=False)


def prepare_training_data(
    catalog: DatasetCatalog,
    request: MarketLMRunRequest,
    prepared_root: Path,
    *,
    status: StatusCallback | None = None,
    force: bool = False,
) -> Path:
    callback = status or (lambda _progress, _message: None)
    request.data.model_validate(request.data.model_dump())
    fingerprint = preparation_fingerprint(catalog, request)
    destination = prepared_root / fingerprint
    prepared_root.mkdir(parents=True, exist_ok=True)
    commit_guard = prepared_root / f"{fingerprint}.commit"
    if not force:
        with interprocess_file_lock(
            commit_guard,
            timeout_seconds=300.0,
            stale_after_seconds=21_600.0,
        ):
            metadata_path = destination / "metadata.json"
            if metadata_path.exists():
                try:
                    load_prepared_metadata(destination)
                except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
                    _rmtree_with_retry(destination)
                else:
                    callback(1.0, "reusing matching prepared MarketLM dataset")
                    return destination

            # A previous Windows worker may have written every tensor and failed
            # only while renaming the final directory. Recover that expensive work.
            candidates = sorted(
                prepared_root.glob(f".{fingerprint}.tmp-*"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            for candidate in candidates:
                try:
                    load_prepared_metadata(candidate)
                except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
                    continue
                if destination.exists():
                    _rmtree_with_retry(destination)
                _publish_prepared_directory(candidate, destination)
                callback(
                    1.0,
                    "recovered completed MarketLM tensors from a failed Windows commit",
                )
                return destination

    descriptor = catalog.prepare(request.data.dataset_key)
    callback(0.03, "scanning canonical market data")
    source = catalog.scan(
        request.data.dataset_key,
        timeframe_seconds=request.data.timeframe_seconds,
        start_timestamp=request.data.start_timestamp,
        end_timestamp=request.data.end_timestamp,
    )
    feature_frame, feature_names, target_names = build_feature_frame(
        source,
        request.data,
        include_targets=True,
    )
    selected_names = ["timestamp", "close", *feature_names, *target_names]
    selected = feature_frame.select(selected_names).with_columns(
        pl.col("timestamp").cast(pl.Int64),
        pl.col("close").cast(pl.Float64),
        *[pl.col(name).cast(pl.Float32, strict=False) for name in feature_names],
        *[pl.col(name).cast(pl.Float32, strict=False) for name in target_names],
    )
    selected = selected.drop_nulls(selected_names).filter(
        pl.all_horizontal(
            [pl.col(name).is_finite() for name in ["close", *feature_names, *target_names]]
        )
    )

    attempt_id = uuid4().hex
    temporary = prepared_root / f".{fingerprint}.tmp-{os.getpid()}-{attempt_id}"
    staging = prepared_root / f".{fingerprint}.stage-{os.getpid()}-{attempt_id}"
    temporary.mkdir(parents=True, exist_ok=False)
    staging.mkdir(parents=True, exist_ok=False)
    staged_features = staging / "staged_features.parquet"
    try:
        callback(0.08, "streaming causal features and future-return labels to Parquet")
        selected.sink_parquet(
            staged_features,
            compression="zstd",
            compression_level=3,
            statistics=True,
            maintain_order=True,
        )
        staged_scan = pl.scan_parquet(staged_features)
        stats = (
            staged_scan.select(
                pl.len().alias("rows"),
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("timestamp").max().alias("last_timestamp"),
            )
            .collect()
            .row(0, named=True)
        )
        rows = int(stats["rows"] or 0)
        callback(0.42, f"prepared {rows:,} finite feature rows")
        minimum_rows = (
            request.data.context_bars
            + request.data.patch_size
            + max(request.data.horizon_steps)
            + 32
        )
        if rows < minimum_rows:
            raise ValueError(
                f"MarketLM preparation produced {rows:,} rows but at least "
                f"{minimum_rows:,} are required; use a wider date range or smaller context"
            )

        train_boundary = int(rows * request.data.train_fraction)
        validation_boundary = int(
            rows * (request.data.train_fraction + request.data.validation_fraction)
        )
        context_bars = request.data.context_bars
        purge_steps = math.ceil(
            request.data.effective_purge_seconds / request.data.timeframe_seconds
        )
        reserve_steps = request.data.patch_size
        split_endpoints = {
            "train": [context_bars - 1, train_boundary - reserve_steps - 1],
            "validation": [
                train_boundary + purge_steps,
                validation_boundary - reserve_steps - 1,
            ],
            "test": [
                validation_boundary + purge_steps,
                rows - reserve_steps - 1,
            ],
        }
        for name, bounds in split_endpoints.items():
            minimum, maximum = bounds
            if minimum > maximum:
                raise ValueError(
                    f"{name} split is empty ({minimum}..{maximum}); "
                    "use more data or a smaller context"
                )

        callback(0.50, "fitting normalization on the chronological training split only")
        feature_mean, feature_std = _lazy_mean_std(
            staged_scan,
            feature_names,
            offset=0,
            length=train_boundary,
        )
        train_min, train_max = split_endpoints["train"]
        target_mean, target_std = _lazy_mean_std(
            staged_scan,
            target_names,
            offset=train_min,
            length=train_max - train_min + 1,
        )

        callback(0.58, "streaming normalized values into memory-mapped tensors")
        features_output = np.lib.format.open_memmap(
            temporary / "features.npy",
            mode="w+",
            dtype=np.float16,
            shape=(rows, len(feature_names)),
        )
        targets_output = np.lib.format.open_memmap(
            temporary / "targets.npy",
            mode="w+",
            dtype=np.float32,
            shape=(rows, len(target_names)),
        )
        directions_output = np.lib.format.open_memmap(
            temporary / "directions.npy",
            mode="w+",
            dtype=np.int8,
            shape=(rows, len(target_names)),
        )
        timestamps_output = np.lib.format.open_memmap(
            temporary / "timestamps.npy",
            mode="w+",
            dtype=np.int64,
            shape=(rows,),
        )
        close_output = np.lib.format.open_memmap(
            temporary / "close.npy",
            mode="w+",
            dtype=np.float64,
            shape=(rows,),
        )

        write_offset = 0
        batch_size = 262_144
        # PyArrow keeps the source file open for the lifetime of ParquetFile.
        # Closing it deterministically is required on Windows before unlinking.
        with pq.ParquetFile(staged_features) as parquet_file:
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=selected_names,
                use_threads=True,
            ):
                batch_rows = batch.num_rows
                stop = write_offset + batch_rows
                feature_values = _batch_matrix(
                    batch, feature_names, np.dtype(np.float32)
                )
                raw_targets = _batch_matrix(
                    batch, target_names, np.dtype(np.float32)
                )
                normalized_features = np.clip(
                    (feature_values - feature_mean) / feature_std,
                    -12.0,
                    12.0,
                )
                normalized_targets = np.clip(
                    (raw_targets - target_mean) / target_std,
                    -20.0,
                    20.0,
                )
                directions = np.full(raw_targets.shape, -1, dtype=np.int8)
                directions[raw_targets < -request.data.cost_threshold_bps] = 0
                directions[
                    np.abs(raw_targets) <= request.data.cost_threshold_bps
                ] = 1
                directions[raw_targets > request.data.cost_threshold_bps] = 2

                features_output[write_offset:stop] = normalized_features.astype(
                    np.float16, copy=False
                )
                targets_output[write_offset:stop] = normalized_targets.astype(
                    np.float32, copy=False
                )
                directions_output[write_offset:stop] = directions
                timestamp_index = batch.schema.get_field_index("timestamp")
                close_index = batch.schema.get_field_index("close")
                timestamps_output[write_offset:stop] = batch.column(
                    timestamp_index
                ).to_numpy(zero_copy_only=False)
                close_output[write_offset:stop] = batch.column(close_index).to_numpy(
                    zero_copy_only=False
                )
                write_offset = stop
                callback(
                    0.58 + 0.37 * write_offset / rows,
                    f"wrote {write_offset:,}/{rows:,} MarketLM tensor rows",
                )
        if write_offset != rows:
            raise RuntimeError(
                f"Parquet stream yielded {write_offset:,} rows, expected {rows:,}"
            )
        for output in (
            features_output,
            targets_output,
            directions_output,
            timestamps_output,
            close_output,
        ):
            _close_memmap(output)
        # Python keeps a for-loop target alive after the loop. Without this delete,
        # `output` retains close.npy and Windows refuses to rename its directory.
        del output
        del (
            features_output,
            targets_output,
            directions_output,
            timestamps_output,
            close_output,
        )
        del staged_scan
        gc.collect()

        metadata = PreparedMarketLMMetadata(
            version=FEATURE_SCHEMA_VERSION,
            fingerprint=fingerprint,
            dataset_key=descriptor.key,
            dataset_updated_at=descriptor.updated_at,
            symbol=descriptor.symbol,
            rows=rows,
            feature_dim=len(feature_names),
            feature_names=feature_names,
            timeframe_seconds=request.data.timeframe_seconds,
            horizons_seconds=list(request.data.horizons_seconds),
            target_names=target_names,
            indicators=normalized_indicators(request.data.indicators),
            patch_size=request.data.patch_size,
            context_patches=request.data.context_patches,
            context_bars=context_bars,
            indicator_warmup_bars=indicator_warmup_bars(request.data.indicators),
            cost_threshold_bps=request.data.cost_threshold_bps,
            feature_mean=feature_mean.astype(float).tolist(),
            feature_std=feature_std.astype(float).tolist(),
            target_mean=target_mean.astype(float).tolist(),
            target_std=target_std.astype(float).tolist(),
            train_boundary=train_boundary,
            validation_boundary=validation_boundary,
            split_endpoints=split_endpoints,
            first_timestamp=int(stats["first_timestamp"]),
            last_timestamp=int(stats["last_timestamp"]),
            source_start_timestamp=request.data.start_timestamp,
            source_end_timestamp=request.data.end_timestamp,
            semantics={
                "feature_time": "Every feature uses only the current or earlier completed bars.",
                "targets": "Negative shifts are used only for supervised future-return labels.",
                "normalization": "Feature and target statistics are fit on the training period only.",
                "splits": "Train, validation and test are chronological and separated by a purge gap.",
                "execution": "Model output is an indicator; authoritative backtests still use next-bar open fills.",
                "materialization": "Engineered rows are streamed through Parquet into bounded-size memory-map writes.",
            },
        )
        atomic_json_write(asdict(metadata), temporary / "metadata.json")
        with interprocess_file_lock(
            commit_guard,
            timeout_seconds=300.0,
            stale_after_seconds=21_600.0,
        ):
            # Another worker may have prepared the same fingerprint while this
            # worker was computing features. Prefer the already-validated cache.
            if destination.exists():
                try:
                    load_prepared_metadata(destination)
                except (FileNotFoundError, ValueError, json.JSONDecodeError):
                    _rmtree_with_retry(destination)
                else:
                    _rmtree_with_retry(temporary, ignore_errors=True)
                    callback(1.0, "reusing concurrently prepared MarketLM dataset")
                    return destination
            _publish_prepared_directory(temporary, destination)
    except Exception:
        _rmtree_with_retry(temporary, ignore_errors=True)
        _rmtree_with_retry(staging, ignore_errors=True)
        raise

    # Staging data is intentionally outside the committed tensor directory. A brief
    # Windows/antivirus lock can therefore delay cleanup without invalidating a
    # completed multi-hour preparation. The repair script removes any leftovers.
    _rmtree_with_retry(staging, ignore_errors=True)
    callback(1.0, "MarketLM training data is ready")
    return destination


class MarketWindowDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Memory-mapped random or deterministic causal windows."""

    def __init__(
        self,
        prepared_dir: str | Path,
        split: SplitName,
        *,
        samples: int,
        seed: int,
        deterministic: bool,
    ) -> None:
        self.prepared_dir = Path(prepared_dir)
        self.metadata = load_prepared_metadata(self.prepared_dir)
        self.features = np.load(self.prepared_dir / "features.npy", mmap_mode="r")
        self.targets = np.load(self.prepared_dir / "targets.npy", mmap_mode="r")
        self.directions = np.load(self.prepared_dir / "directions.npy", mmap_mode="r")
        self.samples = int(samples)
        self.seed = int(seed)
        self.deterministic = deterministic
        self.endpoint_min, self.endpoint_max = self.metadata.split_endpoints[split]
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if deterministic:
            count = min(self.samples, self.endpoint_max - self.endpoint_min + 1)
            self.fixed_endpoints = np.linspace(
                self.endpoint_min,
                self.endpoint_max,
                num=count,
                dtype=np.int64,
            )
        else:
            self.fixed_endpoints = None

    def __len__(self) -> int:
        if self.fixed_endpoints is not None:
            return len(self.fixed_endpoints)
        return self.samples

    def _random_endpoint(self, index: int) -> int:
        value = (index + self.seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 31
        width = self.endpoint_max - self.endpoint_min + 1
        return self.endpoint_min + int(value % width)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        endpoint = (
            int(self.fixed_endpoints[index])
            if self.fixed_endpoints is not None
            else self._random_endpoint(index)
        )
        patch = self.metadata.patch_size
        context = self.metadata.context_patches
        start = endpoint - context * patch + 1
        stop = endpoint + patch + 1
        segment = np.array(self.features[start:stop], dtype=np.float32, copy=True)
        expected_rows = (context + 1) * patch
        if segment.shape[0] != expected_rows:
            raise RuntimeError(f"expected {expected_rows} feature rows, got {segment.shape[0]}")
        patches = torch.from_numpy(segment).view(
            context + 1,
            patch,
            self.metadata.feature_dim,
        )
        targets = torch.from_numpy(
            np.array(self.targets[endpoint], dtype=np.float32, copy=True)
        )
        directions = torch.from_numpy(
            np.array(self.directions[endpoint], dtype=np.int64, copy=True)
        )
        return (
            patches[:-1],
            patches[1:],
            targets,
            directions,
            torch.tensor(endpoint, dtype=torch.int64),
        )
