from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from meteor_quant.datasets import DatasetCatalog
from meteor_quant.marketlm.dataset import (
    MarketWindowDataset,
    load_prepared_metadata,
    prepare_training_data,
)
from meteor_quant.marketlm.inference import MarketLMIndicator
from meteor_quant.marketlm.model import MarketLM
from meteor_quant.marketlm.schemas import (
    IndicatorSelection,
    MarketLMDataConfig,
    MarketLMIndicatorParameters,
    MarketLMModelConfig,
    MarketLMRunRequest,
    MarketLMTrainingConfig,
)
from meteor_quant.strategies.registry import StrategyRegistry


def _request() -> MarketLMRunRequest:
    return MarketLMRunRequest(
        name="test-marketlm",
        data=MarketLMDataConfig(
            dataset_key="btcusdt_1s",
            timeframe_seconds=1,
            indicators=[
                IndicatorSelection(
                    key="wavetrend",
                    parameters={
                        "channel_length": 5,
                        "average_length": 7,
                        "signal_length": 3,
                    },
                ),
                IndicatorSelection(key="rsi", parameters={"period": 7}),
            ],
            horizons_seconds=[5, 10],
            patch_size=2,
            context_patches=4,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_seconds=10,
            cost_threshold_bps=1.0,
        ),
        model=MarketLMModelConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            mlp_hidden=64,
            dropout=0.0,
        ),
        training=MarketLMTrainingConfig(
            max_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=0,
            validation_interval=1,
            checkpoint_interval=1,
            validation_windows=32,
            num_workers=0,
            amp="off",
            device="cpu",
        ),
    )


def _prepared(data_dir: Path, tmp_path: Path) -> tuple[Path, MarketLMRunRequest]:
    catalog = DatasetCatalog(data_dir)
    request = _request()
    prepared = prepare_training_data(catalog, request, tmp_path / "prepared")
    return prepared, request


def test_marketlm_preparation_is_causal_and_memory_mapped(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    prepared, _request_value = _prepared(data_dir, tmp_path)
    metadata = load_prepared_metadata(prepared)

    assert metadata.rows > 1_000
    assert metadata.context_bars == 8
    assert metadata.horizons_seconds == [5, 10]
    assert {"wavetrend_wt1", "wavetrend_wt2", "rsi"} <= set(metadata.feature_names)
    assert np.isfinite(np.asarray(metadata.feature_mean)).all()
    assert np.isfinite(np.asarray(metadata.feature_std)).all()

    dataset = MarketWindowDataset(
        prepared,
        "train",
        samples=4,
        seed=7,
        deterministic=True,
    )
    inputs, next_patches, targets, directions, endpoint = dataset[0]
    assert inputs.shape == (4, 2, metadata.feature_dim)
    assert next_patches.shape == inputs.shape
    assert targets.shape == (2,)
    assert directions.shape == (2,)
    assert endpoint.ndim == 0
    assert torch.isfinite(inputs).all()


def test_marketlm_model_and_registered_indicator(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    prepared, request = _prepared(data_dir, tmp_path)
    metadata = load_prepared_metadata(prepared)
    model = MarketLM(
        feature_dim=metadata.feature_dim,
        patch_size=metadata.patch_size,
        horizons=len(metadata.horizons_seconds),
        config=request.model,
    )
    sample = torch.zeros(
        2,
        metadata.context_patches,
        metadata.patch_size,
        metadata.feature_dim,
    )
    output = model(sample)
    assert output.next_patch.shape == sample.shape
    assert output.quantiles.shape == (2, 2, 3)
    assert output.direction_logits.shape == (2, 2, 3)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "best.pt"
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "request": request.model_dump(mode="json"),
        },
        checkpoint,
    )
    registration = {
        "model_id": "test-model",
        "run_id": "test-model",
        "display_name": "Test model",
        "description": "Test MarketLM indicator",
        "checkpoint_path": str(checkpoint),
        "prepared_dir": str(prepared),
        "timeframe_seconds": 1,
        "horizons_seconds": [5, 10],
        "primary_horizon_seconds": 5,
        "registered_at": "2026-07-13T00:00:00+00:00",
    }
    indicator = MarketLMIndicator(registration)
    frame = DatasetCatalog(data_dir).scan("btcusdt_1s", timeframe_seconds=1)
    predicted = indicator.predict_frame(
        frame,
        MarketLMIndicatorParameters(
            prediction_stride=50,
            batch_size=8,
            device="cpu",
        ),
    )
    assert predicted.height == 1_200
    assert predicted.get_column("marketlm_return_bps").is_not_nan().any()
    assert predicted.get_column("marketlm_forecast_price").is_not_nan().any()
    assert predicted.get_column("marketlm_prob_up").is_not_nan().any()
    price_spec = next(
        spec for spec in indicator.indicator_specs() if spec.key == "marketlm_forecast_price"
    )
    assert price_spec.pane == "price"
    assert price_spec.format == "price"
    assert price_spec.time_offset_seconds == 5
    finite = predicted.filter(
        pl.col("marketlm_return_bps").is_finite() & pl.col("marketlm_forecast_price").is_finite()
    ).row(0, named=True)
    expected_price = float(finite["close"]) * np.exp(
        float(finite["marketlm_return_bps"]) / 10_000.0
    )
    assert float(finite["marketlm_forecast_price"]) == pytest.approx(expected_price)

    registered_dir = tmp_path / "registered"
    registered_dir.mkdir()
    (registered_dir / "test-model.json").write_text(
        json.dumps(registration),
        encoding="utf-8",
    )
    plugin_dir = Path(__file__).resolve().parents[1] / "user_strategies"
    registry = StrategyRegistry(plugin_dir, registered_dir)
    metadata_payload = registry.list_metadata()
    strategy = next(
        item for item in metadata_payload["strategies"] if item["key"] == "marketlm_test_model"
    )
    assert strategy["required_timeframe_seconds"] == 1
    assert strategy["minimum_bars"] == indicator.minimum_bars
    assert strategy["parameter_schema"]["properties"]["prediction_stride"]["default"] == 60


def test_marketlm_training_creates_resumable_checkpoints(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    from meteor_quant.marketlm.training import train_marketlm

    prepared, request = _prepared(data_dir, tmp_path)
    torch.set_num_threads(1)
    run_dir = tmp_path / "training-run"
    run_dir.mkdir()
    updates: list[dict[str, object]] = []

    output = train_marketlm(
        request,
        prepared,
        run_dir,
        update_status=lambda **changes: updates.append(changes),
    )

    assert output == run_dir / "final.pt"
    assert output.exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "checkpoint_step_00000001.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert updates[-1]["state"] == "completed"


def test_marketlm_commits_tensors_before_best_effort_staging_cleanup(
    data_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A persistent Windows staging lock must not discard completed tensors."""

    import meteor_quant.marketlm.dataset as dataset_module

    real_parquet_file = dataset_module.pq.ParquetFile
    real_rmtree = dataset_module._rmtree_with_retry
    lock_state = {"open": False, "cleanup_after_close": False}

    class TrackingParquetFile:
        def __init__(self, *args, **kwargs) -> None:
            self.inner = real_parquet_file(*args, **kwargs)
            lock_state["open"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.close()

        def iter_batches(self, *args, **kwargs):
            return self.inner.iter_batches(*args, **kwargs)

        def close(self, force: bool = False) -> None:
            self.inner.close(force=force)
            lock_state["open"] = False

    def windows_rmtree(path: Path, *, ignore_errors: bool = False) -> None:
        if ".stage-" in path.name:
            assert not lock_state["open"]
            lock_state["cleanup_after_close"] = True
            if ignore_errors:
                return
            raise PermissionError("simulated persistent Windows staging lock")
        real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(dataset_module.pq, "ParquetFile", TrackingParquetFile)
    monkeypatch.setattr(dataset_module, "_rmtree_with_retry", windows_rmtree)

    prepared, _request_value = _prepared(data_dir, tmp_path)

    assert load_prepared_metadata(prepared).rows > 1_000
    assert lock_state == {"open": False, "cleanup_after_close": True}
    assert not (prepared / "staged_features.parquet").exists()
    assert any(".stage-" in path.name for path in prepared.parent.iterdir())


def test_marketlm_publish_falls_back_when_windows_denies_temp_directory_rename(
    data_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A retained/scanned temp handle must not lose completed tensors."""

    import meteor_quant.marketlm.dataset as dataset_module

    real_replace = dataset_module.os.replace
    denied = {"count": 0}

    def windows_replace(source, destination) -> None:
        source_path = Path(source)
        if ".tmp-" in source_path.name:
            denied["count"] += 1
            raise PermissionError(5, "simulated Windows mapped-directory denial")
        real_replace(source, destination)

    monkeypatch.setattr(dataset_module.os, "replace", windows_replace)

    prepared, _request_value = _prepared(data_dir, tmp_path)

    assert denied["count"] >= 1
    assert load_prepared_metadata(prepared).rows > 1_000
    assert not any(".publish-" in path.name for path in prepared.parent.iterdir())


def test_marketlm_closes_every_output_memmap_before_publish(
    data_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    import meteor_quant.marketlm.dataset as dataset_module

    real_close = dataset_module._close_memmap
    closed: list[bool] = []

    def tracking_close(array) -> None:
        real_close(array)
        mapping = getattr(array, "_mmap", None)
        closed.append(mapping is None or mapping.closed)

    monkeypatch.setattr(dataset_module, "_close_memmap", tracking_close)

    prepared, _request_value = _prepared(data_dir, tmp_path)

    assert load_prepared_metadata(prepared).rows > 1_000
    assert closed == [True, True, True, True, True]


def test_marketlm_concurrent_same_fingerprint_reuses_single_committed_cache(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    prepared_root = tmp_path / "prepared-concurrent"
    DatasetCatalog(data_dir).prepare("btcusdt_1s")

    def prepare_once() -> Path:
        return prepare_training_data(
            DatasetCatalog(data_dir),
            _request(),
            prepared_root,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(prepare_once)
        second_future = executor.submit(prepare_once)
        first = first_future.result()
        second = second_future.result()

    assert first == second
    assert load_prepared_metadata(first).rows > 1_000
    assert len([path for path in prepared_root.iterdir() if path.name == first.name]) == 1
    assert not any(".tmp-" in path.name for path in prepared_root.iterdir())


def test_marketlm_recovers_complete_temp_directory_without_recomputing(
    data_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    import shutil

    import meteor_quant.marketlm.dataset as dataset_module

    prepared_root = tmp_path / "recover-prepared"
    prepared = prepare_training_data(
        DatasetCatalog(data_dir),
        _request(),
        prepared_root,
    )
    candidate = prepared_root / f".{prepared.name}.tmp-99999-deadbeef"
    shutil.copytree(prepared, candidate)
    shutil.rmtree(prepared)

    def should_not_recompute(*args, **kwargs):
        raise AssertionError("completed temporary tensors should be recovered")

    monkeypatch.setattr(dataset_module, "build_feature_frame", should_not_recompute)

    recovered = prepare_training_data(
        DatasetCatalog(data_dir),
        _request(),
        prepared_root,
    )

    assert recovered == prepared
    assert load_prepared_metadata(recovered).rows > 1_000
    assert not candidate.exists()
