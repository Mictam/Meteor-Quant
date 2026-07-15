from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from meteor_quant.datasets import DatasetCatalog
from meteor_quant.markethybrid.dataset import MarketHybridWindowDataset
from meteor_quant.markethybrid.inference import MarketHybridIndicator
from meteor_quant.markethybrid.jobs import MarketHybridJobManager
from meteor_quant.markethybrid.model import MarketHybrid
from meteor_quant.markethybrid.schemas import (
    MarketHybridIndicatorParameters,
    MarketHybridJEPAConfig,
    MarketHybridLossWeights,
    MarketHybridModelConfig,
    MarketHybridPredictorConfig,
    MarketHybridRunRequest,
    MarketHybridTrainingConfig,
)
from meteor_quant.markethybrid.training import (
    _float_numpy,
    _validate,
    train_markethybrid,
)
from meteor_quant.marketlm.dataset import load_prepared_metadata, prepare_training_data
from meteor_quant.marketlm.schemas import (
    IndicatorSelection,
    MarketLMDataConfig,
    MarketLMRunRequest,
)
from meteor_quant.strategies.registry import StrategyRegistry


def _request() -> MarketHybridRunRequest:
    return MarketHybridRunRequest(
        name="test-markethybrid",
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
                )
            ],
            horizons_seconds=[4, 8, 16],
            patch_size=2,
            context_patches=4,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_seconds=16,
            cost_threshold_bps=1.0,
        ),
        model=MarketHybridModelConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            mlp_hidden=64,
            dropout=0.0,
            activation_checkpointing=False,
        ),
        predictor=MarketHybridPredictorConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            mlp_hidden=64,
            dropout=0.0,
        ),
        jepa=MarketHybridJEPAConfig(
            target_patch_offsets=[1, 2, 4],
            ema_start=0.9,
            ema_end=0.99,
        ),
        training=MarketHybridTrainingConfig(
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


def _prepared(data_dir: Path, tmp_path: Path) -> tuple[Path, MarketHybridRunRequest]:
    request = _request()
    prepared = prepare_training_data(
        DatasetCatalog(data_dir),
        MarketLMRunRequest(name=request.name, data=request.data, prepare_only=True),
        tmp_path / "prepared",
    )
    return prepared, request


def test_markethybrid_dataset_and_model_shapes(data_dir: Path, tmp_path: Path) -> None:
    prepared, request = _prepared(data_dir, tmp_path)
    metadata = load_prepared_metadata(prepared)
    dataset = MarketHybridWindowDataset(
        prepared,
        request,
        "train",
        samples=2,
        seed=7,
        deterministic=True,
    )
    context, future, next_patch, targets, directions, policy, endpoint = dataset[0]
    assert context.shape == (4, 2, metadata.feature_dim)
    assert future.shape == (4, 2, metadata.feature_dim)
    assert next_patch.shape == context.shape
    assert targets.shape == (3,)
    assert directions.shape == (3,)
    assert policy.shape == (3,)
    assert endpoint.ndim == 0

    model = MarketHybrid(
        feature_dim=metadata.feature_dim,
        patch_size=metadata.patch_size,
        horizons=len(metadata.horizons_seconds),
        request=request,
    )
    output = model(context.unsqueeze(0), future.unsqueeze(0))
    assert output.next_patch.shape == context.unsqueeze(0).shape
    assert output.quantiles.shape == (1, 3, 3)
    assert output.direction_logits.shape == (1, 3, 3)
    assert output.actionable_logits.shape == (1, 3)
    assert output.target_position.shape == (1,)
    assert output.predicted_latents is not None
    assert output.predicted_latents.shape == (1, 3, 32)
    assert output.target_latents is not None
    assert torch.isfinite(output.target_latents).all()


def test_markethybrid_training_and_registered_indicator(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    prepared, request = _prepared(data_dir, tmp_path)
    torch.set_num_threads(1)
    run_dir = tmp_path / "hybrid-run"
    run_dir.mkdir()
    updates: list[dict[str, object]] = []
    final = train_markethybrid(
        request,
        prepared,
        run_dir,
        update_status=lambda **changes: updates.append(changes),
    )
    assert final.exists()
    assert (run_dir / "best.pt").exists()
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_type"] == "markethybrid"
    assert "deployment_model" in checkpoint
    assert updates[-1]["state"] == "completed"

    registration = {
        "model_id": "hybrid-test",
        "run_id": "hybrid-test",
        "display_name": "Hybrid test",
        "description": "Test hybrid indicator",
        "checkpoint_path": str(run_dir / "best.pt"),
        "prepared_dir": str(prepared),
        "timeframe_seconds": 1,
        "horizons_seconds": [4, 8, 16],
        "primary_horizon_seconds": 8,
        "registered_at": "2026-07-13T00:00:00+00:00",
        "model_type": "markethybrid",
    }
    indicator = MarketHybridIndicator(registration)
    predicted = indicator.predict_frame(
        DatasetCatalog(data_dir).scan("btcusdt_1s", timeframe_seconds=1),
        MarketHybridIndicatorParameters(
            prediction_stride=50,
            batch_size=8,
            device="cpu",
            mode="model_policy",
        ),
    )
    assert predicted.height == 1_200
    assert predicted.get_column("markethybrid_return_bps").is_not_nan().any()
    assert predicted.get_column("markethybrid_forecast_price").is_not_nan().any()
    assert predicted.get_column("markethybrid_actionable").is_not_nan().any()
    price_spec = next(
        spec for spec in indicator.indicator_specs() if spec.key == "markethybrid_forecast_price"
    )
    assert price_spec.pane == "price"
    assert price_spec.format == "price"
    assert price_spec.time_offset_seconds == 8
    finite = predicted.filter(
        pl.col("markethybrid_return_bps").is_finite()
        & pl.col("markethybrid_forecast_price").is_finite()
    ).row(0, named=True)
    expected_price = float(finite["close"]) * np.exp(
        float(finite["markethybrid_return_bps"]) / 10_000.0
    )
    assert float(finite["markethybrid_forecast_price"]) == pytest.approx(expected_price)
    assert predicted.get_column("markethybrid_policy_position").is_not_nan().any()
    assert np.isfinite(predicted.get_column("target_fraction").to_numpy()).all()

    alternate_horizon = indicator.predict_frame(
        DatasetCatalog(data_dir).scan("btcusdt_1s", timeframe_seconds=1),
        MarketHybridIndicatorParameters(
            prediction_stride=100,
            batch_size=8,
            device="cpu",
            mode="median_sign_long_short",
            long_threshold_bps=0.0,
            short_threshold_bps=0.0,
        ),
        horizon_seconds=16,
    )
    assert alternate_horizon.get_column("markethybrid_return_bps").is_not_nan().any()
    assert indicator.indicator_specs(16)[0].time_offset_seconds == 16

    marketlm_registered = tmp_path / "marketlm-registered"
    marketlm_registered.mkdir()
    hybrid_registered = tmp_path / "hybrid-registered"
    hybrid_registered.mkdir()
    prepared_with_300s = tmp_path / "prepared-with-300s"
    shutil.copytree(prepared, prepared_with_300s)
    metadata_path = prepared_with_300s / "metadata.json"
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_payload["horizons_seconds"] = [4, 8, 16, 300]
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
    registry_registration = {**registration, "prepared_dir": str(prepared_with_300s)}
    (hybrid_registered / "hybrid-test.json").write_text(
        json.dumps(registry_registration),
        encoding="utf-8",
    )
    plugin_dir = Path(__file__).resolve().parents[1] / "user_strategies"
    registry = StrategyRegistry(
        plugin_dir,
        marketlm_registered,
        hybrid_registered,
    )
    strategy = next(
        item
        for item in registry.list_metadata()["strategies"]
        if item["key"] == "markethybrid_hybrid_test"
    )
    assert strategy["required_timeframe_seconds"] == 1
    assert strategy["source"] == "markethybrid"
    median_sign = next(
        item
        for item in registry.list_metadata()["strategies"]
        if item["key"] == "markethybrid_hybrid_test_median_sign_300s"
    )
    assert median_sign["name"] == "MarketHybrid 300s Median Sign · Hybrid test"
    assert median_sign["required_timeframe_seconds"] == 1
    properties = median_sign["parameter_schema"]["properties"]
    assert properties["median_deadband_bps"]["default"] == 0.0
    price_spec = next(
        spec
        for spec in median_sign["indicator_specs"]
        if spec["key"] == "markethybrid_forecast_price"
    )
    assert price_spec["time_offset_seconds"] == 300


def test_markethybrid_300s_median_sign_targets_hold_until_sign_changes() -> None:
    median = np.asarray([2.0, 1.0, 0.0, -0.5, -3.0, 0.0, 4.0], dtype=np.float64)
    outputs = {
        "median": median,
        "lower": np.zeros_like(median),
        "upper": np.zeros_like(median),
        "probability_up": np.zeros_like(median),
        "probability_down": np.zeros_like(median),
        "actionable": np.zeros_like(median),
        "policy_position": np.zeros_like(median),
        "policy_confidence": np.zeros_like(median),
        "policy_intent": np.zeros_like(median),
    }
    targets, reasons = MarketHybridIndicator._targets(
        outputs,
        MarketHybridIndicatorParameters(
            mode="median_sign_long_short",
            long_threshold_bps=0.0,
            short_threshold_bps=0.0,
            long_target=1.0,
            short_target=-1.0,
            flat_target=0.0,
        ),
    )
    np.testing.assert_array_equal(targets, np.asarray([1, 1, 1, -1, -1, -1, 1]))
    assert reasons == [
        "markethybrid_300s_median_positive",
        "markethybrid_300s_median_positive",
        "markethybrid_300s_median_hold",
        "markethybrid_300s_median_negative",
        "markethybrid_300s_median_negative",
        "markethybrid_300s_median_hold",
        "markethybrid_300s_median_positive",
    ]


def test_markethybrid_prepare_worker_process(data_dir: Path, tmp_path: Path) -> None:
    request = _request().model_copy(update={"prepare_only": True})
    manager = MarketHybridJobManager(Path(__file__).resolve().parents[1], data_dir)
    started = manager.start(request)
    run_id = str(started["run_id"])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = manager.get(run_id)
        if status["state"] in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.1)
    else:
        manager.stop(run_id)
        raise AssertionError("MarketHybrid prepare worker did not finish")
    assert status["state"] == "completed", status.get("log_tail")
    assert Path(str(status["prepared_dir"])).exists()


def test_markethybrid_bfloat16_outputs_cross_numpy_boundary_as_float32() -> None:
    tensor = torch.tensor([1.25, -0.5], dtype=torch.bfloat16)
    converted = _float_numpy(tensor)
    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, np.array([1.25, -0.5], dtype=np.float32))


def test_markethybrid_validation_accepts_bfloat16_autocast_outputs(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    prepared, request = _prepared(data_dir, tmp_path)
    metadata = load_prepared_metadata(prepared)
    dataset = MarketHybridWindowDataset(
        prepared,
        request,
        "validation",
        samples=8,
        seed=11,
        deterministic=True,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MarketHybrid(
        feature_dim=metadata.feature_dim,
        patch_size=metadata.patch_size,
        horizons=len(metadata.horizons_seconds),
        request=request,
    )
    horizons = len(metadata.horizons_seconds)
    metrics = _validate(
        model,
        loader,
        device=torch.device("cpu"),
        amp_enabled=True,
        amp_dtype=torch.bfloat16,
        request=request,
        class_weights=torch.ones((horizons, 3)),
        actionable_pos_weights=torch.ones(horizons),
        loss_weights=MarketHybridLossWeights(),
    )
    assert np.isfinite(metrics["total"])
    assert np.isfinite(metrics["hybrid_score"])


def test_markethybrid_optimized_defaults_and_legacy_mode() -> None:
    defaults = MarketHybridRunRequest()
    assert defaults.data.timeframe_seconds == 15
    assert defaults.data.horizons_seconds == [30, 60, 180, 300, 900]
    assert defaults.data.cost_threshold_bps == 30.0
    assert defaults.policy.position_scale_bps == 30.0
    assert defaults.policy.intent_deadband == 0.25
    assert defaults.training.mode == "staged"
    assert [stage.name for stage in defaults.training.stages] == [
        "representation_pretraining",
        "joint_training",
        "policy_finetuning",
    ]
    assert defaults.training.stages[0].trainable_modules.policy_position_head is False
    assert defaults.training.stages[-1].trainable_modules.jepa_predictor is False

    legacy = MarketHybridTrainingConfig(max_steps=10, warmup_steps=0)
    assert legacy.mode == "single_stage"


def test_markethybrid_staged_training_transitions(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    from meteor_quant.markethybrid.schemas import (
        MarketHybridEarlyStoppingConfig,
        MarketHybridLearningRateMultipliers,
        MarketHybridLossWeights,
        MarketHybridTrainableModules,
        MarketHybridTrainingStageConfig,
    )

    prepared, base = _prepared(data_dir, tmp_path)
    stage_common = {
        "learning_rate": 1e-3,
        "min_learning_rate": 1e-4,
        "warmup_steps": 0,
        "learning_rate_multipliers": MarketHybridLearningRateMultipliers(),
        "early_stopping": MarketHybridEarlyStoppingConfig(enabled=False),
    }
    stages = [
        MarketHybridTrainingStageConfig(
            name="representation_pretraining",
            end_step=1,
            loss_weights=MarketHybridLossWeights(
                policy_position=0,
                policy_confidence=0,
                policy_intent=0,
            ),
            trainable_modules=MarketHybridTrainableModules(
                policy_position_head=False,
                policy_confidence_head=False,
                policy_intent_head=False,
            ),
            **stage_common,
        ),
        MarketHybridTrainingStageConfig(
            name="joint_training",
            end_step=2,
            loss_weights=MarketHybridLossWeights(),
            trainable_modules=MarketHybridTrainableModules(),
            **stage_common,
        ),
        MarketHybridTrainingStageConfig(
            name="policy_finetuning",
            end_step=3,
            loss_weights=MarketHybridLossWeights(),
            trainable_modules=MarketHybridTrainableModules(jepa_predictor=False),
            **stage_common,
        ),
    ]
    request = base.model_copy(
        update={
            "training": MarketHybridTrainingConfig(
                mode="staged",
                max_steps=3,
                stages=stages,
                batch_size=2,
                gradient_accumulation_steps=1,
                validation_interval=1,
                checkpoint_interval=1,
                log_interval=1,
                validation_windows=32,
                num_workers=0,
                warmup_steps=0,
                amp="off",
                device="cpu",
            )
        }
    )
    run_dir = tmp_path / "staged-run"
    run_dir.mkdir()
    updates: list[dict[str, object]] = []
    train_markethybrid(
        request,
        prepared,
        run_dir,
        update_status=lambda **changes: updates.append(changes),
    )
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [record["stage_name"] for record in records] == [
        "representation_pretraining",
        "joint_training",
        "policy_finetuning",
    ]
    assert (run_dir / "best_hybrid.pt").exists()
    assert (run_dir / "best_loss.pt").exists()
    checkpoint = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    assert checkpoint["stage_name"] == "policy_finetuning"
    assert checkpoint["global_step"] == 3
    assert checkpoint["schedule_hash"] == request.training.schedule_hash()
    assert updates[-1]["state"] == "completed"
