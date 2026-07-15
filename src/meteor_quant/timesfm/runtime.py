from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"


class ForecastRuntime(Protocol):
    device: str

    def forecast(
        self,
        inputs: list[NDArray[np.float32]],
        horizon: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    model_id_or_path: str
    cache_dir: str | None
    local_files_only: bool
    max_context: int
    max_horizon: int
    batch_size: int
    torch_compile: bool
    normalize_inputs: bool
    use_continuous_quantile_head: bool
    force_flip_invariance: bool
    infer_is_positive: bool
    fix_quantile_crossing: bool
    require_cuda: bool


class TimesFMRuntime:
    """Thread-safe wrapper around Google's official TimesFM 2.5 PyTorch package."""

    def __init__(self, config: RuntimeConfig) -> None:
        try:
            timesfm = importlib.import_module("timesfm")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "TimesFM is not installed. Run .\\setup-timesfm.ps1 or ./setup-timesfm.sh."
            ) from exc

        if config.require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "TimesFM was configured for CUDA, but this Python environment cannot access CUDA. "
                "Install a CUDA PyTorch wheel and restart Meteor Quant."
            )
        torch.set_float32_matmul_precision("high")

        model_class = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        if model_class is None:
            try:
                torch_backend = importlib.import_module(
                    "timesfm.timesfm_2p5.timesfm_2p5_torch"
                )
                model_class = torch_backend.TimesFM_2p5_200M_torch
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "The installed timesfm package does not expose the TimesFM 2.5 PyTorch model. "
                    "Install timesfm==2.0.2 or newer."
                ) from exc

        source = config.model_id_or_path
        load_kwargs: dict[str, Any] = {
            "torch_compile": config.torch_compile,
            "local_files_only": config.local_files_only,
        }
        if config.cache_dir:
            load_kwargs["cache_dir"] = config.cache_dir
        self._model = model_class.from_pretrained(source, **load_kwargs)
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=config.max_context,
                max_horizon=config.max_horizon,
                per_core_batch_size=config.batch_size,
                normalize_inputs=config.normalize_inputs,
                use_continuous_quantile_head=config.use_continuous_quantile_head,
                force_flip_invariance=config.force_flip_invariance,
                infer_is_positive=config.infer_is_positive,
                fix_quantile_crossing=config.fix_quantile_crossing,
            )
        )
        model_device = getattr(getattr(self._model, "model", None), "device", None)
        self.device = str(model_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._lock = threading.Lock()

    def forecast(
        self,
        inputs: list[NDArray[np.float32]],
        horizon: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        with self._lock:
            point, quantiles = self._model.forecast(horizon=horizon, inputs=list(inputs))
        return (
            np.asarray(point, dtype=np.float32),
            np.asarray(quantiles, dtype=np.float32),
        )


_RUNTIME_CACHE: dict[RuntimeConfig, TimesFMRuntime] = {}
_RUNTIME_CACHE_LOCK = threading.Lock()


def get_timesfm_runtime(config: RuntimeConfig) -> ForecastRuntime:
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(config)
        if runtime is None:
            runtime = TimesFMRuntime(config)
            _RUNTIME_CACHE[config] = runtime
        return runtime


def clear_timesfm_runtime_cache() -> None:
    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE.clear()


def timesfm_capabilities() -> dict[str, Any]:
    installed = importlib.util.find_spec("timesfm") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    payload: dict[str, Any] = {
        "installed": installed,
        "torch_installed": torch_installed,
        "model_id": DEFAULT_MODEL_ID,
        "max_context": 16_384,
        "max_quantile_horizon": 1_024,
        "cached_runtimes": len(_RUNTIME_CACHE),
    }
    if installed:
        try:
            payload["timesfm_version"] = importlib.metadata.version("timesfm")
        except importlib.metadata.PackageNotFoundError:
            payload["timesfm_version"] = "unknown"
    if torch_installed:
        try:
            torch = importlib.import_module("torch")
            payload.update(
                {
                    "torch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda,
                    "device": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else "CPU"
                    ),
                }
            )
        except Exception as exc:
            payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def model_source_identity(model_id_or_path: str) -> dict[str, Any]:
    path = Path(model_id_or_path)
    if not path.exists():
        return {"model_id": model_id_or_path}
    files: list[dict[str, Any]] = []
    candidates = [path] if path.is_file() else [
        path / "model.safetensors",
        path / "config.json",
        path / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            stat = candidate.stat()
            files.append(
                {
                    "path": str(candidate.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"local_model": str(path.resolve()), "files": files}


def installed_timesfm_version() -> str | None:
    try:
        return importlib.metadata.version("timesfm")
    except importlib.metadata.PackageNotFoundError:
        return None
