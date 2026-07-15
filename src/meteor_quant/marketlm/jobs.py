from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from meteor_quant.marketlm.features import default_indicators, indicator_capabilities
from meteor_quant.marketlm.fileio import (
    atomic_json_write,
    read_json,
    update_json_file,
)
from meteor_quant.marketlm.schemas import (
    MarketLMRegisterRequest,
    MarketLMRunRequest,
)

_ACTIVE_STATES = {"queued", "preparing", "training", "stopping"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def update_status_file(path: Path, **changes: Any) -> dict[str, Any]:
    changes["updated_at"] = utc_now()
    return update_json_file(path, **changes)



def read_run_status(status_path: Path) -> dict[str, Any]:
    status = read_json(status_path)
    result_path = status_path.parent / "result.json"
    if result_path.exists():
        try:
            result = read_json(result_path)
        except (OSError, json.JSONDecodeError):
            return status
        status.update(result)
    return status

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class MarketLMJobManager:
    def __init__(self, project_root: Path, data_dir: Path) -> None:
        self.project_root = project_root
        self.data_dir = data_dir
        self.root = data_dir / "marketlm"
        self.runs_dir = self.root / "runs"
        self.prepared_dir = self.root / "prepared"
        self.registered_dir = self.root / "registered"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        self.registered_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._logs: dict[str, Any] = {}

    @property
    def torch_available(self) -> bool:
        return importlib.util.find_spec("torch") is not None

    def capabilities(self) -> dict[str, Any]:
        cuda_available = False
        torch_version: str | None = None
        cuda_version: str | None = None
        gpu_name: str | None = None
        if self.torch_available:
            try:
                import torch

                torch_version = torch.__version__
                cuda_version = torch.version.cuda
                cuda_available = torch.cuda.is_available()
                gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
            except Exception:
                pass
        return {
            "torch_available": self.torch_available,
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "gpu_name": gpu_name,
            "indicators": indicator_capabilities(),
            "default_indicators": [item.model_dump(mode="json") for item in default_indicators()],
            "timeframes_seconds": [1, 5, 15, 60, 300, 900, 3600],
            "model_presets": {
                "quick": {
                    "d_model": 192,
                    "n_layers": 6,
                    "n_heads": 6,
                    "mlp_hidden": 576,
                    "dropout": 0.08,
                    "rope_base": 10000.0,
                },
                "quality_4060ti": {
                    "d_model": 384,
                    "n_layers": 12,
                    "n_heads": 6,
                    "mlp_hidden": 1152,
                    "dropout": 0.08,
                    "rope_base": 10000.0,
                },
                "large_4060ti": {
                    "d_model": 512,
                    "n_layers": 16,
                    "n_heads": 8,
                    "mlp_hidden": 1536,
                    "dropout": 0.08,
                    "rope_base": 10000.0,
                },
            },
        }

    def start(self, request: MarketLMRunRequest) -> dict[str, Any]:
        if not self.torch_available and not request.prepare_only:
            raise RuntimeError(
                "PyTorch is not installed. Run .\\setup-marketlm.ps1 or install the marketlm extra."
            )
        active = [item for item in self.list_runs() if item.get("state") in _ACTIVE_STATES]
        if active:
            raise RuntimeError(
                f"MarketLM job {active[0]['run_id']} is already {active[0]['state']}; "
                "stop or finish it before starting another GPU job"
            )
        run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        kind = "prepare" if request.prepare_only else "train"
        spec = {
            "run_id": run_id,
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "prepared_root": str(self.prepared_dir),
            "request": request.model_dump(mode="json"),
        }
        atomic_json_write(spec, run_dir / "spec.json")
        status: dict[str, Any] = {
            "run_id": run_id,
            "name": request.name,
            "kind": kind,
            "state": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "pid": None,
            "progress": 0.0,
            "message": "queued",
            "step": 0,
            "max_steps": 0 if request.prepare_only else request.training.max_steps,
            "metrics": {},
            "prepared_dir": None,
            "checkpoint_path": None,
            "error": None,
        }
        atomic_json_write(status, run_dir / "status.json")
        log_handle = (run_dir / "worker.log").open("a", encoding="utf-8")
        creationflags = 0
        if sys.platform == "win32" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "meteor_quant.marketlm.worker",
                str(run_dir),
            ],
            cwd=self.project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        self._processes[run_id] = process
        self._logs[run_id] = log_handle
        return {**status, "pid": process.pid, "message": "worker started"}

    def list_runs(self) -> list[dict[str, Any]]:
        self._reconcile()
        results: list[dict[str, Any]] = []
        for path in self.runs_dir.glob("*/status.json"):
            try:
                item = read_run_status(path)
                item["registered"] = (self.registered_dir / f"{item['run_id']}.json").exists()
                results.append(item)
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        results.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return results

    def get(self, run_id: str) -> dict[str, Any]:
        self._reconcile()
        path = self._run_dir(run_id) / "status.json"
        if not path.exists():
            raise KeyError("MarketLM run not found")
        result = read_run_status(path)
        result["registered"] = (self.registered_dir / f"{run_id}.json").exists()
        metrics_path = self._run_dir(run_id) / "metrics.jsonl"
        if metrics_path.exists():
            records = []
            for line in metrics_path.read_text(encoding="utf-8").splitlines()[-200:]:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            result["metric_history"] = records
        log_path = self._run_dir(run_id) / "worker.log"
        if log_path.exists():
            result["log_tail"] = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        return result

    def stop(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        status_path = run_dir / "status.json"
        if not status_path.exists():
            raise KeyError("MarketLM run not found")
        status = read_run_status(status_path)
        if status.get("state") not in _ACTIVE_STATES:
            return status
        update_status_file(status_path, state="stopping", message="stop requested")
        process = self._processes.get(run_id)
        if process is not None and process.poll() is None:
            if sys.platform == "win32" and hasattr(signal, "CTRL_BREAK_EVENT"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
        else:
            pid = status.get("pid")
            if isinstance(pid, int) and pid > 0:
                with suppress(OSError):
                    os.kill(pid, signal.SIGTERM)
        return self.get(run_id)

    def register(
        self,
        run_id: str,
        request: MarketLMRegisterRequest,
    ) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        status = self.get(run_id)
        if status.get("state") != "completed":
            raise ValueError("only completed MarketLM runs can be registered")
        checkpoint_path = run_dir / f"{request.checkpoint}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path.name}")
        prepared_value = status.get("prepared_dir")
        if not isinstance(prepared_value, str) or not Path(prepared_value).exists():
            raise ValueError("prepared MarketLM tensors are missing for this run")
        spec = read_json(run_dir / "spec.json")
        run_request = MarketLMRunRequest.model_validate(spec["request"])
        horizons = run_request.data.horizons_seconds
        primary = request.primary_horizon_seconds or horizons[0]
        if primary not in horizons:
            raise ValueError(f"primary horizon must be one of {horizons}")
        payload = {
            "model_id": run_id,
            "run_id": run_id,
            "display_name": request.display_name or run_request.name,
            "description": request.description
            or "Custom MarketLM forecast indicator trained in Meteor Quant.",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "prepared_dir": str(Path(prepared_value).resolve()),
            "timeframe_seconds": run_request.data.timeframe_seconds,
            "horizons_seconds": horizons,
            "primary_horizon_seconds": primary,
            "registered_at": utc_now(),
        }
        atomic_json_write(payload, self.registered_dir / f"{run_id}.json")
        return payload

    def unregister(self, model_id: str) -> None:
        path = self.registered_dir / f"{model_id}.json"
        if not path.exists():
            raise KeyError("registered MarketLM model not found")
        path.unlink()

    def list_registered(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for path in self.registered_dir.glob("*.json"):
            try:
                payload = read_json(path)
                payload["available"] = (
                    Path(str(payload["checkpoint_path"])).exists()
                    and Path(str(payload["prepared_dir"])).exists()
                )
                output.append(payload)
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        output.sort(key=lambda item: str(item.get("registered_at", "")), reverse=True)
        return output

    def _run_dir(self, run_id: str) -> Path:
        if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
            raise KeyError("invalid MarketLM run id")
        path = (self.runs_dir / run_id).resolve()
        if not path.is_relative_to(self.runs_dir.resolve()):
            raise KeyError("invalid MarketLM run id")
        return path

    def _reconcile(self) -> None:
        for status_path in self.runs_dir.glob("*/status.json"):
            try:
                status = read_run_status(status_path)
                run_id = str(status["run_id"])
                pid = status.get("pid")
                if (
                    status.get("state") in _ACTIVE_STATES
                    and run_id not in self._processes
                    and (not isinstance(pid, int) or not _pid_alive(pid))
                ):
                    update_status_file(
                        status_path,
                        state="failed",
                        message="worker is no longer running",
                        error="worker is no longer running",
                    )
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        for run_id, process in list(self._processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle = self._logs.pop(run_id, None)
            if log_handle is not None:
                log_handle.close()
            self._processes.pop(run_id, None)
            status_path = self._run_dir(run_id) / "status.json"
            if not status_path.exists():
                continue
            status = read_run_status(status_path)
            if status.get("state") in _ACTIVE_STATES:
                state = "stopped" if status.get("state") == "stopping" else "failed"
                update_status_file(
                    status_path,
                    state=state,
                    message=f"worker exited with code {return_code}",
                    error=None if state == "stopped" else f"worker exited with code {return_code}",
                )
