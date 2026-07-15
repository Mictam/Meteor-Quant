from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

from meteor_quant.datasets import DatasetCatalog
from meteor_quant.marketlm.dataset import prepare_training_data
from meteor_quant.marketlm.fileio import atomic_json_write
from meteor_quant.marketlm.jobs import read_json, update_status_file, utc_now
from meteor_quant.marketlm.schemas import MarketLMRunRequest
from meteor_quant.marketlm.training import train_marketlm


def run_worker(run_dir: Path) -> None:
    spec = read_json(run_dir / "spec.json")
    request = MarketLMRunRequest.model_validate(spec["request"])
    status_path = run_dir / "status.json"

    pending_status: dict[str, Any] = {}

    def update(**changes: Any) -> None:
        pending_status.update(changes)
        try:
            update_status_file(status_path, **pending_status)
        except (OSError, TimeoutError) as exc:
            print(
                f"warning: status update deferred after Windows file contention: {exc}",
                flush=True,
            )
        else:
            pending_status.clear()

    def write_result(**changes: Any) -> None:
        changes["updated_at"] = utc_now()
        atomic_json_write(changes, run_dir / "result.json")

    def preparation_progress(progress: float, message: str) -> None:
        update(
            state="preparing",
            progress=min(max(progress * (1.0 if request.prepare_only else 0.15), 0.0), 1.0),
            message=message,
        )

    try:
        update(
            state="preparing",
            progress=0.0,
            message="preparing MarketLM data",
            pid=os.getpid(),
        )
        catalog = DatasetCatalog(Path(spec["data_dir"]))
        prepared = prepare_training_data(
            catalog,
            request,
            Path(spec["prepared_root"]),
            status=preparation_progress,
        )
        update(prepared_dir=str(prepared))
        if request.prepare_only:
            terminal = {
                "state": "completed",
                "progress": 1.0,
                "message": "MarketLM data preparation completed",
                "prepared_dir": str(prepared),
            }
            update(**terminal)
            write_result(**terminal)
            return
        final_path = train_marketlm(
            request,
            prepared,
            run_dir,
            update_status=update,
        )
        write_result(
            state="completed",
            progress=1.0,
            step=request.training.max_steps,
            max_steps=request.training.max_steps,
            message="training completed",
            prepared_dir=str(prepared),
            checkpoint_path=str(run_dir / "best.pt"),
            final_checkpoint_path=str(final_path),
        )
    except KeyboardInterrupt:
        current = read_json(status_path)
        if current.get("state") not in {"stopped", "completed"}:
            update(state="stopped", message="worker stopped")
        write_result(state="stopped", message="worker stopped")
        raise SystemExit(130) from None
    except Exception as exc:
        traceback.print_exc()
        terminal = {
            "state": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_result(**terminal)
        update(**terminal)
        raise


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m meteor_quant.marketlm.worker RUN_DIR")
    run_worker(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
