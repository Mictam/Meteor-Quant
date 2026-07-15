from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from meteor_quant.marketlm.fileio import atomic_json_write, read_json
from meteor_quant.marketlm.jobs import read_run_status, update_status_file


def test_atomic_json_retries_windows_replace_and_leaves_no_shared_temp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import meteor_quant.marketlm.fileio as fileio

    status_path = tmp_path / "status.json"
    atomic_json_write({"state": "queued"}, status_path)
    real_replace = fileio.os.replace
    failures = {"count": 0}

    def windows_replace(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == status_path and failures["count"] < 3:
            failures["count"] += 1
            raise PermissionError("simulated WinError 5")
        real_replace(source, destination)

    monkeypatch.setattr(fileio.os, "replace", windows_replace)
    monkeypatch.setattr(fileio.time, "sleep", lambda _seconds: None)

    update_status_file(status_path, state="preparing", progress=0.937)

    assert failures["count"] == 3
    assert read_json(status_path)["progress"] == 0.937
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_concurrent_status_updates_are_serialized(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    atomic_json_write({"state": "training"}, status_path)

    def write(index: int) -> None:
        update_status_file(status_path, **{f"update_{index}": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(32)))

    status = read_json(status_path)
    assert status["state"] == "training"
    for index in range(32):
        assert status[f"update_{index}"] == index
    assert not (tmp_path / ".status.json.lock").exists()


def test_terminal_result_snapshot_survives_status_file_contention(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status_path = run_dir / "status.json"
    atomic_json_write(
        {
            "run_id": "run",
            "state": "training",
            "progress": 0.937,
            "updated_at": "2026-07-13T18:00:00+00:00",
        },
        status_path,
    )
    atomic_json_write(
        {
            "state": "completed",
            "progress": 1.0,
            "checkpoint_path": "best.pt",
            "updated_at": "2026-07-13T19:00:00+00:00",
        },
        run_dir / "result.json",
    )

    status = read_run_status(status_path)

    assert status["run_id"] == "run"
    assert status["state"] == "completed"
    assert status["progress"] == 1.0
    assert status["checkpoint_path"] == "best.pt"
