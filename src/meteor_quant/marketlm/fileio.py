from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

T = TypeVar("T")
_RETRYABLE_WINDOWS_ERRORS = {5, 32, 33}


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


def _retryable_file_error(exc: OSError) -> bool:
    if isinstance(exc, PermissionError):
        return True
    winerror = getattr(exc, "winerror", None)
    return isinstance(winerror, int) and winerror in _RETRYABLE_WINDOWS_ERRORS


def retry_file_operation(
    operation: Callable[[], T],
    *,
    path: Path,
    attempts: int = 60,
    initial_delay_seconds: float = 0.02,
    maximum_delay_seconds: float = 0.25,
) -> T:
    """Retry transient Windows sharing violations and access-denied errors."""

    delay_seconds = initial_delay_seconds
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except FileNotFoundError:
            raise
        except OSError as exc:
            if not _retryable_file_error(exc):
                raise
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.5, maximum_delay_seconds)
    raise PermissionError(
        f"could not complete file operation after {attempts} attempts: {path}"
    ) from last_error


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


def _lock_is_stale(path: Path, *, stale_after_seconds: float) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    age_seconds = max(time.time() - stat.st_mtime, 0.0)
    if age_seconds >= stale_after_seconds:
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return age_seconds >= 2.0
    return age_seconds >= 1.0 and not _pid_alive(pid)


@contextmanager
def interprocess_file_lock(
    path: Path,
    *,
    timeout_seconds: float = 30.0,
    stale_after_seconds: float = 120.0,
) -> Iterator[None]:
    """Cross-process lock based on exclusive lock-file creation.

    The critical sections protected by this lock are intentionally tiny. A lock older
    than two minutes is treated as abandoned, even if Windows has already reused its
    recorded PID.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(path)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError:
            if _lock_is_stale(lock_path, stale_after_seconds=stale_after_seconds):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    if not _retryable_file_error(exc):
                        raise
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for file lock: {lock_path}") from None
            time.sleep(0.02)

    try:
        owner = {
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "created_at_unix": time.time(),
        }
        os.write(descriptor, json.dumps(owner, sort_keys=True).encode("utf-8"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            retry_file_operation(
                lambda: lock_path.unlink(),
                path=lock_path,
                attempts=20,
            )
        except FileNotFoundError:
            pass
        except PermissionError:
            # A stale lock is self-healing on the next acquisition.
            pass



def atomic_file_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Write through a unique sibling and atomically replace the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with interprocess_file_lock(
        path, timeout_seconds=60.0, stale_after_seconds=3600.0
    ):
        temporary = path.parent / (
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.tmp"
        )
        try:
            writer(temporary)
            retry_file_operation(
                lambda: os.replace(temporary, path),
                path=path,
            )
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

def _atomic_json_write_unlocked(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.tmp"
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        retry_file_operation(
            lambda: os.replace(temporary, path),
            path=path,
        )
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    with interprocess_file_lock(path):
        _atomic_json_write_unlocked(payload, path)


def read_json(path: Path, *, attempts: int = 30) -> dict[str, Any]:
    delay_seconds = 0.01
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return cast(
                dict[str, Any],
                json.loads(path.read_text(encoding="utf-8")),
            )
        except (PermissionError, json.JSONDecodeError, FileNotFoundError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.5, 0.2)
    raise RuntimeError(f"could not read JSON file: {path}") from last_error


def update_json_file(path: Path, **changes: Any) -> dict[str, Any]:
    with interprocess_file_lock(path):
        current = read_json(path) if path.exists() else {}
        current.update(changes)
        _atomic_json_write_unlocked(current, path)
        return current
