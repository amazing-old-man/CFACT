"""Small, deterministic persistence helpers."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = f"{array.dtype.str}:{array.shape}".encode()
    return hashlib.sha256(descriptor + array.tobytes()).hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Scene workers often publish the same shared admission record. A fixed
    # ``.tmp`` suffix lets two writers truncate/replace one another's staging
    # file. Keep every staging path unique and in the destination directory so
    # ``replace`` remains atomic on the same filesystem.
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def natural_key(value: str) -> tuple[Any, ...]:
    parts: list[Any] = []
    current = ""
    numeric = False
    for char in value:
        if char.isdigit() != numeric and current:
            parts.append(int(current) if numeric else current)
            current = ""
        current += char
        numeric = char.isdigit()
    if current:
        parts.append(int(current) if numeric else current)
    return tuple(parts)


def validate_unique(records: Iterable[dict[str, Any]], key: str) -> None:
    values = [record[key] for record in records]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {key} values")


class RunLock:
    """Host-local lock preventing duplicate writers for one run directory."""

    def __init__(self, path: Path):
        self.path = path
        self.owner_id = uuid.uuid4().hex
        self._guard = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()

    @staticmethod
    def _process_start_ticks(pid: int) -> int | None:
        try:
            # Linux /proc stat field 22 is the process start time since boot.
            return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            return None

    def acquire(self) -> None:
        if self._guard is not None:
            raise RuntimeError(f"run already active: {self.path}")
        # Never unlink the guard inode: doing so lets waiters lock different files.
        guard = self.path.with_name(self.path.name + ".guard").open("a+")
        try:
            try:
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"run already active: {self.path}") from error
            if self.path.exists():
                try:
                    existing = json.loads(self.path.read_text())
                    pid = int(existing["pid"])
                    if existing.get("hostname", socket.gethostname()) != socket.gethostname():
                        raise RuntimeError(f"cannot verify lock owned by another host: {self.path}")
                    if pid <= 0:
                        raise ValueError("invalid owner pid")
                    expected_start = existing.get("process_start_ticks")
                    os.kill(pid, 0)
                    if expected_start is not None:
                        actual_start = self._process_start_ticks(pid)
                        if actual_start is None:
                            raise RuntimeError(f"cannot inspect live lock owner: {self.path}")
                        if actual_start != int(expected_start):
                            raise ProcessLookupError
                except ProcessLookupError:
                    pass  # A dead local owner can be recovered while holding the kernel guard.
                except (KeyError, ValueError, PermissionError) as error:
                    raise RuntimeError(f"cannot verify run lock {self.path}") from error
                else:
                    raise RuntimeError(f"run already active with pid {pid}: {self.path.parent}")
            atomic_json(
                self.path,
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "owner_id": self.owner_id,
                    "process_start_ticks": self._process_start_ticks(os.getpid()),
                },
            )
            self._guard = guard
        except BaseException:
            guard.close()
            raise

    def release(self) -> None:
        if self._guard is None:
            return
        try:
            if self.path.exists():
                payload = json.loads(self.path.read_text())
                if payload.get("owner_id") == self.owner_id:
                    self.path.unlink()
        finally:
            self._guard.close()
            self._guard = None
