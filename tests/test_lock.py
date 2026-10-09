from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from cfact.io.serialization import RunLock


def test_run_lock_rejects_same_live_process(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    first = RunLock(path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="run already active"):
            RunLock(path).acquire()
    finally:
        first.release()
    assert not path.exists()


def test_run_lock_replaces_reused_pid_with_different_start_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.lock"
    path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "owner_id": "old-owner",
                "process_start_ticks": -1,
            }
        )
    )
    lock = RunLock(path)
    lock.acquire()
    try:
        payload = json.loads(path.read_text())
        assert payload["owner_id"] == lock.owner_id
        assert payload["process_start_ticks"] != -1
    finally:
        lock.release()


def test_kernel_guard_blocks_even_before_owner_metadata_is_written(tmp_path):
    import fcntl

    path = tmp_path / "run.lock"
    with path.with_name("run.lock.guard").open("a+") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not path.exists()
        with pytest.raises(RuntimeError, match="run already active"):
            RunLock(path).acquire()
        assert not path.exists()
    lock = RunLock(path)
    lock.acquire()
    lock.release()


@pytest.mark.parametrize("record", ["", '{"pid":123,"hostname":"another-host"}'])
def test_unverifiable_owner_is_preserved(tmp_path, record):
    path = tmp_path / "run.lock"
    path.write_text(record)
    with pytest.raises(RuntimeError, match="cannot verify"):
        RunLock(path).acquire()
    assert path.read_text() == record
