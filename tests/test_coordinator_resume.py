"""Interruption recovery keeps both physics evidence and deterministic media spool."""

from __future__ import annotations
import json
import os
from pathlib import Path

import pytest

from cfact.collection.coordinator import _quarantine_interrupted_job  # noqa: E402
from cfact.io.serialization import RunLock  # noqa: E402


JOB = {"split": "development", "task": "CoffeeSetupMug", "root_id": "demo_12"}
ROOT = "CoffeeSetupMug__demo_12__abcdef123456"


def test_resume_quarantines_incomplete_data_and_video_spools(tmp_path: Path) -> None:
    run = tmp_path / "run"
    root = run / "data/development/CoffeeSetupMug" / ROOT
    root.mkdir(parents=True)
    (root / "completed.json").write_text('{"status":"PARTIAL"}')
    (root / "run.lock").write_text(json.dumps({"pid": 999999999, "process_start_ticks": 1}))
    (root / "execution.npz").write_bytes(b"preserve prior physical evidence")
    spool = run / "videos/G1/development/CoffeeSetupMug" / ROOT / "n_root__o_01__quad.frames"
    spool.mkdir(parents=True)
    (spool / "000001.jpg").write_bytes(b"preserve video frame")
    history = run / "history/development/CoffeeSetupMug/demo_12"
    history.mkdir(parents=True)
    (history / "frame_0000000.npz").write_bytes(b"preserve true replay history")
    original = run / "videos/original/CoffeeSetupMug/demo_12/source_original.mp4"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"source remains in place")
    old_result = run / "runtime/job_results/development/CoffeeSetupMug/demo_12.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text('{"status":"PARTIAL"}')
    old_log = run / "runtime/logs/development/CoffeeSetupMug/demo_12.log"
    old_log.parent.mkdir(parents=True)
    old_log.write_text("first attempt\n")
    alive = run / "runtime/scene_alive/development__CoffeeSetupMug__demo_12.json"
    alive.parent.mkdir(parents=True)
    alive.write_text('{"pid":999999999}')

    archive = _quarantine_interrupted_job(run, JOB)

    assert archive is not None
    assert not root.exists() and not spool.exists()
    assert (
        archive / root.relative_to(run) / "execution.npz"
    ).read_bytes() == b"preserve prior physical evidence"
    assert (archive / spool.relative_to(run) / "000001.jpg").read_bytes() == b"preserve video frame"
    assert (archive / history.relative_to(run) / "frame_0000000.npz").read_bytes() == b"preserve true replay history"
    assert (archive / old_result.relative_to(run)).is_file()
    assert (archive / old_log.relative_to(run)).is_file()
    assert (archive / alive.relative_to(run)).is_file()
    assert original.read_bytes() == b"source remains in place"
    assert _quarantine_interrupted_job(run, JOB) is None


def test_resume_refuses_live_root_owner(tmp_path: Path) -> None:
    run = tmp_path / "run"
    root = run / "data/development/CoffeeSetupMug" / ROOT
    root.mkdir(parents=True)
    (root / "run.lock").write_text(
        json.dumps(
            {"pid": os.getpid(), "process_start_ticks": RunLock._process_start_ticks(os.getpid())}
        )
    )
    (root / "execution.npz").write_bytes(b"live data")

    with pytest.raises(RuntimeError, match="worker still alive"):
        _quarantine_interrupted_job(run, JOB)
    assert (root / "execution.npz").read_bytes() == b"live data"
