import json
from importlib.resources import files

import h5py
import pytest

from cfact.collection.manifest_runner import read_profile, validate_manifest, status
from cfact.collection.coordinator import _run_batch


def test_smoke_profile_requires_sixty_distinct_requests_and_preserves_defaults(tmp_path):
    profile = read_profile("coverage")
    raw = tmp_path / "raw.hdf5"
    with h5py.File(raw, "w") as out:
        for name in ("demo_0", "demo_1", "demo_2", "donor"):
            out.create_group("data/" + name)
    tasks = list(json.loads(files("cfact.resources").joinpath("tasks.json").read_text()))
    manifest = {"tasks": [{"task": task, "raw_hdf5": str(raw),
                           "roots": ["demo_0", "demo_1", "demo_2"], "donors": ["donor"]}
                          for task in tasks]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert len(validate_manifest(path, profile)["tasks"]) == 20
    manifest["tasks"][0]["roots"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="three distinct"):
        validate_manifest(path, profile)
    defaults = json.loads(files("cfact.resources").joinpath("collection.json").read_text())
    assert defaults["execution"]["local_timeout_steps"] == 160
    assert defaults["execution"]["continuation_steps"] == 150
    assert defaults["tree_limits"]["maximum_decision_depth"] == 3
    assert profile["max_edges"] == 60 * 2 * 4
    assert profile["max_workers"] == 6


def test_pause_does_not_dispatch_pending_root(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "runtime").mkdir(parents=True)
    (run / "runtime/pause_requested.json").write_text("{}")
    config = tmp_path / "config.json"
    config.write_text('{"resource":{}}')
    monkeypatch.setattr("cfact.collection.coordinator._gpu_status", lambda: [])
    monkeypatch.setattr("cfact.collection.coordinator.subprocess.Popen",
                        lambda *a, **kw: pytest.fail("paused job was dispatched"))
    jobs = [{"task": "OpenDrawer", "root_id": "r", "split": "formal"}]
    assert _run_batch(run=run, jobs=jobs, config=config, split_manifest=tmp_path / "m",
                      workers=1, devices=[0], resume=False) == []
    saved = json.loads((run / "runtime/paused.json").read_text())
    assert saved["pending_jobs"] == jobs
    assert saved["active_workers"] == 0


def test_completed_blocked_root_is_not_replayed_on_resume(tmp_path, monkeypatch):
    marker = tmp_path / "root_result.json"
    marker.write_text('{"status":"BLOCKED_SOURCE_REPLAY"}')
    config = tmp_path / "config.json"
    config.write_text('{"resource":{}}')
    monkeypatch.setattr("cfact.collection.coordinator._result_paths", lambda *a: [marker])
    monkeypatch.setattr("cfact.collection.coordinator._gpu_status", lambda: [])
    monkeypatch.setattr("cfact.collection.coordinator.subprocess.Popen",
                        lambda *a, **kw: pytest.fail("completed blocked root was replayed"))
    assert _run_batch(run=tmp_path, jobs=[{"root_id": "r"}], config=config,
                      split_manifest=tmp_path / "m", workers=1, devices=[0], resume=True) == []
    assert status(tmp_path)["outcomes"] == []


def test_status_does_not_call_building_release_complete(tmp_path):
    path = tmp_path / "runs/collection/release.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"BUILDING"}')
    assert not status(tmp_path)["release_complete"]
    path.write_text('{"status":"COMPLETE"}')
    assert status(tmp_path)["release_complete"]
