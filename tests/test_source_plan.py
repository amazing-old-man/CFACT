import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
import pytest
from cfact.sources.manifest import build_plan
from cfact.collection.coordinator import _verify_prepared_sources


def source_fixture(tmp_path):
    from cfact.sources.prepare import PREPARED_SCHEMA, OPERATION_SCHEMA, preparer_code_identity

    path = tmp_path / "source.hdf5"
    with h5py.File(path, "w") as out:
        out.attrs["cfact_prepared_identity"] = json.dumps({
            "schema": PREPARED_SCHEMA, "operation_schema": OPERATION_SCHEMA,
            "preparer_code_sha256": preparer_code_identity(), "episodes": ["demo_0", "demo_1"],
            "raw_sha256": "synthetic_unit_fixture",
            "controller_semantics_sha256": {name: hashlib.sha256(
                json.dumps({"synthetic_unit_fixture": True}).encode()).hexdigest()
                for name in ["demo_0", "demo_1"]},
        })
        data = out.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "OpenDrawer"})
        for index in range(2):
            group = data.create_group(f"demo_{index}")
            group.create_dataset("actions", data=np.full((2, 12), index))
            group.create_dataset("states", data=np.zeros((2, 8)))
            group.attrs["model_file"] = "<mujoco/>"
            group.attrs["ep_meta"] = "{}"
            info = group.create_group("datagen_info")
            info.attrs["cfact_controller_semantics"] = json.dumps({"synthetic_unit_fixture": True})
            for key in ["eef_pose", "target_pose", "object_poses/handle"]:
                info.create_dataset(key, data=np.repeat(np.eye(4)[None], 2, axis=0))
            info.create_dataset("gripper_action", data=np.zeros((2, 1)))
            info.create_dataset("subtask_term_signals/stage_contact_handle", data=np.ones(2))
    spec = tmp_path / "sources.json"
    spec.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": "OpenDrawer",
                        "prepared_hdf5": path.name,
                        "roots": ["demo_0"],
                        "donors": ["demo_1"],
                    }
                ]
            }
        )
    )
    return spec, path


def test_portable_source_plan_verifies_and_detects_changed_inputs(tmp_path):
    spec, path = source_fixture(tmp_path)
    report = build_plan(spec, tmp_path / "plan")
    config = json.loads(Path(report["config"]).read_text())
    manifest = json.loads(Path(report["manifest"]).read_text())
    assert report["roots"] == 1
    route = manifest["requests_by_id"][manifest["queue_order"][0]]
    assert route["prepared_group"] == "demo_0"
    assert len(route["donor_source_ids"]) == 1
    result = _verify_prepared_sources(manifest, tmp_path / "check.json", config)
    assert result["status"] == "PASS"
    with h5py.File(path, "a") as out:
        out["data/demo_0/actions"][0, 0] = 99
    result = _verify_prepared_sources(manifest, tmp_path / "check.json", config)
    assert result["status"] == "SOURCE_INTEGRITY_FAILED"
    with pytest.raises(FileExistsError):
        build_plan(spec, tmp_path / "plan")


def test_root_donor_overlap_is_rejected_before_writes(tmp_path):
    spec, _ = source_fixture(tmp_path)
    record = json.loads(spec.read_text())
    record["tasks"][0]["donors"] = ["demo_0"]
    spec.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="disjoint"):
        build_plan(spec, tmp_path / "plan")
    assert not (tmp_path / "plan").exists()


def test_dataset_output_root_precedence(tmp_path, monkeypatch):
    spec, _ = source_fixture(tmp_path)
    data_disk = tmp_path / "data_disk"
    monkeypatch.setenv("CFACT_OUTPUT_ROOT", str(data_disk))
    first = build_plan(spec, tmp_path / "plan_env")
    assert json.loads(Path(first["config"]).read_text())["output_root"] == str(data_disk)
    explicit = tmp_path / "explicit"
    record = json.loads(spec.read_text())
    record["output_root"] = str(explicit)
    spec.write_text(json.dumps(record))
    second = build_plan(spec, tmp_path / "plan_explicit")
    assert json.loads(Path(second["config"]).read_text())["output_root"] == str(explicit)


def test_dual_view_storage_guard_measures_dataset_disk_and_keeps_budget(tmp_path, monkeypatch):
    from collections import namedtuple
    from cfact.collection.coordinator import _storage_admission

    run = tmp_path / "data_disk/release/_work/collection"
    run.mkdir(parents=True)
    (run / "measured.npz").write_bytes(b"x" * 500)
    plan = tmp_path / "code_disk/plan"
    plan.mkdir(parents=True)
    calls = []

    def usage(path):
        calls.append(path)
        return namedtuple("Usage", "total used free")(10000, 1000, 9000)

    monkeypatch.setattr("cfact.collection.coordinator.shutil.disk_usage", usage)
    config = {
        "release_root": str(plan),
        "storage": {"schema": "cfact.dual-view.v1"},
        "resource": {"maximum_output_gb": 0.0000004, "minimum_host_free_gib": 0},
    }
    report = _storage_admission(run, config)
    assert calls == [run.parent.parent]
    assert report["release_bytes"] == 500
    assert report["maximum_release_bytes"] == 400
    assert report["status"] == "STORAGE_GUARD_STOP"
