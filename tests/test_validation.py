import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cfact.data.validation import validate_prepared_dataset


def _prepared(path: Path, *, signal_length: int = 3) -> None:
    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "OpenDrawer"})
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((3, 12)))
        demo.create_dataset("states", data=np.zeros((3, 20)))
        demo.attrs["model_file"] = "<mujoco/>"
        demo.attrs["ep_meta"] = "{}"
        info = demo.create_group("datagen_info")
        info.attrs["env_interface_name"] = "MG_OpenDrawer"
        info.attrs["env_interface_type"] = "robosuite"
        info.create_dataset("eef_pose", data=np.zeros((3, 4, 4)))
        info.create_dataset("target_pose", data=np.zeros((3, 4, 4)))
        info.create_dataset("gripper_action", data=np.zeros((3, 1)))
        info.create_dataset("object_poses/handle", data=np.zeros((3, 4, 4)))
        info.create_dataset(
            "subtask_term_signals/stage_contact_handle",
            data=np.zeros(signal_length),
        )


def test_prepared_dataset_validator_accepts_complete_schema(tmp_path: Path):
    path = tmp_path / "prepared.hdf5"
    _prepared(path)
    result = validate_prepared_dataset(
        path, expected_task="OpenDrawer", expected_interface="MG_OpenDrawer"
    )
    assert result["status"] == "PREPARED_SCHEMA_VALID"
    assert result["steps"] == 3


def test_prepared_dataset_validator_rejects_misaligned_signal(tmp_path: Path):
    path = tmp_path / "prepared.hdf5"
    _prepared(path, signal_length=2)
    with pytest.raises(ValueError, match="datagen lengths mismatch"):
        validate_prepared_dataset(
            path, expected_task="OpenDrawer", expected_interface="MG_OpenDrawer"
        )
