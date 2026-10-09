"""Structural validators for raw and MimicGen-prepared replay datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py

from cfact.io.serialization import natural_key


def _leading_lengths(group: h5py.Group) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            for child, length in _leading_lengths(value).items():
                lengths[f"{name}/{child}"] = length
        elif value.shape:
            lengths[name] = int(value.shape[0])
    return lengths


def validate_prepared_dataset(
    path: Path,
    *,
    expected_task: str,
    expected_interface: str,
) -> dict[str, Any]:
    """Reject a prepared dataset with incomplete replay or datagen fields."""

    required_arrays = {"eef_pose", "target_pose", "gripper_action"}
    required_groups = {"object_poses", "subtask_term_signals"}
    episodes = []
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset:
            raise ValueError("prepared dataset has no data group")
        data = dataset["data"]
        env_args = data.attrs.get("env_args")
        if isinstance(env_args, bytes):
            env_args = env_args.decode()
        env_args = json.loads(env_args) if isinstance(env_args, str) else env_args
        if not isinstance(env_args, dict) or env_args.get("env_name") != expected_task:
            raise ValueError("prepared dataset task metadata mismatch")
        demo_ids = sorted(data.keys(), key=natural_key)
        if not demo_ids:
            raise ValueError("prepared dataset contains no episodes")
        for demo_id in demo_ids:
            episode = data[demo_id]
            for name in ("actions", "states"):
                if name not in episode or episode[name].ndim != 2 or not len(episode[name]):
                    raise ValueError(f"{demo_id} has invalid {name}")
            length = int(len(episode["actions"]))
            if len(episode["states"]) != length:
                raise ValueError(f"{demo_id} action/state length mismatch")
            if "model_file" not in episode.attrs or "ep_meta" not in episode.attrs:
                raise ValueError(f"{demo_id} lacks replay metadata")
            if "datagen_info" not in episode:
                raise ValueError(f"{demo_id} lacks datagen_info")
            info = episode["datagen_info"]
            interface = info.attrs.get("env_interface_name")
            if isinstance(interface, bytes):
                interface = interface.decode()
            if interface != expected_interface:
                raise ValueError(
                    f"{demo_id} interface mismatch: {interface} != {expected_interface}"
                )
            missing_arrays = required_arrays - set(info.keys())
            missing_groups = required_groups - set(info.keys())
            if missing_arrays or missing_groups:
                raise ValueError(
                    f"{demo_id} incomplete datagen_info: "
                    f"arrays={sorted(missing_arrays)}, groups={sorted(missing_groups)}"
                )
            for group_name in required_groups:
                if not isinstance(info[group_name], h5py.Group) or not info[group_name].keys():
                    raise ValueError(f"{demo_id} has empty {group_name}")
            lengths = _leading_lengths(info)
            wrong = {name: value for name, value in lengths.items() if value != length}
            if wrong:
                raise ValueError(f"{demo_id} datagen lengths mismatch: {wrong}")
            episodes.append(
                {
                    "episode_id": demo_id,
                    "steps": length,
                    "object_pose_keys": sorted(info["object_poses"].keys()),
                    "subtask_signal_keys": sorted(info["subtask_term_signals"].keys()),
                }
            )
    return {
        "schema_version": "1.0.0",
        "task": expected_task,
        "interface": expected_interface,
        "episode_count": len(episodes),
        "steps": sum(row["steps"] for row in episodes),
        "episodes": episodes,
        "status": "PREPARED_SCHEMA_VALID",
    }
