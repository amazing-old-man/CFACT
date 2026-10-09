"""Synthetic API integration fixture; never counted as physical CFACT output."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("lerobot")

from cfact.export.dataset import write_dataset, verify_rows
from cfact.export.shards import open_local_shard


def test_multi_episode_offsets_cameras_and_independent_tree_export(tmp_path):
    segments = []
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    for number, count in enumerate([2, 3, 4]):
        directory = tmp_path / f"fixture/{number}"
        directory.mkdir(parents=True)
        times = np.arange(count + 1) / 20 + number
        actions = np.arange(count * 12).reshape(count, 12) / 100 + number
        np.savez(directory / "trace.npz", sim_time=times, commanded_actions=actions,
                 robot_qpos=np.ones((count + 1, 7)) * number,
                 robot_qvel=np.ones((count + 1, 7)) * .5,
                 actual_eef_pose=np.repeat(np.eye(4)[None], count + 1, 0),
                 robot_qpos_joint_names=np.array([f"j{x}" for x in range(7)]))
        paths = {}
        for camera in cameras:
            paths[camera] = []
            for step in range(count + 1):
                filename = directory / f"{camera}-{step}.png"
                Image.new("RGB", (32, 32), (step * 20, number * 30, 90)).save(filename)
                paths[camera].append(filename.relative_to(tmp_path).as_posix())
        media = {"cameras": cameras, "clean_camera_frame_paths": paths}
        (directory / "media.json").write_text(json.dumps(media))
        segments.append({"segment_id": f"synthetic_{number}", "task": "SyntheticAPIFixture",
            "root_id": "tree_0" if number < 2 else "tree_1", "edge_id": f"edge_{number}",
            "segment": "local", "fps": 20, "action_count": count,
            "camera_frame_indices": list(range(count + 1)),
            "trace_path": f"fixture/{number}/trace.npz", "media": f"fixture/{number}/media.json"})
    rows = write_dataset(tmp_path, segments, Path("lerobot/by_task/SyntheticAPIFixture"))
    assert [r["episode_index"] for r in rows] == [0, 1, 2]
    assert [r["dataset_start_index"] for r in rows] == [0, 2, 5]
    assert verify_rows(tmp_path, rows)["all_rows_checked"] == 9
    tree = write_dataset(tmp_path, segments[:2], Path("lerobot/by_tree_exports/SyntheticAPIFixture/tree_0"))
    task_reader, tree_reader = open_local_shard(tmp_path, rows[0]), open_local_shard(tmp_path, tree[0])
    assert len(tree_reader) == 5
    for index in range(5):
        for key in ("action", "observation.state", "timestamp", "episode_index"):
            np.testing.assert_array_equal(task_reader[index][key].numpy(), tree_reader[index][key].numpy())
    assert write_dataset(tmp_path, segments, Path("lerobot/by_task/SyntheticAPIFixture")) == rows
