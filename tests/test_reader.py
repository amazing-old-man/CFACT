import json
import numpy as np
import pytest
from cfact.data.reader import TreeDataset


def test_installed_reader_retains_transition_alignment_and_tree(tmp_path):
    root = tmp_path / "data/formal/OpenDrawer/root1"
    root.mkdir(parents=True)
    (root / "root_result.json").write_text(
        json.dumps(
            {
                "task": "OpenDrawer",
                "root_id": "root1",
                "split": "formal",
                "status": "PHYSICS_COMPLETE",
            }
        )
    )
    (root / "root_manifest.json").write_text(json.dumps({"source_episode": "request1"}))
    edge = root / "nodes/n_root/edges/edge1"
    edge.mkdir(parents=True)
    row = {"candidate_id": "edge1", "edge_path": str(edge.relative_to(tmp_path))}
    (root / "tree.json").write_text(
        json.dumps({"edges": [row], "child_pools": [], "third_pools": []})
    )
    np.savez(
        edge / "execution.npz",
        commanded_actions=np.ones((2, 12)),
        raw_planned_actions=np.ones((2, 12)),
        controller_target_poses=np.repeat(np.eye(4)[None], 2, axis=0),
        local_event_flags=np.array([False, True]),
        sim_time=np.array([0.0, 0.05, 0.10]),
        robot_qpos=np.zeros((3, 7)),
    )
    reader = TreeDataset(tmp_path)
    assert reader.roots(task="OpenDrawer")[0]["request_id"] == "request1"
    transition = reader.read_transition("OpenDrawer", "root1", "edge1", 1)
    assert transition["state_t"]["sim_time"] == 0.05
    assert transition["state_t_plus_1"]["sim_time"] == 0.10
    assert transition["local_event_reached"]
    assert len(reader.read_tree_path("OpenDrawer", "root1", ["edge1"])) == 1
    with pytest.raises(ValueError):
        reader._inside("../outside.npz")
    with pytest.raises(IndexError):
        reader.read_transition("OpenDrawer", "root1", "edge1", 2)
