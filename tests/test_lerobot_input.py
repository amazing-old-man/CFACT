import gzip
import json
from pathlib import Path

import h5py
import numpy as np

from cfact.adapters import lerobot as adapter


MODALITY = {
    "state": {},
    "action": {
        "base_motion": {"original_key": "action", "start": 0, "end": 4},
        "control_mode": {"original_key": "action", "start": 4, "end": 5},
        "end_effector_position": {"original_key": "action", "start": 5, "end": 8},
        "end_effector_rotation": {"original_key": "action", "start": 8, "end": 11},
        "gripper_close": {"original_key": "action", "start": 11, "end": 12},
    },
}


def _fixture(root: Path) -> Path:
    dataset = root / "lerobot"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    episode = dataset / "extras" / "episode_000000"
    episode.mkdir(parents=True)
    (dataset / "meta" / "modality.json").write_text(json.dumps(MODALITY))
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {"action": {"shape": [12]}},
            }
        )
    )
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 2}) + "\n"
    )
    (dataset / "extras" / "dataset_meta.json").write_text(
        json.dumps(
            {
                "env_args": {"env_name": "OpenDrawer", "env_kwargs": {"seed": 3}},
                "total": 1,
            }
        )
    )
    np.savez_compressed(episode / "states.npz", states=np.arange(10).reshape(2, 5))
    with gzip.open(episode / "model.xml.gz", "wt", encoding="utf-8") as stream:
        stream.write("<mujoco model='fixture'/>")
    (episode / "ep_meta.json").write_text(json.dumps({"layout_id": 1}))
    (dataset / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"fixture")
    return dataset


def test_action_reordering_preserves_official_components():
    source = np.arange(12, dtype=float).reshape(1, 12)
    reordered = adapter.reorder_lerobot_actions(source, MODALITY)
    assert reordered.tolist() == [[5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]]


def test_inspection_uses_replay_extras_not_video(tmp_path: Path):
    dataset = _fixture(tmp_path)
    inspection = adapter.inspect_lerobot_dataset(dataset, expected_task="OpenDrawer")
    summary = inspection.summary()
    assert summary["episode_count"] == 1
    assert summary["state_rows"] == 2
    assert summary["video_used_as_state"] is False
    assert inspection.episodes[0].parquet_path.name == "episode_000000.parquet"


def test_raw_hdf5_import_preserves_state_xml_metadata_and_action_order(tmp_path: Path, monkeypatch):
    inspection = adapter.inspect_lerobot_dataset(_fixture(tmp_path), expected_task="OpenDrawer")
    actions = np.vstack([np.arange(12), np.arange(12) + 20])
    monkeypatch.setattr(
        adapter,
        "_read_parquet_transition_data",
        lambda _: (actions, np.array([0.0, 1.0]), np.array([False, True])),
    )
    guard_calls = []
    output = tmp_path / "derived" / "raw.hdf5"
    result = adapter.import_raw_replay_hdf5(
        inspection, output, progress_guard=lambda: guard_calls.append(True)
    )
    assert result["requires_official_prepare_step"] is True
    assert result["source_successful_episode_count"] == 1
    assert guard_calls == [True]
    with h5py.File(output, "r") as dataset:
        demo = dataset["data/demo_0"]
        assert demo["states"].shape == (2, 5)
        assert demo["actions"][0].tolist() == [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
        assert demo["source_next_reward"][:].tolist() == [0.0, 1.0]
        assert demo["source_next_done"][:].tolist() == [False, True]
        assert demo.attrs["source_terminal_reward"] == 1.0
        assert bool(demo.attrs["source_terminal_done"]) is True
        assert "fixture" in demo.attrs["model_file"]
        assert json.loads(demo.attrs["ep_meta"])["layout_id"] == 1
        assert demo.attrs["lerobot_episode_index"] == 0
