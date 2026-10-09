import json
import numpy as np
import pytest
from test_release import acquisition  # noqa: F401

from cfact.storage.schema import make_child_context, validate_child_context, training_observation


def test_legacy_time_uses_evidence_and_disagrees_loudly():
    state = np.array([3.25, 4.0, 5.0])
    result = validate_child_context({}, state, terminal_time=3.25)
    assert result["sim_time_s"] == 3.25
    assert result["time_origin"] == "legacy_official_flat_state_verified_by_local_trace"
    with pytest.raises(ValueError, match="time_mismatch"):
        validate_child_context({}, state, terminal_time=0.0)


def test_child_constructor_deep_copies_actual_context():
    context = {"task_goal": {"target_q": -0.1}}
    cursor = {"issued_source_index": 21, "physical_joint_q": -0.04}
    record = make_child_context(sim_state=np.array([1.5, 0.0]), operation_context=context,
                                source_cursor=cursor, snapshot={"path": "snapshot.npz"},
                                history_refs=[{"path": "ancestor"}])
    context["task_goal"]["target_q"] = 3.0
    cursor["issued_source_index"] = 100
    assert record["operation_context"]["task_goal"]["target_q"] == -0.1
    assert record["source_cursor"]["issued_source_index"] == 21
    json.dumps(record)


def test_training_observation_does_not_accept_outcomes_or_future_fields():
    arrays = {"sim_time": np.arange(3), "robot_qpos": np.ones((3, 2)),
              "task_success": np.ones(3), "future_reward": np.ones(3)}
    assert set(training_observation(arrays, 1)) == {"sim_time", "robot_qpos"}


def test_publisher_recovers_legacy_child_without_rewriting_raw(acquisition, tmp_path):  # noqa: F811
    from cfact.storage.tree import publish_tree

    run, root, config, scope = acquisition
    path = root / "nodes/n_child/edges/c/child_context.json"
    record = json.loads(path.read_text())
    record.pop("sim_time_s")
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    published = publish_tree(root, run, tmp_path / "published", config, scope)
    assert path.read_bytes() == before
    leaf = next(e for e in published["tree"]["edges"] if e["generator"] == "G2")
    node = next(n for n in published["tree"]["nodes"] if n["node_id"] == leaf["actual_child"])
    saved = json.loads((tmp_path / "published" / node["context"]).read_text())
    assert saved["sim_time_s"] == 0.1
    assert saved["time_origin"].startswith("legacy_official_flat_state")
