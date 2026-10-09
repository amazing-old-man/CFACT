"""New-format recovery records must survive publication without fabrication."""
import json
import shutil
import numpy as np
import pytest
from test_release import acquisition as acquisition, put
from test_backend_snapshot import fake_env

from cfact.execution.snapshot import capture_snapshot, save_snapshot, load_snapshot
from cfact.storage.schema import make_child_context
from cfact.storage.tree import publish_tree
from cfact.storage.integrity import input_fingerprint
from cfact.io.review_video import QuadReviewWriter


def new_child(root, *, save=True):
    edge = root / "nodes/n_child/edges/c"
    with np.load(edge / "child_context.npz", allow_pickle=False) as values:
        state = values["sim_state"].copy()
    context = make_child_context(
        sim_state=state, operation_context={"task_goal": {"target_q": -.1}},
        source_cursor={"next_action_index": 10}, snapshot={"path": "child_snapshot.json"},
        history_refs=[],
    )
    put(edge / "child_context.json", context)
    snapshot = capture_snapshot(fake_env())
    snapshot["states"], snapshot["sim_time_s"] = state, float(state[0])
    if save:
        save_snapshot(edge / "child_snapshot.json", snapshot)
    return edge, snapshot


def test_new_child_missing_snapshot_is_rejected(acquisition, tmp_path):
    run, root, config, scope = acquisition
    new_child(root, save=False)
    with pytest.raises(ValueError, match="backend snapshot is missing"):
        publish_tree(root, run, tmp_path / "release", config, scope)


def test_declared_parent_snapshot_is_required(acquisition, tmp_path):
    run, root, config, scope = acquisition
    path = root / "nodes/n_root/context.json"
    context = json.loads(path.read_text())
    context["snapshot"] = "parent_snapshot.json"
    put(path, context)
    with pytest.raises(ValueError, match="backend snapshot is missing"):
        publish_tree(root, run, tmp_path / "release", config, scope)


@pytest.mark.parametrize("field", ["states", "sim_time_s"])
def test_snapshot_must_match_actual_child_state_and_time(acquisition, tmp_path, field):
    run, root, config, scope = acquisition
    edge, snapshot = new_child(root)
    snapshot[field] = snapshot[field] + 1
    save_snapshot(edge / "child_snapshot.json", snapshot)
    with pytest.raises(ValueError, match="Backend snapshot.*differs"):
        publish_tree(root, run, tmp_path / "release", config, scope)


def test_complete_snapshot_is_copied_to_portable_node_and_legacy_is_limited(acquisition, tmp_path):
    run, root, config, scope = acquisition
    edge, snapshot = new_child(root)
    output = tmp_path / "release"
    published = publish_tree(root, run, output, config, scope)
    leaf = next(row for row in published["tree"]["edges"] if row["generator"] == "G2")
    node = next(row for row in published["tree"]["nodes"] if row["node_id"] == leaf["actual_child"])
    context = json.loads((output / node["context"]).read_text())
    copied = output / context["snapshot"]["path"]
    assert copied.name == "backend_snapshot.json" and copied.is_file()
    assert copied.stat().st_ino != (edge / "child_snapshot.json").stat().st_ino
    np.testing.assert_array_equal(load_snapshot(copied)["states"], snapshot["states"])
    parent = next(row for row in published["tree"]["nodes"] if row["parent_edge"] is None)
    parent_context = json.loads((output / parent["context"]).read_text())
    assert parent_context["snapshot"]["status"] == "LEGACY_UNAVAILABLE"


def test_nonempty_local_or_fixed_trace_cannot_disappear_from_lineage(acquisition, tmp_path):
    run, root, config, scope = acquisition
    # Leave valid clean camera frames but remove recorded review media. Their
    # filenames are deliberately distinct from the six-digit review spool.
    media = json.loads((root / "nodes/n_root/edges/a/media.json").read_text())
    (run / media["video"]).unlink()
    with pytest.raises(ValueError, match="Nonempty measured.*missing review media"):
        publish_tree(root, run, tmp_path / "release", config, scope)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="system ffmpeg required")
def test_blocked_root_preserves_and_fingerprints_actual_replay_prefix(acquisition, tmp_path):
    run, root, config, scope = acquisition
    # This root never reached a decision parent; do not manufacture a node.
    shutil.rmtree(root / "nodes")
    history = run / "history/blocked_root"
    history.mkdir(parents=True)
    rows = []
    for index in range(3):
        timestamp = index * .05
        name = f"frame_{index:07d}.npz"
        arrays = {"rgb": np.full((3, 8, 8, 3), index * 40, dtype=np.uint8),
                  "sim_state": np.array([timestamp, index]), "sim_time_s": timestamp}
        if index:
            arrays["action_from_previous"] = np.zeros(12)
        np.savez(history / name, **arrays)
        rows.append({"frame": name, "segment": "source_prefix", "phase": "source_prefix",
                     "sim_time_s": timestamp, "has_action_from_previous": bool(index)})
    (history / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    put(history / "index.json", {"camera_names": list(QuadReviewWriter.CAMERAS),
                                "width": 8, "height": 8, "frames": "frames.jsonl"})
    manifest_path = root / "root_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_replay_history"] = str(history)
    put(manifest_path, manifest)
    put(root / "root_result.json", {"task": "OpenDrawer", "status": "BLOCKED_SOURCE_REPLAY",
                                    "edges": [], "nodes": [], "charged_control_steps": 2,
                                    "source_replay_history": str(history)})
    fingerprint = input_fingerprint(run, [root])
    output = tmp_path / "release"
    result = publish_tree(root, run, output, config, scope)
    assert result["tree"]["nodes"] == [] and result["tree"]["edges"] == []
    assert result["tree"]["source_replay_prefix_status"] == "AVAILABLE"
    assert result["tree"]["source_replay_prefix_media"]["control_steps"] == 2
    copied = (output / result["tree"]["measured_history"]).parent
    assert (copied / "frame_0000002.npz").read_bytes() == (history / "frame_0000002.npz").read_bytes()
    with (history / "frame_0000002.npz").open("ab") as stream:
        stream.write(b"changed evidence")
    assert input_fingerprint(run, [root]) != fingerprint
