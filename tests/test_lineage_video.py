from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from test_release import acquisition as acquisition, fake_episode_writer, put

from cfact.export.lineage_video import _review_panel, render_lineage, render_replay_prefix
from cfact.io.review_video import QuadReviewWriter
from cfact.io.video import FFmpegRGBWriter, ffmpeg_available
from cfact.io.serialization import array_digest
from cfact.sources.manifest import sha256_file


def test_prefix_review_uses_three_recorded_cameras_without_cropping():
    frames = np.empty((3, 360, 640, 3), dtype=np.uint8)
    frames[0] = [255, 0, 0]
    frames[1] = [0, 255, 0]
    frames[2] = [0, 0, 255]
    panel = _review_panel(frames, 0.05, "source_prefix")
    assert panel.shape == (360, 640, 3)
    np.testing.assert_array_equal(panel[90, 160], [255, 0, 0])
    np.testing.assert_array_equal(panel[90, 480], [0, 255, 0])
    np.testing.assert_array_equal(panel[270, 160], [0, 0, 255])


@pytest.mark.skipif(not ffmpeg_available(), reason="system ffmpeg H264 encoder required")
def test_prefix_is_rendered_from_current_replay_npz_and_keeps_identity(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    rows = []
    # A completed earlier replay followed by the actual parent replay.
    for index, (sim_time, has_action) in enumerate(((0., False), (0.05, True),
                                                  (0., False), (0.05, True), (0.1, True))):
        frame = np.full((3, 36, 64, 3), 20 * index, dtype=np.uint8)
        name = f"frame_{index:07d}.npz"
        np.savez_compressed(history / name, rgb=frame, sim_time_s=sim_time, sim_state=[sim_time])
        rows.append({"frame": name, "segment": "source_prefix", "phase": "source_prefix",
                     "sim_time_s": sim_time, "has_action_from_previous": has_action})
    (history / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (history / "index.json").write_text(json.dumps({
        "camera_names": list(QuadReviewWriter.CAMERAS), "width": 64, "height": 36,
        "frames": "frames.jsonl",
    }))
    result = render_replay_prefix(history, tmp_path / "prefix.mp4")
    assert result["frame_count"] == 3
    assert result["control_steps"] == 2
    assert result["frame_sim_time_s"] == [0., 0.05, 0.1]
    assert result["discarded_frames_from_earlier_replays"] == 2
    assert result["source_original_video_is_distinct"]
    assert result["render_source"] == "actual_current_source_replay_history"
    assert result["initial_state_available"]


def release_fixture(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    video = release / "review.mp4"
    with FFmpegRGBWriter(video, width=64, height=36, fps=20) as writer:
        for index in range(8):
            writer.append_data(np.full((36, 64, 3), index * 25, dtype=np.uint8))
    lineage = {
        "lineage_id": "path1", "root_id": "root1", "edge_ids": ["edge1", "edge2"],
        "playlist": "path1.json", "segments": [
            {"segment": "source_prefix", "video": "review.mp4", "start_s": 0., "end_s": 0.1},
            {"edge_id": "edge1", "segment": "local", "video": "review.mp4",
             "start_s": 0.1, "end_s": 0.2},
            {"edge_id": "edge2", "segment": "local", "video": "review.mp4",
             "start_s": 0.2, "end_s": 0.3},
            {"edge_id": "edge2", "segment": "fixed_continuation", "video": "review.mp4",
             "start_s": 0.3, "end_s": 0.4},
        ],
    }
    (release / "release.json").write_text(json.dumps({"lineage_paths": "lineages.jsonl"}))
    (release / "lineages.jsonl").write_text(json.dumps(lineage) + "\n")
    (release / "path1.json").write_text(json.dumps(lineage))
    return release, lineage


@pytest.mark.skipif(not ffmpeg_available() or shutil.which("ffprobe") is None,
                    reason="system ffmpeg and ffprobe required")
def test_complete_lineage_renders_shared_prefix_ancestor_local_and_leaf_tail(tmp_path):
    release, _ = release_fixture(tmp_path)
    result = render_lineage(release, "path1", tmp_path / "complete.mp4")
    assert result["frame_count"] == 8
    assert result["source_prefix_included"]
    assert result["shared_ancestor_media_copied"] is False
    assert [row["output_start_frame"] for row in result["segments"]] == [0, 2, 4, 6]
    assert result["resolution"] == [64, 36]


@pytest.mark.skipif(not ffmpeg_available() or shutil.which("ffprobe") is None,
                    reason="system ffmpeg and ffprobe required")
def test_lineage_renderer_rejects_a_sibling_and_nonleaf_fixed_tail(tmp_path):
    release, lineage = release_fixture(tmp_path)
    lineage["segments"][2]["edge_id"] = "sibling"
    (release / "path1.json").write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match="not_an_ancestor"):
        render_lineage(release, "path1", tmp_path / "bad.mp4")
    lineage["segments"][2]["edge_id"] = "edge2"
    lineage["segments"][1]["segment"] = "fixed_continuation"
    (release / "path1.json").write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match="only_selected_leaf"):
        render_lineage(release, "path1", tmp_path / "bad.mp4")


def _add_saved_history(acquisition):
    """Make the two-level acquisition fixture carry real saved media ledgers."""
    run, root, config, scope = acquisition
    # Give the fixture a two-action prefix so every recorded time is physical
    # (nonnegative), and shift its existing local/tail/child records together.
    for path in root.glob("nodes/*/edges/*/**/execution.npz"):
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        arrays["sim_time"] += 0.1
        arrays["robot_qpos"] += 0.1
        np.savez(path, **arrays)
    for path in [*root.glob("nodes/*/context.npz"), *root.glob("nodes/*/edges/*/child_context.npz")]:
        with np.load(path, allow_pickle=False) as stored:
            state = stored["sim_state"].copy()
        # The independently assembled history fixture below uses decimal
        # timestamps; keep its saved state bytes identical at shared nodes.
        state[0] = round(float(state[0] + 0.1), 10)
        np.savez(path, sim_state=state)
        context_path = path.with_suffix(".json")
        context = json.loads(context_path.read_text())
        context["sim_time_s"] = float(state[0])
        hash_key = "state_sha256" if "state_sha256" in context else "sim_state_sha256"
        context[hash_key] = array_digest(state)
        context["source_action_prefix"] = 2
        put(context_path, context)
    for path in root.glob("nodes/*/edges/*/media.json"):
        media = json.loads(path.read_text())
        media["frame_sim_time_s"] = [value + 0.1 for value in media["frame_sim_time_s"]]
        put(path, media)
        with FFmpegRGBWriter(run / media["video"], width=8, height=8, fps=20) as writer:
            for index in range(len(media["frame_sim_time_s"])):
                writer.append_data(np.full((8, 8, 3), 40 + index * 40, dtype=np.uint8))
    history = run / "history" / "recorded_root"
    history.mkdir(parents=True)
    edge_a = root / "nodes/n_root/edges/a"
    edge_b = root / "nodes/n_root/edges/b"
    edge_c = root / "nodes/n_child/edges/c"
    states = [
        ("source_prefix", "source_prefix", 0., 0., False),
        ("source_prefix", "source_prefix", .05, .5, True),
        ("source_prefix", "source_prefix", .1, 1., True),
        (str(edge_a), "local_operation", .1, 1., False),
        (str(edge_a), "local_operation", .15, 2., True),
        (str(edge_a), "fixed_continuation", .2, 3., True),
        (str(edge_b), "local_operation", .1, 1., False),
        (str(edge_b), "local_operation", .15, 2., True),
        (str(edge_c), "local_operation", .15, 2., False),
        (str(edge_c), "local_operation", .2, 2., True),
    ]
    rows = []
    controls = 0
    for index, (segment, phase, sim_time, state_value, has_action) in enumerate(states):
        frame = f"frame_{index:07d}.npz"
        arrays = {"sim_state": np.array([sim_time, state_value]),
                  "sim_time_s": np.asarray(sim_time),
                  "rgb": np.full((3, 8, 8, 3), index * 20, dtype=np.uint8)}
        if has_action:
            arrays["action_from_previous"] = np.zeros(12)
            controls += 1
        np.savez_compressed(history / frame, **arrays)
        rows.append({"frame": frame, "segment": segment, "phase": phase,
                     "sim_time_s": sim_time, "has_action_from_previous": has_action,
                     "control_step_total": controls})
    (history / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    put(history / "index.json", {
        "schema": "cfact.control-history.v1", "width": 8, "height": 8,
        "camera_names": list(QuadReviewWriter.CAMERAS), "frames": "frames.jsonl",
        "frame_count": len(rows), "control_steps": controls,
        "history_origin": "actual_current_replay_and_branch_execution",
        "source_original_video_is_distinct": True,
    })

    def references(indices):
        return [{**rows[index], "directory": str(history)} for index in indices]

    for path, indices in (
        (root / "nodes/n_root/context.json", [0, 1, 2]),
        (root / "nodes/n_child/context.json", [1, 2, 4]),
        (edge_a / "child_context.json", [1, 2, 4]),
        (edge_b / "child_context.json", [1, 2, 7]),
        (edge_c / "child_context.json", [2, 4, 9]),
    ):
        context = json.loads(path.read_text())
        context["history_refs"] = references(indices)
        context["history_complete"] = True
        if path == root / "nodes/n_root/context.json":
            context["source_replay_history"] = str(history)
        put(path, context)
    original = run / "videos/original/OpenDrawer/req/official.mp4"
    with FFmpegRGBWriter(original, width=8, height=8, fps=20) as writer:
        writer.append_data(np.full((8, 8, 3), 255, dtype=np.uint8))
    put(original.parent / "index.json", {
        "source_videos": [{"camera": "source_original_camera", "sha256": sha256_file(original),
                           "video_path": str(original.relative_to(run))}],
    })
    return history, original


@pytest.mark.skipif(not ffmpeg_available() or shutil.which("ffprobe") is None,
                    reason="system ffmpeg and ffprobe required")
def test_publisher_preserves_history_and_shared_lineage_boundaries_on_resume(
    acquisition, tmp_path, monkeypatch
):
    from cfact.export.release import publish

    history, original = _add_saved_history(acquisition)
    run, _, _, _ = acquisition
    input_inventory = {path: sha256_file(path) for path in run.rglob("*") if path.is_file()}
    monkeypatch.setattr("cfact.export.release.write_segment", fake_episode_writer)
    output = tmp_path / "published"
    report = publish(run, output, verify=False)
    tree = json.loads((output / report["trees"][0]).read_text())
    prefix = tree["source_replay_prefix_media"]
    assert prefix["frame_sim_time_s"] == [0., .05, .1]
    assert prefix["control_steps"] == 2
    assert prefix["source_original_video_is_distinct"]
    assert not (output / prefix["video"]).is_symlink()
    copied_history = (output / tree["measured_history"]).parent
    for source in history.glob("*.npz"):
        copied = copied_history / source.name
        assert copied.read_bytes() == source.read_bytes()
        assert copied.stat().st_ino != source.stat().st_ino
        assert copied.stat().st_nlink == 1
    original_copy = next((output / "videos/source").glob("*.mp4"))
    assert original_copy.read_bytes() == original.read_bytes()
    assert original_copy.stat().st_ino != original.stat().st_ino
    assert original_copy.read_bytes() != (output / prefix["video"]).read_bytes()
    reference = next(edge for edge in tree["edges"] if edge["generator"] == "Reference")
    child = next(node for node in tree["nodes"] if node["node_id"] == reference["actual_child"])
    assert child["decision_node"] and child["parent_edge"] == reference["edge_id"]
    child_context = json.loads((output / child["context"]).read_text())
    assert [row["sim_time_s"] for row in child_context["history_refs"]] == [.05, .1, .15]
    for row in child_context["history_refs"]:
        assert row["directory"] == str(copied_history.relative_to(output))
        with np.load(output / row["directory"] / row["frame"], allow_pickle=False) as frame:
            assert float(frame["sim_time_s"]) == row["sim_time_s"]
            assert frame["rgb"].shape == (3, 8, 8, 3)
    lineages = [json.loads(line) for line in (output / report["lineage_paths"]).read_text().splitlines()]
    descendant = next(line for line in lineages if len(line["edge_ids"]) == 2)
    assert [part["segment"] for part in descendant["segments"]] == ["source_prefix", "local", "local"]
    assert all(line["segments"][0]["video"] == prefix["video"] for line in lineages)
    # Two prefix actions + one ancestor local + one child local = five
    # states, with each shared parent/actual-child boundary appearing once.
    rendered = render_lineage(output, descendant["lineage_id"], tmp_path / "complete_lineage.mp4")
    assert rendered["frame_count"] == 5
    assert [part["output_end_frame_exclusive"] - part["output_start_frame"]
            for part in rendered["segments"]] == [2, 1, 2]
    assert rendered["source_prefix_included"]
    assert all(part["segment"] != "fixed_continuation" for part in rendered["segments"])
    files_before = {path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}

    def unexpected_render(*args, **kwargs):
        pytest.fail("Completed publish resume must not encode replay prefix again")

    monkeypatch.setattr("cfact.export.lineage_video.render_replay_prefix", unexpected_render)
    assert publish(run, output, resume=True, verify=False) == report
    assert all(path.stat().st_mtime_ns == modified for path, modified in files_before.items())
    assert all(sha256_file(path) == digest for path, digest in input_inventory.items())
