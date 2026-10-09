"""Storage semantics with deliberately distinct local-child and fixed-terminal states."""

import json
from pathlib import Path
import multiprocessing
import numpy as np
import pytest
from PIL import Image
from cfact import ReleaseDataset
from cfact.io.serialization import array_digest, RunLock
from cfact.storage.layout import inside, stable_id
from cfact.export.release import publish
from cfact.storage.tree import publish_tree


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def trace(path, times):
    count = len(times) - 1
    arrays = {
        "sim_time": np.asarray(times, dtype=float),
        "commanded_actions": np.zeros((count, 12)),
        "raw_planned_actions": np.zeros((count, 12)),
        "controller_target_poses": np.repeat(np.eye(4)[None], count, 0),
        "local_event_flags": np.zeros(count, dtype=bool),
        "robot_qpos": np.repeat(np.array(times)[:, None], 7, 1),
        "robot_qvel": np.zeros((count + 1, 7)),
        "actual_eef_pose": np.repeat(np.eye(4)[None], count + 1, 0),
        "object_poses": np.repeat(np.eye(4)[None, None], count + 1, 0),
        "object_velocities": np.zeros((count + 1, 1, 3)),
        "fixture_poses": np.repeat(np.eye(4)[None], count + 1, 0),
        "relevant_fixture_qpos": np.zeros((count + 1, 1)),
        "relevant_fixture_qvel": np.zeros((count + 1, 1)),
        "object_to_gripper_transform": np.repeat(np.eye(4)[None], count + 1, 0),
        "task_success": np.zeros(count + 1, dtype=bool),
        "robot_qpos_joint_names": np.array([f"j{i}" for i in range(7)]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


@pytest.fixture
def acquisition(tmp_path):
    run = tmp_path / "raw"
    root = run / "data/formal/OpenDrawer/original_root"
    config = {
        "generator_revision": "test-protocol",
        "run_seed": 1,
        "execution": {"control_hz": 20},
        "video": {"fps": 20},
        "tasks": [{"task": "OpenDrawer"}],
    }
    route = {
        "source_id": "original-root",
        "original_episode_id": "demo_0",
        "prepared_group": "demo_0",
        "prepared_sha256": "a" * 64,
        "task": "OpenDrawer",
    }
    donor = {
        **route,
        "source_id": "donor",
        "original_episode_id": "demo_1",
        "prepared_group": "demo_1",
    }
    scope = {"requests_by_id": {"req": route}}
    put(run / "config_frozen.json", config)
    put(run / "source_split_frozen.json", scope)
    put(root / "root_manifest.json", {"source_episode": "req", "donor_source_routes": [donor]})
    summaries = []
    for node, state, parent in [
        ("n_root", [0.0, 1.0], None),
        ("n_child", [0.05, 2.0], "n_root/edges/a"),
    ]:
        path = root / "nodes" / node
        context = {
            "node_id": node,
            "sim_state_sha256": array_digest(np.array(state)),
            "sim_time_s": state[0],
            "source_action_prefix": 9,
        }
        if parent:
            context["parent_edge"] = str((root / "nodes" / parent).relative_to(run))
        put(path / "context.json", context)
        np.savez(path / "context.npz", sim_state=np.array(state))
    for node, name, method, start, fixed, label in [
        ("n_root", "a", "Reference", 0.0, True, "horizon_censored"),
        ("n_root", "b", "G1", 0.0, False, "no_supported_continuation"),
        ("n_child", "c", "G2", 0.05, False, "observed_operation_failure"),
    ]:
        edge = root / "nodes" / node / "edges" / name
        local_times = [start, start + 0.05]
        fixed_times = [start + 0.05, start + 0.1] if fixed else [start + 0.05]
        trace(edge / "execution.npz", local_times)
        trace(edge / "continuation/execution.npz", fixed_times)
        child = np.array([local_times[-1], 2.0])
        np.savez(edge / "child_context.npz", sim_state=child)
        put(
            edge / "child_context.json",
            {"state_sha256": array_digest(child), "sim_time_s": local_times[-1]},
        )
        put(
            edge / "option.json",
            {
                "source_method": method,
                "operation_type": "open_door",
                "candidate_id": name,
                "source_episode": "donor" if method == "G1" else "req",
            },
        )
        put(
            edge / "local_result.json",
            {"termination_reason": label, "outcome_labels": {label: True}},
        )
        put(edge / "continuation/result.json", {"status": label})
        times = local_times + fixed_times[1:]
        frames = run / "frames" / (node + name)
        frames.mkdir(parents=True)
        names = []
        for i, _ in enumerate(times):
            filename = f"{i}.jpg"
            Image.new("RGB", (8, 8), (i * 50, 20, 30)).save(frames / filename)
            names.append(filename)
        video = run / "videos" / (node + name + ".mp4")
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(b"unit fixture, not a physical video")
        put(
            edge / "media.json",
            {
                "fps": 20,
                "frame_sim_time_s": times,
                "frame_phase": ["local"] * len(times),
                "cameras": ["camera"],
                "frame_spool": str(frames),
                "clean_camera_frame_paths": {"camera": names},
                "video": str(video.relative_to(run)),
            },
        )
        summaries.append(
            {
                "candidate_id": name,
                "edge_path": str(edge.relative_to(run)),
                "status": "PHYSICS_COMPLETE",
            }
        )
    put(
        root / "root_result.json",
        {"task": "OpenDrawer", "status": "PHYSICS_COMPLETE", "edges": summaries},
    )
    return run, root, config, scope


def test_tree_actual_child_zero_terminal_labels_lineage(acquisition, tmp_path):
    run, root, config, scope = acquisition
    output = tmp_path / "release"
    result = publish_tree(root, run, output, config, scope)
    tree = result["tree"]
    assert len(tree["nodes"]) == 4
    assert len(result["terminals"]) == 6
    assert sum(t["zero_actions"] for t in result["terminals"]) == 2
    reference = next(e for e in tree["edges"] if e["generator"] == "Reference")
    child = next(n for n in tree["nodes"] if n["node_id"] == reference["actual_child"])
    with np.load(output / child["state"]) as state:
        assert state["sim_state"][0] == 0.05  # fixed end would be .1
    donor = next(e for e in tree["edges"] if e["generator"] == "G1")
    assert donor["donor_source_id"] != donor["root_source_id"]
    labels = [
        json.loads((output / e["result"]).read_text())["fixed_continuation"]["status"]
        for e in tree["edges"]
    ]
    assert set(labels) == {
        "horizon_censored",
        "no_supported_continuation",
        "observed_operation_failure",
    }
    lineage = next(row for row in result["lineages"] if len(row["edge_ids"]) == 2)
    assert [s["segment"] for s in lineage["segments"]] == ["local", "local"]
    for segment in result["segments"]:
        assert inside(output, segment["trace_path"]).is_file()
    before = tree["root_id"]
    config["execution"]["control_hz"] = 10
    other = publish_tree(root, run, tmp_path / "other", config, scope)
    assert other["tree"]["root_id"] != before


def test_publish_resume_and_bidirectional_mapping(acquisition, tmp_path, monkeypatch):
    run, _, _, _ = acquisition
    writes = []

    def writer(output, segment):
        writes.append(segment["segment_id"])
        return {
            **{k: v for k, v in segment.items() if k != "camera_frame_indices"},
            "frame_count": segment["action_count"],
            "shard_id": segment["segment_id"],
            "compatible_schema": "v3",
            "shard_path": "lerobot/v3/" + segment["segment_id"],
            "episode_index": 0,
            "repo_id": "local/test",
            "dataset_start_index": 0,
        }

    monkeypatch.setattr("cfact.export.release.write_segment", writer)
    output = tmp_path / "release"
    report = publish(run, output, verify=False)
    assert report["episodes"] == 4
    assert report["zero_action_terminals"] == 2
    dataset = ReleaseDataset(output)
    row = dataset.episodes()[0]
    assert dataset.resolve_episode(row["shard_id"], 0)["edge_id"] == row["edge_id"]
    for terminal in dataset.terminals():
        if terminal["zero_actions"]:
            assert terminal["episode_index"] is None and terminal["shard_id"] is None
        else:
            assert (
                dataset.resolve_episode(terminal["shard_id"], terminal["episode_index"])[
                    "segment_id"
                ]
                == terminal["segment_id"]
            )
    assert dataset.trace(row["edge_id"], row["segment"])["sim_time"].shape == (2,)
    count = len(writes)
    files = {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
    assert publish(run, output, resume=True, verify=False) == report
    assert len(writes) == count
    assert all(p.stat().st_mtime_ns == timestamp for p, timestamp in files.items())
    with pytest.raises(FileExistsError):
        publish(run, output, verify=False)
    root_result = next(run.glob("data/*/*/*/root_result.json"))
    root_result.write_text(root_result.read_text() + "\n")
    with pytest.raises(ValueError, match="Resume source"):
        publish(run, output, resume=True, verify=False)


def _hold_lock(path, ready, finish):
    lock = RunLock(Path(path))
    lock.acquire()
    ready.set()
    finish.wait(15)
    lock.release()


def test_concurrent_publishers_are_excluded(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready, finish = ctx.Event(), ctx.Event()
    path = tmp_path / ".publish.lock"
    process = ctx.Process(target=_hold_lock, args=(str(path), ready, finish))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(RuntimeError):
            RunLock(path).acquire()
    finally:
        finish.set()
        process.join(10)
    assert process.exitcode == 0


def test_release_paths_cannot_escape(tmp_path):
    for value in ["/tmp/file", "../file", "trees/../../file"]:
        with pytest.raises(ValueError):
            inside(tmp_path, value)
    assert stable_id("root", {"b": 2, "a": 1}) == stable_id("root", {"a": 1, "b": 2})


def test_pending_review_frames_export_and_resume_without_input_writes(tmp_path):
    import shutil
    from cfact.storage.media import review_video

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("system FFmpeg required")
    run, output = tmp_path / "raw", tmp_path / "release"
    spool = run / "review.frames"
    spool.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (16, 8), (index * 60, 20, 40)).save(spool / f"{index:06d}.jpg")
    media = {"fps": 20, "frame_sim_time_s": [0, 0.05, 0.1], "encoding_status": "PENDING"}
    before = {p.name: p.read_bytes() for p in spool.iterdir()}
    target = Path("videos/edges/e/review.mp4")
    result = review_video(run, output, media, spool, target)
    assert result["review_status"] == "ENCODED_RECORDED_REVIEW_FRAMES"
    assert int(result["review_ffprobe"]["nb_frames"]) == 3
    timestamp = (output / target).stat().st_mtime_ns
    assert review_video(run, output, media, spool, target) == result
    assert (output / target).stat().st_mtime_ns == timestamp
    assert before == {p.name: p.read_bytes() for p in spool.iterdir()}
    missing = review_video(run, output, media, run / "missing", Path("missing.mp4"))
    assert missing == {"review_video": None, "review_status": "REVIEW_MEDIA_MISSING"}


def test_completed_edge_cannot_substitute_fixed_terminal_for_missing_child(acquisition, tmp_path):
    run, root, config, scope = acquisition
    (root / "nodes/n_root/edges/a/child_context.npz").unlink()
    with pytest.raises(ValueError, match="no saved actual child"):
        publish_tree(root, run, tmp_path / "release", config, scope)


def fake_episode_writer(output, segment):
    return {
        **{k: v for k, v in segment.items() if k != "camera_frame_indices"},
        "frame_count": segment["action_count"],
        "compatible_schema": "v3-test",
        "shard_id": segment["segment_id"],
        "shard_path": "lerobot/test/" + segment["segment_id"],
        "episode_index": 0,
        "repo_id": "local/test",
        "dataset_start_index": 0,
    }


@pytest.mark.parametrize("change", ["input_camera", "output_state", "publisher", "verification"])
def test_resume_refuses_changed_evidence_or_publisher(acquisition, tmp_path, monkeypatch, change):
    run, _, _, _ = acquisition
    output = tmp_path / "release"
    monkeypatch.setattr("cfact.export.release.write_segment", fake_episode_writer)
    publish(run, output, verify=False)
    verification = False
    if change == "input_camera":
        frame = next((run / "frames").rglob("*.jpg"))
        Image.new("RGB", (8, 8), (255, 0, 0)).save(frame)
    elif change == "output_state":
        next((output / "trees").rglob("state.npz")).write_bytes(b"broken")
    elif change == "publisher":
        monkeypatch.setattr("cfact.export.release.implementation_digest", lambda: "changed")
    else:
        verification = True
    with pytest.raises(ValueError, match="Resume|integrity"):
        publish(run, output, resume=True, verify=verification)


def test_root_identity_includes_donor_population(acquisition, tmp_path):
    run, root, config, scope = acquisition
    first = publish_tree(root, run, tmp_path / "first", config, scope)
    manifest = json.loads((root / "root_manifest.json").read_text())
    manifest["donor_source_routes"][0]["original_episode_id"] = "demo_2"
    put(root / "root_manifest.json", manifest)
    second = publish_tree(root, run, tmp_path / "second", config, scope)
    assert first["tree"]["root_id"] != second["tree"]["root_id"]


def test_camera_ledger_mismatch_is_rejected(acquisition, tmp_path):
    run, root, config, scope = acquisition
    path = root / "nodes/n_root/edges/a/media.json"
    media = json.loads(path.read_text())
    media["clean_camera_frame_paths"]["camera"].pop()
    put(path, media)
    with pytest.raises(ValueError, match="frame counts differ"):
        publish_tree(root, run, tmp_path / "release", config, scope)


def test_shared_source_is_canonical_across_root_and_donor_routes(tmp_path):
    from cfact.storage.tree import source_record

    route = {
        "prepared_sha256": "a" * 64,
        "prepared_group": "demo_0",
        "source_id": "same",
        "source_identity": {"package_sha256": "b" * 64, "original_episode_id": "demo_0"},
    }
    first = source_record(route, "OpenDrawer", tmp_path / "release", tmp_path, None)
    route["prepared_sha256"] = "c" * 64
    route["prepared_group"] = "demo_99"
    second = source_record(route, "OpenDrawer", tmp_path / "release", tmp_path, None)
    assert first["source_id"] == second["source_id"]


def test_missing_local_shard_does_not_invoke_official_hub_fallback(tmp_path, monkeypatch):
    from cfact.export.shards import open_local_shard

    def unexpected_api():
        pytest.fail("Missing local metadata must not reach the official Hub fallback")

    monkeypatch.setattr("cfact.export.shards.official_api", unexpected_api)
    with pytest.raises(FileNotFoundError, match="Incomplete local"):
        open_local_shard(tmp_path, {"shard_path": "missing", "repo_id": "local/test"})


def test_g1_donor_must_resolve_to_recorded_source(acquisition, tmp_path):
    run, root, config, scope = acquisition
    path = root / "nodes/n_root/edges/b/option.json"
    option = json.loads(path.read_text())
    option["source_episode"] = "unknown-donor"
    put(path, option)
    with pytest.raises(ValueError, match="G1 edge donor"):
        publish_tree(root, run, tmp_path / "release", config, scope)
