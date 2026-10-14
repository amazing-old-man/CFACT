"""Task and tree datasets written through the pinned official LeRobot v3 API."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import uuid

import numpy as np
from PIL import Image

from cfact.export.lerobot import _encode_h264_frames
from cfact.export.shards import features_for, official_api, open_local_shard
from cfact.io.serialization import atomic_json, RunLock
from cfact.sources.manifest import digest_json, sha256_file


def write_dataset(output: Path, segments: list[dict], relative: Path) -> list[dict]:
    segments = [s for s in segments if s["action_count"]]
    if not segments:
        return []
    target = output / relative
    receipt = target.parent / (target.name + ".receipt.json")
    identity = digest_json({"segments": segments, "trace_hashes": {
        s["segment_id"]: sha256_file(output / s["trace_path"]) for s in segments}})
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["input_sha256"] != identity:
            raise ValueError("Dataset export input changed")
        verify_rows(output, saved["rows"])
        return saved["rows"]
    if target.exists():
        raise FileExistsError(f"Dataset without receipt requires inspection: {target}")
    stage = target.with_name(target.name + ".partial")
    if stage.exists():
        stage.rename(stage.with_name(stage.name + ".interrupted-" + uuid.uuid4().hex[:8]))
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    dataset = None
    reference_features = None
    frames = 0
    repo_id = "local/" + target.name
    import lerobot.datasets.lerobot_dataset as api

    original_encoder = api.encode_video_frames
    api.encode_video_frames = _encode_h264_frames
    try:
        for segment in segments:
            media = json.loads((output / segment["media"]).read_text())
            with np.load(output / segment["trace_path"], allow_pickle=False) as stored:
                trace = {key: stored[key] for key in stored.files}
            features = features_for(trace, media, output)
            signature = digest_json({"features": features, "fps": segment["fps"]})
            if dataset is None:
                reference_features = signature
                dataset = official_api().create(repo_id=repo_id, fps=int(segment["fps"]),
                    features=features, root=stage, robot_type="RoboCasa365_SingleArm",
                    use_videos=True, image_writer_processes=0, image_writer_threads=0,
                    batch_encoding_size=1)
            elif signature != reference_features:
                raise ValueError("Incompatible camera/robot schema within one task dataset")
            for index in range(segment["action_count"]):
                frame = {"task": segment["task"],
                    "observation.state": np.asarray(trace["robot_qpos"][index], dtype=np.float32),
                    "observation.robot_qvel": np.asarray(trace["robot_qvel"][index], dtype=np.float32),
                    "observation.eef_pose": np.asarray(trace["actual_eef_pose"][index], dtype=np.float32).reshape(-1),
                    "observation.sim_time_s": np.asarray([trace["sim_time"][index]], dtype=np.float32),
                    "action": np.asarray(trace["commanded_actions"][index], dtype=np.float32)}
                camera_index = segment["camera_frame_indices"][index]
                for camera in media["cameras"]:
                    with Image.open(output / media["clean_camera_frame_paths"][camera][camera_index]) as image:
                        frame["observation.images." + camera] = np.asarray(image.convert("RGB"))
                dataset.add_frame(frame)
            dataset.save_episode()
            row = {k: v for k, v in segment.items() if k not in ("action_count", "camera_frame_indices")}
            row.update({"shard_path": relative.as_posix(), "dataset_path": relative.as_posix(),
                        "shard_id": target.name, "repo_id": repo_id, "episode_index": len(rows),
                        "dataset_start_index": frames, "frame_count": segment["action_count"],
                        "compatible_schema": reference_features, "lerobot_version": "0.4.0"})
            rows.append(row)
            frames += segment["action_count"]
        dataset.finalize()
    finally:
        api.encode_video_frames = original_encoder
    verify_rows(output, [{**r, "shard_path": stage.relative_to(output).as_posix()} for r in rows])
    stage.rename(target)
    atomic_json(receipt, {"input_sha256": identity, "rows": rows})
    return rows


def verify_rows(output: Path, rows: list[dict]) -> dict:
    readers = {}
    checked = 0
    for row in rows:
        if row["shard_path"] not in readers:
            readers[row["shard_path"]] = open_local_shard(output, row)
        reader = readers[row["shard_path"]]
        with np.load(output / row["trace_path"], allow_pickle=False) as trace:
            count = len(trace["commanded_actions"])
            if count != row["frame_count"] or len(trace["sim_time"]) != count + 1:
                raise ValueError("Episode mapping does not preserve the measured transition count")
            for step in range(count):
                sample = reader[row["dataset_start_index"] + step]
                if int(sample["episode_index"].item()) != row["episode_index"]:
                    raise ValueError("Official episode offset mismatch")
                if not np.isclose(float(sample["timestamp"].item()), step / reader.fps, atol=1e-5):
                    raise ValueError("Official episode timestamp mismatch")
                for key, value in {
                    "action": trace["commanded_actions"][step],
                    "observation.state": trace["robot_qpos"][step],
                    "observation.robot_qvel": trace["robot_qvel"][step],
                    "observation.eef_pose": trace["actual_eef_pose"][step].reshape(-1),
                    "observation.sim_time_s": [trace["sim_time"][step]],
                }.items():
                    observed = sample[key].numpy()
                    if key == "observation.sim_time_s":
                        observed = observed.reshape(1)
                    if not np.array_equal(observed, np.asarray(value, dtype=np.float32)):
                        raise ValueError(f"Official reader differs from physical trace: {key}")
                for key, feature in reader.meta.features.items():
                    if feature["dtype"] == "video":
                        height, width, channels = feature["shape"]
                        if tuple(sample[key].shape) != (channels, height, width) or not np.isfinite(sample[key].numpy()).all():
                            raise ValueError("Official video decode failed")
                        camera=key.removeprefix("observation.images.")
                        media=json.loads((output/row["media"]).read_text())
                        with Image.open(output/media["clean_camera_frame_paths"][camera][step]) as saved:
                            native=np.asarray(saved.convert("RGB"),dtype=float)
                        decoded=np.moveaxis(sample[key].numpy(),0,-1)*255
                        if np.mean(np.abs(decoded-native))>8.:
                            raise ValueError("Official decoded RGB differs from mapped native frame beyond H264 tolerance")
                checked += 1
    for path, reader in readers.items():
        expected = sum(r["frame_count"] for r in rows if r["shard_path"] == path)
        if len(reader) != expected:
            raise ValueError("Official dataset frame count differs from mapping")
    return {"status": "PASS" if rows else "NO_EPISODES", "reader": "LeRobotDataset",
            "version": "0.4.0", "datasets": len(readers), "episodes": len(rows), "all_rows_checked": checked}


def export_run(run: Path, *, full_media: bool = True, by_tree: bool = False) -> dict:
    from cfact.storage.tree import publish_tree
    from cfact.storage.integrity import input_fingerprint, release_inventory, verify_release
    from cfact.export.release import write_parquet
    from cfact.export.lineage_video import render_lineage
    from cfact.collection.snapshot import implementation_digest

    native = run / "runtime/collection/_work/collection"
    with RunLock(run / "runtime/export.lock"):
        roots = sorted(p.parent for p in native.glob("data/*/*/*/root_result.json"))
        if not roots:
            raise ValueError("No measured root records; export cannot fabricate a tree")
        identity = input_fingerprint(native, roots)
        marker = run / "release.json"
        if marker.exists():
            prior = json.loads(marker.read_text())
            if prior["input_fingerprint"] != identity or prior["publisher_sha256"] != implementation_digest():
                raise ValueError("Run input or exporter changed; cannot resume this export")
            if prior["status"] == "COMPLETE":
                verify_release(run, prior)
                upgrade = (full_media and not prior["full_media"]) or (by_tree and not prior["by_tree_exports"])
                if not upgrade:
                    return prior
            full_media = full_media or prior["full_media"]
            by_tree = by_tree or prior["by_tree_exports"]
        config = json.loads((native / "config_frozen.json").read_text())
        scope = json.loads((native / "source_split_frozen.json").read_text())
        published = [publish_tree(root, native, run, config, scope) for root in roots]
        segments = [s for tree in published for s in tree["segments"]]
        rows = []
        for task in sorted({s["task"] for s in segments}):
            rows.extend(write_dataset(run, [s for s in segments if s["task"] == task], Path("lerobot/by_task") / task))
        lineages = [line for tree in published for line in tree["lineages"]]
        record = {"schema": "cfact.dual-view.v1", "status": "BUILDING",
                  "input_fingerprint": identity, "publisher_sha256": implementation_digest(),
                  "full_media": full_media, "by_tree_exports": by_tree,
                  "trees": [p["tree"]["tree_path"] for p in published],
                  "episode_tree_map": "mappings/episode_tree_map.parquet",
                  "terminal_observation_map": "mappings/terminal_observation_map.parquet",
                  "lineage_paths": "mappings/lineage_paths.jsonl",
                  "file_integrity": "mappings/file_integrity.json", "lerobot_version": "0.4.0"}
        write_parquet(run / record["episode_tree_map"], rows)
        by_segment = {r["segment_id"]: r for r in rows}
        terminals = [t for p in published for t in p["terminals"]]
        for terminal in terminals:
            row = by_segment.get(terminal["segment_id"])
            if row:
                terminal.update({k: row[k] for k in ("compatible_schema", "shard_id", "shard_path", "repo_id", "episode_index")})
            elif not terminal["zero_actions"]:
                raise ValueError("Nonempty terminal has no official episode mapping")
        write_parquet(run / record["terminal_observation_map"], terminals)
        atomic_json(marker, record)
        (run / record["lineage_paths"]).write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lineages))
        videos = []
        if full_media:
            for line in lineages:
                path = run / "media/lineages" / (line["lineage_id"] + ".mp4")
                receipt = path.with_suffix(".mp4.json")
                if path.exists() and receipt.exists():
                    info = json.loads(receipt.read_text())
                    if info["sha256"] != sha256_file(path):
                        raise ValueError("Lineage video changed")
                else:
                    info = render_lineage(run, line["lineage_id"], path)
                info["video"] = path.relative_to(run).as_posix()
                atomic_json(receipt, info)
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
                videos.append(info)
        tree_reports = []
        edge_map = {}
        for segment in segments:
            entry = edge_map.setdefault(segment["edge_id"], {"tree_id": segment["root_id"], "segments": [],
                "lineages": [line["lineage_id"] for line in lineages if segment["edge_id"] in line["edge_ids"]]})
            entry["segments"].append({"segment_id": segment["segment_id"], "kind": segment["segment"],
                                      "episode": by_segment.get(segment["segment_id"])})
        atomic_json(run / "mappings/edge_segment_episode_lineage.json", edge_map)
        for published_tree in published:
            tree = published_tree["tree"]
            tree_rows = [r for r in rows if r["root_id"] == tree["root_id"]]
            index = {"task": tree["task"], "tree_id": tree["root_id"], "tree": tree["tree_path"],
                     "source": tree["source_manifest"], "edges": tree["edges"],
                     "episodes": tree_rows, "is_independent_dataset": False,
                     "lineages": [x for x in lineages if x["root_id"] == tree["root_id"]]}
            atomic_json(run / "lerobot/by_tree" / tree["task"] / tree["root_id"] / "index.json", index)
            atomic_json((run / tree["tree_path"]).with_name("lineage.json"), index["lineages"])
            if by_tree:
                selected = [s for s in published_tree["segments"] if s["action_count"]]
                exported = write_dataset(run, selected, Path("lerobot/by_tree_exports") / tree["task"] / tree["root_id"])
                tree_reports.append(verify_rows(run, exported))
        official = verify_rows(run, rows)
        atomic_json(run / "reports/official_readback.json", {**official, "tree_datasets": tree_reports})
        atomic_json(run / "reports/full_media.json", {"status": "PASS" if full_media else "PLAYLIST_ONLY",
                    "lineages": len(lineages), "materialized_videos": len(videos),
                    "extra_disk_bytes": sum((run / "media/lineages" / (x["lineage_id"] + ".mp4")).stat().st_size for x in lineages) if full_media else 0})
        record.update({"status": "COMPLETE", "roots": len(published), "edges": sum(len(p["tree"]["edges"]) for p in published),
                       "episodes": len(rows), "frames": sum(r["frame_count"] for r in rows),
                       "official_readback": "reports/official_readback.json"})
        atomic_json(run / record["file_integrity"], release_inventory(run))
        atomic_json(marker, record)
        return record
