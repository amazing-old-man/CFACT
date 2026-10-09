"""Official LeRobot 0.4.0 (dataset v3) segment writer and readback verification."""

from __future__ import annotations
from importlib.metadata import version
import json
import hashlib
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
from cfact.export.lerobot import _encode_h264_frames
from cfact.storage.layout import copy_file, stable_id, write_json

LEROBOT_VERSION = "0.4.0"


def official_api():
    if version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(
            f"Use the export environment with lerobot=={LEROBOT_VERSION}; "
            f"found {version('lerobot')}"
        )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def open_local_shard(output: Path, row: dict):
    """Fail on missing local files before the official loader can try Hub fallback."""
    from cfact.storage.layout import inside
    import pyarrow.parquet as pq

    root = inside(output, row["shard_path"])
    required = [root / "meta" / name for name in ("info.json", "tasks.parquet", "stats.json")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete local LeRobot shard: {path}")
    info = json.loads(required[0].read_text())
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"Missing local episode metadata: {root}")
    for path in episode_files:
        for episode in pq.read_table(path).to_pylist():
            expected = [
                info["data_path"].format(
                    chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
                )
            ]
            for key, feature in info["features"].items():
                if feature["dtype"] == "video":
                    expected.append(
                        info["video_path"].format(
                            video_key=key,
                            chunk_index=episode[f"videos/{key}/chunk_index"],
                            file_index=episode[f"videos/{key}/file_index"],
                        )
                    )
            for relative in expected:
                if not inside(root, relative).is_file():
                    raise FileNotFoundError(f"Missing local shard file: {relative}")
    return official_api()(row["repo_id"], root=root, video_backend="pyav", download_videos=False)


def features_for(trace: dict, media: dict, output: Path) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (trace["robot_qpos"].shape[1],),
            "names": trace["robot_qpos_joint_names"].tolist(),
        },
        "observation.robot_qvel": {
            "dtype": "float32",
            "shape": (trace["robot_qvel"].shape[1],),
            "names": None,
        },
        "observation.eef_pose": {"dtype": "float32", "shape": (16,), "names": None},
        "observation.sim_time_s": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {
            "dtype": "float32",
            "shape": (trace["commanded_actions"].shape[1],),
            "names": None,
        },
    }
    for camera in media["cameras"]:
        with Image.open(output / media["clean_camera_frame_paths"][camera][0]) as image:
            width, height = image.size
        features["observation.images." + camera] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def verify_shard(output: Path, row: dict, *, all_rows: bool = False) -> dict:
    reader = open_local_shard(output, row)
    with np.load(output / row["trace_path"], allow_pickle=False) as trace:
        count = len(trace["commanded_actions"])
        if count != row["frame_count"]:
            raise ValueError("Mapping frame count differs from raw measured trace")
        steps = range(count) if all_rows else sorted({0, count // 2, count - 1})
        checked = 0
        for step in steps:
            sample = reader[row["dataset_start_index"] + step]
            if int(sample["episode_index"].item()) != row["episode_index"]:
                raise ValueError("Official loader crossed an episode boundary")
            for name, values in [
                ("action", trace["commanded_actions"]),
                ("observation.state", trace["robot_qpos"]),
            ]:
                if not np.array_equal(
                    sample[name].numpy(), np.asarray(values[step], dtype=np.float32)
                ):
                    raise ValueError(f"Official loader {name} differs from measured trace")
            for name, feature in reader.meta.features.items():
                if feature["dtype"] == "video":
                    shape = tuple(feature["shape"])
                    expected = (shape[2], shape[0], shape[1])
                    if (
                        tuple(sample[name].shape) != expected
                        or not np.isfinite(sample[name].numpy()).all()
                    ):
                        raise ValueError(f"Official loader image check failed: {name}")
            checked += 1
    return {"segment_id": row["segment_id"], "decoded_rows": checked, "status": "PASS"}


def write_segment(output: Path, segment: dict) -> dict:
    """One uninterrupted segment per independently finalized shard; K=0 emits no rows."""
    if segment["action_count"] <= 0:
        raise ValueError("Zero-action segments have terminal mappings, not LeRobot episodes")
    LeRobotDataset = official_api()
    media = json.loads((output / segment["media"]).read_text())
    with np.load(output / segment["trace_path"], allow_pickle=False) as arrays:
        trace = {k: arrays[k] for k in arrays.files}
    features = features_for(trace, media, output)
    schema = stable_id(
        "v3", {"features": features, "fps": segment["fps"], "writer": LEROBOT_VERSION}
    )
    shard = stable_id("shard", segment["segment_id"])
    relative = str(Path("lerobot") / schema / shard)
    target = output / relative
    receipt = output / "mappings/shards" / (shard + ".json")
    row = {k: v for k, v in segment.items() if k not in ["camera_frame_indices", "action_count"]}
    row.update(
        {
            "compatible_schema": schema,
            "shard_id": shard,
            "shard_path": relative,
            "repo_id": "local/" + shard,
            "episode_index": 0,
            "dataset_start_index": 0,
            "frame_count": segment["action_count"],
            "lerobot_version": LEROBOT_VERSION,
        }
    )
    if receipt.exists():
        if json.loads(receipt.read_text()) != row:
            raise ValueError(f"Shard resume mapping differs: {receipt}")
        if not target.is_dir():
            raise FileNotFoundError(f"Committed shard is missing: {target}")
        return row
    if target.exists():
        # Crash after final rename but before receipt: validate and commit without encoding again.
        verify_shard(output, row)
        write_json(receipt, row)
        return row
    stage = target.with_name(target.name + ".partial")
    if stage.exists():
        # A finalized staging shard can be recovered; an unfinished one is retained for diagnosis.
        tentative = {**row, "shard_path": str(stage.relative_to(output))}
        try:
            verify_shard(output, tentative)
        except Exception:
            stage.rename(stage.with_name(stage.name + ".interrupted-" + uuid.uuid4().hex[:8]))
        else:
            stage.rename(target)
            write_json(receipt, row)
            return row
    target.parent.mkdir(parents=True, exist_ok=True)
    import lerobot.datasets.lerobot_dataset as api

    original_encoder = api.encode_video_frames
    api.encode_video_frames = _encode_h264_frames
    try:
        dataset = LeRobotDataset.create(
            repo_id=row["repo_id"],
            fps=int(segment["fps"]),
            features=features,
            root=stage,
            robot_type="RoboCasa365_SingleArm",
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=0,
            batch_encoding_size=1,
        )
        for index in range(segment["action_count"]):
            frame = {
                "task": segment["task"],
                "observation.state": np.asarray(trace["robot_qpos"][index], dtype=np.float32),
                "observation.robot_qvel": np.asarray(trace["robot_qvel"][index], dtype=np.float32),
                "observation.eef_pose": np.asarray(
                    trace["actual_eef_pose"][index], dtype=np.float32
                ).reshape(-1),
                "observation.sim_time_s": np.asarray([trace["sim_time"][index]], dtype=np.float32),
                "action": np.asarray(trace["commanded_actions"][index], dtype=np.float32),
            }
            camera_index = segment["camera_frame_indices"][index]
            for camera in media["cameras"]:
                with Image.open(
                    output / media["clean_camera_frame_paths"][camera][camera_index]
                ) as image:
                    frame["observation.images." + camera] = np.asarray(image.convert("RGB"))
            dataset.add_frame(frame)
        dataset.save_episode()
        dataset.finalize()
    finally:
        api.encode_video_frames = original_encoder
    verify_shard(output, {**row, "shard_path": str(stage.relative_to(output))})
    stage.rename(target)
    write_json(receipt, row)
    return row


def reuse_official_dataset(source: Path, output: Path, segments: list[dict]) -> list[dict]:
    """Reuse finalized official v3 media/rows only when all measured traces match exactly."""
    from cfact.sources.manifest import sha256_file

    official_api()
    readback = json.loads((source / "lerobot_readback.json").read_text())
    if readback["status"] != "PASS" or readback["dataset_version"] != "v3.0":
        raise ValueError("Only a finalized official v3 export with readback evidence can be reused")
    old_rows = [
        json.loads(line)
        for line in (source / "indices/episode_tree_map.jsonl").read_text().splitlines()
    ]
    lookup = {
        (stable_id("acquisition", row["edge_data_path"]), row["segment_kind"]): row
        for row in old_rows
    }
    active = [s for s in segments if s["action_count"]]
    expected = {(s["acquisition_edge_key"], s["segment"]) for s in active}
    if len(lookup) != len(old_rows) or len(expected) != len(active) or set(lookup) != expected:
        raise ValueError("Reused dataset episode inventory must exactly match this release")
    info = json.loads((source / "meta/info.json").read_text())
    schema = stable_id(
        "v3", {"features": info["features"], "fps": info["fps"], "writer": LEROBOT_VERSION}
    )
    shard = stable_id(
        "shard",
        {
            "info": sha256_file(source / "meta/info.json"),
            "traces": [s["trace_sha256"] for s in active],
        },
    )
    relative = str(Path("lerobot") / schema / shard)
    for folder in ["meta", "data", "videos"]:
        for path in sorted((source / folder).rglob("*")):
            if path.is_file():
                copy_file(path, output / relative / path.relative_to(source))
    rows = []
    for segment in active:
        old = lookup[(segment["acquisition_edge_key"], segment["segment"])]
        if (
            old["trace_sha256"] != segment["trace_sha256"]
            or old["frame_count"] != segment["action_count"]
        ):
            raise ValueError("Reused official episode does not match measured trace")
        media = json.loads((output / segment["media"]).read_text())
        records = []
        for index, sim_time in enumerate(media["frame_sim_time_s"]):
            records.append(
                {
                    "index": index,
                    "sim_time_s": float(sim_time),
                    "phase": media["frame_phase"][index],
                    "camera_sha256": {
                        camera: sha256_file(output / paths[index])
                        for camera, paths in media["clean_camera_frame_paths"].items()
                    },
                }
            )
        digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        if digest != old.get("camera_frame_hashes_sha256"):
            raise ValueError(
                "Reused official export camera provenance differs from measured frames"
            )
        row = {
            k: v for k, v in segment.items() if k not in ["camera_frame_indices", "action_count"]
        }
        row.update(
            {
                "compatible_schema": schema,
                "shard_id": shard,
                "shard_path": relative,
                "repo_id": readback["lerobot_repo_id"],
                "episode_index": old["episode_index"],
                "dataset_start_index": old["dataset_start_index"],
                "frame_count": old["frame_count"],
                "lerobot_version": LEROBOT_VERSION,
            }
        )
        rows.append(row)
    return rows
