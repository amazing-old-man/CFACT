#!/usr/bin/env python3
"""Export measured CF20 edges to LeRobot Dataset v3 with clean true-camera video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
VIDEO_PREFIX = "observation.images."


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _episode_sources(run: Path) -> Iterator[tuple[dict[str, Any], Path, Path]]:
    seen: set[str] = set()
    for root_path in sorted((run / "data").glob("*/*/*/root_result.json")):
        root_result = json.loads(root_path.read_text())
        if root_result.get("split") != "formal":
            continue
        for edge in root_result.get("edges", []):
            if edge.get("status") != "PHYSICS_COMPLETE" or not edge.get("edge_path"):
                continue
            edge_rel = str(edge["edge_path"])
            if edge_rel in seen:
                raise ValueError(f"duplicate physical edge path: {edge_rel}")
            seen.add(edge_rel)
            edge_path = run / edge_rel
            local_path = edge_path / "execution.npz"
            fixed_path = edge_path / "continuation/execution.npz"
            media_path = edge_path / "media.json"
            if all(path.is_file() for path in (local_path, fixed_path, media_path)):
                yield root_result, edge_path, root_path


def _load_trace(edge_path: Path, segment: str) -> dict[str, np.ndarray]:
    """Load one uninterrupted segment; local and fixed traces stay separate episodes."""
    trace_path = edge_path / (
        "execution.npz" if segment == "local" else "continuation/execution.npz"
    )
    trace = _npz(trace_path)
    action_count = len(trace["commanded_actions"])
    state_keys = (
        "sim_time",
        "robot_qpos",
        "robot_qvel",
        "actual_eef_pose",
        "object_poses",
        "object_velocities",
        "fixture_poses",
        "relevant_fixture_qpos",
        "relevant_fixture_qvel",
        "object_to_gripper_transform",
        "task_success",
    )
    for key in state_keys:
        if key not in trace:
            raise KeyError(f"{trace_path}: missing measured state field {key}")
        if len(trace[key]) != action_count + 1:
            raise ValueError(f"{trace_path}: {key} is not K+1 for K={action_count}")
    for key in ("commanded_actions", "raw_planned_actions", "controller_target_poses"):
        if key not in trace or len(trace[key]) != action_count:
            raise ValueError(f"{trace_path}: {key} is not K={action_count}")
    if "local_event_flags" not in trace or len(trace["local_event_flags"]) != action_count:
        raise ValueError(f"{trace_path}: transition event flags are not K={action_count}")
    if not np.isfinite(trace["sim_time"]).all() or np.any(np.diff(trace["sim_time"]) < -1e-9):
        raise ValueError(f"{trace_path}: simulation times are nonfinite or run backwards")
    return trace


def _media_images(
    edge_path: Path, action_times: np.ndarray, *, include_hashes: bool = False
) -> tuple[dict[float, dict[str, Path]], list[dict[str, Any]]]:
    media = json.loads((edge_path / "media.json").read_text())
    if media.get("encoding_status") != "COMPLETE":
        raise ValueError(f"{edge_path}: review video encoding is not complete")
    frame_times = np.asarray(media["frame_sim_time_s"], dtype=np.float64)
    paths = media.get("clean_camera_frame_paths")
    spool_value = media.get("frame_spool")
    if not isinstance(paths, dict) or not spool_value:
        raise ValueError(f"{edge_path}: clean camera frame spool is missing")
    spool = Path(spool_value)
    by_time: dict[float, dict[str, Path]] = {}
    frame_records: list[dict[str, Any]] = []
    phases = media.get("frame_phase", [])
    if len(phases) != len(frame_times):
        raise ValueError(f"{edge_path}: camera time and phase ledger lengths differ")
    for index, sim_time in enumerate(frame_times):
        key = round(float(sim_time), 8)
        row = by_time.setdefault(key, {})
        camera_paths: dict[str, Path] = {}
        for camera in CAMERAS:
            values = paths.get(camera, [])
            if index >= len(values):
                raise ValueError(f"{edge_path}: {camera} frame list is short")
            frame_path = spool / values[index]
            if not frame_path.is_file():
                raise FileNotFoundError(frame_path)
            camera_paths[camera] = frame_path
            row.setdefault(camera, frame_path)
        if include_hashes:
            frame_records.append(
                {
                    "index": index,
                    "sim_time_s": float(sim_time),
                    "phase": phases[index],
                    "camera_sha256": {camera: _sha(camera_paths[camera]) for camera in CAMERAS},
                }
            )
    for sim_time in action_times:
        key = round(float(sim_time), 8)
        if key not in by_time:
            nearest = min(by_time, key=lambda t: abs(t - key)) if by_time else None
            if nearest is None or abs(nearest - key) > 1e-5:
                raise ValueError(
                    f"{edge_path}: no clean camera frame for simulation time {sim_time}"
                )
            key = nearest
        missing = [camera for camera in CAMERAS if camera not in by_time[key]]
        if missing:
            raise ValueError(f"{edge_path}: missing clean cameras {missing} at {sim_time}")
    return by_time, frame_records


def _pose(arr: np.ndarray) -> np.ndarray:
    value = np.asarray(arr, dtype=np.float32)
    if value.shape == (1, 4, 4):
        value = value[0]
    if value.shape != (4, 4):
        raise ValueError(f"expected one 4x4 pose, got {value.shape}")
    return value


def _fixture_vector(row: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(row, dtype=np.float32).reshape(-1)
    if len(value) > width:
        raise ValueError(f"fixture vector length {len(value)} exceeds schema width {width}")
    padded = np.zeros((width,), dtype=np.float32)
    mask = np.zeros((width,), dtype=np.bool_)
    padded[: len(value)] = value
    mask[: len(value)] = True
    return padded, mask


def _edge_sources(run: Path) -> Iterator[tuple[dict[str, Any], Path, Path]]:
    return _episode_sources(run)


def _encode_h264_frames(
    imgs_dir: Path | str, video_path: Path | str, fps: int, overwrite: bool = False
) -> None:
    """Encode without B-frame reordering so one-frame episodes can concatenate."""
    imgs_dir, video_path = Path(imgs_dir), Path(video_path)
    if video_path.exists() and not overwrite:
        return
    frames = sorted(imgs_dir.glob("frame-*.png"))
    if not frames or any(p.name != f"frame-{i:06d}.png" for i, p in enumerate(frames)):
        raise ValueError(f"Expected contiguous LeRobot PNG frames starting at zero: {imgs_dir}")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            str(imgs_dir / "frame-%06d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "10",
            "-bf",
            "0",
            "-crf",
            "24",
            "-tune",
            "fastdecode",
            "-threads",
            "1",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )


def _matrix_safe_episode_stats(
    episode_data: dict[str, Any], features: dict[str, Any], quantile_list=None, *, compute
) -> dict[str, Any]:
    """Keep each matrix entry a feature, reducing only over episode time.

    LeRobot's vector statistic accumulator collapses all leading dimensions for
    multi-frame matrices, whereas its single-frame path retains matrix shape.
    Flatten feature axes for statistics only, then restore the declared shape.
    Recorded rows, frame counts and image statistics remain unchanged.
    """
    data = dict(episode_data)
    specs = dict(features)
    shapes = {}
    for key, value in episode_data.items():
        if features[key]["dtype"] in {"image", "video", "string"}:
            continue
        array = np.asarray(value)
        if array.ndim > 2:
            shape = tuple(features[key]["shape"])
            if tuple(array.shape[1:]) != shape:
                raise ValueError(f"{key}: recorded shape {array.shape[1:]} != {shape}")
            shapes[key] = shape
            data[key] = array.reshape(len(array), -1)
            specs[key] = {**features[key], "shape": (int(np.prod(shape)),), "names": None}
    result = compute(data, specs, quantile_list=quantile_list)
    for key, shape in shapes.items():
        result[key] = {
            name: value if name == "count" else np.asarray(value).reshape(shape)
            for name, value in result[key].items()
        }
    return result


def export(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run.resolve()
    output = args.output.resolve()
    config = json.loads((run / "config_frozen.json").read_text())
    fps = int(
        (config.get("dataset_export") or {}).get("fps", config.get("video", {}).get("fps", 20))
    )
    release_root = Path(config.get("release_root", run)).resolve()
    if not output.resolve().is_relative_to(run):
        raise ValueError("LeRobot output must be inside the isolated run directory")

    def storage_admission() -> dict[str, Any]:
        usage = shutil.disk_usage(release_root)
        baseline_path = run / "runtime" / "storage_baseline.json"
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text())
            used = int(baseline.get("release_bytes", 0)) + max(
                0, int(baseline["filesystem_free_bytes"]) - int(usage.free)
            )
        else:
            used = sum(path.stat().st_size for path in release_root.rglob("*") if path.is_file())
        maximum = int(config["resource"]["maximum_output_gb"] * 1_000_000_000)
        minimum_free = int(config["resource"]["minimum_host_free_gib"] * 1024**3)
        return {
            "status": "PASS"
            if used <= maximum and usage.free >= minimum_free
            else "STORAGE_GUARD_STOP",
            "release_bytes": used,
            "maximum_release_bytes": maximum,
            "filesystem_free_bytes": usage.free,
            "minimum_filesystem_free_bytes": minimum_free,
        }

    if output.exists():
        raise FileExistsError(f"Export output already exists; choose a new directory: {output}")
    initial_storage = storage_admission()
    if initial_storage["status"] != "PASS":
        raise RuntimeError(f"STORAGE_GUARD_STOP before LeRobot export: {initial_storage}")
    package_dir = args.python_packages.resolve() if args.python_packages else None
    if package_dir and package_dir.is_dir():
        sys.path.insert(0, str(package_dir))
    cache_root = release_root / "tools" / "huggingface_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    # Keep all third-party dataset caches inside the accounted release volume;
    # the host home quota is not a data staging area.
    for key, value in {
        "HF_HOME": cache_root,
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "HF_HUB_CACHE": cache_root / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hub",
        "HF_ASSETS_CACHE": cache_root / "assets",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
    }.items():
        os.environ[key] = str(value)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_api
    from functools import partial
    from lerobot.datasets.compute_stats import compute_episode_stats

    lerobot_dataset_api.compute_episode_stats = partial(
        _matrix_safe_episode_stats, compute=compute_episode_stats
    )
    from lerobot.datasets.video_utils import encode_video_frames as native_encode_video_frames
    import av

    try:
        av.codec.Codec("h264", "w")
        selected_vcodec = "h264"
    except Exception:
        selected_vcodec = "libsvtav1"

    def encode_h264_batch(
        imgs_dir: Path | str, video_path: Path | str, fps: int, overwrite: bool = False
    ) -> None:
        if selected_vcodec == "h264":
            _encode_h264_frames(imgs_dir, video_path, fps, overwrite)
        else:
            native_encode_video_frames(
                imgs_dir,
                video_path,
                fps,
                vcodec=selected_vcodec,
                pix_fmt="yuv420p",
                g=10,
                crf=24,
                fast_decode=1,
                overwrite=overwrite,
            )

    # LeRobot 0.4 exposes batch encoding but not RGBEncoderConfig yet. Keep
    # the official dataset writer; H264 disables B frames for short-episode concat.
    lerobot_dataset_api.encode_video_frames = encode_h264_batch

    source_scope = json.loads((run / "source_split_frozen.json").read_text())
    route_by_id = source_scope["requests_by_id"]
    robot_widths: set[int] = set()
    robot_velocity_widths: set[int] = set()
    action_widths: set[int] = set()
    robot_qpos_schema: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None = None
    robot_qvel_units: tuple[str, ...] | None = None
    max_joint_width = 1
    image_size: tuple[int, int] | None = None
    source_original_hashes: dict[str, str] = {}

    # Pass one is validation and schema discovery only. Full trajectories and
    # camera frames are released after each edge; the export is not held in RAM.
    for root_result, edge_path, root_path in _edge_sources(run):
        root_manifest_path = root_path.parent / "root_manifest.json"
        root_manifest = json.loads(root_manifest_path.read_text())
        request_id = str(root_manifest["source_episode"])
        route = route_by_id.get(request_id)
        if route is None or route["task"] != root_result["task"]:
            raise ValueError(f"formal edge has no matching source route: {request_id}")
        local = _load_trace(edge_path, "local")
        fixed = _load_trace(edge_path, "fixed_continuation")
        active = [("local", local), ("fixed_continuation", fixed)]
        active = [(name, trace) for name, trace in active if len(trace["commanded_actions"]) > 0]
        if not active:
            continue
        if not np.isclose(local["sim_time"][-1], fixed["sim_time"][0], atol=1e-8, rtol=0):
            raise ValueError(
                f"{edge_path}: fixed continuation does not start at measured local child"
            )
        action_times = np.concatenate([trace["sim_time"][:-1] for _, trace in active])
        image_map, _ = _media_images(edge_path, action_times)
        first_trace = active[0][1]
        first_time = round(float(first_trace["sim_time"][0]), 8)
        with Image.open(image_map[first_time][CAMERAS[0]]) as image:
            current_size = image.size
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(f"camera resolution changed from {image_size} to {current_size}")
        for _, trace in active:
            robot_widths.add(int(trace["robot_qpos"].shape[-1]))
            robot_velocity_widths.add(int(trace["robot_qvel"].shape[-1]))
            action_widths.add(int(trace["commanded_actions"].shape[-1]))
            names = tuple(str(value) for value in trace["robot_qpos_joint_names"].tolist())
            types = tuple(str(value) for value in trace["robot_qpos_joint_types"].tolist())
            qpos_units = tuple(str(value) for value in trace["robot_qpos_units"].tolist())
            qvel_units = tuple(str(value) for value in trace["robot_qvel_units"].tolist())
            if (
                len(names) != trace["robot_qpos"].shape[-1]
                or len(names) != trace["robot_qvel"].shape[-1]
                or len(types) != len(names)
                or len(qpos_units) != len(names)
                or len(qvel_units) != len(names)
            ):
                raise ValueError(
                    f"{edge_path}: named robot-joint schema does not match qpos/qvel dimensions"
                )
            schema = (names, types, qpos_units)
            if robot_qpos_schema is None:
                robot_qpos_schema, robot_qvel_units = schema, qvel_units
            elif robot_qpos_schema != schema or robot_qvel_units != qvel_units:
                raise ValueError(
                    f"{edge_path}: robot-joint names or units vary across formal traces"
                )
            max_joint_width = max(
                max_joint_width,
                int(trace["relevant_fixture_qpos"].shape[-1]),
                int(trace["relevant_fixture_qvel"].shape[-1]),
            )
    if image_size is None:
        raise RuntimeError("no nonempty measured formal segments are available for LeRobot export")
    if len(robot_widths) != 1:
        raise ValueError(f"robot qpos widths differ across tasks: {sorted(robot_widths)}")
    if len(robot_velocity_widths) != 1:
        raise ValueError(f"robot qvel widths differ across tasks: {sorted(robot_velocity_widths)}")
    if len(action_widths) != 1:
        raise ValueError(f"commanded action widths differ across tasks: {sorted(action_widths)}")
    robot_width = next(iter(robot_widths))
    robot_velocity_width = next(iter(robot_velocity_widths))
    action_width = next(iter(action_widths))
    if action_width != 12:
        raise ValueError(
            f"formal action schema requires 12 documented RoboCasa action dimensions, got {action_width}"
        )
    if robot_qpos_schema is None or robot_qvel_units is None:
        raise RuntimeError("robot joint names and units are missing from physical traces")
    robot_joint_names, robot_joint_types, robot_qpos_units = robot_qpos_schema
    height, width = image_size[1], image_size[0]

    # Outcomes and event labels stay separate from execution-time observations.
    # The local event is a transition label (K values), not a K+1 state feature.
    action_names = [
        "eef_delta_x_normalized_base_frame",
        "eef_delta_y_normalized_base_frame",
        "eef_delta_z_normalized_base_frame",
        "eef_axis_angle_x_normalized_base_frame",
        "eef_axis_angle_y_normalized_base_frame",
        "eef_axis_angle_z_normalized_base_frame",
        "gripper_command",
        "base_command_0",
        "base_command_1",
        "base_command_2",
        "base_command_3",
        "controller_mode",
    ]
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (robot_width,),
            "names": list(robot_joint_names),
        },
        "observation.robot_qvel": {
            "dtype": "float32",
            "shape": (robot_velocity_width,),
            "names": list(robot_joint_names),
        },
        "observation.eef_pose": {"dtype": "float32", "shape": (4, 4), "names": None},
        "observation.object_pose": {"dtype": "float32", "shape": (4, 4), "names": None},
        "observation.object_velocity": {"dtype": "float32", "shape": (3,), "names": None},
        "observation.fixture_pose": {"dtype": "float32", "shape": (4, 4), "names": None},
        "observation.fixture_qpos": {
            "dtype": "float32",
            "shape": (max_joint_width,),
            "names": [f"fixture_joint_{i}" for i in range(max_joint_width)],
        },
        "observation.fixture_qvel": {
            "dtype": "float32",
            "shape": (max_joint_width,),
            "names": [f"fixture_joint_{i}" for i in range(max_joint_width)],
        },
        "observation.fixture_joint_mask": {
            "dtype": "bool",
            "shape": (max_joint_width,),
            "names": None,
        },
        "observation.object_to_gripper_transform": {
            "dtype": "float32",
            "shape": (4, 4),
            "names": None,
        },
        "observation.sim_time_s": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (action_width,), "names": action_names},
        "action.raw_planned": {"dtype": "float32", "shape": (action_width,), "names": action_names},
        "action.controller_target_pose": {"dtype": "float32", "shape": (4, 4), "names": None},
        "label.local_event_reached": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for camera in CAMERAS:
        features[f"{VIDEO_PREFIX}{camera}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=output,
        robot_type="RoboCasa365_SingleArm",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )
    episode_index: list[dict[str, Any]] = []
    terminal_index: list[dict[str, Any]] = []
    sample_expectations: list[dict[str, Any]] = []
    dataset_start = 0
    failures: list[dict[str, Any]] = []
    for root_result, edge_path, root_path in _edge_sources(run):
        root_manifest = json.loads((root_path.parent / "root_manifest.json").read_text())
        request_id = str(root_manifest["source_episode"])
        route = route_by_id[request_id]
        task = str(root_result["task"])
        local = _load_trace(edge_path, "local")
        fixed = _load_trace(edge_path, "fixed_continuation")
        active = [("local", local), ("fixed_continuation", fixed)]
        active = [(name, trace) for name, trace in active if len(trace["commanded_actions"]) > 0]
        if not active:
            continue
        action_times = np.concatenate([trace["sim_time"][:-1] for _, trace in active])
        image_map, frame_records = _media_images(edge_path, action_times, include_hashes=True)
        tail_result_path = edge_path / "continuation/result.json"
        tail_status = (
            json.loads(tail_result_path.read_text()).get("status", "unknown")
            if tail_result_path.is_file()
            else "unknown"
        )
        edge_result_path = edge_path / "local_result.json"
        edge_result = json.loads(edge_result_path.read_text()) if edge_result_path.is_file() else {}
        root_key = str(root_result["root_id"])
        node_id = edge_path.parent.parent.name
        candidate_id = edge_path.name
        option_path = edge_path / "option.json"
        option = json.loads(option_path.read_text()) if option_path.is_file() else {}
        source_method = option.get("source_method", "unknown")
        media = json.loads((edge_path / "media.json").read_text())
        video_rel = media["video"]
        original_index_path = run / "videos/original" / task / request_id / "index.json"
        original = (
            json.loads(original_index_path.read_text()) if original_index_path.is_file() else {}
        )

        for segment, trace in active:
            action = trace["commanded_actions"]
            segment_id = len(episode_index)
            episode_start = dataset_start
            matching_frames = [
                i
                for i, t in enumerate(media["frame_sim_time_s"])
                if round(float(t), 8) == round(float(trace["sim_time"][-1]), 8)
            ]
            if not matching_frames:
                nearest = min(
                    range(len(media["frame_sim_time_s"])),
                    key=lambda i: abs(
                        float(media["frame_sim_time_s"][i]) - float(trace["sim_time"][-1])
                    ),
                )
                if (
                    abs(float(media["frame_sim_time_s"][nearest]) - float(trace["sim_time"][-1]))
                    > 1e-5
                ):
                    raise ValueError(f"{edge_path}: no exact {segment} terminal camera timestamp")
                matching_frames = [nearest]
            terminal_video_frame = matching_frames[-1]
            try:
                for step in range(len(action)):
                    current_time = float(trace["sim_time"][step])
                    current_key = round(current_time, 8)
                    if current_key not in image_map:
                        current_key = min(image_map, key=lambda t: abs(t - current_time))
                    image_values = {}
                    for camera in CAMERAS:
                        with Image.open(image_map[current_key][camera]) as image:
                            image_values[f"{VIDEO_PREFIX}{camera}"] = np.asarray(
                                image.convert("RGB"), dtype=np.uint8
                            )
                    current_fixture_qpos, joint_mask = _fixture_vector(
                        trace["relevant_fixture_qpos"][step], max_joint_width
                    )
                    current_fixture_qvel, _ = _fixture_vector(
                        trace["relevant_fixture_qvel"][step], max_joint_width
                    )
                    frame = {
                        "task": task,
                        **image_values,
                        "observation.state": np.asarray(
                            trace["robot_qpos"][step], dtype=np.float32
                        ),
                        "observation.robot_qvel": np.asarray(
                            trace["robot_qvel"][step], dtype=np.float32
                        ),
                        "observation.eef_pose": np.asarray(
                            trace["actual_eef_pose"][step], dtype=np.float32
                        ),
                        "observation.object_pose": _pose(trace["object_poses"][step]),
                        "observation.object_velocity": np.asarray(
                            trace["object_velocities"][step, 0], dtype=np.float32
                        ),
                        "observation.fixture_pose": _pose(trace["fixture_poses"][step]),
                        "observation.fixture_qpos": current_fixture_qpos,
                        "observation.fixture_qvel": current_fixture_qvel,
                        "observation.fixture_joint_mask": joint_mask,
                        "observation.object_to_gripper_transform": np.asarray(
                            trace["object_to_gripper_transform"][step], dtype=np.float32
                        ),
                        "observation.sim_time_s": np.asarray([current_time], dtype=np.float32),
                        "action": np.asarray(action[step], dtype=np.float32),
                        "action.raw_planned": np.asarray(
                            trace["raw_planned_actions"][step], dtype=np.float32
                        ),
                        "action.controller_target_pose": np.asarray(
                            trace["controller_target_poses"][step], dtype=np.float32
                        ),
                        "label.local_event_reached": np.asarray(
                            [trace["local_event_flags"][step]], dtype=np.bool_
                        ),
                    }
                    ds.add_frame(frame)
                ds.save_episode()
                dataset_start += len(action)
                live_storage = storage_admission()
                if live_storage["status"] != "PASS":
                    raise RuntimeError(f"STORAGE_GUARD_STOP during LeRobot export: {live_storage}")
                for source_video in original.get("source_videos", []):
                    video_path = source_video["video_path"]
                    expected_hash = source_video["sha256"]
                    prior_hash = source_original_hashes.get(video_path)
                    if prior_hash is not None and prior_hash != expected_hash:
                        raise ValueError(f"conflicting source video hashes for {video_path}")
                    source_original_hashes[video_path] = expected_hash
                episode_row = {
                    "dataset_id": args.repo_id,
                    "shard_id": "shard-000",
                    "episode_index": segment_id,
                    "dataset_start_index": episode_start,
                    "frame_count": len(action),
                    "fps": fps,
                    "task": task,
                    "request_id": request_id,
                    "source_id": route["source_id"],
                    "source_class": route["source_class"],
                    "source_video_status": route.get("source_video_status"),
                    "prepared_source_group": route.get("prepared_group"),
                    "prepared_source_sha256": route.get("prepared_sha256"),
                    "role_in_release": route.get("role_in_release", "FORMAL_REQUEST"),
                    "fixture_joint_names": [
                        str(value) for value in trace["fixture_joint_names"].tolist()
                    ],
                    "fixture_joint_types": [
                        str(value) for value in trace["fixture_joint_types"].tolist()
                    ],
                    "fixture_joint_qpos_units": [
                        str(value) for value in trace["fixture_joint_units"].tolist()
                    ],
                    "fixture_joint_qvel_units": [
                        str(value) for value in trace["fixture_joint_qvel_units"].tolist()
                    ],
                    "robot_joint_names": list(robot_joint_names),
                    "robot_joint_types": [
                        str(value) for value in trace["robot_qpos_joint_types"].tolist()
                    ],
                    "robot_qpos_units": list(robot_qpos_units),
                    "robot_qvel_units": list(robot_qvel_units),
                    "root_id": root_key,
                    "node_id": node_id,
                    "edge_id": candidate_id,
                    "rollout_id": f"{candidate_id}:{segment}",
                    "segment_kind": segment,
                    "source_method": source_method,
                    "donor_source": option.get("source_episode"),
                    "local_action_count": len(local["commanded_actions"]),
                    "fixed_action_count": len(fixed["commanded_actions"]),
                    "local_event_termination": edge_result.get("termination_reason"),
                    "fixed_continuation_status": tail_status,
                    "official_success_at_segment_terminal": bool(trace["task_success"][-1]),
                    "physical_gpu_index": root_manifest.get("physical_gpu_index"),
                    "physical_gpu_uuid": root_manifest.get("physical_gpu_uuid"),
                    "edge_data_path": str(edge_path.relative_to(run)),
                    "trace_path": str(
                        (
                            edge_path
                            / (
                                "execution.npz"
                                if segment == "local"
                                else "continuation/execution.npz"
                            )
                        ).relative_to(run)
                    ),
                    "trace_terminal_state_index": len(action),
                    "trace_sha256": _sha(
                        edge_path
                        / ("execution.npz" if segment == "local" else "continuation/execution.npz")
                    ),
                    "source_video_index": str(original_index_path.relative_to(run))
                    if original_index_path.is_file()
                    else None,
                    "camera_observation_provenance": media.get(
                        "camera_observation_provenance", "native_clean_camera_frames"
                    ),
                    "camera_render_phase_count": len(frame_records),
                    "camera_frame_hashes_sha256": hashlib.sha256(
                        json.dumps(frame_records, sort_keys=True).encode()
                    ).hexdigest(),
                }
                episode_index.append(episode_row)
                terminal_index.append(
                    {
                        "dataset_id": args.repo_id,
                        "shard_id": "shard-000",
                        "episode_index": segment_id,
                        "request_id": request_id,
                        "root_id": root_key,
                        "node_id": node_id,
                        "edge_id": candidate_id,
                        "segment_kind": segment,
                        "terminal_state_index": len(action),
                        "terminal_state_sim_time_s": float(trace["sim_time"][-1]),
                        "terminal_state_success": bool(trace["task_success"][-1]),
                        "trace_path": episode_row["trace_path"],
                        "trace_sha256": episode_row["trace_sha256"],
                        "review_video_path": video_rel,
                        "review_video_frame_index": terminal_video_frame,
                        "review_video_sim_time_s": float(
                            media["frame_sim_time_s"][terminal_video_frame]
                        ),
                        "camera_views": CAMERAS,
                    }
                )
                sample_steps = sorted({0, len(action) // 2, len(action) - 1})
                for sample_step in sample_steps:
                    sample_expectations.append(
                        {
                            "dataset_index": episode_start + sample_step,
                            "episode_index": segment_id,
                            "expected_state": np.asarray(
                                trace["robot_qpos"][sample_step], dtype=np.float32
                            ),
                            "expected_action": np.asarray(action[sample_step], dtype=np.float32),
                        }
                    )
            except Exception as error:
                has_pending = (
                    ds.has_pending_frames()
                    if hasattr(ds, "has_pending_frames")
                    else bool(getattr(ds, "episode_buffer", {}).get("size", 0))
                )
                if has_pending:
                    ds.clear_episode_buffer(delete_images=True)
                failures.append(
                    {
                        "episode_index": segment_id,
                        "edge_path": str(edge_path.relative_to(run)),
                        "segment_kind": segment,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                raise
    ds.finalize()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset as LeRobotReadback

    reader = LeRobotReadback(args.repo_id, root=output, video_backend="pyav")
    if int(reader.meta.total_episodes) != len(episode_index):
        raise RuntimeError(
            f"LeRobot loader episode count {reader.meta.total_episodes} != {len(episode_index)}"
        )
    if len(reader) != dataset_start or int(reader.meta.total_frames) != dataset_start:
        raise RuntimeError(
            f"LeRobot loader frame count mismatch: loader={len(reader)} expected={dataset_start}"
        )
    readback_rows = []
    for expectation in sample_expectations:
        index = expectation["dataset_index"]
        sample = reader[index]
        if int(sample["episode_index"].item()) != int(expectation["episode_index"]):
            raise RuntimeError(f"episode boundary mismatch at dataset frame {index}")
        action_value = sample["action"].detach().cpu().numpy().reshape(-1)
        state_value = sample["observation.state"].detach().cpu().numpy().reshape(-1)
        if not np.array_equal(action_value, expectation["expected_action"]):
            raise RuntimeError(
                f"action readback differs from the measured command at frame {index}"
            )
        if not np.array_equal(state_value, expectation["expected_state"]):
            raise RuntimeError(f"state readback differs from the measured qpos at frame {index}")
        for camera in CAMERAS:
            key = f"{VIDEO_PREFIX}{camera}"
            value = sample[key]
            if tuple(value.shape) != (3, height, width):
                raise RuntimeError(
                    f"LeRobot decoded shape mismatch for {key}: {tuple(value.shape)}"
                )
        readback_rows.append(
            {
                "dataset_index": index,
                "episode_index": expectation["episode_index"],
                "decoded_all_three_cameras": True,
                "action_and_state_match_measured_arrays": True,
            }
        )
    if len(source_original_hashes) != len(source_original_hashes.keys()):
        raise AssertionError("source original video path mapping is not unique")
    for video_rel, expected in source_original_hashes.items():
        video_path = run / video_rel
        if not video_path.is_file() or _sha(video_path) != expected:
            raise RuntimeError(f"original source video changed or missing: {video_rel}")

    provenance_dir = output / "provenance"
    indices_dir = output / "indices"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)
    with (provenance_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in episode_index:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (indices_dir / "episode_tree_map.jsonl").open("w", encoding="utf-8") as stream:
        for row in episode_index:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (indices_dir / "terminal_observation_map.jsonl").open("w", encoding="utf-8") as stream:
        for row in terminal_index:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    request_rows = []
    for request_id in source_scope["queue_order"]:
        route = route_by_id[request_id]
        root_dirs = list(
            (run / "data/formal" / route["task"]).glob(f"{route['task']}__{request_id}__*")
        )
        completed = (
            json.loads((root_dirs[0] / "completed.json").read_text())
            if root_dirs and (root_dirs[0] / "completed.json").is_file()
            else None
        )
        request_rows.append(
            {
                "request_id": request_id,
                "source_id": route["source_id"],
                "task": route["task"],
                "source_class": route["source_class"],
                "request_status": route["request_status"],
                "source_video_status": route.get("source_video_status"),
                "job_status": None if completed is None else completed.get("status"),
                "exported_episode_count": sum(
                    row["request_id"] == request_id for row in episode_index
                ),
                "root_dir": None if not root_dirs else str(root_dirs[0].relative_to(run)),
            }
        )
    with (provenance_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for row in request_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (indices_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for row in request_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.Table.from_pylist(episode_index),
            indices_dir / "episode_tree_map.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(terminal_index),
            indices_dir / "terminal_observation_map.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(request_rows), indices_dir / "requests.parquet", compression="zstd"
        )
    except ImportError as error:
        raise RuntimeError("pyarrow is required to write the promised LeRobot tree maps") from error
    (provenance_dir / "joint_schema.json").write_text(
        json.dumps(
            {
                "robot_joint_names": list(robot_joint_names),
                "robot_joint_types": list(robot_joint_types),
                "robot_qpos_units": list(robot_qpos_units),
                "robot_qvel_units": list(robot_qvel_units),
                "fixture_joint_values": "per episode in indices/episode_tree_map.parquet and in raw execution.npz",
                "fixture_qpos_units": "metres for slide joints; radians for hinge joints",
                "fixture_qvel_units": "metres/second for slide joints; radians/second for hinge joints",
            },
            indent=2,
        )
        + "\n"
    )
    (output / "DATA_FORMAT.md").write_text(
        "# LeRobot v3 字段与对齐\n\n"
        "每个 LeRobot episode 恰为一个连续片段：`local` 或 `fixed_continuation`。K 个实测 action 写 K 行；第 t 行是实际状态 s[t] 与发出的 action a[t]。实测 s[K] 不制造 action，保存在 `indices/terminal_observation_map.parquet` 并通过原始 execution trace 索引。两个片段各自成 episode，不跨 child state 边界。\n\n"
        "| Feature | dtype / shape | 字段含义与单位 |\n|---|---|---|\n"
        "| `observation.state` | float32, J | 实际机器人 joint qpos；按 `provenance/joint_schema.json` 的名字、类型与逐关节单位排列 |\n"
        "| `observation.robot_qvel` | float32, J | 同序关节速度；hinge 为 rad/s，slide 为 m/s |\n"
        "| `observation.eef_pose` | float32, 4×4 | world frame 齐次变换；平移 m，旋转矩阵无量纲 |\n"
        "| `observation.object_pose` | float32, 4×4 | 任务对象 world pose；平移 m |\n"
        "| `observation.object_velocity` | float32, 3 | 对象 world 线速度，m/s |\n"
        "| `observation.fixture_pose` | float32, 4×4 | 活动 fixture world pose；平移 m |\n"
        "| `observation.fixture_qpos` / `fixture_qvel` | float32, Jf | 活动 fixture 关节值/速度；零填充位置由 `fixture_joint_mask` 标出；实际名称及每关节单位在 episode tree map 与原始 trace |\n"
        "| `observation.fixture_joint_mask` | bool, Jf | fixture joint 有效维掩码 |\n"
        "| `observation.object_to_gripper_transform` | float32, 4×4 | 夹爪坐标系到对象的相对变换；平移 m |\n"
        "| `observation.sim_time_s` | float32, 1 | 模拟器时间，秒 |\n"
        f"| `observation.images.robot0_agentview_left/right` | video, {height}×{width}×3 RGB | 两个分开的真实 simulator camera 视图；按采集 receipt，JPEG q80 暂存 |\n"
        f"| `observation.images.robot0_eye_in_hand` | video, {height}×{width}×3 RGB | 真实腕部 camera 视图；按采集 receipt，JPEG q80 暂存 |\n"
        "| `action` / `action.raw_planned` | float32, 12 | `action` 是 env.step 收到的命令；planned 是原始提案值。前6维为 base frame 归一化末端位置/轴角增量，第7维夹爪，第8–11维固定底盘命令通道，第12维控制器模式 -1 |\n"
        "| `action.controller_target_pose` | float32, 4×4 | 控制器目标末端姿态；不是实测状态 |\n"
        "| `label.local_event_reached` | bool, 1 | 与 action 对齐的局部事件边界标签；不是 observation |\n\n"
        "官方任务成功、fixed outcome、超时/错误状态留在原始 tree result 与 provenance 中，不塞入 observation。review video 是回看用四宫格，不是训练图像特征。历史 R2 兼容样例从旧 review panel 裁切，单独记录 `DERIVED_FROM_LOSSY_REVIEW_PANEL_CROP`；缺失源原视频保持 `SOURCE_VIDEO_MISSING`。\n"
    )

    readback = {
        "status": "PASS",
        "lerobot_repo_id": args.repo_id,
        "exporter_sha256": _sha(Path(__file__)),
        "source_run": str(run),
        "source_config_sha256": _sha(run / "config_frozen.json"),
        "source_manifest_sha256": _sha(run / "source_split_frozen.json"),
        "matrix_statistics": "per_matrix_element_over_episode_time",
        "lerobot_root": str(output),
        "dataset_version": reader.meta.info.get("codebase_version"),
        "episode_count": int(reader.meta.total_episodes),
        "frame_count": int(reader.meta.total_frames),
        "fps": int(reader.meta.fps),
        "camera_features": [f"{VIDEO_PREFIX}{camera}" for camera in CAMERAS],
        "camera_feature_resolution_hw": [height, width],
        "camera_observation_provenance": media["camera_observation_provenance"],
        "review_video_resolution": media["resolution"],
        "video_encoder": selected_vcodec,
        "h264_b_frames": 0 if selected_vcodec == "h264" else None,
        "storage_admission_before_export": initial_storage,
        "decoded_camera_samples": len(readback_rows),
        "decoded_episode_endpoints": readback_rows,
        "source_original_video_count": len(source_original_hashes),
        "source_original_videos_verified_sha256": bool(source_original_hashes) or None,
        "episode_tree_map": "indices/episode_tree_map.parquet",
        "terminal_observation_map": "indices/terminal_observation_map.parquet",
        "decoded_segment_samples": len(readback_rows),
        "export_errors": failures,
        "provenance": {
            "episodes": "provenance/episodes.jsonl",
            "requests": "provenance/requests.jsonl",
        },
    }
    (output / "lerobot_readback.json").write_text(
        json.dumps(readback, indent=2, sort_keys=True) + "\n"
    )
    readme = [
        "# CFACT measured action trees",
        "",
        "LeRobot Dataset v3.0 export of measured physical counterfactual transitions from RoboCasa365.",
        "Each LeRobot episode is one uninterrupted local or fixed-continuation segment; no episode crosses a branch or segment boundary.",
        "Rows contain execution-time measurements s[t] and the command actually sent a[t]. The measured s[K] endpoint is referenced by the terminal observation map.",
        f"The three image features are separate true simulator camera views at {width}x{height}, staged as JPEG q80; resolution/transforms follow the acquisition receipt; review panels are excluded from training images.",
        "Task success and event outcomes are kept as tree labels or explicitly named label features, not observation fields.",
        "Per-episode tree lineage is in `indices/episode_tree_map.parquet`; terminal s[K] references are in `indices/terminal_observation_map.parquet`.",
        "All requested source roots and their export counts are in `indices/requests.parquet`.",
        "Original source-camera videos remain byte-identical in the collection run when they were available.",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme))
    cleanup = {"status": "PENDING", "removed_clean_camera_spools": 0, "removed_bytes": 0}
    if args.cleanup_camera_spools:
        removed, removed_bytes = 0, 0
        for _, edge_path, _ in _edge_sources(run):
            media_path = edge_path / "media.json"
            if not media_path.is_file():
                continue
            media = json.loads(media_path.read_text())
            spool_value = media.get("frame_spool")
            if not spool_value:
                continue
            spool = Path(spool_value)
            clean = (spool / "clean_cameras").resolve()
            if not clean.is_relative_to(run):
                raise ValueError(
                    f"refusing to clean a camera spool outside the source run: {clean}"
                )
            if clean.is_dir():
                removed_bytes += sum(
                    path.stat().st_size for path in clean.rglob("*") if path.is_file()
                )
                shutil.rmtree(clean)
                removed += 1
        cleanup = {
            "status": "COMPLETE",
            "removed_clean_camera_spools": removed,
            "removed_bytes": removed_bytes,
        }
    readback["clean_camera_spool_cleanup"] = cleanup
    (output / "lerobot_readback.json").write_text(
        json.dumps(readback, indent=2, sort_keys=True) + "\n"
    )
    return readback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--python-packages", type=Path)
    parser.add_argument("--cleanup-camera-spools", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args), sort_keys=True))


if __name__ == "__main__":
    main()
