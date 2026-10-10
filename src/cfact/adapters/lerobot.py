"""Thin, provenance-preserving adapter for official RoboCasa LeRobot bundles.

The official conversion stores raw MuJoCo states, model XML, episode metadata,
and dataset environment metadata under ``extras/``. This module reconstructs a
raw replay HDF5 from those authoritative fields and reorders only the recorded
action vector. It deliberately does not infer simulator state from videos or
invent missing control signals.
"""

from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np


HDF5_ACTION_LAYOUT: dict[str, tuple[int, int]] = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}
EPISODE_PATTERN = re.compile(r"^episode_(\d+)$")


@dataclass(frozen=True)
class LeRobotEpisode:
    index: int
    parquet_path: Path
    states_path: Path
    model_xml_path: Path
    episode_meta_path: Path
    state_rows: int
    state_width: int
    state_bytes: int
    model_xml_bytes: int


@dataclass(frozen=True)
class LeRobotInspection:
    root: Path
    task: str
    env_args: dict[str, Any]
    dataset_meta: dict[str, Any]
    modality: dict[str, Any]
    action_width: int
    episodes: tuple[LeRobotEpisode, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "source_format": "robocasa_lerobot_with_replay_extras",
            "task": self.task,
            "episode_count": len(self.episodes),
            "episode_indices": [episode.index for episode in self.episodes],
            "action_width": self.action_width,
            "state_widths": sorted({episode.state_width for episode in self.episodes}),
            "state_rows": sum(episode.state_rows for episode in self.episodes),
            "estimated_raw_hdf5_bytes": estimate_raw_hdf5_bytes(self),
            "physical_replay_fields": [
                "extras/dataset_meta.json:env_args",
                "extras/episode_*/states.npz",
                "extras/episode_*/model.xml.gz",
                "extras/episode_*/ep_meta.json",
                "data/*/episode_*.parquet:action",
            ],
            "video_used_as_state": False,
        }


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def locate_lerobot_root(path: Path) -> Path:
    """Accept either the LeRobot root or a directory containing ``lerobot/``."""

    path = path.resolve()
    candidates = (path, path / "lerobot")
    for candidate in candidates:
        if (
            (candidate / "meta" / "info.json").is_file()
            and (candidate / "meta" / "modality.json").is_file()
            and (candidate / "extras" / "dataset_meta.json").is_file()
        ):
            return candidate
    raise ValueError(f"no complete RoboCasa LeRobot root found under {path}")


def validate_action_modality(modality: dict[str, Any]) -> int:
    """Validate the five official PandaOmron action components and slices."""

    action = modality.get("action")
    if not isinstance(action, dict):
        raise ValueError("modality.json must contain an action mapping")
    missing = set(HDF5_ACTION_LAYOUT) - set(action)
    if missing:
        raise ValueError(f"missing action modalities: {sorted(missing)}")
    occupied: set[int] = set()
    width = 0
    for key in HDF5_ACTION_LAYOUT:
        item = action[key]
        start, end = item.get("start"), item.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError(f"invalid LeRobot action slice for {key}: {start}:{end}")
        expected_width = HDF5_ACTION_LAYOUT[key][1] - HDF5_ACTION_LAYOUT[key][0]
        if end - start != expected_width:
            raise ValueError(f"wrong action width for {key}: {end - start} != {expected_width}")
        indices = set(range(start, end))
        if occupied.intersection(indices):
            raise ValueError(f"overlapping LeRobot action slice for {key}")
        occupied.update(indices)
        width = max(width, end)
        if item.get("original_key", "action") != "action":
            raise ValueError(f"unsupported action source key for {key}")
    if occupied != set(range(width)):
        raise ValueError("LeRobot action slices must cover one contiguous vector")
    return width


def reorder_lerobot_actions(
    actions: np.ndarray,
    modality: dict[str, Any],
) -> np.ndarray:
    """Map official LeRobot action ordering to RoboCasa raw-HDF5 ordering."""

    width = validate_action_modality(modality)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != width:
        raise ValueError(f"expected action array [T,{width}], got {actions.shape}")
    reordered = np.empty((actions.shape[0], 12), dtype=np.float64)
    for key, (target_start, target_end) in HDF5_ACTION_LAYOUT.items():
        source_start = modality["action"][key]["start"]
        source_end = modality["action"][key]["end"]
        reordered[:, target_start:target_end] = actions[:, source_start:source_end]
    return reordered


def _episode_metadata_indices(path: Path) -> set[int] | None:
    if not path.is_file():
        return None
    indices = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        index = row.get("episode_index")
        if not isinstance(index, int):
            raise ValueError(f"missing integer episode_index at {path}:{line_number}")
        indices.add(index)
    return indices


def inspect_lerobot_dataset(
    source: Path,
    *,
    expected_task: str | None = None,
    episode_indices: list[int] | None = None,
) -> LeRobotInspection:
    """Validate replay-critical structure without loading videos or parquet bodies."""

    root = locate_lerobot_root(source)
    dataset_meta = _read_json(root / "extras" / "dataset_meta.json")
    env_args = dataset_meta.get("env_args")
    if isinstance(env_args, str):
        env_args = json.loads(env_args)
    if not isinstance(env_args, dict):
        raise ValueError("extras/dataset_meta.json must contain env_args")
    task = env_args.get("env_name")
    if not isinstance(task, str) or not task:
        raise ValueError("env_args.env_name is required")
    if expected_task is not None and task != expected_task:
        raise ValueError(f"dataset task mismatch: {task} != {expected_task}")
    modality = _read_json(root / "meta" / "modality.json")
    action_width = validate_action_modality(modality)
    info = _read_json(root / "meta" / "info.json")
    feature_shape = info.get("features", {}).get("action", {}).get("shape")
    if feature_shape is not None and list(feature_shape) != [action_width]:
        raise ValueError(f"info.json action shape {feature_shape} != [{action_width}]")

    episodes = []
    for directory in sorted((root / "extras").glob("episode_*")):
        if not directory.is_dir():
            continue
        match = EPISODE_PATTERN.match(directory.name)
        if match is None:
            raise ValueError(f"invalid episode directory name: {directory.name}")
        index = int(match.group(1))
        if episode_indices is not None and index not in episode_indices:
            continue
        basename = f"episode_{index:06d}"
        parquet = list((root / "data").glob(f"*/{basename}.parquet"))
        if len(parquet) != 1:
            raise ValueError(f"episode {index} requires exactly one parquet file")
        states_path = directory / "states.npz"
        model_path = directory / "model.xml.gz"
        ep_meta_path = directory / "ep_meta.json"
        with np.load(states_path, allow_pickle=False) as stored:
            if "states" not in stored:
                raise ValueError(f"states key missing in {states_path}")
            states = np.asarray(stored["states"])
            if states.ndim != 2 or not len(states):
                raise ValueError(f"invalid states array in {states_path}: {states.shape}")
            state_rows, state_width = map(int, states.shape)
            state_bytes = int(states.nbytes)
        with gzip.open(model_path, "rb") as stream:
            model_xml = stream.read()
        ET.fromstring(model_xml)
        episode_meta = _read_json(ep_meta_path)
        if not isinstance(episode_meta, dict):
            raise ValueError(f"episode metadata must be an object: {ep_meta_path}")
        episodes.append(
            LeRobotEpisode(
                index=index,
                parquet_path=parquet[0],
                states_path=states_path,
                model_xml_path=model_path,
                episode_meta_path=ep_meta_path,
                state_rows=state_rows,
                state_width=state_width,
                state_bytes=state_bytes,
                model_xml_bytes=len(model_xml),
            )
        )
    if not episodes:
        if episode_indices is not None:
            raise ValueError("requested Source episodes missing or duplicated")
        raise ValueError("dataset contains no replay episodes")
    actual_indices = {episode.index for episode in episodes}
    metadata_indices = _episode_metadata_indices(root / "meta" / "episodes.jsonl")
    if episode_indices is not None:
        requested = set(episode_indices)
        if len(requested) != len(episode_indices) or requested != actual_indices:
            raise ValueError("requested Source episodes missing or duplicated")
        if metadata_indices is not None and not requested.issubset(metadata_indices):
            raise ValueError("requested Source episodes absent from metadata")
    elif metadata_indices is not None and metadata_indices != actual_indices:
        raise ValueError("meta/episodes.jsonl does not match replay extras")
    if len(actual_indices) != len(episodes):
        raise ValueError("duplicate episode indices")
    return LeRobotInspection(
        root=root,
        task=task,
        env_args=env_args,
        dataset_meta=dataset_meta,
        modality=modality,
        action_width=action_width,
        episodes=tuple(sorted(episodes, key=lambda item: item.index)),
    )


def estimate_raw_hdf5_bytes(inspection: LeRobotInspection) -> int:
    """Conservative low-dimensional output estimate, excluding source videos."""

    total = 1024 * 1024
    for episode in inspection.episodes:
        total += episode.state_bytes
        total += episode.state_rows * 12 * np.dtype(np.float64).itemsize
        total += episode.model_xml_bytes
        total += 1024 * 1024
    return int(total)


def _read_parquet_transition_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read actions and authoritative source outcome columns."""

    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "LeRobot import requires the optional 'input' dependencies "
            "(pandas and pyarrow); inspection does not"
        ) from error
    frame = pd.read_parquet(path, columns=["action", "next.reward", "next.done"])
    if "action" not in frame or frame.empty:
        raise ValueError(f"parquet action column is missing or empty: {path}")
    if "next.reward" not in frame or "next.done" not in frame:
        raise ValueError(f"parquet source outcome columns are missing: {path}")
    return (
        np.stack(frame["action"].to_list()),
        np.asarray(frame["next.reward"], dtype=np.float64),
        np.asarray(frame["next.done"], dtype=np.bool_),
    )


def import_raw_replay_hdf5(
    inspection: LeRobotInspection,
    destination: Path,
    *,
    max_episodes: int | None = None,
    progress_guard: Callable[[], None] | None = None,
    preserve_episode_indices: bool = False,
) -> dict[str, Any]:
    """Create a new raw replay HDF5; never overwrite an existing destination."""

    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    episodes = inspection.episodes[:max_episodes] if max_episodes else inspection.episodes
    if not episodes:
        raise ValueError("max_episodes selected no episodes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_successful_episodes = 0
    try:
        with h5py.File(temporary, "w") as output:
            data = output.create_group("data")
            for key, value in inspection.dataset_meta.items():
                if key == "env_args":
                    value = inspection.env_args
                if value is None or isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, sort_keys=True)
                data.attrs[key] = value
            data.attrs["total"] = sum(episode.state_rows for episode in episodes)
            data.attrs["env_args"] = json.dumps(inspection.env_args, sort_keys=True)
            for output_index, episode in enumerate(episodes):
                source_actions, source_rewards, source_dones = _read_parquet_transition_data(
                    episode.parquet_path
                )
                actions = reorder_lerobot_actions(source_actions, inspection.modality)
                with np.load(episode.states_path, allow_pickle=False) as stored:
                    states = np.asarray(stored["states"])
                if not np.isfinite(actions).all() or not np.isfinite(states).all():
                    raise ValueError(
                        f"episode {episode.index} contains nonfinite actions or states"
                    )
                if not np.isfinite(source_rewards).all():
                    raise ValueError(f"episode {episode.index} contains nonfinite source rewards")
                if len(actions) != len(states):
                    raise ValueError(
                        f"episode {episode.index} action/state length mismatch: "
                        f"{len(actions)} != {len(states)}"
                    )
                if len(source_rewards) != len(states) or len(source_dones) != len(states):
                    raise ValueError(
                        f"episode {episode.index} source outcome/state length mismatch"
                    )
                with gzip.open(episode.model_xml_path, "rt", encoding="utf-8") as stream:
                    model_xml = stream.read()
                episode_meta = _read_json(episode.episode_meta_path)
                demo_index = episode.index if preserve_episode_indices else output_index
                group = data.create_group(f"demo_{demo_index}")
                group.create_dataset("actions", data=actions, compression="gzip")
                group.create_dataset("states", data=states, compression="gzip")
                group.create_dataset("source_next_reward", data=source_rewards, compression="gzip")
                group.create_dataset("source_next_done", data=source_dones, compression="gzip")
                group.attrs["model_file"] = model_xml
                group.attrs["ep_meta"] = json.dumps(episode_meta, sort_keys=True)
                group.attrs["lerobot_episode_index"] = episode.index
                group.attrs["original_episode_id"] = f"episode_{episode.index:06d}"
                group.attrs["source_terminal_reward"] = float(source_rewards[-1])
                group.attrs["source_terminal_done"] = bool(source_dones[-1])
                source_successful_episodes += int(bool(np.any(source_rewards > 0)))
                output.flush()
                if progress_guard is not None:
                    progress_guard()
        temporary.replace(destination)
    except Exception:
        # Preserve the partial file for diagnosis and capacity accounting.
        raise
    return {
        "schema_version": "1.0.0",
        "source_format": "robocasa_lerobot_with_replay_extras",
        "output_format": "robocasa_raw_replay_hdf5",
        "task": inspection.task,
        "episode_count": len(episodes),
        "source_episode_indices": [episode.index for episode in episodes],
        "output_bytes": destination.stat().st_size,
        "source_successful_episode_count": source_successful_episodes,
        "video_used_as_state": False,
        "requires_official_prepare_step": True,
    }
