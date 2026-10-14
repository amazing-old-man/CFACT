"""Source replay and waypoint helpers shared by collection workers."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import h5py
import numpy as np
from cfact.execution.events import HysteresisEventDetector
from cfact.adapters.robocasa import robocasa_env_wrapper_class


def _json_attr(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value) if isinstance(value, str) else dict(value)


def _text_attr(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _load_root(dataset: Path, root_id: str) -> dict[str, Any]:
    with h5py.File(dataset, "r") as source:
        group = source[f"data/{root_id}"]
        success_path = "datagen_info/subtask_term_signals/success"
        terminal_success = (
            bool(group[success_path][-1])
            if success_path in group and len(group[success_path])
            else None
        )
        identity_raw = source.attrs.get("cfact_prepared_identity")
        identity = _json_attr(identity_raw) if identity_raw is not None else {}
        provenance = {
            "prepared_episode_id": root_id,
            "prepared_file_name": dataset.name,
            "raw_sha256": identity.get("raw_sha256"),
            "preparer_code_sha256": identity.get("preparer_code_sha256"),
            "original_episode_id": str(group.attrs.get("original_episode_id", root_id)),
        }
        return {
            "actions": np.asarray(group["actions"][()]),
            "terminal_recorded_state": np.asarray(group["states"][-1]),
            "terminal_recorded_state_index": len(group["states"]) - 1,
            "source_terminal_official_success": terminal_success,
            "source_provenance": provenance,
            "initial": {
                "states": np.asarray(group["states"][0]),
                "model": _text_attr(group.attrs["model_file"]),
                "ep_meta": group.attrs.get("ep_meta"),
            },
            "env_meta": _json_attr(source["data"].attrs["env_args"]),
        }


def _create_env(
    env_meta: dict[str, Any], seed: int, camera: str, render_gpu_device_id: int = 0,
    *, width: int = 224, height: int = 224
) -> Any:
    from cfact.runtime.readonly_assets import install_generated_xml_redirect, install_write_boundary
    from cfact.paths import workspace_root
    install_generated_xml_redirect(workspace_root())
    install_write_boundary(workspace_root())
    wrapper = robocasa_env_wrapper_class()
    kwargs = dict(env_meta["env_kwargs"])
    for key in (
        "env_name",
        "camera_names",
        "camera_height",
        "camera_width",
        "camera_heights",
        "camera_widths",
        "reward_shaping",
        "render_gpu_device_id",
    ):
        kwargs.pop(key, None)
    kwargs["seed"] = seed
    # Use the same physical GPU index as MUJOCO_EGL_DEVICE_ID.
    kwargs["render_gpu_device_id"] = render_gpu_device_id
    return wrapper.create_for_data_processing(
        env_name=env_meta["env_name"],
        camera_names=[camera],
        camera_height=height,
        camera_width=width,
        reward_shaping=False,
        **kwargs,
    )


def _probe_parent(
    env: Any,
    interface: Any,
    initial: dict[str, Any],
    actions: np.ndarray,
    event_signal: str,
    minimum_duration: int,
) -> dict[str, Any]:
    """Replay only enough recorded history to measure a real parent boundary."""
    env.reset_to(initial)
    detector = HysteresisEventDetector(minimum_duration, minimum_duration)
    successes: list[int] = []
    for step, action in enumerate(actions):
        env.step(action)
        signals = {
            str(name): bool(value) for name, value in interface.get_subtask_term_signals().items()
        }
        if bool(env.is_success()["task"]):
            successes.append(step)
        for observed in detector.update(step, signals):
            if observed.event_name == event_signal:
                return {
                    "event": observed.to_dict(),
                    "parent_action_count": step + 1,
                    "first_success_step_before_parent": successes[0] if successes else None,
                    "parent_state": np.asarray(env.env.sim.get_state().flatten()).copy(),
                    "parent_signals": signals,
                }
    raise RuntimeError(f"source positive-control replay never confirmed {event_signal}")


def _build_trajectory(
    WaypointSequence: Any,
    WaypointTrajectory: Any,
    current_eef_pose: np.ndarray,
    poses: np.ndarray,
    gripper: np.ndarray,
    interpolation_steps: int,
) -> Any:
    trajectory = WaypointTrajectory()
    trajectory.add_waypoint_sequence(
        WaypointSequence.from_poses(
            poses=np.asarray(current_eef_pose)[None],
            gripper_actions=np.asarray(gripper[:1]),
            action_noise=0.0,
        )
    )
    transformed = WaypointTrajectory()
    transformed.add_waypoint_sequence(
        WaypointSequence.from_poses(
            poses=np.asarray(poses),
            gripper_actions=np.asarray(gripper),
            action_noise=0.0,
        )
    )
    trajectory.merge(
        transformed, num_steps_interp=interpolation_steps, num_steps_fixed=0, action_noise=0.0
    )
    trajectory.pop_first()
    return trajectory
