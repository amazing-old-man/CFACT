"""Outcome-blind modern operation candidate construction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np


@dataclass(frozen=True)
class SourceSegment:
    episode_id: str
    event_first_step: int
    end_step: int
    eef_pose_at_entry: np.ndarray
    object_pose_at_entry: np.ndarray
    source_poses: np.ndarray
    gripper_actions: np.ndarray
    base_action_max_abs: tuple[float, float, float, float]
    mode_values: tuple[float, ...]
    # The manipulated object's measured source motion, separate from the
    # fixture frame used by MimicGen to redirect end-effector waypoints.
    motion_object_ref: str | None = None
    motion_object_poses: np.ndarray | None = None
    operation_type: str | None = None
    stage: str | None = None
    object_roles: dict[str, str] = field(default_factory=dict)
    source_action_indices: np.ndarray | None = None
    gripper_semantics: str = "negative_open_positive_close"


def segment_timeline(
    segment: SourceSegment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Return motion, gripper and source cursor on one recorded timeline.

    The leading pose is the entry state. Older in-memory segments have one
    fewer object measurement; its entry is duplicated explicitly, never
    resampled independently of the donor's gripper commands.
    """
    poses = np.asarray(segment.source_poses, dtype=float)
    grips = np.asarray(segment.gripper_actions, dtype=float)
    if grips.ndim == 1:
        grips = grips[:, None]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("invalid_source_pose_timeline")
    if grips.ndim != 2 or len(grips) != len(poses):
        raise ValueError("source_motion_gripper_timeline_mismatch")
    if not np.isfinite(poses).all() or not np.isfinite(grips).all():
        raise ValueError("source_timeline_non_finite")
    motion = segment.motion_object_poses
    if motion is not None:
        motion = np.asarray(motion, dtype=float)
        if motion.ndim != 3 or motion.shape[1:] != (4, 4) or not np.isfinite(motion).all():
            raise ValueError("invalid_source_object_timeline")
        if len(motion) == len(poses) - 1:
            motion = np.concatenate([motion[:1], motion], axis=0)
        elif len(motion) != len(poses):
            raise ValueError("source_object_motion_timeline_mismatch")
    if segment.source_action_indices is None:
        # -1 denotes the entry waypoint, which issues no source action.
        indices = np.concatenate(([-1], np.arange(
            segment.event_first_step, segment.event_first_step + len(poses) - 1
        )))
    else:
        indices = np.asarray(segment.source_action_indices, dtype=int)
        if indices.shape != (len(poses),):
            raise ValueError("source_action_index_timeline_mismatch")
    if np.any(indices[1:] < 0) or np.any(np.diff(indices) < 0):
        raise ValueError("source_action_indices_not_monotone")
    return poses.copy(), grips.copy(), None if motion is None else motion.copy(), indices.copy()


def first_sustained_true(signal: np.ndarray, minimum_duration: int) -> int | None:
    """Return the first index of a consecutive true run of the given length."""

    if minimum_duration < 1:
        raise ValueError("minimum_duration must be positive")
    start: int | None = None
    for index, value in enumerate(np.asarray(signal, dtype=bool)):
        if value:
            if start is None:
                start = index
            if index - start + 1 >= minimum_duration:
                return start
        else:
            start = None
    return None


def load_source_segment(
    dataset: Path,
    episode_id: str,
    *,
    event_signal: str,
    event_minimum_duration: int,
    object_ref: str,
    start_step: int | None = None,
    motion_object_ref: str | None = None,
    operation_type: str | None = None,
    stage: str | None = None,
    object_roles: dict[str, str] | None = None,
) -> SourceSegment:
    """Load one source suffix without consulting terminal outcome labels.

    By default the prepared source event chooses the boundary. ``start_step`` is
    reserved for a boundary observed by replaying the root in the real physics
    environment; this keeps proposal slicing aligned with the physical parent.
    """

    with h5py.File(dataset, "r") as source:
        group = source[f"data/{episode_id}"]
        info = group["datagen_info"]
        signal = np.asarray(info[f"subtask_term_signals/{event_signal}"][()], dtype=bool)
        source_event_start = first_sustained_true(signal, event_minimum_duration)
        if source_event_start is None:
            raise ValueError(f"{episode_id} has no sustained {event_signal} event")
        target = np.asarray(info["target_pose"][()])
        eef = np.asarray(info["eef_pose"][()])
        gripper = np.asarray(info["gripper_action"][()])
        objects = np.asarray(info[f"object_poses/{object_ref}"][()])
        motion = (
            None
            if motion_object_ref is None
            else np.asarray(info[f"object_poses/{motion_object_ref}"][()])
        )
        actions = np.asarray(group["actions"][()])
    start = source_event_start if start_step is None else int(start_step)
    if start < 0 or start >= len(target):
        raise ValueError(f"start step {start} is outside {episode_id} with {len(target)} steps")
    source_poses = np.concatenate([eef[start : start + 1], target[start:]], axis=0)
    # Entry is s[start], before a[start]. Keep the preceding discrete command
    # through the connector; an opening at a[start] remains a later event.
    prior_command = max(0, start - 1)
    gripper_actions = np.concatenate(
        [gripper[prior_command : prior_command + 1], gripper[start:]], axis=0
    )
    return SourceSegment(
        episode_id=episode_id,
        event_first_step=int(start),
        end_step=int(len(target)),
        eef_pose_at_entry=np.array(eef[start], copy=True),
        object_pose_at_entry=np.array(objects[start], copy=True),
        source_poses=source_poses,
        gripper_actions=gripper_actions,
        base_action_max_abs=tuple(
            float(value) for value in np.abs(actions[start:, 7:11]).max(axis=0)
        ),
        mode_values=tuple(float(value) for value in np.unique(actions[start:, 11])),
        motion_object_ref=motion_object_ref,
        motion_object_poses=None if motion is None else np.concatenate(
            [motion[start : start + 1], motion[start:]], axis=0
        ),
        operation_type=operation_type,
        stage=stage,
        object_roles=dict(object_roles or {}),
        source_action_indices=np.concatenate(([-1], np.arange(start, len(target)))),
    )


def recorded_held_entry_step(
    dataset: Path,
    episode_id: str,
    *,
    contact_signal: str,
    object_ref: str = "obj",
    minimum_lift_m: float = 0.03,
) -> int:
    """Locate the first sustained held transport state in measured source data."""
    with h5py.File(dataset, "r") as source:
        info = source[f"data/{episode_id}/datagen_info"]
        contacts = np.asarray(info[f"subtask_term_signals/{contact_signal}"], dtype=bool)
        eef = np.asarray(info["eef_pose"], dtype=float)
        obj = np.asarray(info[f"object_poses/{object_ref}"], dtype=float)
        gripper = np.asarray(info["gripper_action"], dtype=float).reshape(-1)
    contact_at = first_sustained_true(contacts, 2)
    if contact_at is None:
        raise ValueError("source_contact_never_sustained")
    origin = obj[contact_at, :3, 3]
    prior = None
    stable = 0
    for index in range(contact_at, len(eef)):
        relative = np.linalg.inv(eef[index]) @ obj[index]
        shift = np.inf if prior is None else float(np.linalg.norm(relative[:3, 3] - prior[:3, 3]))
        stable = stable + 1 if shift <= 0.003 else 0
        prior = relative
        moved = (
            obj[index, 2, 3] - origin[2] >= minimum_lift_m
            if minimum_lift_m > 0
            else np.linalg.norm(obj[index, :3, 3] - origin) >= 0.04
        )
        if stable >= 5 and moved and gripper[index] > 0 and contacts[index]:
            return index
    raise ValueError("source_grasp_entry_not_observed")


def _validated_pose_inverse(pose: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite 4x4 pose and its inverse, or a stable rejection reason."""

    value = np.asarray(pose, dtype=float)
    if value.shape != (4, 4):
        raise ValueError(f"{label}_invalid_shape:{value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label}_non_finite")
    try:
        inverse = np.linalg.inv(value)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{label}_singular") from error
    if not np.all(np.isfinite(inverse)):
        raise ValueError(f"{label}_inverse_non_finite")
    return value, inverse


def _relative_entry_pose(segment: SourceSegment) -> np.ndarray:
    object_pose, object_inverse = _validated_pose_inverse(
        segment.object_pose_at_entry,
        label="object_pose_at_entry",
    )
    del object_pose
    eef_pose, _ = _validated_pose_inverse(
        segment.eef_pose_at_entry,
        label="eef_pose_at_entry",
    )
    relative = object_inverse @ eef_pose
    if not np.all(np.isfinite(relative)):
        raise ValueError("relative_entry_pose_non_finite")
    return relative


def relative_entry_descriptor(segment: SourceSegment) -> np.ndarray:
    relative = _relative_entry_pose(segment)
    return np.concatenate([relative[:3, 3], relative[:3, :3].reshape(-1)])


def donor_distance(root: SourceSegment, donor: SourceSegment) -> dict[str, float]:
    root_rel = _relative_entry_pose(root)
    donor_rel = _relative_entry_pose(donor)
    translation = float(np.linalg.norm(root_rel[:3, 3] - donor_rel[:3, 3]))
    relative_rotation = root_rel[:3, :3] @ donor_rel[:3, :3].T
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation = float(np.arccos(cosine))
    return {
        "translation_m": translation,
        "rotation_rad": rotation,
        "combined": translation + 0.05 * rotation,
    }


def select_compatible_donor(
    root: SourceSegment,
    donors: list[SourceSegment],
    *,
    allow_none: bool = False,
) -> tuple[SourceSegment | None, list[dict[str, Any]]]:
    """Choose the nearest compatible donor using only entry geometry."""

    rows: list[dict[str, Any]] = []
    try:
        _relative_entry_pose(root)
    except ValueError as error:
        reason = f"invalid_root_entry_geometry:{error}"
        rows = [
            {
                "donor_id": donor.episode_id,
                "compatible": False,
                "rejection_reasons": [reason],
                "geometry_distance_available": False,
            }
            for donor in donors
        ]
        if allow_none:
            return None, rows
        raise ValueError("root has invalid entry geometry") from error
    for donor in donors:
        reasons: list[str] = []
        if root.operation_type != donor.operation_type:
            reasons.append("operation_type_mismatch")
        if root.stage != donor.stage:
            reasons.append("operation_stage_mismatch")
        if root.object_roles != donor.object_roles:
            reasons.append("object_role_mismatch")
        if root.motion_object_ref != donor.motion_object_ref:
            reasons.append("motion_object_role_mismatch")
        if root.gripper_semantics != donor.gripper_semantics:
            reasons.append("gripper_controller_semantics_mismatch")
        if any(value != 0.0 for value in donor.base_action_max_abs):
            reasons.append("nonzero_mobile_base_action")
        if donor.mode_values != root.mode_values:
            reasons.append("controller_mode_mismatch")
        root_gripper = np.asarray(root.gripper_actions[0]).reshape(-1)
        donor_gripper = np.asarray(donor.gripper_actions[0]).reshape(-1)
        if root_gripper.shape != donor_gripper.shape or not np.array_equal(
            np.sign(root_gripper), np.sign(donor_gripper)
        ):
            reasons.append("entry_gripper_semantics_mismatch")
        try:
            distance = donor_distance(root, donor)
        except ValueError as error:
            reasons.append(f"invalid_donor_entry_geometry:{error}")
            distance = None
        rows.append(
            {
                "donor_id": donor.episode_id,
                "compatible": not reasons,
                "rejection_reasons": reasons,
                "geometry_distance_available": distance is not None,
                **({} if distance is None else distance),
            }
        )
    eligible = [row for row in rows if row["compatible"]]
    if not eligible:
        if allow_none:
            return None, rows
        raise ValueError("no compatible donor under frozen entry checks")
    chosen = min(eligible, key=lambda row: (row["combined"], row["donor_id"]))
    return next(item for item in donors if item.episode_id == chosen["donor_id"]), rows


def transform_segment(
    segment: SourceSegment,
    current_object_pose: np.ndarray,
    official_transform: Callable[..., np.ndarray],
) -> np.ndarray:
    """Redirect a source suffix with MimicGen's official object-pose transform."""

    return official_transform(
        obj_pose=np.asarray(current_object_pose),
        src_eef_poses=np.asarray(segment.source_poses),
        src_obj_pose=np.asarray(segment.object_pose_at_entry),
    )


def _dominant_local_motion(poses: np.ndarray, object_pose: np.ndarray) -> tuple[int, int, float]:
    rotation = np.asarray(object_pose[:3, :3])
    origin = np.asarray(object_pose[:3, 3])
    local = (rotation.T @ (poses[:, :3, 3] - origin).T).T
    displacement = local - local[0]
    peak_by_axis = np.max(np.abs(displacement), axis=0)
    axis = int(np.argmax(peak_by_axis))
    peak = int(np.argmax(np.abs(displacement[:, axis])))
    sign = float(np.sign(displacement[peak, axis]) or 1.0)
    return axis, peak, sign


def _dominant_local_rotation(poses: np.ndarray, object_pose: np.ndarray) -> tuple[int, int, float]:
    """Find the dominant observed rotation axis in the operation-object frame.

    The skew part of the relative rotation is proportional to ``sin(angle) *
    axis``.  Modern operation suffixes only contain modest rotations, so this
    representation is sufficient for choosing an axis without adding a SciPy
    runtime dependency.
    """

    object_rotation = np.asarray(object_pose[:3, :3])
    local_rotations = np.einsum("ij,njk->nik", object_rotation.T, np.asarray(poses[:, :3, :3]))
    relative = np.einsum("ij,njk->nik", local_rotations[0].T, local_rotations)
    rotation_signal = 0.5 * np.stack(
        [
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ],
        axis=1,
    )
    peak_by_axis = np.max(np.abs(rotation_signal), axis=0)
    axis = int(np.argmax(peak_by_axis))
    peak = int(np.argmax(np.abs(rotation_signal[:, axis])))
    sign = float(np.sign(rotation_signal[peak, axis]) or 1.0)
    return axis, peak, sign


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    rotation = np.eye(3, dtype=float)
    first = (axis + 1) % 3
    second = (axis + 2) % 3
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation[first, first] = cosine
    rotation[second, second] = cosine
    rotation[first, second] = -sine
    rotation[second, first] = sine
    return rotation


def deform_candidate_poses(
    reference_poses: np.ndarray,
    *,
    method: str,
    primitive_family: str,
    object_pose: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a frozen, unit-aware G0 or operation-family G2 deformation."""

    poses = np.array(reference_poses, copy=True)
    count = len(poses)
    if count < 3:
        raise ValueError("operation segment is too short for a physical deformation")
    rotation = np.asarray(object_pose[:3, :3])
    dominant_axis, peak, motion_sign = _dominant_local_motion(poses, object_pose)
    if method == "reference" or method == "G1":
        return poses, {
            "kind": "official_object_pose_redirect",
            "coordinate_frame": "operation_object_local",
        }
    if method == "G0":
        other_axes = [axis for axis in range(3) if axis != dominant_axis]
        axis = other_axes[0]
        excursion_m = 0.003
        weights = np.sin(np.linspace(0.0, np.pi, count))
        poses[:, :3, 3] += weights[:, None] * excursion_m * rotation[:, axis]
        return poses, {
            "kind": "smooth_local_excursion",
            "coordinate_frame": "operation_object_local",
            "axis": axis,
            "excursion_m": excursion_m,
            "endpoint_offset_m": 0.0,
        }
    if method != "G2":
        raise ValueError(f"unknown candidate method: {method}")
    if primitive_family == "button":
        magnitude_m = 0.004
        parameter = "press_depth_delta_m"
    elif primitive_family == "linear_joint":
        magnitude_m = 0.010
        parameter = "joint_motion_extension_m"
    elif primitive_family == "placement":
        # Shape the approach in the target-fixture frame without moving either
        # operation boundary.  This represents a distinct insertion path while
        # preserving the task target and final withdrawal pose.
        clearance_m = 0.008
        weights = np.sin(np.linspace(0.0, np.pi, count)) ** 2
        poses[:, :3, 3] += weights[:, None] * clearance_m * rotation[:, 2]
        poses[0] = reference_poses[0]
        poses[-1] = reference_poses[-1]
        return poses, {
            "kind": "insertion_approach_geometry",
            "coordinate_frame": "operation_target_local",
            "target_normal_axis": 2,
            "approach_clearance_m": clearance_m,
            "endpoint_offset_m": 0.0,
        }
    elif primitive_family in {"rotary_joint", "lever", "knob"}:
        rotation_axis, rotation_peak, rotation_sign = _dominant_local_rotation(poses, object_pose)
        if rotation_peak <= 0 or rotation_peak >= count - 1:
            rotation_peak = max(1, min(count - 2, count // 2))
        weights = np.zeros(count, dtype=float)
        weights[: rotation_peak + 1] = np.linspace(0.0, 1.0, rotation_peak + 1)
        weights[rotation_peak:] = np.linspace(1.0, 0.0, count - rotation_peak)
        angle_delta_rad = 0.08
        object_rotation = np.asarray(object_pose[:3, :3])
        object_origin = np.asarray(object_pose[:3, 3])
        for index, weight in enumerate(weights):
            local_delta = _axis_rotation(
                rotation_axis,
                float(weight * angle_delta_rad * rotation_sign),
            )
            world_delta = object_rotation @ local_delta @ object_rotation.T
            poses[index, :3, 3] = object_origin + world_delta @ (
                poses[index, :3, 3] - object_origin
            )
            poses[index, :3, :3] = world_delta @ poses[index, :3, :3]
        # Preserve exact operation entry/exit boundaries for sibling fairness.
        poses[0] = reference_poses[0]
        poses[-1] = reference_poses[-1]
        return poses, {
            "kind": "operation_geometry",
            "coordinate_frame": "operation_object_local",
            "dominant_rotation_axis": rotation_axis,
            "motion_sign": rotation_sign,
            "peak_waypoint_index": rotation_peak,
            "angle_delta_rad": angle_delta_rad,
            "endpoint_angle_offset_rad": 0.0,
        }
    else:
        raise ValueError(f"no frozen G2 deformation for {primitive_family}")
    weights = np.zeros(count, dtype=float)
    if peak <= 0 or peak >= count - 1:
        peak = max(1, min(count - 2, count // 2))
    weights[: peak + 1] = np.linspace(0.0, 1.0, peak + 1)
    weights[peak:] = np.linspace(1.0, 0.0, count - peak)
    poses[:, :3, 3] += weights[:, None] * magnitude_m * motion_sign * rotation[:, dominant_axis]
    return poses, {
        "kind": "operation_geometry",
        "coordinate_frame": "operation_object_local",
        "dominant_axis": dominant_axis,
        "motion_sign": motion_sign,
        "peak_waypoint_index": peak,
        parameter: magnitude_m,
        "endpoint_offset_m": 0.0,
    }
