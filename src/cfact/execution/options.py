"""Measured, step-aligned execution primitives for object-effect options.

No method label is read by this module. All stopping decisions use observations
made after env.step; a planned target is never written into simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def connect_and_retime_poses(
    current_pose: np.ndarray,
    poses: np.ndarray,
    gripper: np.ndarray,
    *,
    maximum_translation_m: float = 0.012,
    maximum_rotation_rad: float = 0.08,
    maximum_waypoints: int = 160,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Join the measured tool pose to a plan in physical control ticks.

    Each output interval is one 20 Hz control tick. The discrete gripper event
    stays at the end of its original interval; it is never moved into a join.
    This checks the *targets*, while the executor still records actual motion.
    """
    current = np.asarray(current_pose, dtype=float)
    targets = np.asarray(poses, dtype=float)
    grips = np.asarray(gripper, dtype=float)
    if (
        current.shape != (4, 4)
        or targets.ndim != 3
        or targets.shape[1:] != (4, 4)
        or grips.ndim != 2
        or len(targets) != len(grips)
        or len(targets) == 0
        or not np.isfinite(current).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(grips).all()
    ):
        raise ValueError("invalid_connection_inputs")
    output_poses = [current.copy()]
    output_grips = [grips[0].copy()]
    original_indices = [-1]
    added = 0
    for index, target in enumerate(targets):
        origin = output_poses[-1]
        delta = target[:3, 3] - origin[:3, 3]
        relative = origin[:3, :3].T @ target[:3, :3]
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        intervals = max(
            1,
            int(np.ceil(np.linalg.norm(delta) / maximum_translation_m)),
            int(np.ceil(np.linalg.norm(rotvec) / maximum_rotation_rad)),
        )
        if len(output_poses) - 1 + intervals > maximum_waypoints:
            raise ValueError("connection_or_path_exceeds_control_horizon")
        for part in range(1, intervals + 1):
            fraction = part / intervals
            pose = np.eye(4)
            pose[:3, :3] = origin[:3, :3] @ Rotation.from_rotvec(rotvec * fraction).as_matrix()
            pose[:3, 3] = origin[:3, 3] + fraction * delta
            output_poses.append(pose)
            output_grips.append(grips[index if part == intervals else max(index - 1, 0)].copy())
            original_indices.append(index if part == intervals else max(index - 1, 0))
        added += intervals - 1
    return (
        np.asarray(output_poses),
        np.asarray(output_grips),
        {
            "connection_control_steps": int(max(0, original_indices.index(0) - 1)),
            "retiming_added_control_steps": int(added),
            "source_waypoint_index": original_indices,
            "translation_limit_m_per_tick": maximum_translation_m,
            "rotation_limit_rad_per_tick": maximum_rotation_rad,
            "control_hz": 20,
        },
    )


def articulation_followthrough(
    *,
    joint_type: str,
    joint_name: str,
    target_q: float,
    joint_unit: str,
    stage: str,
    actual_child: dict[str, Any],
    current_q: float,
    joint_range: tuple[float, float],
    axis_world: np.ndarray,
    anchor_world: np.ndarray,
    current_handle_pose: np.ndarray,
    current_eef_pose: np.ndarray,
    samples: int = 60,
) -> np.ndarray:
    """Plan a contact preserving tool path from the measured joint state.

    The returned values are tool targets only. MuJoCo remains responsible for
    moving the named joint, and every executed qpos is recorded separately.
    """
    if joint_type not in {"slide", "hinge"}:
        raise ValueError("unsupported_joint_type")
    if not joint_name or joint_unit != ("m" if joint_type == "slide" else "rad"):
        raise ValueError("invalid_explicit_articulation_binding")
    if not stage or not np.isfinite(float(actual_child["sim_time_s"])):
        raise ValueError("missing_actual_child_phase_or_time")
    lower, upper = map(float, joint_range)
    axis = np.array(axis_world, dtype=float, copy=True)
    anchor = np.asarray(anchor_world, dtype=float)
    handle = np.asarray(current_handle_pose, dtype=float)
    eef = np.asarray(current_eef_pose, dtype=float)
    if (
        handle.shape != (4, 4)
        or eef.shape != (4, 4)
        or axis.shape != (3,)
        or anchor.shape != (3,)
        or not np.isfinite([lower, upper, current_q]).all()
        or not np.isfinite(axis).all()
        or not np.isfinite(anchor).all()
        or lower >= upper
        or not lower <= target_q <= upper
        or samples < 2
    ):
        raise ValueError("invalid_joint_followthrough_geometry")
    # Start at the measured child even when soft-limit compliance places it
    # just outside the nominal interval. The explicit target stays bounded.
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        raise ValueError("zero_joint_axis")
    axis /= axis_norm
    relation = np.linalg.inv(handle) @ eef
    result = np.repeat(handle[None], samples, axis=0)
    for index, q in enumerate(np.linspace(current_q, target_q, samples)):
        delta = float(q - current_q)
        if joint_type == "slide":
            result[index, :3, 3] = handle[:3, 3] + axis * delta
        else:
            rotation = Rotation.from_rotvec(axis * delta).as_matrix()
            result[index, :3, :3] = rotation @ handle[:3, :3]
            result[index, :3, 3] = anchor + rotation @ (handle[:3, 3] - anchor)
    return np.einsum("nij,jk->nik", result, relation)


@dataclass
class MeasuredTrace:
    object_ref: str
    fixture_ref: str
    fixture_qpos_indices: tuple[int, ...] = ()
    fixture_qvel_indices: tuple[int, ...] = ()
    fixture_joint_names: tuple[str, ...] = ()
    fixture_joint_types: tuple[str, ...] = ()
    fixture_joint_units: tuple[str, ...] = ()
    fixture_joint_qvel_units: tuple[str, ...] = ()
    robot_qpos_joint_names: tuple[str, ...] = ()
    robot_qpos_joint_types: tuple[str, ...] = ()
    robot_qpos_units: tuple[str, ...] = ()
    robot_qvel_units: tuple[str, ...] = ()
    sim_time: list[float] = field(default_factory=list)
    robot_qpos: list[np.ndarray] = field(default_factory=list)
    robot_qvel: list[np.ndarray] = field(default_factory=list)
    actual_eef_pose: list[np.ndarray] = field(default_factory=list)
    object_poses: list[np.ndarray] = field(default_factory=list)
    fixture_poses: list[np.ndarray] = field(default_factory=list)
    fixture_qpos: list[np.ndarray] = field(default_factory=list)
    fixture_qvel: list[np.ndarray] = field(default_factory=list)
    object_to_gripper_transform: list[np.ndarray] = field(default_factory=list)
    task_success: list[bool] = field(default_factory=list)
    subtask_signals: list[dict[str, bool]] = field(default_factory=list)
    commanded_actions: list[np.ndarray] = field(default_factory=list)
    raw_planned_actions: list[np.ndarray] = field(default_factory=list)
    controller_target_poses: list[np.ndarray] = field(default_factory=list)
    transition_reasons: list[str] = field(default_factory=list)
    local_event_flags: list[bool] = field(default_factory=list)
    robot_contacts_json: list[str] = field(default_factory=list)
    observation_object_poses: list[np.ndarray] = field(default_factory=list)
    observation_fixture_poses: list[np.ndarray] = field(default_factory=list)
    observation_eef_poses: list[np.ndarray] = field(default_factory=list)
    observation_metadata: list[dict] = field(default_factory=list)
    fixture_activation_json: list[str] = field(default_factory=list)
    _capture_failed: bool = field(default=False, init=False, repr=False)

    def capture(self, env: Any, interface: Any) -> None:
        if self._capture_failed:
            raise ValueError("failed_observation_capture")
        # Any exception poisons this rollout: partially appended labels must
        # never be serialized or made apparently complete by a later capture.
        self._capture_failed = True
        sim = env.env.sim
        info = interface.get_datagen_info()
        objects = info.object_poses
        if self.object_ref not in objects or self.fixture_ref not in objects:
            raise KeyError(f"missing measured role: {self.object_ref}/{self.fixture_ref}")
        obj = np.asarray(objects[self.object_ref], dtype=float).copy()
        fixture = np.asarray(objects[self.fixture_ref], dtype=float).copy()
        eef = np.asarray(info.eef_pose, dtype=float).copy()
        # Keep execution caches for the existing detectors / candidate policy.
        # Only serialized observation labels use the refreshed current state.
        from cfact.execution.observation import observation_poses
        current = observation_poses(env, interface)
        if current is not None:
            poses, current_eef, metadata = current
            current_obj = np.array(poses[self.object_ref], copy=True)
            current_fixture = np.array(poses[self.fixture_ref], copy=True)
            current_eef = np.array(current_eef, copy=True)
            if any(p.shape != (4, 4) or not np.isfinite(p).all()
                   for p in (current_obj, current_fixture, current_eef)):
                raise ValueError("observation_invalid_current_pose")
            current_metadata = metadata
        if obj.shape != (4, 4) or fixture.shape != (4, 4) or eef.shape != (4, 4):
            raise ValueError("measured pose has invalid shape")
        robot = env.env.robots[0]
        pos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        vel_idx = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
        joint_ids = np.asarray(robot._ref_joint_indexes, dtype=int)
        joint_names = tuple(str(name) for name in robot.robot_joints)
        if (
            len(joint_names) != len(pos_idx)
            or len(joint_names) != len(vel_idx)
            or len(joint_names) != len(joint_ids)
        ):
            raise ValueError("robot joint name, qpos, and qvel dimensions do not match")
        model = getattr(sim.model, "_model", sim.model)
        joint_type_values = [int(model.jnt_type[int(jid)]) for jid in joint_ids]
        joint_type_names = tuple(
            "hinge"
            if value == int(mujoco.mjtJoint.mjJNT_HINGE)
            else "slide"
            if value == int(mujoco.mjtJoint.mjJNT_SLIDE)
            else f"mujoco_joint_type_{value}"
            for value in joint_type_values
        )
        qpos_units = tuple(
            "rad"
            if value == int(mujoco.mjtJoint.mjJNT_HINGE)
            else "m"
            if value == int(mujoco.mjtJoint.mjJNT_SLIDE)
            else "native_joint_position"
            for value in joint_type_values
        )
        qvel_units = tuple(
            "rad/s"
            if value == int(mujoco.mjtJoint.mjJNT_HINGE)
            else "m/s"
            if value == int(mujoco.mjtJoint.mjJNT_SLIDE)
            else "native_joint_velocity"
            for value in joint_type_values
        )
        metadata = (joint_names, joint_type_names, qpos_units, qvel_units)
        prior = (
            self.robot_qpos_joint_names,
            self.robot_qpos_joint_types,
            self.robot_qpos_units,
            self.robot_qvel_units,
        )
        if self.robot_qpos_joint_names and prior != metadata:
            raise ValueError("robot joint schema changed within one rollout")
        self.robot_qpos_joint_names, self.robot_qpos_joint_types = joint_names, joint_type_names
        self.robot_qpos_units, self.robot_qvel_units = qpos_units, qvel_units
        self.sim_time.append(float(sim.data.time))
        self.robot_qpos.append(np.asarray(sim.data.qpos[pos_idx], dtype=float).copy())
        self.robot_qvel.append(np.asarray(sim.data.qvel[vel_idx], dtype=float).copy())
        self.actual_eef_pose.append(eef)
        self.object_poses.append(obj)
        self.fixture_poses.append(fixture)
        self.fixture_qpos.append(
            np.asarray(sim.data.qpos[list(self.fixture_qpos_indices)], dtype=float).copy()
        )
        self.fixture_qvel.append(
            np.asarray(sim.data.qvel[list(self.fixture_qvel_indices)], dtype=float).copy()
        )
        self.object_to_gripper_transform.append(np.linalg.inv(eef) @ obj)
        self.task_success.append(bool(env.is_success()["task"]))
        self.subtask_signals.append({str(k): bool(v) for k, v in info.subtask_term_signals.items()})
        model = getattr(sim.model, "_model", None)
        data = getattr(sim.data, "_data", None)
        if model is None or data is None or not hasattr(robot, "eef_site_id"):
            self.robot_contacts_json.append('"NOT_CHECKED"')
        else:
            site_id = int(robot.eef_site_id["right"])
            robot_root = int(model.body_rootid[model.site_bodyid[site_id]])
            contacts = []
            for index in range(data.ncon):
                contact = data.contact[index]
                geom_a, geom_b = int(contact.geom1), int(contact.geom2)
                body_a, body_b = int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b])
                if (
                    model.body_rootid[body_a] == robot_root
                    or model.body_rootid[body_b] == robot_root
                ):
                    contacts.append(
                        {
                            "geom_ids": [geom_a, geom_b],
                            "body_ids": [body_a, body_b],
                            "distance_m": float(contact.dist),
                        }
                    )
            self.robot_contacts_json.append(json.dumps(contacts, sort_keys=True))
        if current is not None:
            self.observation_object_poses.append(current_obj)
            self.observation_fixture_poses.append(current_fixture)
            self.observation_eef_poses.append(current_eef)
            self.observation_metadata.append(current_metadata)
        from cfact.generation.task_adapters import activation_state
        self.fixture_activation_json.append(json.dumps(activation_state(env), sort_keys=True))
        self._capture_failed = False

    def after_step(
        self,
        action: np.ndarray,
        env: Any,
        interface: Any,
        event: bool,
        *,
        target_pose: np.ndarray | None = None,
        raw_planned_action: np.ndarray | None = None,
        transition_reason: str = "follow_executable_waypoint",
    ) -> None:
        self.commanded_actions.append(np.asarray(action, dtype=float).copy())
        self.raw_planned_actions.append(
            np.asarray(
                action if raw_planned_action is None else raw_planned_action, dtype=float
            ).copy()
        )
        self.controller_target_poses.append(
            np.asarray(
                np.full((4, 4), np.nan) if target_pose is None else target_pose, dtype=float
            ).copy()
        )
        self.transition_reasons.append(transition_reason)
        self.capture(env, interface)
        self.local_event_flags.append(bool(event))
        if len(self.sim_time) != len(self.commanded_actions) + 1:
            raise AssertionError("K actions must have K+1 measured frames")

    def arrays(self) -> dict[str, np.ndarray]:
        if self._capture_failed:
            raise ValueError("failed_observation_capture")
        times = np.asarray(self.sim_time)
        refreshed = bool(self.observation_metadata)
        if refreshed and any(len(rows) != len(times) for rows in (
            self.observation_metadata, self.observation_object_poses,
            self.observation_fixture_poses, self.observation_eef_poses,
        )):
            raise ValueError("incomplete_current_observation_labels")
        objects = np.asarray(self.observation_object_poses if refreshed else self.object_poses)
        eef = np.asarray(self.observation_eef_poses if refreshed else self.actual_eef_pose)
        fixtures = np.asarray(self.observation_fixture_poses if refreshed else self.fixture_poses)
        relation = np.linalg.inv(eef) @ objects if refreshed else np.asarray(self.object_to_gripper_transform)
        velocity = np.full((len(times), 3), np.nan, dtype=float)
        if len(times) > 1:
            dt = np.diff(times)
            if np.any(dt <= 0):
                raise ValueError("non-increasing physical time")
            velocity[1:] = np.diff(objects[:, :3, 3], axis=0) / dt[:, None]
        return {
            "sim_time": times,
            "commanded_actions": np.asarray(self.commanded_actions),
            "raw_planned_actions": np.asarray(self.raw_planned_actions),
            "controller_target_poses": np.asarray(self.controller_target_poses),
            "controller_transition_reasons": np.asarray(self.transition_reasons),
            "controller_mode": np.asarray([a[-1] for a in self.commanded_actions]),
            "object_pose_names": np.asarray([self.object_ref]),
            "fixture_pose_names": np.asarray([self.fixture_ref]),
            "fixture_joint_names": np.asarray(self.fixture_joint_names),
            "fixture_joint_types": np.asarray(self.fixture_joint_types),
            "fixture_joint_units": np.asarray(self.fixture_joint_units),
            "fixture_joint_qvel_units": np.asarray(self.fixture_joint_qvel_units),
            "robot_qpos_joint_names": np.asarray(self.robot_qpos_joint_names),
            "robot_qpos_joint_types": np.asarray(self.robot_qpos_joint_types),
            "robot_qpos_units": np.asarray(self.robot_qpos_units),
            "robot_qvel_units": np.asarray(self.robot_qvel_units),
            "robot_qpos": np.asarray(self.robot_qpos),
            "robot_qvel": np.asarray(self.robot_qvel),
            "actual_eef_pose": eef,
            "object_poses": objects[:, None],
            "object_velocities": velocity[:, None],
            "fixture_poses": fixtures,
            "relevant_fixture_qpos": np.asarray(self.fixture_qpos),
            "relevant_fixture_qvel": np.asarray(self.fixture_qvel),
            "object_to_gripper_transform": relation,
            "observation_capture_json": np.asarray([json.dumps(row, sort_keys=True) for row in self.observation_metadata]),
            "execution_cache_object_poses": np.asarray(self.object_poses),
            "execution_cache_fixture_poses": np.asarray(self.fixture_poses),
            "execution_cache_eef_poses": np.asarray(self.actual_eef_pose),
            "task_success": np.asarray(self.task_success, dtype=bool),
            "subtask_signals_json": np.asarray(
                [json.dumps(row, sort_keys=True) for row in self.subtask_signals]
            ),
            "local_event_flags": np.asarray(self.local_event_flags, dtype=bool),
            "robot_contacts_json": np.asarray(self.robot_contacts_json),
            "fixture_activation_json": np.asarray(self.fixture_activation_json),
        }


@dataclass
class LocalEventDetector:
    """Object/handle subgoal in world metres, guarded by new motion.

    A stable relative EEF/object transform is required only for held transport.
    Endpoints already true in the parent are never considered a new event.
    """

    goal_xyz: np.ndarray
    initial_xyz: np.ndarray
    held: bool
    distance_m: float = 0.06
    minimum_motion_m: float = 0.02
    stable_steps: int = 3
    grip_slip_m: float = 0.03
    joint_goal: float | None = None
    joint_index: int | None = None
    joint_tolerance: float = 0.04
    _count: int = 0
    _grasp_count: int = 0
    _grasp_established: bool = False
    _grasp_reference_xyz: np.ndarray | None = None

    def update(self, trace: MeasuredTrace) -> tuple[bool, dict[str, float | bool]]:
        obj = trace.object_poses[-1]
        current = obj[:3, 3]
        distance = float(np.linalg.norm(current - self.goal_xyz))
        motion = float(np.linalg.norm(current - self.initial_xyz))
        relative_shift = 0.0
        if self.held and len(trace.object_to_gripper_transform) >= 2:
            step_shift = float(
                np.linalg.norm(
                    trace.object_to_gripper_transform[-1][:3, 3]
                    - trace.object_to_gripper_transform[-2][:3, 3]
                )
            )
            self._grasp_count = self._grasp_count + 1 if step_shift <= 0.005 else 0
            if self._grasp_count >= self.stable_steps and not self._grasp_established:
                self._grasp_established = True
                self._grasp_reference_xyz = trace.object_to_gripper_transform[-1][:3, 3].copy()
        if self._grasp_reference_xyz is not None:
            relative_shift = float(
                np.linalg.norm(
                    trace.object_to_gripper_transform[-1][:3, 3] - self._grasp_reference_xyz
                )
            )
        grip_valid = (
            not self.held or not self._grasp_established or relative_shift <= self.grip_slip_m
        )
        joint_error = None
        if self.joint_goal is not None:
            if self.joint_index is None or self.joint_index >= len(trace.fixture_qpos[-1]):
                raise ValueError("joint_subgoal_has_no_measured_fixture_joint")
            joint_error = abs(float(trace.fixture_qpos[-1][self.joint_index]) - self.joint_goal)
        if joint_error is not None and not self.held:
            # Named fixture qpos is the physical measure for articulated parts;
            # a hinge center or knob origin need not translate as it turns.
            actual_joint_motion = abs(
                float(trace.fixture_qpos[-1][self.joint_index])
                - float(trace.fixture_qpos[0][self.joint_index])
            )
            reached = joint_error <= self.joint_tolerance and actual_joint_motion > 1e-8
        else:
            reached = (
                distance <= self.distance_m
                and motion >= self.minimum_motion_m
                and grip_valid
                and (not self.held or self._grasp_established)
                and (joint_error is None or joint_error <= self.joint_tolerance)
            )
        self._count = self._count + 1 if reached else 0
        return self._count >= self.stable_steps, {
            "object_goal_distance_m": distance,
            "object_motion_m": motion,
            "eef_object_relative_shift_m": relative_shift,
            "contact_relation_valid": grip_valid,
            "grasp_established": self._grasp_established,
            "joint_goal_error": joint_error,
            "stable_steps": self._count,
            "threshold_m": self.distance_m,
        }


@dataclass
class ReleaseEventDetector:
    """A newly commanded opening followed by measured object/tool separation.

    This establishes a local release boundary. It does not assert contact with a
    support surface; official task success is recorded separately.
    """

    target_xyz: np.ndarray
    gripper_action_index: int
    target_tolerance_m: float = 0.10
    separation_m: float = 0.025
    object_speed_limit_m_s: float = 0.035
    stable_steps: int = 4
    _initial_relative_xyz: np.ndarray | None = None
    _open_seen: bool = False
    _count: int = 0

    def update(self, trace: MeasuredTrace) -> tuple[bool, dict[str, float | bool]]:
        if self._initial_relative_xyz is None:
            self._initial_relative_xyz = trace.object_to_gripper_transform[0][:3, 3].copy()
        action = trace.commanded_actions[-1]
        if self.gripper_action_index >= len(action):
            raise ValueError("gripper action index outside command")
        self._open_seen |= bool(action[self.gripper_action_index] < 0)
        separation = float(
            np.linalg.norm(
                trace.object_to_gripper_transform[-1][:3, 3] - self._initial_relative_xyz
            )
        )
        xyz = trace.object_poses[-1][:3, 3]
        goal_error = float(np.linalg.norm(xyz - self.target_xyz))
        if len(trace.object_poses) >= 2:
            dt = trace.sim_time[-1] - trace.sim_time[-2]
            speed = float(np.linalg.norm(xyz - trace.object_poses[-2][:3, 3]) / dt)
        else:
            speed = float("inf")
        reached = (
            self._open_seen
            and separation >= self.separation_m
            and goal_error <= self.target_tolerance_m
            and speed <= self.object_speed_limit_m_s
        )
        self._count = self._count + 1 if reached else 0
        return self._count >= self.stable_steps, {
            "opening_command_seen": self._open_seen,
            "object_eef_relative_separation_m": separation,
            "object_goal_distance_m": goal_error,
            "object_speed_m_s": speed,
            "stable_steps": self._count,
            "contact_relation_valid": True,
            "support_contact_directly_observed": False,
        }


class MeasuredArmContactGuard:
    """Stop after a measured arm/body penetration; preserve the triggering step.

    This guard covers robot bodies outside the gripper against scene and robot
    bodies at sampled control frames. It is not a continuous collision checker.
    Gripper contacts require a separate stage-specific policy and remain audited.
    """

    def __init__(self, env: Any, *, penetration_limit_m: float):
        if not 0 < penetration_limit_m <= 0.01:
            raise ValueError("invalid_arm_contact_guard_limit_m")
        model = env.env.sim.model._model
        robot = env.env.robots[0]
        site_id = int(robot.eef_site_id["right"])
        root = int(model.body_rootid[model.site_bodyid[site_id]])
        self.model = model
        self.arm_bodies = {
            body
            for body in range(model.nbody)
            if model.body_rootid[body] == root
            and "gripper" not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "")
        }
        self.penetration_limit_m = float(penetration_limit_m)

    def check(self, trace: MeasuredTrace) -> dict[str, Any] | None:
        contacts = json.loads(trace.robot_contacts_json[-1])
        if contacts == "NOT_CHECKED":
            return None
        violating = []
        for contact in contacts:
            body_ids = [int(value) for value in contact["body_ids"]]
            if (
                not any(body in self.arm_bodies for body in body_ids)
                or float(contact["distance_m"]) > -self.penetration_limit_m
            ):
                continue
            violating.append(
                {
                    **contact,
                    "body_names": [
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
                        for body in body_ids
                    ],
                }
            )
        if not violating:
            return None
        return {
            "scope": "measured_non_gripper_robot_body_contacts",
            "threshold_m": self.penetration_limit_m,
            "control_step": len(trace.commanded_actions),
            "sim_time_s": trace.sim_time[-1],
            "continuous_between_frames": "NOT_CHECKED",
            "contacts": sorted(violating, key=lambda row: row["distance_m"])[:8],
        }


def execute_option(
    env: Any,
    interface: Any,
    trajectory: Any,
    trace: MeasuredTrace,
    detector: LocalEventDetector,
    *,
    max_steps: int,
    frame_callback: Any = None,
    contact_guard: MeasuredArmContactGuard | None = None,
    source_action_indices: list[int] | None = None,
    source_episode: str | None = None,
    initial_source_cursor: int = 0,
) -> dict[str, Any]:
    """Use MimicGen waypoints/controller, stopping at the first measured event."""
    trace.capture(env, interface)
    start_time = trace.sim_time[0]
    event_values: dict[str, Any] = {}
    initial_satisfied = False
    if isinstance(detector, LocalEventDetector):
        initial_satisfied = (
            abs(float(trace.fixture_qpos[0][detector.joint_index]) - detector.joint_goal)
            <= detector.joint_tolerance
            if detector.joint_goal is not None and detector.joint_index is not None
            else float(np.linalg.norm(trace.object_poses[0][:3, 3] - detector.goal_xyz))
            <= detector.distance_m
        )
    guard_measurement: dict[str, Any] | None = None
    reason = "waypoints_exhausted_without_event"
    waypoint_count = 0
    for seq in trajectory.waypoint_sequences:
        for waypoint in seq:
            if len(trace.commanded_actions) >= max_steps:
                reason = "local_timeout"
                break
            if frame_callback is not None:
                frame_callback(env, trace.sim_time[-1], "local_operation")
            if waypoint.noise is not None and np.any(np.asarray(waypoint.noise) != 0):
                raise ValueError("execution disallows unseeded waypoint noise")
            action = interface.action_codec.command(waypoint.pose, waypoint.gripper_action)
            env.step(action)
            waypoint_count += 1
            # Capture first, then inspect the newly measured state.
            trace.after_step(
                action,
                env,
                interface,
                False,
                target_pose=np.asarray(waypoint.pose),
                raw_planned_action=action,
            )
            guard_measurement = contact_guard.check(trace) if contact_guard is not None else None
            if guard_measurement is not None:
                reason = "EXECUTION_ABORTED_BY_DECLARED_GUARD"
                event_values = {"contact_relation_valid": False, "guard_triggered": True}
                break
            reached, event_values = detector.update(trace)
            trace.local_event_flags[-1] = reached
            # The official predicate is authoritative when both events become
            # true on the same physical step. Keep the local event flag as
            # observed evidence, but do not offer a terminal child for recursion.
            if trace.task_success[-1]:
                reason = "official_task_success"
                break
            if reached:
                reason = "local_event_reached"
                break
            if not event_values["contact_relation_valid"]:
                reason = "contact_lost_after_operation_start"
                break
        if reason != "waypoints_exhausted_without_event":
            break
    if frame_callback is not None:
        frame_callback(env, trace.sim_time[-1], "child_boundary")
    issued_source = (
        []
        if source_action_indices is None
        else [int(index) for index in source_action_indices[:waypoint_count] if int(index) >= 0]
    )
    return {
        "termination_reason": reason,
        "event_measurement": event_values,
        "guard_measurement": guard_measurement,
        "local_event_reached": reason == "local_event_reached",
        "control_steps": len(trace.commanded_actions),
        "consumed_waypoints": waypoint_count,
        "initial_state_goal_satisfied": initial_satisfied,
        "source_cursor": {
            "episode_id": source_episode,
            "issued_waypoints": waypoint_count,
            "last_issued_action_index": (issued_source[-1] if issued_source else None),
            "next_action_index": (
                issued_source[-1] + 1 if issued_source else initial_source_cursor
            ),
        },
        "physical_progress": {
            "object_displacement_m": float(
                np.linalg.norm(trace.object_poses[-1][:3, 3] - trace.object_poses[0][:3, 3])
            ),
            "joint_qpos_initial": np.asarray(trace.fixture_qpos[0]).tolist(),
            "joint_qpos_final": np.asarray(trace.fixture_qpos[-1]).tolist(),
        },
        "event_action_index": (waypoint_count - 1 if any(trace.local_event_flags) else None),
        "event_state_index": (waypoint_count if any(trace.local_event_flags) else None),
        "plan_dispatch_finished": waypoint_count
        == sum(len(seq) for seq in trajectory.waypoint_sequences),
        "budget_terminated": reason == "local_timeout",
        "elapsed_sim_time_s": trace.sim_time[-1] - start_time,
        "official_success_at_child": trace.task_success[-1],
    }
