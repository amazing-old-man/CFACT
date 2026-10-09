"""Isolated MuJoCo kinematic audit for an OSC tool-target path.

This module never changes the physical simulator data. A reported geometric pass
only describes the IK branch in this copied world, not the OSC execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cfact.execution.connector import BoundedJointConnector, ConnectorBudget


@dataclass(frozen=True)
class KinematicLimits:
    position_tolerance_m: float = 0.003
    rotation_tolerance_rad: float = 0.035
    joint_limit_margin_rad: float = 0.002
    maximum_joint_step_rad: float = 0.18
    interpolation_resolution_rad: float = 0.035
    maximum_ik_iterations: int = 35
    damping: float = 0.025


def _pose_error(target: np.ndarray, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.r_[
        target[:3, 3] - position, Rotation.from_matrix(target[:3, :3] @ rotation.T).as_rotvec()
    ]


class KinematicPathValidator:
    """Check a single worker's current scene with a private MjData.

    Allowed contacts use exact body pairs supplied for the operation stage.
    Other nonrobot scene contacts are reported but do not veto a robot path.
    """

    def __init__(
        self,
        env: Any,
        *,
        allowed_body_pairs: set[tuple[int, int]],
        limits: KinematicLimits = KinematicLimits(),
        held_object_body_id: int | None = None,
        allow_inherited_held_contacts: bool = False,
    ):
        sim = env.env.sim
        self.model = sim.model._model
        self.data = mujoco.MjData(self.model)
        physical = sim.data._data
        self.data.qpos[:] = physical.qpos
        self.data.qvel[:] = physical.qvel
        if self.model.nmocap:
            self.data.mocap_pos[:] = physical.mocap_pos
            self.data.mocap_quat[:] = physical.mocap_quat
        self.data.time = physical.time
        mujoco.mj_forward(self.model, self.data)
        robot = env.env.robots[0]
        self.qpos_indices = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        self.dof_indices = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
        self.site_id = int(robot.eef_site_id["right"])
        self.robot_root = int(self.model.body_rootid[self.model.site_bodyid[self.site_id]])
        self.allowed_body_pairs = {tuple(sorted(map(int, pair))) for pair in allowed_body_pairs}
        self.limits = limits
        self.joint_ids = np.asarray(
            [
                next(j for j in range(self.model.njnt) if self.model.jnt_qposadr[j] == address)
                for address in self.qpos_indices
            ],
            dtype=int,
        )
        if len(self.qpos_indices) != 7 or len(set(self.dof_indices)) != 7:
            raise ValueError("unsupported_arm_joint_mapping")
        self.lower = self.model.jnt_range[self.joint_ids, 0] + limits.joint_limit_margin_rad
        self.upper = self.model.jnt_range[self.joint_ids, 1] - limits.joint_limit_margin_rad
        if not np.all(self.model.jnt_limited[self.joint_ids]):
            raise ValueError("unbounded_arm_joint")
        self.snapshot_qpos = np.asarray(physical.qpos).copy()
        self.snapshot_time = float(physical.time)
        self.initial_q = np.asarray(physical.qpos[self.qpos_indices]).copy()
        self.held_object_root = None
        self.held_object_qpos_address = None
        self.attachment_active = False
        self.attachment_transform = None
        self.release_waypoint = None
        if held_object_body_id is not None:
            body = int(held_object_body_id)
            if self.model.body_jntnum[body] != 1:
                raise ValueError("held_object_requires_single_free_joint")
            joint = int(self.model.body_jntadr[body])
            if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("held_object_requires_free_joint")
            self.held_object_root = int(self.model.body_rootid[body])
            self.held_object_qpos_address = int(self.model.jnt_qposadr[joint])
            # MuJoCo's derived xpos/site_xpos can lag qpos after env.step.
            # Calibrate both frames from the same forward-updated private data.
            site_position = np.asarray(self.data.site_xpos[self.site_id]).copy()
            site_rotation = np.asarray(self.data.site_xmat[self.site_id]).reshape(3, 3).copy()
            object_position = np.asarray(self.data.xpos[body]).copy()
            object_rotation = np.asarray(self.data.xmat[body]).reshape(3, 3).copy()
            if not np.allclose(
                object_position,
                self.data.qpos[self.held_object_qpos_address : self.held_object_qpos_address + 3],
                atol=1e-7,
            ):
                raise ValueError("held_object_free_joint_frame_mismatch")
            site_pose = np.eye(4)
            site_pose[:3, :3] = site_rotation
            site_pose[:3, 3] = site_position
            object_pose = np.eye(4)
            object_pose[:3, :3] = object_rotation
            object_pose[:3, 3] = object_position
            self.attachment_transform = np.linalg.inv(site_pose) @ object_pose
            self.attachment_active = True
        # During a first join, an already held object may still touch its
        # departure surface. Authorize only exact, shallow parent body pairs,
        # and never allow their penetration to deepen by more than 0.1 mm.
        # Arm/gripper contacts are never inherited by this rule.
        self.inherited_held_contact_bounds: dict[tuple[int, int], float] = {}
        if allow_inherited_held_contacts and self.held_object_root is not None:
            contacts: dict[tuple[int, int], float] = {}
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                a = int(self.model.geom_bodyid[contact.geom1])
                b = int(self.model.geom_bodyid[contact.geom2])
                ar = int(self.model.body_rootid[a])
                br = int(self.model.body_rootid[b])
                if (
                    ar == self.held_object_root
                    and br not in {self.robot_root, self.held_object_root}
                ) or (
                    br == self.held_object_root
                    and ar not in {self.robot_root, self.held_object_root}
                ):
                    pair = tuple(sorted((a, b)))
                    contacts[pair] = min(contacts.get(pair, 0.0), float(contact.dist))
            self.inherited_held_contact_bounds = {
                pair: distance for pair, distance in contacts.items() if -0.001 <= distance <= 0.0
            }
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def _set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_indices] = q
        mujoco.mj_forward(self.model, self.data)
        if self.attachment_active:
            site_position, site_rotation = self._tcp()
            site_pose = np.eye(4)
            site_pose[:3, :3] = site_rotation
            site_pose[:3, 3] = site_position
            object_pose = site_pose @ self.attachment_transform
            address = self.held_object_qpos_address
            self.data.qpos[address : address + 3] = object_pose[:3, 3]
            xyzw = Rotation.from_matrix(object_pose[:3, :3]).as_quat()
            self.data.qpos[address + 3 : address + 7] = np.r_[xyzw[3], xyzw[:3]]
            mujoco.mj_forward(self.model, self.data)

    def _tcp(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.data.site_xpos[self.site_id]).copy(),
            np.asarray(self.data.site_xmat[self.site_id]).reshape(3, 3).copy(),
        )

    def _contacts(self) -> list[dict[str, Any]]:
        bad = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a = int(self.model.geom_bodyid[contact.geom1])
            b = int(self.model.geom_bodyid[contact.geom2])
            relevant = (
                self.model.body_rootid[a] == self.robot_root
                or self.model.body_rootid[b] == self.robot_root
                or (
                    self.attachment_active
                    and self.held_object_root
                    in (self.model.body_rootid[a], self.model.body_rootid[b])
                )
            )
            pair = tuple(sorted((a, b)))
            inherited = self.inherited_held_contact_bounds.get(pair)
            if (
                not relevant
                or pair in self.allowed_body_pairs
                or (inherited is not None and contact.dist >= inherited - 0.0001)
            ):
                continue
            bad.append(
                {
                    "geom_ids": [int(contact.geom1), int(contact.geom2)],
                    "body_ids": [a, b],
                    "distance_m": float(contact.dist),
                    "body_names": [
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, a),
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b),
                    ],
                }
            )
        return bad

    def _solve(
        self, target: np.ndarray, start: np.ndarray
    ) -> tuple[np.ndarray | None, dict[str, float | str | int]]:
        q = start.copy()
        limit = self.limits
        best = np.inf
        for iteration in range(limit.maximum_ik_iterations):
            self._set_q(q)
            position, rotation = self._tcp()
            error = _pose_error(target, position, rotation)
            ep = float(np.linalg.norm(error[:3]))
            er = float(np.linalg.norm(error[3:]))
            if ep <= limit.position_tolerance_m and er <= limit.rotation_tolerance_rad:
                return q.copy(), {
                    "status": "IK_REACHED",
                    "iterations": iteration,
                    "position_error_m": ep,
                    "rotation_error_rad": er,
                }
            weighted = np.r_[error[:3], error[3:] * 0.35]
            merit = float(np.linalg.norm(weighted))
            if merit < best:
                best = merit
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            jac = np.vstack(
                (self._jacp[:, self.dof_indices], self._jacr[:, self.dof_indices] * 0.35)
            )
            delta = jac.T @ np.linalg.solve(jac @ jac.T + limit.damping**2 * np.eye(6), weighted)
            delta = np.clip(delta, -0.12, 0.12)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                candidate = np.clip(q + scale * delta, self.lower, self.upper)
                self._set_q(candidate)
                pos2, rot2 = self._tcp()
                err2 = _pose_error(target, pos2, rot2)
                if np.linalg.norm(np.r_[err2[:3], err2[3:] * 0.35]) < merit:
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                break
        self._set_q(q)
        pos, rot = self._tcp()
        error = _pose_error(target, pos, rot)
        return None, {
            "status": "IK_NO_SOLUTION_WITHIN_BUDGET",
            "iterations": iteration + 1,
            "position_error_m": float(np.linalg.norm(error[:3])),
            "rotation_error_rad": float(np.linalg.norm(error[3:])),
        }

    def validate(
        self, poses: np.ndarray, gripper_actions: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        targets = np.asarray(poses, dtype=float)
        if targets.ndim != 3 or targets.shape[1:] != (4, 4) or not np.isfinite(targets).all():
            raise ValueError("invalid_tool_target_path")
        if self.held_object_root is not None:
            grips = np.asarray(gripper_actions)
            if grips.ndim != 2 or len(grips) != len(targets) or grips.shape[1] < 1:
                raise ValueError("held_path_requires_aligned_gripper_actions")
        else:
            grips = None
        rows = []
        qpath = [self.initial_q.copy()]
        initial_contacts = self._contacts()
        status = "GEOMETRIC_PASS"
        for index, target in enumerate(targets[1:], start=1):
            solution, residual = self._solve(target, qpath[-1])
            if solution is None:
                status = "IK_NO_SOLUTION_WITHIN_BUDGET"
                rows.append({"waypoint": index, **residual})
                break
            difference = float(np.max(np.abs(solution - qpath[-1])))
            if difference > self.limits.maximum_joint_step_rad:
                status = "JOINT_STEP_LIMIT"
                rows.append(
                    {
                        "waypoint": index,
                        "maximum_joint_delta_rad": difference,
                        **residual,
                        "reason": "JOINT_STEP_LIMIT",
                    }
                )
                break
            subdivisions = max(
                1, int(np.ceil(difference / self.limits.interpolation_resolution_rad))
            )
            collision = None
            for part in range(1, subdivisions + 1):
                fraction = part / subdivisions
                q = qpath[-1] * (1 - fraction) + solution * fraction
                self._set_q(q)
                collision = self._contacts()
                if collision:
                    break
            if collision:
                status = "ROBOT_CONTACT_NOT_ALLOWED"
                rows.append(
                    {
                        "waypoint": index,
                        "subdivision": part,
                        "subdivisions": subdivisions,
                        "contacts": collision[:8],
                        **residual,
                        "reason": "ROBOT_CONTACT_NOT_ALLOWED",
                    }
                )
                break
            qpath.append(solution)
            if self.attachment_active and grips is not None and grips[index, 0] < 0:
                self.attachment_active = False
                self.release_waypoint = index
            rows.append(
                {
                    "waypoint": index,
                    "maximum_joint_delta_rad": difference,
                    "subdivisions": subdivisions,
                    **residual,
                }
            )
        report = {
            "backend": "mujoco_kinematic_copy",
            "backend_version": mujoco.__version__,
            "solver_status": status,
            "evaluated_waypoints": len(rows),
            "total_waypoints": len(targets),
            "initial_robot_contacts_not_allowed": initial_contacts[:8],
            "allowed_body_pairs": [list(pair) for pair in sorted(self.allowed_body_pairs)],
            "robot_root_body_id": self.robot_root,
            "tcp_site_id": self.site_id,
            "arm_qpos_indices": self.qpos_indices.tolist(),
            "arm_dof_indices": self.dof_indices.tolist(),
            "interpolation_resolution_rad": self.limits.interpolation_resolution_rad,
            "limits": vars(self.limits),
            "failures": [x for x in rows if x.get("status") != "IK_REACHED" or x.get("reason")],
            "maximum_position_residual_m": max(
                (float(x.get("position_error_m", 0)) for x in rows), default=0
            ),
            "maximum_rotation_residual_rad": max(
                (float(x.get("rotation_error_rad", 0)) for x in rows), default=0
            ),
            "control_mapping": "OSC tracks Cartesian targets; joint path is not commanded",
            "executed_joint_path_equivalent": False,
            "dynamic_validity": "NOT_CHECKED",
            "continuous_collision": "NOT_CHECKED",
            "held_object_attachment": (
                "KINEMATIC_FREE_JOINT_COPY"
                if self.held_object_root is not None
                else "NOT_APPLICABLE"
            ),
            "held_object_root_body_id": self.held_object_root,
            "held_object_release_waypoint": self.release_waypoint,
            "held_object_initial_tool_transform": (
                self.attachment_transform.tolist()
                if self.attachment_transform is not None
                else None
            ),
            "post_release_object_dynamics": "NOT_CHECKED",
        }
        return np.asarray(qpath), report

    def diagnose_first_join(
        self, target_pose: np.ndarray, *, budget: ConnectorBudget = ConnectorBudget(), seed: int = 0
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Search only the first free-space join in the copied current world.

        The returned q path is a diagnostic. OSC does not command these joints;
        callers must not treat it as an executed or validated Cartesian plan.
        """
        goal, ik = self._solve(np.asarray(target_pose, dtype=float), self.initial_q)
        if goal is None:
            return None, {
                "backend": "bounded_bidirectional_joint_rrt_connect",
                "status": "GOAL_IK_NO_SOLUTION_WITHIN_BUDGET",
                "goal_ik": ik,
                "seed": int(seed),
                "executed_joint_path_equivalent": False,
                "continuous_collision": "NOT_CHECKED",
            }
        if np.any(self.initial_q < self.lower) or np.any(self.initial_q > self.upper):
            return None, {
                "backend": "bounded_bidirectional_joint_rrt_connect",
                "status": "PARENT_OUTSIDE_CONSERVATIVE_JOINT_MARGIN",
                "initial_arm_qpos": self.initial_q.tolist(),
                "joint_lower_with_margin": self.lower.tolist(),
                "joint_upper_with_margin": self.upper.tolist(),
                "seed": int(seed),
                "executed_joint_path_equivalent": False,
                "continuous_collision": "NOT_CHECKED",
            }
        self._set_q(self.initial_q)
        start_contacts = self._contacts()
        if start_contacts:
            return None, {
                "backend": "bounded_bidirectional_joint_rrt_connect",
                "status": "START_CONTACT_NOT_ALLOWED",
                "contacts": start_contacts[:8],
                "seed": int(seed),
                "executed_joint_path_equivalent": False,
                "continuous_collision": "NOT_CHECKED",
            }
        self._set_q(goal)
        goal_contacts = self._contacts()
        if goal_contacts:
            return None, {
                "backend": "bounded_bidirectional_joint_rrt_connect",
                "status": "GOAL_CONTACT_NOT_ALLOWED",
                "contacts": goal_contacts[:8],
                "seed": int(seed),
                "executed_joint_path_equivalent": False,
                "continuous_collision": "NOT_CHECKED",
            }

        def valid(q: np.ndarray) -> bool:
            self._set_q(q)
            return not self._contacts()

        connector = BoundedJointConnector(self.lower, self.upper, valid, budget=budget, seed=seed)
        path, report = connector.plan(self.initial_q, goal)
        report.update(
            {
                "goal_ik": ik,
                "allowed_body_pairs": [list(pair) for pair in sorted(self.allowed_body_pairs)],
                "held_object_attachment": (
                    "KINEMATIC_FREE_JOINT_COPY"
                    if self.held_object_root is not None
                    else "NOT_APPLICABLE"
                ),
                "executed_joint_path_equivalent": False,
                "scope": "first free-space join in current frozen world only",
                "inherited_held_parent_contacts": [
                    {
                        "body_ids": list(pair),
                        "parent_distance_m": distance,
                        "maximum_extra_penetration_m": 0.0001,
                    }
                    for pair, distance in sorted(self.inherited_held_contact_bounds.items())
                ],
            }
        )
        return path, report


def observed_gripper_target_pairs(env: Any) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    """Authorize the measured gripper/object body contact at this parent only.

    This derives target bodies from current contacts. If no current target contact
    exists, it grants no contact rather than widening permission to a fixture.
    """
    model = env.env.sim.model._model
    data = env.env.sim.data._data
    site_id = int(env.env.robots[0].eef_site_id["right"])
    robot_root = int(model.body_rootid[model.site_bodyid[site_id]])
    gripper_bodies = {
        body
        for body in range(model.nbody)
        if model.body_rootid[body] == robot_root
        and "gripper" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "")
    }
    target_bodies = set()
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        a = int(model.geom_bodyid[contact.geom1])
        b = int(model.geom_bodyid[contact.geom2])
        if a in gripper_bodies and model.body_rootid[b] != robot_root:
            target_bodies.add(b)
            contacts.append([a, b])
        elif b in gripper_bodies and model.body_rootid[a] != robot_root:
            target_bodies.add(a)
            contacts.append([b, a])
    pairs = {
        tuple(sorted((gripper, target))) for gripper in gripper_bodies for target in target_bodies
    }
    audit = {
        "observed_gripper_target_body_contacts": contacts,
        "authorized_target_bodies": [
            {"id": body, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)}
            for body in sorted(target_bodies)
        ],
        "gripper_body_count": len(gripper_bodies),
        "policy": "only gripper bodies against exactly observed target bodies",
    }
    return pairs, audit
