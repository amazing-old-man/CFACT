"""Action semantics for the locked robosuite OSC / hybrid mobile backend.

Targets are decoded from the *pre-action* pose and controller goal, never from
the observed next pose. The source command is not rewritten. Branch commands
use an explicit stationary-base, achieved-pose policy.
"""

from __future__ import annotations

import hashlib
import json
from types import MethodType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

CODEC_SCHEMA = "cfact.action_codec.v1"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


class ActionCodec:
    """Interpret the configured channels; encode only the arm when requested."""

    def __init__(self, interface: Any, arm: str = "right") -> None:
        self.interface = interface
        self.arm = arm

    @property
    def composite(self):
        # reset_to(model=...) replaces the controller objects, so never cache it.
        return self.interface.env.robots[0].composite_controller

    @property
    def controller(self):
        return self.composite.get_controller(self.arm)

    @property
    def channels(self) -> dict[str, tuple[int, int]]:
        return dict(self.composite._action_split_indexes)

    def semantics(self) -> dict[str, Any]:
        c = self.controller
        if getattr(c, "impedance_mode", "fixed") != "fixed" or not getattr(c, "use_ori", True):
            raise ValueError("action_codec_requires_locked_fixed_impedance_OSC_POSE")
        if c.input_type not in {"delta", "absolute"} or c.input_ref_frame not in {"base", "world"}:
            raise ValueError("unsupported_controller_modality")
        part_semantics = {}
        for name, part in getattr(self.composite, "part_controllers", {self.arm: c}).items():
            part_semantics[name] = {
                "class": f"{type(part).__module__}.{type(part).__name__}",
                **{key: getattr(part, key) for key in (
                    "control_dim", "input_type", "input_ref_frame", "input_min", "input_max",
                    "output_min", "output_max", "impedance_mode", "control_freq",
                ) if hasattr(part, key)},
            }
        grippers = {
            name: {"class": f"{type(gripper).__module__}.{type(gripper).__name__}",
                   **{key: getattr(gripper, key) for key in ("dof", "speed") if hasattr(gripper, key)}}
            for name, gripper in getattr(self.composite, "grippers", {}).items()
        }
        return _plain({
            "schema": CODEC_SCHEMA,
            "controller_class": f"{type(c).__module__}.{type(c).__name__}",
            "composite_class": f"{type(self.composite).__module__}.{type(self.composite).__name__}",
            "channels": self.channels,
            "parts": part_semantics, "grippers": grippers,
            "action_dimension": len(self.composite.action_limits[0]),
            "input_type": c.input_type,
            "input_ref_frame": c.input_ref_frame,
            "input_min": c.input_min, "input_max": c.input_max,
            "output_min": c.output_min, "output_max": c.output_max,
            "position_limits": c.position_limits, "orientation_limits": c.orientation_limits,
            "rotation_composition": "left_multiply_in_controller_frame",
            "clipping": "per_input_channel_before_affine_scaling_for_delta_only",
            "inverse_scope": "reachable_targets; clipped commands need not have a global inverse",
            "source_policy": "preserve_all_recorded_channels",
            "branch_policy": "stationary_base_torso_head; base_mode=-1; achieved_pose",
            "goal_update": "base_mode>0:desired; otherwise:achieved" if self.hybrid else getattr(c, "_goal_update_mode", "achieved"),
        })

    @property
    def hybrid(self) -> bool:
        return getattr(self.composite, "name", "") == "HYBRID_MOBILE_BASE"

    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.semantics(), sort_keys=True).encode()).hexdigest()

    def _frames(self):
        c = self.controller
        measured = np.asarray(self.interface.get_robot_eef_pose(), dtype=float)
        if c.input_ref_frame == "world":
            return measured, np.eye(4)
        # set_goal consumes the stored origin before run_controller refreshes it.
        # This matters for moving-base demonstrations and desired-goal mode.
        origin = np.eye(4)
        if c.origin_pos is None or c.origin_ori is None:
            raise ValueError("controller_origin_uninitialized")
        origin[:3, 3], origin[:3, :3] = c.origin_pos, c.origin_ori
        return np.linalg.inv(origin) @ measured, origin

    def _reference(self, action: np.ndarray | None = None):
        measured, origin = self._frames()
        desired = (self.hybrid and action is not None and len(action) > 6 and action[-1] > 0)
        if not self.hybrid:
            desired = getattr(self.controller, "_goal_update_mode", "achieved") == "desired"
        reference = measured.copy()
        if desired:
            if self.controller.goal_pos is not None:
                reference[:3, 3] = self.controller.goal_pos
            if self.controller.goal_ori is not None:
                reference[:3, :3] = self.controller.goal_ori
        return reference, origin

    def decode(self, action: np.ndarray) -> np.ndarray:
        """World-frame arm goal that set_goal will produce at this state."""
        self.semantics()
        a = np.asarray(action, dtype=float)
        start, end = self.channels[self.arm]
        arm = a if a.shape == (6,) else a[start:end]
        if arm.shape != (6,) or not np.isfinite(a).all():
            raise ValueError("invalid_action_channels")
        c = self.controller
        reference, origin = self._reference(a)
        goal = np.eye(4)
        if c.input_type == "absolute":
            goal[:3, 3] = arm[:3]
            goal[:3, :3] = Rotation.from_rotvec(arm[3:]).as_matrix()
        else:
            if c.position_limits is not None or np.array(c.orientation_limits).any():
                raise ValueError("backend_OSC_goal_limits_not_implemented")
            scale = np.abs(c.output_max - c.output_min) / np.abs(c.input_max - c.input_min)
            delta = (np.clip(arm, c.input_min, c.input_max) - (c.input_max + c.input_min) / 2) * scale
            delta += (c.output_max + c.output_min) / 2
            goal[:3, 3] = reference[:3, 3] + delta[:3]
            # Match the locked OSC's quaternion conversion precision as well
            # as its composition order; its axisangle2quat uses float32.
            from robosuite.utils import transform_utils

            delta_rotation = transform_utils.quat2mat(transform_utils.axisangle2quat(delta[3:]))
            goal[:3, :3] = delta_rotation @ reference[:3, :3]
        return origin @ goal

    def encode(self, target_pose: np.ndarray) -> np.ndarray:
        """Arm input for the branch policy, with clipping in input axes."""
        self.semantics()
        c = self.controller
        reference, origin = self._reference()
        target = np.asarray(target_pose, dtype=float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("invalid_target_pose")
        target = np.linalg.inv(origin) @ target
        if c.input_type == "absolute":
            return np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_rotvec()]
        delta = np.r_[target[:3, 3] - reference[:3, 3],
                      Rotation.from_matrix(target[:3, :3] @ reference[:3, :3].T).as_rotvec()]
        scale = np.abs(c.output_max - c.output_min) / np.abs(c.input_max - c.input_min)
        if np.any(scale == 0):
            raise ValueError("noninvertible_action_scale")
        command = (delta - (c.output_max + c.output_min) / 2) / scale + (c.input_max + c.input_min) / 2
        return np.clip(command, c.input_min, c.input_max)

    def gripper(self, action: np.ndarray) -> np.ndarray:
        key = f"{self.arm}_gripper"
        if key not in self.channels:
            raise ValueError("missing_gripper_channel")
        start, end = self.channels[key]
        return np.asarray(action)[start:end].copy()

    def command(self, target_pose: np.ndarray, gripper: np.ndarray) -> np.ndarray:
        """Assemble every channel with the documented branch base/mode policy."""
        return self.composite.create_action_vector({
            self.arm: self.encode(target_pose), f"{self.arm}_gripper": np.asarray(gripper),
            "base_mode": -1,
        })


def install_action_codec(interface: Any) -> Any:
    """Attach CFACT-owned methods without modifying upstream classes/modules."""
    codec = ActionCodec(interface)
    interface.action_codec = codec

    def decode(self, action, relative=None):
        return self.action_codec.decode(action)

    def encode(self, target_pose, relative=None):
        return self.action_codec.encode(target_pose)

    def gripper(self, action):
        return self.action_codec.gripper(action)

    interface.action_to_target_pose = MethodType(decode, interface)
    interface.target_pose_to_action = MethodType(encode, interface)
    interface.action_to_gripper_action = MethodType(gripper, interface)
    return interface
