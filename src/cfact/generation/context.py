"""The shared operation and compiled-plan contract for every generator.

These records describe goals and source cursors, never measured physical
progress.  The execution record owns the latter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


OPERATION_SCHEMA = "cfact.operation_context.v1"
CANDIDATE_PLAN_SCHEMA = "cfact.candidate_plan.v1"


@dataclass(frozen=True)
class OperationContext:
    parent_context_id: str
    object_roles: dict[str, str]
    task_goal: dict[str, Any]
    stage: str
    source_cursor: int
    controller_semantics: dict[str, Any]
    current_fixture_pose: np.ndarray
    current_object_pose: np.ndarray
    current_eef_pose: np.ndarray
    joint_binding: dict[str, Any] = field(default_factory=dict)
    holding_object: bool = False
    schema: str = OPERATION_SCHEMA

    def __post_init__(self) -> None:
        if not self.parent_context_id or not self.stage or self.source_cursor < 0:
            raise ValueError("invalid_operation_context_identity")
        for name in ("current_fixture_pose", "current_object_pose", "current_eef_pose"):
            pose = np.array(getattr(self, name), dtype=float, copy=True)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"invalid_operation_context_{name}")
            pose.setflags(write=False)
            object.__setattr__(self, name, pose)
        goal_joint = self.task_goal.get("joint")
        bound_joint = self.joint_binding.get("joint", self.joint_binding.get("joint_name"))
        if goal_joint is not None and bound_joint is not None and goal_joint != bound_joint:
            raise ValueError("task_goal_joint_binding_mismatch")

    def record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_context_id": self.parent_context_id,
            "object_roles": dict(self.object_roles),
            "task_goal": dict(self.task_goal),
            "stage": self.stage,
            "source_cursor": self.source_cursor,
            "controller_semantics": dict(self.controller_semantics),
            "joint_binding": dict(self.joint_binding),
            "holding_object": self.holding_object,
            "current_fixture_pose": self.current_fixture_pose.tolist(),
            "current_object_pose": self.current_object_pose.tolist(),
            "current_eef_pose": self.current_eef_pose.tolist(),
        }


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    source_method: str
    operation_context: OperationContext
    poses: np.ndarray
    gripper_actions: np.ndarray
    source_action_indices: np.ndarray
    source_waypoint_indices: np.ndarray
    events: tuple[dict[str, Any], ...]
    edit: dict[str, Any]
    local_goal: dict[str, Any]
    task_goal: dict[str, Any]
    plan_difference: dict[str, Any]
    source_episode: str | None
    schema: str = CANDIDATE_PLAN_SCHEMA

    def record(self) -> dict[str, Any]:
        """JSON metadata; dense targets belong in the executable NPZ record."""
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "source_method": self.source_method,
            "operation_context": self.operation_context.record(),
            "source_episode": self.source_episode,
            "source_action_indices": self.source_action_indices.tolist(),
            "source_waypoint_indices": self.source_waypoint_indices.tolist(),
            "events": list(self.events),
            "edit": dict(self.edit),
            "local_goal": dict(self.local_goal),
            "task_goal": dict(self.task_goal),
            "plan_difference": dict(self.plan_difference),
            "physical_progress": "recorded_only_from_post_step_measurements",
            "compile_order": [
                "source_timeline",
                "parent_reanchor",
                "complete_task_goal_constraint",
                "candidate_edit",
                "shared_discrete_event_mapping",
            ] if self.edit.get("kind") != "effect_goal_compilation" else [
                "restored_current_parent", "propose_effect_goals", "compile_current_geometry",
                "shared_discrete_event_mapping"],
        }
