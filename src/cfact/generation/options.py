"""Outcome-blind G0/G1/G2 proposals for the counterfactual action trees.

The module is deliberately independent of MuJoCo.  It turns a frozen source
fragment plus the *current* parent geometry into executable waypoint targets;
the existing candidate-pool runner is still responsible for restoration and
``env.step``.  This keeps proposal construction inspectable and prevents a
generator from probing a branch's future before it is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from cfact.generation.segments import (
    SourceSegment,
    segment_timeline,
    select_compatible_donor,
    transform_segment,
)
from cfact.generation.context import CandidatePlan, OperationContext


@dataclass(frozen=True)
class OperationContract:
    """Stage-local permissions, in metres/radians, fixed before execution."""

    operation_type: str
    primitive_family: str
    phase: str
    object_roles: dict[str, str]
    holding_object: bool
    allows_gripper_timing: bool
    allowed_g2_subtypes: tuple[str, ...]


def operation_contract(
    *,
    operation_type: str,
    primitive_family: str,
    event_object_ref: str,
    operation_object_ref: str,
    holding_object: bool,
) -> OperationContract:
    """Return a conservative contract from current, not future, context."""

    is_placement = primitive_family == "placement"
    is_contact = primitive_family in {"linear_joint", "rotary_joint", "lever", "knob", "button"}
    phase = (
        "held_transport"
        if holding_object and is_placement
        else ("constrained_contact" if is_contact else "approach")
    )
    # A transport segment never gains a deliberately early opening.  An
    # independently timed gripper rule is only permitted for explicit grasp or
    # support/release contracts, neither of which this suffix adapter infers.
    return OperationContract(
        operation_type=operation_type,
        primitive_family=primitive_family,
        phase=phase,
        object_roles={"event": event_object_ref, "operation": operation_object_ref},
        holding_object=holding_object,
        allows_gripper_timing=False,
        allowed_g2_subtypes=(
            ("entry", "goal")
            if is_placement
            else ("entry", "contact")
            if is_contact
            else ("entry",)
        ),
    )


def _smooth_weights(count: int, *, power: float = 1.0) -> np.ndarray:
    if count < 3:
        raise ValueError("operation segment is too short for an option")
    return np.sin(np.linspace(0.0, np.pi, count)) ** power


def _local_axis(object_pose: np.ndarray, axis: int) -> np.ndarray:
    vector = np.asarray(object_pose, dtype=float)[:3, :3][:, axis]
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("operation_frame_has_invalid_axis")
    return vector / norm


@dataclass(frozen=True)
class OptionProposal:
    candidate_id: str
    generator_revision: str
    source_method: str
    subtype: str
    parent_context_id: str
    operation_type: str
    decision_group_id: str
    source_episode: str | None
    source_fragment: tuple[int, int] | None
    object_roles: dict[str, str]
    reference_frame: str
    current_contact_mode: str
    intended_object_path: np.ndarray | None
    intended_fixture_subgoal: dict[str, Any]
    entry_constraints: dict[str, Any]
    support_constraints: dict[str, Any]
    task_constraints: dict[str, Any]
    gripper_event_policy: dict[str, Any]
    goal_binding: dict[str, Any]
    motion_template: str
    connection_plan: dict[str, Any]
    control_limits: dict[str, Any]
    end_event: str
    local_timeout: int
    precondition_checks: dict[str, str]
    poses: np.ndarray
    gripper_actions: np.ndarray
    mode_descriptor: dict[str, Any] | None = None
    provenance: tuple[str, ...] = ()
    rejection_reason: str | None = None
    operation_context: OperationContext | None = None
    candidate_plan: CandidatePlan | None = None
    candidate_edit: dict[str, Any] = field(default_factory=dict)
    source_action_indices: np.ndarray | None = None
    plan_difference: dict[str, Any] = field(default_factory=dict)

    @property
    def executable(self) -> bool:
        return self.rejection_reason is None and all(
            value in {"PASS", "NOT_APPLICABLE"} for value in self.precondition_checks.values()
        )


def _stable_option_id(
    parent: str, method: str, subtype: str, poses: np.ndarray, goal: dict[str, Any], ordinal: int
) -> str:
    import hashlib
    import json

    digest = hashlib.sha256()
    digest.update(parent.encode())
    digest.update(method.encode())
    digest.update(subtype.encode())
    digest.update(np.asarray(poses, dtype="<f8").tobytes())
    digest.update(json.dumps(goal, sort_keys=True).encode())
    return f"o_{ordinal:02d}_{digest.hexdigest()[:12]}"


def _discrete_time_indices(values: np.ndarray, length: int) -> np.ndarray:
    """One time map, holding commands and preserving every discrete change."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) < 1 or length < 1:
        raise ValueError("empty_gripper_timeline")
    samples = np.linspace(0, len(values) - 1, length)
    indices = np.floor(samples).astype(int)
    events = np.flatnonzero(np.any(values[1:] != values[:-1], axis=1)) + 1
    if len(events) + 1 > length:
        raise ValueError("time_map_cannot_preserve_all_gripper_events")
    # Each source event has a separate slot. Moving an event later by less
    # than one target interval is allowed; synthesizing an intermediate
    # controller command or dropping an event is not.
    prior = 0
    for ordinal, event in enumerate(events):
        remaining = len(events) - ordinal - 1
        slot = min(length - 1 - remaining, max(prior + 1, int(np.searchsorted(samples, event))))
        indices[slot] = int(event)
        prior = slot
    indices[-1] = len(values) - 1
    # Preserve assigned event order even when several changes share an
    # interval in a compressed source timeline.
    indices = np.maximum.accumulate(indices)
    selected = values[indices]
    expected = values[np.concatenate(([0], events))]
    observed = selected[np.concatenate(([0], np.flatnonzero(
        np.any(selected[1:] != selected[:-1], axis=1)
    ) + 1))]
    if not np.array_equal(expected, observed):
        raise ValueError("time_map_cannot_preserve_all_gripper_events")
    return indices


def _resample_gripper(values: np.ndarray, length: int) -> np.ndarray:
    return np.asarray(values[_discrete_time_indices(values, length)], dtype=float)


def _remap_source_indices(option: OptionProposal, length: int) -> np.ndarray:
    mapping = _discrete_time_indices(option.gripper_actions, length)
    if option.source_action_indices is not None:
        return np.asarray(option.source_action_indices, dtype=int)[mapping].copy()
    start = 0 if option.source_fragment is None else option.source_fragment[0]
    return (start + mapping).astype(int)


def compile_candidate_plan(
    option: OptionProposal,
    context: OperationContext | None = None,
) -> CandidatePlan:
    """The common final compiler used by proposals and physical execution.

    Geometry is already reanchored and constrained before candidate edits.
    This validates the shared command timeline and emits the source and event
    mapping consumed by the executor; it never infers physical progress.
    """
    context = context or option.operation_context
    if context is None:
        raise ValueError("operation_context_required_for_candidate_compilation")
    poses = np.array(option.poses, dtype=float, copy=True)
    grips = np.array(option.gripper_actions, dtype=float, copy=True)
    if grips.ndim == 1:
        grips = grips[:, None]
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0
            or len(grips) != len(poses) or grips.ndim != 2
            or not np.isfinite(poses).all() or not np.isfinite(grips).all()):
        raise ValueError("invalid_candidate_command_timeline")
    indices = _remap_source_indices(option, len(poses))
    if len(indices) != len(poses):
        raise ValueError("candidate_source_index_timeline_mismatch")
    events = tuple({
        "kind": "gripper_command_change",
        "candidate_waypoint_index": int(index),
        "source_action_index": int(indices[index]),
        "command": grips[index].tolist(),
        "diagnostic_state": "after_action",
    } for index in np.flatnonzero(np.any(grips[1:] != grips[:-1], axis=1)) + 1)
    task_goal = dict(context.task_goal)
    if option.goal_binding.get("q_task") is not None:
        if task_goal.get("q_task") is not None and not np.isclose(
            float(task_goal["q_task"]), float(option.goal_binding["q_task"]), atol=1e-12, rtol=0
        ):
            raise ValueError("candidate_complete_task_goal_changed")
        task_goal.update({key: option.goal_binding[key] for key in
                          ("joint", "q_task", "joint_unit") if key in option.goal_binding})
    return CandidatePlan(
        candidate_id=option.candidate_id,
        source_method=option.source_method,
        operation_context=context,
        poses=poses,
        gripper_actions=grips,
        source_action_indices=indices,
        source_waypoint_indices=np.arange(len(poses), dtype=int),
        events=events,
        edit=dict(option.candidate_edit),
        local_goal=dict(option.goal_binding),
        task_goal=task_goal,
        plan_difference=dict(option.plan_difference),
        source_episode=option.source_episode,
    )


def _compiled(option: OptionProposal) -> OptionProposal:
    return replace(option, candidate_plan=compile_candidate_plan(option))


def _object_motion_to_eef(
    source_motion: np.ndarray,
    source_fixture: np.ndarray,
    current_fixture: np.ndarray,
    current_object: np.ndarray,
    current_eef: np.ndarray,
    *,
    held: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Redirect measured object/handle motion and preserve the current contact transform.

    All inputs and outputs are world-frame homogeneous transforms. This is a
    planning assumption for one local skill; the executor measures any slip.
    """
    source_motion = np.asarray(source_motion, dtype=float)
    if source_motion.ndim != 3 or source_motion.shape[1:] != (4, 4):
        raise ValueError("source_object_motion_unavailable")
    if held:
        # Fixture-local path, translated to coincide with the actual object at
        # the parent. This preserves donor path shape without changing grasp.
        source_local = np.einsum("ij,njk->nik", np.linalg.inv(source_fixture), source_motion)
        current_local = np.linalg.inv(current_fixture) @ current_object
        alignment = current_local @ np.linalg.inv(source_local[0])
        desired = np.einsum("ij,njk->nik", current_fixture @ alignment, source_local)
    else:
        # Handle articulation is represented by its actual source handle path;
        # the current handle frame sets the initial contact point.
        alignment = current_object @ np.linalg.inv(source_motion[0])
        desired = np.einsum("ij,njk->nik", alignment, source_motion)
    object_to_eef = np.linalg.inv(current_object) @ current_eef
    eef = np.einsum("nij,jk->nik", desired, object_to_eef)
    return desired, eef


def compile_held_path_with_release(
    object_path: np.ndarray,
    source_eef_path: np.ndarray,
    gripper_actions: np.ndarray,
    current_object_pose: np.ndarray,
    current_eef_pose: np.ndarray,
) -> np.ndarray:
    """Follow the object while held, then follow the recorded tool retreat.

    The grasp relation is a planning assumption only until the first opening
    command. Afterwards the tool path is aligned to that release pose and can
    move away from a stationary object. Physics decides whether release works.
    """
    object_path = np.asarray(object_path, dtype=float)
    source_eef_path = np.asarray(source_eef_path, dtype=float)
    gripper_actions = np.asarray(gripper_actions, dtype=float).reshape(-1)
    if (
        object_path.ndim != 3
        or object_path.shape[1:] != (4, 4)
        or source_eef_path.ndim != 3
        or source_eef_path.shape[1:] != (4, 4)
        or len(object_path) < 2
        or len(source_eef_path) < 2
        or len(source_eef_path) != len(object_path)
        or len(gripper_actions) != len(object_path)
    ):
        raise ValueError("invalid_held_transport_and_release_path")
    relation = np.linalg.inv(np.asarray(current_object_pose)) @ np.asarray(current_eef_pose)
    tools = np.einsum("nij,jk->nik", object_path, relation)
    openings = np.flatnonzero(gripper_actions < 0)
    if not len(openings):
        return tools
    release = int(openings[0])
    retreat = source_eef_path
    alignment = tools[release] @ np.linalg.inv(retreat[release])
    tools[release:] = np.einsum("ij,njk->nik", alignment, retreat[release:])
    return tools


def placement_shift_weights(gripper_actions: np.ndarray, length: int) -> np.ndarray:
    """Finish a placement goal shift before opening, then retain the goal."""
    grip = np.asarray(gripper_actions).reshape(-1)
    if len(grip) != length:
        raise ValueError("gripper_schedule_path_length_mismatch")
    openings = np.flatnonzero(grip < 0)
    endpoint = int(openings[0]) if len(openings) else length - 1
    weights = np.ones(length, dtype=float)
    weights[: endpoint + 1] = np.linspace(0.0, 1.0, endpoint + 1)
    return weights


def _bind_placement_path(
    path: np.ndarray, gripper: np.ndarray, goal: dict[str, Any]
) -> np.ndarray:
    path = np.array(path, dtype=float, copy=True)
    if "target_pose_world" in goal:
        target = np.asarray(goal["target_pose_world"], dtype=float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("invalid_complete_placement_goal")
        shift = target[:3, 3] - path[-1, :3, 3]
        path[:, :3, 3] += placement_shift_weights(gripper, len(path))[:, None] * shift
    return path


def _g0_weights(gripper: np.ndarray, *, held: bool) -> tuple[np.ndarray, str | None]:
    """Keep a sine excursion in transport, with fixed entry and release."""
    length = len(gripper)
    openings = np.flatnonzero(np.asarray(gripper).reshape(length, -1)[:, 0] < 0)
    endpoint = int(openings[0]) if held and len(openings) else length - 1
    weights = np.zeros(length, dtype=float)
    if endpoint < 2:
        return weights, "transport_too_short_under_fixed_entry_and_release_constraints"
    weights[:endpoint + 1] = _smooth_weights(endpoint + 1)
    weights[0] = weights[endpoint] = 0.0
    return weights, None


def _compile_segment_basis(
    source: SourceSegment,
    context: OperationContext,
    official_transform: Callable[..., np.ndarray],
    *,
    object_motion_mode: bool = False,
    static_object: bool = False,
) -> tuple[SourceSegment, np.ndarray, np.ndarray, np.ndarray | None]:
    """Shared source timeline, reanchor and complete-goal constraint pass."""
    source_poses, grips, motion, indices = segment_timeline(source)
    source = replace(source, source_poses=source_poses, gripper_actions=grips,
                     motion_object_poses=motion, source_action_indices=indices)
    poses = transform_segment(source, context.current_fixture_pose, official_transform)
    object_path = None
    if static_object:
        object_path = np.repeat(context.current_object_pose[None], len(poses), axis=0)
    elif motion is not None:
        object_path, object_tools = _object_motion_to_eef(
            motion, source.object_pose_at_entry, context.current_fixture_pose,
            context.current_object_pose, context.current_eef_pose, held=context.holding_object,
        )
        if object_motion_mode and not context.holding_object:
            poses = object_tools
    if context.holding_object and object_path is not None:
        object_path = _bind_placement_path(object_path, grips, context.task_goal)
        poses = compile_held_path_with_release(
            object_path, poses, grips, context.current_object_pose, context.current_eef_pose
        )
    return source, poses, grips, object_path


def bind_inherited_placement_goal(
    options: list[OptionProposal], context: OperationContext
) -> list[OptionProposal]:
    """Apply an inherited task goal once, preserving each candidate edit.

    New callers pass context before proposal construction. This also provides
    a safe boundary for callers binding a goal after construction: translate
    both planned object and tool together, so a G0 transport edit survives.
    """
    result = []
    for ordinal, option in enumerate(options):
        if option.source_method == "G2" or option.intended_object_path is None:
            continue
        path = _bind_placement_path(option.intended_object_path, option.gripper_actions,
                                    context.task_goal)
        delta = path[:, :3, 3] - option.intended_object_path[:, :3, 3]
        poses = option.poses.copy()
        poses[:, :3, 3] += delta
        updated = replace(
            option, poses=poses, intended_object_path=path,
            goal_binding=dict(context.task_goal), operation_context=context,
            candidate_id=_stable_option_id(option.parent_context_id, option.source_method,
                                           option.subtype, poses, context.task_goal, ordinal),
        )
        result.append(_compiled(updated))
    return result


def build_object_effect_options(
    *,
    root: SourceSegment,
    donors: Iterable[SourceSegment],
    current_fixture_pose: np.ndarray,
    current_object_pose: np.ndarray,
    current_eef_pose: np.ndarray,
    contract: OperationContract,
    official_transform: Callable[..., np.ndarray],
    parent_context_id: str,
    end_event: str,
    local_timeout: int = 160,
    max_raw_per_generator: int = 12,
    compatibility_limits: dict[str, float] | None = None,
    operation_context: OperationContext | None = None,
    g0_amplitude_m: float = 0.003,
) -> tuple[list[OptionProposal], list[dict[str, Any]]]:
    """Produce variable-length, outcome-blind object-effect options.

    Geometric support validation requires task-specific fixture bounds. Until
    provided, placement goal variants are marked UNSUPPORTED and never execute.
    """
    if max_raw_per_generator < 1 or local_timeout < 1:
        raise ValueError("invalid candidate budget")
    context = operation_context or OperationContext(
        parent_context_id=parent_context_id,
        object_roles=dict(contract.object_roles),
        task_goal={"kind": "source_task_goal", "persistent": True},
        stage=contract.phase,
        source_cursor=root.event_first_step,
        controller_semantics={"gripper": root.gripper_semantics},
        current_fixture_pose=current_fixture_pose,
        current_object_pose=current_object_pose,
        current_eef_pose=current_eef_pose,
        holding_object=contract.holding_object,
    )
    if context.parent_context_id != parent_context_id:
        raise ValueError("proposal_parent_context_mismatch")
    if context.holding_object != contract.holding_object:
        raise ValueError("proposal_current_holding_state_mismatch")
    current_fixture_pose = context.current_fixture_pose
    current_object_pose = context.current_object_pose
    current_eef_pose = context.current_eef_pose
    held = contract.holding_object
    options: list[OptionProposal] = []
    audit: list[dict[str, Any]] = []
    root, reference_poses, reference_gripper, reference_obj = _compile_segment_basis(
        root, context, official_transform
    )
    common = dict(
        generator_revision="shared_operation_compiler_v1",
        parent_context_id=parent_context_id,
        operation_type=contract.operation_type,
        decision_group_id=f"{parent_context_id}/{contract.phase}",
        object_roles={**contract.object_roles, "motion": root.motion_object_ref or "unknown"},
        reference_frame="world_from_fixture_local",
        current_contact_mode=contract.phase,
        intended_fixture_subgoal={},
        entry_constraints={"frame": "fixture", "phase": contract.phase},
        support_constraints={},
        task_constraints={"official_task_unchanged": True},
        gripper_event_policy={"kind": "source_schedule", "independent_timing": False},
        motion_template="official_mimicgen_waypoints",
        connection_plan={"kind": "controller_interpolation"},
        control_limits={"base_actions": [0, 0, 0, 0], "mode_action": -1},
        end_event=end_event,
        local_timeout=local_timeout,
        operation_context=context,
    )

    def add(
        method: str,
        subtype: str,
        source: SourceSegment | None,
        poses: np.ndarray,
        gripper: np.ndarray,
        objpath: np.ndarray | None,
        goal: dict[str, Any],
        checks: dict[str, str],
        reason: str | None = None,
        provenance: tuple[str, ...] = (),
        edit: dict[str, Any] | None = None,
        difference: dict[str, Any] | None = None,
    ) -> None:
        index = len(options)
        option = OptionProposal(
                candidate_id=_stable_option_id(
                    parent_context_id, method, subtype, poses, goal, index
                ),
                source_method=method,
                subtype=subtype,
                source_episode=None if source is None else source.episode_id,
                source_fragment=None
                if source is None
                else (source.event_first_step, source.end_step),
                intended_object_path=None if objpath is None else np.asarray(objpath).copy(),
                goal_binding=goal,
                precondition_checks=checks,
                poses=np.asarray(poses).copy(),
                gripper_actions=np.asarray(gripper).copy(),
                provenance=provenance,
                mode_descriptor=(
                    {
                        "kind": subtype,
                        "source_episode": source.episode_id,
                        "measured_object_path_waypoints": None
                        if source.motion_object_poses is None
                        else len(source.motion_object_poses),
                        "source_fixture_frame": contract.object_roles["operation"],
                    }
                    if source is not None
                    else {"kind": subtype}
                ),
                rejection_reason=reason,
                candidate_edit=dict(edit or {"kind": "none"}),
                plan_difference=dict(difference or {}),
                source_action_indices=(None if source is None else
                    segment_timeline(source)[3][
                        _discrete_time_indices(source.gripper_actions, len(poses))
                    ]),
                **common,
            )
        options.append(_compiled(option))

    add(
        "Reference",
        "source_reference",
        root,
        reference_poses,
        reference_gripper,
        reference_obj,
        dict(context.task_goal),
        {
            "finite_targets": "PASS",
            "object_motion_source": "PASS" if reference_obj is not None else "NOT_APPLICABLE",
        },
        provenance=(root.episode_id,),
    )
    g0_axes = (0, 1) if held else (0,)
    for axis in g0_axes:
        if sum(o.source_method == "G0" for o in options) >= min(2, max_raw_per_generator):
            break
        poses = reference_poses.copy()
        if held:
            direction = _local_axis(current_fixture_pose, axis)
        else:
            movement = reference_poses[-1, :3, 3] - reference_poses[0, :3, 3]
            norm = np.linalg.norm(movement)
            direction = movement / norm if norm >= 0.01 else _local_axis(current_fixture_pose, axis)
        weights, degeneracy = _g0_weights(reference_gripper, held=held)
        offsets = weights[:, None] * float(g0_amplitude_m) * direction
        poses[:, :3, 3] += offsets
        object_path = None if reference_obj is None else reference_obj.copy()
        if held and object_path is not None:
            object_path[:, :3, 3] += offsets
        difference = {
            "metric": "compiled_target_translation_from_reference",
            "requested_amplitude_m": float(g0_amplitude_m),
            "maximum_translation_m": float(np.linalg.norm(offsets, axis=1).max()),
            "transport_maximum_translation_m": float(np.linalg.norm(offsets, axis=1).max()),
            "retreat_maximum_translation_m": 0.0 if held else None,
            "degeneracy_reason": degeneracy,
            "actual_command_difference": "measured_separately_after_execution",
        }
        add(
            "G0",
            "local_translation",
            root,
            poses,
            reference_gripper,
            object_path,
            dict(context.task_goal),
            {"finite_targets": "PASS", "gripper_window": "NOT_APPLICABLE"},
            provenance=(root.episode_id,),
            edit={"kind": "smooth_local_excursion", "coordinate_frame": "fixture_local",
                  "axis": axis, "direction_world": direction.tolist(),
                  "amplitude_m": float(g0_amplitude_m), "shape": "sine",
                  "fixed_boundaries": "entry_and_release" if held else "entry_and_exit",
                  "applied_after": "complete_task_goal_constraint"},
            difference=difference,
        )
    selected, donor_audit = select_diverse_object_modes(
        root, donors, limit=max_raw_per_generator, compatibility_limits=compatibility_limits
    )
    audit.extend(donor_audit)
    for donor in selected:
        if donor.motion_object_poses is None:
            audit.append(
                {"donor_id": donor.episode_id, "status": "UNSUPPORTED", "reason": "eef_only_source"}
            )
            continue
        try:
            donor, eef_path, gripper, object_path = _compile_segment_basis(
                donor, context, official_transform, object_motion_mode=True,
                static_object=contract.primitive_family == "button",
            )
            if held and float(gripper[0, 0]) <= 0:
                raise ValueError("donor_entry_command_incompatible_with_current_held_stage")
        except (ValueError, np.linalg.LinAlgError) as error:
            audit.append(
                {
                    "donor_id": donor.episode_id,
                    "status": "REJECT",
                    "reason": f"object_motion_transform:{error}",
                }
            )
            continue
        # Compatibility was checked before choosing a donor. Its geometry,
        # commands and release/retreat share exactly the donor's time map.
        add(
            "G1",
            "object_motion_mode" if held else "handle_motion_mode",
            donor,
            eef_path,
            gripper,
            object_path,
            dict(context.task_goal),
            {
                "source_object_motion": "PASS",
                "current_contact_transform": "PASS",
                "finite_targets": "PASS",
            },
            provenance=(donor.episode_id,),
            edit={"kind": "compatible_donor_segment", "motion_and_events": "same_source_timeline",
                  "current_grasp_relation": "preserved", "donor_episode": donor.episode_id},
        )
    # Buttons have static geom poses. Their G2 path is the source-derived
    # press axis with the frozen 4 mm local depth excursion and zero endpoint
    # offset from the modern button protocol.
    if contract.primitive_family == "button":
        reference = next((o for o in options if o.source_method == "Reference"), None)
        if reference is not None and len(reference.poses) >= 3:
            motion = np.diff(reference.poses[:, :3, 3], axis=0)
            lengths = np.linalg.norm(motion, axis=1)
            usable = np.flatnonzero(lengths > 1e-9)
            if len(usable):
                press_axis = motion[int(usable[0])] / lengths[int(usable[0])]
                weights = np.sin(np.linspace(0.0, np.pi, len(reference.poses)))
                poses = reference.poses.copy()
                poses[:, :3, 3] += weights[:, None] * 0.004 * press_axis
                object_path = np.repeat(current_object_pose[None], len(poses), axis=0)
                goal = {
                    "kind": "button_press_depth_delta",
                    "persistent": True,
                    "button_ref": contract.object_roles["event"],
                    "press_axis_world": press_axis.round(9).tolist(),
                    "press_depth_delta_m": 0.004,
                    "endpoint_offset_m": 0.0,
                }
                add(
                    "G2",
                    "button_press_depth_delta",
                    root,
                    poses,
                    _resample_gripper(root.gripper_actions, len(poses)),
                    object_path,
                    goal,
                    {"source_press_axis": "PASS", "official_button_protocol": "PASS"},
                    provenance=(root.episode_id, "modern_candidate_protocol_v1"),
                    edit={"kind": "button_press_depth_delta", "coordinate_frame": "world",
                          "press_axis_world": press_axis.tolist(), "press_depth_delta_m": 0.004,
                          "endpoint_offset_m": 0.0},
                )
            else:
                audit.append(
                    {
                        "status": "UNSUPPORTED",
                        "reason": "source_press_axis_unobserved",
                        "candidate_id": "G2_button_press_depth_delta",
                    }
                )
    # G2 draws candidate goals from observed source endpoints in fixture
    # coordinates. A task geometry adapter must supply support and clearance
    # checks before these are eligible for physical execution.
    if held and reference_obj is not None:
        for donor in selected[:max_raw_per_generator]:
            if donor.motion_object_poses is None:
                continue
            try:
                donor_poses, donor_grips, donor_motion, donor_indices = segment_timeline(donor)
                donor = replace(donor, source_poses=donor_poses, gripper_actions=donor_grips,
                                motion_object_poses=donor_motion,
                                source_action_indices=donor_indices)
                object_path, eef_path = _object_motion_to_eef(
                    donor.motion_object_poses,
                    donor.object_pose_at_entry,
                    current_fixture_pose,
                    current_object_pose,
                    current_eef_pose,
                    held=True,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            goal = {
                "kind": "fixture_relative_object_goal",
                "persistent": True,
                "source_episode": donor.episode_id,
                "target_pose_world": object_path[-1].round(9).tolist(),
            }
            gripper = donor.gripper_actions.copy()
            eef_path = compile_held_path_with_release(
                object_path, transform_segment(donor, current_fixture_pose, official_transform),
                gripper, current_object_pose, current_eef_pose
            )
            add(
                "G2",
                "observed_fixture_goal",
                donor,
                eef_path,
                gripper,
                object_path,
                goal,
                {"source_goal_geometry": "PASS", "support_and_clearance": "UNSUPPORTED"},
                "fixture_support_geometry_adapter_unavailable",
                (donor.episode_id,),
                edit={"kind": "observed_fixture_goal", "coordinate_frame": "fixture_local"},
            )
    return options, audit


def select_option_pool(
    options: Iterable[OptionProposal], *, max_executed: int = 9, max_g0: int = 2
) -> tuple[list[OptionProposal], list[dict[str, Any]]]:
    """Coverage selection using only frozen plans, goals and provenance."""
    import hashlib

    if max_executed < 1:
        raise ValueError("max_executed must be positive")
    accepted: list[OptionProposal] = []
    rejected: list[dict[str, Any]] = []
    unique: dict[str, OptionProposal] = {}
    for option in options:
        if not option.executable:
            rejected.append(
                {
                    "candidate_id": option.candidate_id,
                    "reason": option.rejection_reason or "precondition_not_pass",
                }
            )
            continue
        digest = hashlib.sha256(
            np.asarray(option.poses, dtype="<f8").tobytes()
            + np.asarray(option.gripper_actions, dtype="<f8").tobytes()
            + repr(sorted(option.goal_binding.items())).encode()
        ).hexdigest()
        if digest in unique:
            rejected.append(
                {
                    "candidate_id": option.candidate_id,
                    "reason": "exact_preexecution_duplicate",
                    "canonical_id": unique[digest].candidate_id,
                }
            )
            continue
        unique[digest] = option
    groups = {
        method: [o for o in unique.values() if o.source_method == method]
        for method in ("Reference", "G0", "G1", "G2")
    }
    # One representative of every nonempty source first. Remaining slots use
    # plan-space separation; method labels are only a deterministic tie break.
    for method in groups:
        if groups[method] and len(accepted) < max_executed:
            accepted.append(groups[method].pop(0))
    while len(accepted) < max_executed:
        remaining = [
            o
            for method, rows in groups.items()
            for o in rows
            if method != "G0" or sum(a.source_method == "G0" for a in accepted) < max_g0
        ]
        if not remaining:
            break

        def separation(option: OptionProposal) -> tuple[float, str]:
            endpoint = (
                option.intended_object_path[-1, :3, 3]
                if option.intended_object_path is not None
                else option.poses[-1, :3, 3]
            )
            dist = (
                min(
                    float(
                        np.linalg.norm(
                            endpoint
                            - (
                                a.intended_object_path[-1, :3, 3]
                                if a.intended_object_path is not None
                                else a.poses[-1, :3, 3]
                            )
                        )
                    )
                    for a in accepted
                )
                if accepted
                else 0.0
            )
            return (-dist, option.candidate_id)

        chosen = min(remaining, key=separation)
        accepted.append(chosen)
        groups[chosen.source_method].remove(chosen)
    for rows in groups.values():
        for option in rows:
            rejected.append({"candidate_id": option.candidate_id, "reason": "pool_budget"})
    return accepted, rejected


def select_child_frontier(
    eligible: Sequence[tuple[OptionProposal, dict[str, Any]]],
    *,
    family_priority: Sequence[str],
    maximum: int,
) -> list[tuple[OptionProposal, dict[str, Any]]]:
    """Choose actual event children with fixed source-family coverage.

    Eligibility is decided by the physical event guard before this call. The
    ranking reads only frozen generator labels and proposal order; neither
    fixed-tail success nor child state quality is a ranking input. This avoids
    spending both child-pool slots on the first two proposal families.
    """
    if maximum < 0:
        raise ValueError("maximum child pools must be nonnegative")
    if len(set(family_priority)) != len(family_priority):
        raise ValueError("frontier family priority has duplicate labels")
    rows = list(eligible)
    selected: list[tuple[OptionProposal, dict[str, Any]]] = []
    selected_ids: set[str] = set()
    for family in family_priority:
        for option, edge in rows:
            if option.source_method == family and option.candidate_id not in selected_ids:
                selected.append((option, edge))
                selected_ids.add(option.candidate_id)
                break
        if len(selected) >= maximum:
            return selected[:maximum]
    for option, edge in rows:
        if option.candidate_id not in selected_ids:
            selected.append((option, edge))
            selected_ids.add(option.candidate_id)
            if len(selected) >= maximum:
                break
    return selected[:maximum]


def solve_g2_support_goals(
    options: list[OptionProposal],
    *,
    support_regions: list[dict[str, Any]],
    object_bbox_local: np.ndarray,
    current_object_pose: np.ndarray,
    current_eef_pose: np.ndarray,
    maximum: int = 6,
) -> tuple[list[OptionProposal], list[dict[str, Any]]]:
    """Sample legal fixture-local placement goals using the object's full footprint.

    Region poses are world-from-region transforms in metres. These are static
    geometry and contact-contract checks, not a physical success guarantee.
    """
    import dataclasses

    reference = next(
        (
            o
            for o in options
            if o.source_method == "Reference" and o.intended_object_path is not None
        ),
        None,
    )
    if reference is None:
        return options, [{"status": "UNSUPPORTED", "reason": "reference_object_path_missing"}]
    bbox = np.asarray(object_bbox_local, dtype=float)
    if bbox.ndim != 2 or bbox.shape[1] != 3 or len(bbox) < 4:
        return options, [{"status": "UNSUPPORTED", "reason": "object_bbox_missing"}]
    base_path = np.asarray(reference.intended_object_path)
    endpoint = base_path[-1]
    relative_tool = np.linalg.inv(np.asarray(current_object_pose)) @ np.asarray(current_eef_pose)
    records: list[dict[str, Any]] = []
    produced: list[OptionProposal] = []
    for region in support_regions:
        region_pose = np.asarray(region["pose"], dtype=float)
        size = np.asarray(region["size_m"], dtype=float)
        local_corners = (
            np.linalg.inv(region_pose)
            @ np.column_stack((bbox @ endpoint[:3, :3].T + endpoint[:3, 3], np.ones(len(bbox)))).T
        ).T[:, :3]
        extent = np.max(
            np.abs(local_corners[:, :2] - (np.linalg.inv(region_pose) @ endpoint)[:2, 3]), axis=0
        )
        free = size / 2 - extent - 0.01
        if np.any(free <= 0):
            records.append(
                {
                    "region": region["name"],
                    "status": "REJECT",
                    "reason": "object_footprint_exceeds_support_region",
                }
            )
            continue
        for fraction in (-0.5, 0.0, 0.5):
            if len(produced) >= maximum:
                break
            desired_local = np.linalg.inv(region_pose) @ endpoint
            desired_local[:2, 3] = [fraction * free[0], 0.0]
            target = region_pose @ desired_local
            world_corners = bbox @ target[:3, :3].T + target[:3, 3]
            corners_local = (
                np.linalg.inv(region_pose) @ np.column_stack((world_corners, np.ones(len(bbox)))).T
            ).T[:, :3]
            footprint_ok = bool(np.all(np.abs(corners_local[:, :2]) <= size / 2 - 0.005))
            bottom_z = float(corners_local[:, 2].min())
            support_z_ok = abs(bottom_z) <= 0.06
            tool = target @ relative_tool
            tool_local = np.linalg.inv(region_pose) @ tool
            tool_clearance_ok = bool(np.all(np.abs(tool_local[:2, 3]) <= size / 2 + 0.03))
            if not (footprint_ok and support_z_ok and tool_clearance_ok):
                records.append(
                    {
                        "region": region["name"],
                        "fraction": fraction,
                        "status": "REJECT",
                        "footprint_ok": footprint_ok,
                        "support_z_ok": support_z_ok,
                        "tool_clearance_ok": tool_clearance_ok,
                        "object_bottom_relative_region_m": bottom_z,
                    }
                )
                continue
            shift = target[:3, 3] - endpoint[:3, 3]
            if np.linalg.norm(shift) < 0.005:
                records.append(
                    {
                        "region": region["name"],
                        "fraction": fraction,
                        "status": "REJECT",
                        "reason": "same_as_reference_goal",
                    }
                )
                continue
            path = base_path.copy()
            gripper = _resample_gripper(reference.gripper_actions, len(path))
            weights = placement_shift_weights(gripper, len(path))
            path[:, :3, 3] += weights[:, None] * shift
            poses = compile_held_path_with_release(
                path, reference.poses, gripper, current_object_pose, current_eef_pose
            )
            goal = {
                "kind": "fixture_region_object_goal",
                "persistent": True,
                "region": region["name"],
                "target_pose_world": target.round(9).tolist(),
                "support_frame": "region",
                "support_margin_m": 0.005,
            }
            ordinal = len(options) + len(produced)
            produced.append(
                _compiled(dataclasses.replace(
                    reference,
                    candidate_id=_stable_option_id(
                        reference.parent_context_id,
                        "G2",
                        "support_region_goal",
                        poses,
                        goal,
                        ordinal,
                    ),
                    source_method="G2",
                    subtype="support_region_goal",
                    intended_object_path=path,
                    poses=poses,
                    gripper_actions=gripper,
                    source_action_indices=_remap_source_indices(reference, len(path)),
                    candidate_edit={"kind": "placement_local_subgoal", "frame": "support_region"},
                    source_episode=reference.source_episode,
                    source_fragment=reference.source_fragment,
                    intended_fixture_subgoal={
                        "region": region["name"],
                        "object_bottom_relative_region_m": bottom_z,
                    },
                    support_constraints={
                        "footprint_within_region": True,
                        "support_height_tolerance_m": 0.06,
                        "tool_point_clearance_m": 0.03,
                    },
                    goal_binding=goal,
                    precondition_checks={
                        "object_footprint": "PASS",
                        "support_height": "PASS",
                        "tool_point_clearance": "PASS",
                        "finite_targets": "PASS",
                    },
                    provenance=(reference.source_episode or "source", region["name"]),
                ))
            )
            records.append(
                {
                    "region": region["name"],
                    "fraction": fraction,
                    "status": "PASS",
                    "candidate_id": produced[-1].candidate_id,
                    "object_bottom_relative_region_m": bottom_z,
                }
            )
    return [o for o in options if o.source_method != "G2"] + produced, records


def solve_g2_articulation_subgoals(
    options: list[OptionProposal],
    *,
    joint_name: str,
    joint_type: str,
    current_q: float,
    joint_range: tuple[float, float],
    axis_world: np.ndarray,
    anchor_world: np.ndarray,
    current_handle_pose: np.ndarray,
    current_eef_pose: np.ndarray,
    target_q: float | None = None,
    fractions: tuple[float, ...] = (0.25, 0.45, 0.65),
) -> tuple[list[OptionProposal], list[dict[str, Any]]]:
    """Plan handle/tool paths for intermediate joint targets; physics moves q.

    This does not set simulator qpos. The joint target is a planned subgoal and
    subsequent execution must measure actual joint and handle motion.
    """
    import dataclasses

    reference = next((o for o in options if o.source_method == "Reference"), None)
    if reference is None or joint_type not in {"slide", "hinge"}:
        return options, [{"status": "UNSUPPORTED", "reason": "joint_contract_missing"}]
    axis = np.array(axis_world, dtype=float, copy=True)
    anchor = np.asarray(anchor_world, dtype=float)
    lower, upper = map(float, joint_range)
    if (axis.shape != (3,) or anchor.shape != (3,)
            or not np.isfinite([lower, upper, current_q, *axis, *anchor]).all()
            or lower >= upper or np.linalg.norm(axis) < 1e-8):
        return options, [{"status": "REJECT", "reason": "invalid_joint_geometry"}]
    # MuJoCo joint limits are soft constraints: a measured parent may sit
    # outside the nominal interval. Preserve that measured start unchanged;
    # only the proposed subgoal and complete task target must be in range.
    axis /= np.linalg.norm(axis)
    context = reference.operation_context
    contextual_goal = {} if context is None else context.task_goal
    explicit_goal = target_q if target_q is not None else contextual_goal.get("q_task")
    if explicit_goal is None:
        return options, [{"status": "REJECT", "reason": "complete_task_joint_goal_missing"}]
    q_task = float(explicit_goal)
    if contextual_goal.get("q_task") is not None and not np.isclose(
        q_task, float(contextual_goal["q_task"]), atol=1e-12, rtol=0
    ):
        return options, [{"status": "REJECT", "reason": "complete_task_joint_goal_mismatch"}]
    if not lower <= q_task <= upper:
        return options, [{"status": "REJECT", "reason": "task_target_outside_joint_range"}]
    if abs(q_task - current_q) < 0.02:
        return options, [{"status": "INITIAL_GOAL_SATISFIED",
                          "reason": "no_task_joint_range_remaining",
                          "initial_goal_satisfied": True, "actual_new_motion": False,
                          "q_task": q_task, "q_parent": float(current_q)}]
    handle = np.asarray(current_handle_pose, dtype=float)
    eef = np.asarray(current_eef_pose, dtype=float)
    handle_to_eef = np.linalg.inv(handle) @ eef
    result = []
    audit = []
    for ordinal, fraction in enumerate(fractions):
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            audit.append({"fraction": fraction, "status": "REJECT", "reason": "invalid_fraction"})
            continue
        desired_q = current_q + float(fraction) * (q_task - current_q)
        if not lower <= desired_q <= upper:
            audit.append(
                {"fraction": fraction, "status": "REJECT", "reason": "outside_joint_range"}
            )
            continue
        q_path = np.linspace(current_q, desired_q, 24)
        handle_path = np.repeat(handle[None], len(q_path), axis=0)
        for index, q in enumerate(q_path):
            delta = q - current_q
            if joint_type == "slide":
                handle_path[index, :3, 3] = handle[:3, 3] + axis * delta
            else:
                skew = np.array(
                    [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
                )
                rot = np.eye(3) + np.sin(delta) * skew + (1 - np.cos(delta)) * (skew @ skew)
                handle_path[index, :3, :3] = rot @ handle[:3, :3]
                handle_path[index, :3, 3] = anchor + rot @ (handle[:3, 3] - anchor)
        poses = np.einsum("nij,jk->nik", handle_path, handle_to_eef)
        maximum_step = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).max()
        if maximum_step > 0.05:
            audit.append(
                {
                    "fraction": fraction,
                    "status": "REJECT",
                    "reason": "tool_step_exceeds_local_limit",
                    "maximum_step_m": float(maximum_step),
                }
            )
            continue
        # An intermediate joint goal consumes the corresponding source
        # prefix. Mapping the entire source here would both release contact
        # too early and falsely mark the continuation as fully dispatched.
        source_end = max(1, int(round(float(fraction) * (len(reference.gripper_actions) - 1))))
        source_end = min(source_end, len(reference.gripper_actions) - 1)
        source_gripper = reference.gripper_actions[:source_end + 1]
        time_indices = _discrete_time_indices(source_gripper, len(poses))
        gripper = np.asarray(source_gripper[time_indices]).copy()
        source_indices = _remap_source_indices(reference, len(reference.gripper_actions))[
            :source_end + 1
        ][time_indices]
        goal = {
            "kind": "articulation_local_subgoal",
            "persistent": True,
            "joint": joint_name,
            "target_q": desired_q,
            "q_parent": float(current_q),
            "q_sub": float(desired_q),
            "q_task": q_task,
            "fraction": float(fraction),
            "joint_unit": "m" if joint_type == "slide" else "rad",
        }
        option = dataclasses.replace(
            reference,
            candidate_id=_stable_option_id(
                reference.parent_context_id,
                "G2",
                "articulation_subgoal",
                poses,
                goal,
                len(options) + ordinal,
            ),
            source_method="G2",
            subtype="articulation_subgoal",
            source_episode=reference.source_episode,
            source_fragment=reference.source_fragment,
            intended_object_path=handle_path,
            intended_fixture_subgoal={
                "joint": joint_name,
                "current_q": current_q,
                "desired_q": desired_q,
                "q_task": q_task,
                "q_sub": float(desired_q),
                "fraction": float(fraction),
                "range": [lower, upper],
                "axis_world": axis.tolist(),
                "anchor_world_m": anchor.tolist(),
            },
            goal_binding=goal,
            poses=poses,
            gripper_actions=gripper,
            source_action_indices=source_indices,
            candidate_edit={"kind": "bound_joint_local_subgoal", "joint": joint_name,
                            "q_parent": float(current_q), "q_sub": float(desired_q),
                            "q_task": q_task, "fraction": float(fraction),
                            "source_prefix_end_waypoint": source_end},
            precondition_checks={
                "joint_range": "PASS",
                "joint_axis": "PASS",
                "tool_handle_relation": "PASS",
                "local_step_bound": "PASS",
            },
            provenance=(joint_name,),
        )
        result.append(_compiled(option))
        audit.append(
            {
                "fraction": fraction,
                "status": "PASS",
                "candidate_id": option.candidate_id,
                "desired_q": desired_q,
                "q_task": q_task,
                "q_parent": float(current_q),
                "planned_direction": float(np.sign(q_task - current_q)),
            }
        )
    return [o for o in options if o.source_method != "G2"] + result, audit


def solve_g2_coffee_receptacle_goals(
    options: list[OptionProposal],
    *,
    pour_site_world_m: np.ndarray,
    current_object_pose: np.ndarray,
    current_eef_pose: np.ndarray,
) -> tuple[list[OptionProposal], list[dict[str, Any]]]:
    """Choose mug centres inside RoboCasa's real 4 cm pour-site predicate."""
    import dataclasses

    reference = next(
        (
            x
            for x in options
            if x.source_method == "Reference" and x.intended_object_path is not None
        ),
        None,
    )
    if reference is None:
        return options, [{"status": "UNSUPPORTED", "reason": "source_mug_path_missing"}]
    site = np.asarray(pour_site_world_m, dtype=float)
    base = reference.intended_object_path
    endpoint = base[-1]
    if abs(endpoint[2, 3] - site[2]) >= 0.10:
        return options, [{"status": "REJECT", "reason": "source_target_outside_receptacle_z"}]
    generated = []
    audit = []
    for index, offset in enumerate(((-0.015, 0.0), (0.015, 0.0), (0.0, -0.015), (0.0, 0.015))):
        target = endpoint.copy()
        target[0, 3] = site[0] + offset[0]
        target[1, 3] = site[1] + offset[1]
        if np.linalg.norm(target[:2, 3] - site[:2]) >= 0.04:
            continue
        shift = target[:3, 3] - endpoint[:3, 3]
        if np.linalg.norm(shift) < 0.005:
            continue
        path = base.copy()
        gripper = _resample_gripper(reference.gripper_actions, len(path))
        path[:, :3, 3] += placement_shift_weights(gripper, len(path))[:, None] * shift
        poses = compile_held_path_with_release(
            path, reference.poses, gripper, current_object_pose, current_eef_pose
        )
        goal = {
            "kind": "coffee_receptacle_object_goal",
            "persistent": True,
            "target_pose_world": target.round(9).tolist(),
            "pour_site_world_m": site.round(9).tolist(),
            "official_xy_tolerance_m": 0.04,
            "official_z_tolerance_m": 0.10,
        }
        option = dataclasses.replace(
            reference,
            candidate_id=_stable_option_id(
                reference.parent_context_id,
                "G2",
                "coffee_receptacle_goal",
                poses,
                goal,
                len(options) + index,
            ),
            source_method="G2",
            subtype="coffee_receptacle_goal",
            source_episode=reference.source_episode,
            source_fragment=reference.source_fragment,
            intended_object_path=path,
            poses=poses,
            gripper_actions=gripper,
            source_action_indices=_remap_source_indices(reference, len(path)),
            candidate_edit={"kind": "placement_local_subgoal", "frame": "receptacle"},
            goal_binding=goal,
            intended_fixture_subgoal={
                "pour_site_world_m": site.tolist(),
                "xy_offset_m": list(offset),
            },
            support_constraints={"source_support_height_retained": True},
            precondition_checks={
                "official_receptacle_xy": "PASS",
                "official_receptacle_z": "PASS",
                "finite_targets": "PASS",
            },
            provenance=(reference.source_episode or "source", "pour_site"),
        )
        generated.append(_compiled(option))
        audit.append(
            {"candidate_id": option.candidate_id, "status": "PASS", "xy_offset_m": list(offset)}
        )
    return [x for x in options if x.source_method != "G2"] + generated, audit


def select_diverse_object_modes(
    root: SourceSegment,
    donors: Iterable[SourceSegment],
    *,
    limit: int = 12,
    compatibility_limits: dict[str, float] | None = None,
) -> tuple[list[SourceSegment], list[dict[str, Any]]]:
    """Choose nearby compatible donors, then cover distinct measured paths."""
    pool = list(donors)
    _, audit = select_compatible_donor(root, pool, allow_none=True)
    limits = compatibility_limits or {}
    for row in audit:
        if not row["compatible"]:
            continue
        violations = []
        if row["translation_m"] > limits.get("entry_translation_m", float("inf")):
            violations.append("entry_translation_outside_reference_neighborhood")
        if row["rotation_rad"] > limits.get("entry_rotation_rad", float("inf")):
            violations.append("entry_rotation_outside_reference_neighborhood")
        row["reference_neighborhood_limits"] = dict(limits)
        if violations:
            row["compatible"] = False
            row["rejection_reasons"].extend(violations)
    compatible = [
        item
        for item in pool
        if item.motion_object_poses is not None
        and next(row for row in audit if row["donor_id"] == item.episode_id)["compatible"]
    ]
    if not compatible:
        return [], audit

    def feature(segment: SourceSegment) -> np.ndarray:
        motion = np.asarray(segment.motion_object_poses)
        local = np.einsum(
            "ij,nj->ni",
            np.linalg.inv(segment.object_pose_at_entry)[:3, :3],
            (motion[:, :3, 3] - segment.object_pose_at_entry[:3, 3]),
        )
        samples = np.linspace(0, len(local) - 1, 5, dtype=int)
        displacement = local[samples] - local[0]
        return np.concatenate([displacement.reshape(-1), [len(local) * 0.001]])

    features = {item.episode_id: feature(item) for item in compatible}
    first = min(
        compatible,
        key=lambda item: (
            next(row["combined"] for row in audit if row["donor_id"] == item.episode_id),
            item.episode_id,
        ),
    )
    selected = [first]
    while len(selected) < min(limit, len(compatible)):
        others = [
            item for item in compatible if item.episode_id not in {x.episode_id for x in selected}
        ]
        if not others:
            break
        # Path coverage is secondary to entry compatibility. An extreme path
        # is not allowed to win solely because it is far from the others.
        chosen = min(
            others,
            key=lambda item: (
                next(row["combined"] for row in audit if row["donor_id"] == item.episode_id)
                - 0.25
                * min(
                    float(np.linalg.norm(features[item.episode_id] - features[x.episode_id]))
                    for x in selected
                ),
                item.episode_id,
            ),
        )
        selected.append(chosen)
    selected_ids = {item.episode_id for item in selected}
    for row in audit:
        row["selected_as_object_mode"] = row["donor_id"] in selected_ids
        if row["donor_id"] in features:
            row["object_path_fixture_samples_m"] = (
                features[row["donor_id"]][:-1].reshape(5, 3).round(6).tolist()
            )
            row["duration_waypoints"] = int(round(features[row["donor_id"]][-1] / 0.001))
            row["duration_feature_scale_m_per_waypoint"] = 0.001
    return selected, audit
