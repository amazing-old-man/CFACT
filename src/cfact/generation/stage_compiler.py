"""Outcome-blind goals compiled from a restored, current physical parent.

This module only returns controller targets. It never writes simulator state.
Source indices describe consumed source actions, not task progress.
"""
from dataclasses import dataclass, asdict
import numpy as np
from cfact.generation.options import OptionProposal, _stable_option_id, _object_motion_to_eef
from cfact.execution.options import articulation_followthrough


@dataclass(frozen=True)
class EffectGoal:
    task: str
    entity: str
    stage: str
    mode: str
    units: str
    magnitude: float
    horizon: int
    task_goal: dict

    def record(self):
        return {"schema": "cfact.effect_goal.v2", **asdict(self),
                "proposed_before_compilation": True, "is_ground_truth": False}


def source_suffix(source, cursor):
    """Return AVAILABLE/EXHAUSTED/INCOMPATIBLE without dropping source identity."""
    if source is None:
        return "EXHAUSTED", None
    indices = source.source_action_indices
    if indices is None or len(indices) != len(source.source_poses):
        return "INCOMPATIBLE", None
    remaining = np.flatnonzero(np.asarray(indices) >= cursor)
    if not len(remaining) or int(remaining[0]) >= len(indices) - 3:
        return "EXHAUSTED", None
    if source.motion_object_poses is None:
        return "INCOMPATIBLE", None
    return "AVAILABLE", int(remaining[0])


def contact_suffix_compatibility(source, cut, current_object_pose, current_eef_pose, limits):
    """Check the existing declared reference-entry bounds without deleting Source.

    An old suffix can start in a later release/other-entity phase. Its tool/body
    relation must fit the currently measured contact entry to serve as a suffix.
    Missing declared limits remain unknown rather than creating a rejection.
    """
    required={"entry_translation_m","entry_rotation_rad"}
    if not required.issubset(limits):
        return "AVAILABLE",{"status":"NOT_CHECKED_NO_DECLARED_ENTRY_LIMITS"}
    actual=np.linalg.inv(current_object_pose)@current_eef_pose
    prior=np.linalg.inv(source.motion_object_poses[cut])@source.source_poses[cut]
    translation=float(np.linalg.norm(actual[:3,3]-prior[:3,3]))
    rotation=float(np.arccos(np.clip((np.trace(actual[:3,:3].T@prior[:3,:3])-1)/2,-1,1)))
    good=translation<=limits["entry_translation_m"] and rotation<=limits["entry_rotation_rad"]
    return "AVAILABLE" if good else "INCOMPATIBLE",{
        "status":"MEASURED_SOURCE_AND_CURRENT_GEOMETRY","compatible":good,
        "translation_m":translation,"rotation_rad":rotation,"limits":dict(limits),
        "current_tool_to_body_relation":actual.tolist(),"recorded_suffix_entry_relation":prior.tolist(),
        "source_entry_index":int(cut),"remaining_source_waypoints":len(source.source_poses)-cut,
        "source_preserved":True,"reason":None if good else "SOURCE_ENTRY_OUTSIDE_DECLARED_CONTACT_NEIGHBORHOOD"}


def propose_goals(task, context, *, horizon=40):
    """Generate goals before any robot candidate; original task stays immutable."""
    family = task.primitive_family
    if task.name in {"OpenBlenderLid", "CloseBlenderLid"}:
        return []
    if context.stage in {"TERMINAL_SUCCESS", "UNSUPPORTED", "RELEASE_AND_RETREAT"}:
        return []
    if family == "placement":
        if not context.holding_object:
            return []
        modes = [("transport", .06), ("transport_left", .06), ("hold", 0.)]
        unit = "m"
    elif family == "button":
        modes = [("press", .025), ("withdraw", -.025), ("hold", 0.)]
        unit = "m"
    elif context.joint_binding:
        kind = context.joint_binding["joint_type"]
        if kind not in {"slide", "hinge"}:
            return []
        bound = .06 if kind == "slide" else .30
        modes = [("progress", bound), ("reverse", -bound), ("hold", 0.)]
        unit = "m" if kind == "slide" else "rad"
    else:
        return []
    return [EffectGoal(task.name, context.object_roles["manipulated"], context.stage,
                       mode, unit, magnitude, horizon, dict(context.task_goal))
            for mode, magnitude in modes]


def compile_goal(goal, context, *, geometry=None, source=None, source_status="EXHAUSTED",
                 gripper=1., gripper_timeline=None, contact_verified=False, sim_time=0.):
    """Compile one goal, including a bounded source-independent contact primitive."""
    n = goal.horizon + 1
    obj = context.current_object_pose
    tool = context.current_eef_pose
    poses = np.repeat(tool[None], n, axis=0)
    path = np.repeat(obj[None], n, axis=0)
    binding = {"kind": "bounded_effect_goal", "effect_goal": goal.record()}
    fixture_goal = {}
    origin = "CURRENT_STATE_PRIMITIVE"
    if context.joint_binding:
        if not contact_verified:
            raise ValueError("NO_VERIFIED_CONTACT_FOR_CURRENT_STATE_ARTICULATION")
        if not geometry:
            raise ValueError("ACTUAL_JOINT_GEOMETRY_REQUIRED")
        g = geometry
        direction = np.sign(g["target_q"] - g["current_q"]) or 1.
        target = float(np.clip(g["current_q"] + direction * goal.magnitude, *g["joint_range"]))
        poses = articulation_followthrough(
            **{k: v for k, v in g.items() if k != "target_q"}, target_q=target, joint_unit=goal.units, stage=context.stage,
            actual_child={"sim_time_s": sim_time}, current_handle_pose=obj,
            current_eef_pose=tool, samples=n)
        relation = np.linalg.inv(tool) @ obj
        path = poses @ relation
        fixture_goal = {"joint": g["joint_name"], "desired_q": target,
                        "current_q": g["current_q"], "range": list(g["joint_range"])}
        binding.update(kind="articulation_local_subgoal", joint=g["joint_name"],
                       target_q=target, q_task=context.task_goal["target_q"], joint_unit=goal.units)
    elif context.holding_object:
        target = context.task_goal.get("target_pose_world")
        if target is not None:
            delta = np.asarray(target)[:3, 3] - obj[:3, 3]
        elif source is not None and source.motion_object_poses is not None:
            prior, _ = _object_motion_to_eef(source.motion_object_poses,
                source.object_pose_at_entry, context.current_fixture_pose, obj, tool, held=True)
            delta = prior[-1, :3, 3] - obj[:3, 3]
            origin = "SOURCE_PRIOR_RECOMPILED"
        elif goal.mode == "hold":
            delta = np.zeros(3)
        else:
            raise ValueError("NO_BOUNDED_PLACEMENT_GOAL_OR_SOURCE_PRIOR")
        length = float(np.linalg.norm(delta))
        delta = delta * min(1., goal.magnitude / max(length, 1e-9))
        if goal.mode == "transport_left":
            delta += context.current_fixture_pose[:3, 0] * .025
        if goal.mode == "hold":
            delta[:] = 0.
            origin = "CURRENT_STATE_PRIMITIVE"
        path[:, :3, 3] += np.linspace(0., 1., n)[:, None] * delta
        poses = path @ (np.linalg.inv(obj) @ tool)
    elif goal.mode in {"press", "withdraw", "hold"}:
        if source is None:
            raise ValueError("NO_VERIFIED_BUTTON_APPROACH_PRIOR")
        # Source provides the direction only; current state provides the origin.
        delta = source.source_poses[min(5, len(source.source_poses)-1), :3, 3] - source.source_poses[0, :3, 3]
        norm = np.linalg.norm(delta)
        if norm < 1e-6 and goal.mode != "hold":
            raise ValueError("BUTTON_APPROACH_DIRECTION_UNAVAILABLE")
        poses[:, :3, 3] += np.linspace(0., 1., n)[:, None] * delta/max(norm, 1e-9)*goal.magnitude
        origin = "SOURCE_PRIOR_RECOMPILED"
    else:
        raise ValueError("NO_SUPPORTED_CONTINUATION")
    grips = np.full((n, 1), float(gripper))
    if gripper_timeline is not None:
        timeline = np.asarray(gripper_timeline).reshape(-1)
        # Preserve the prior's actual discrete closing event; do not interpolate it.
        grips[:,0] = timeline[np.minimum(np.arange(n),len(timeline)-1)]
        origin = "SOURCE_PRIOR_RECOMPILED"
    cid = _stable_option_id(context.parent_context_id, "G2", goal.mode, poses, binding, 90)
    return OptionProposal(
        candidate_id=cid, generator_revision="cfact-recursive-goals-v2",
        source_method="G2", subtype="effect_"+goal.mode,
        parent_context_id=context.parent_context_id, operation_type=goal.task,
        decision_group_id=context.parent_context_id,
        source_episode=None if source is None else source.episode_id,
        source_fragment=None, object_roles=dict(context.object_roles),
        reference_frame="current_world", current_contact_mode="measured_parent",
        intended_object_path=path, intended_fixture_subgoal=fixture_goal,
        entry_constraints={"contact_verified": contact_verified}, support_constraints={},
        task_constraints={"original_task_goal": dict(context.task_goal)},
        gripper_event_policy={"kind": "maintain_measured_parent_command"},
        goal_binding=binding, motion_template=origin,
        connection_plan={"kind": "current_state_bounded_primitive"}, control_limits={},
        end_event="bounded_effect_or_horizon", local_timeout=goal.horizon+1,
        precondition_checks={"finite_targets": "PASS"}, poses=poses, gripper_actions=grips,
        provenance=(origin,), operation_context=context,
        candidate_edit={"kind": "effect_goal_compilation", "effect_goal": goal.record(),
                        "continuation_origin": origin, "source_suffix_status": source_status},
        source_action_indices=np.full(n, -1, dtype=int))


def select_goal_pool(options, *, maximum):
    """Reserve mode coverage so legacy family order cannot suppress new goals."""
    goals = [o for o in options if o.executable and o.candidate_edit.get("kind") == "effect_goal_compilation"]
    reference = [o for o in options if o.executable and o.source_method == "Reference"]
    chosen = (reference[:1] + goals)[:maximum]
    ids = {o.candidate_id for o in chosen}
    return chosen, [{"candidate_id": o.candidate_id, "reason": o.rejection_reason or "mode_pool_budget"}
                    for o in options if o.candidate_id not in ids]
