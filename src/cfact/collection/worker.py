#!/usr/bin/env python3
"""Physical object-effect collection for one independent task/episode root.

Every sibling restores the same complete parent simulator state. The candidate
pool is frozen before physical execution. Local event children are captured
before the fixed continuation; only the child state can seed a later pool.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cfact.paths import workspace_root

from cfact.collection.runtime import _build_trajectory, _create_env, _load_root, _probe_parent
from cfact.adapters.action_codec import install_action_codec
from cfact.execution.current import current_info, child_geometry
from cfact.execution.snapshot import (
    capture_snapshot,
    restore_snapshot,
    save_snapshot,
    load_snapshot,
)
from cfact.execution.binding import (
    declared_operation_joint,
    resolve_role_binding,
    bind_joint,
    validate_joint_binding,
    trace_joint_spec,
    source_joint_task_goal,
)
from cfact.execution.history import ControlStepLedger, ledger
from cfact.execution.observation import observe
from cfact.storage.schema import make_child_context
from cfact.generation.context import OperationContext
from cfact.generation.options import compile_candidate_plan
from cfact.sources.prepare import validate_prepared_identity
from cfact.generation.options import (
    build_object_effect_options,
    operation_contract,
    select_option_pool,
    select_child_frontier,
    solve_g2_support_goals,
)
from cfact.generation.options import (
    solve_g2_articulation_subgoals,
    solve_g2_coffee_receptacle_goals,
    bind_inherited_placement_goal,
)
from cfact.generation.segments import load_source_segment, recorded_held_entry_step
from cfact.execution.options import (
    LocalEventDetector,
    ReleaseEventDetector,
    MeasuredTrace,
    MeasuredArmContactGuard,
    execute_option,
    connect_and_retime_poses,
    articulation_followthrough,
)
from cfact.io.review_video import QuadReviewWriter
from cfact.execution.kinematics import KinematicPathValidator, observed_gripper_target_pairs
from cfact.execution.connector import ConnectorBudget
from cfact.io.serialization import RunLock, array_digest, atomic_json
from cfact.tasks import target_task


PROJECT = workspace_root()


def _relative(path: Path, run: Path) -> str:
    return str(path.relative_to(run))


def _check_storage(run: Path, cfg: dict[str, Any]) -> None:
    resource = cfg["resource"]
    total = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    from cfact.storage.policy import target_filesystem
    storage = target_filesystem(run, resource)
    free = storage["filesystem_free_bytes"]
    maximum = int(resource["maximum_output_gb"] * 1_000_000_000)
    minimum_free = int(resource["minimum_host_free_gib"] * 1024**3)
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "run_and_spool_bytes": total,
        "maximum_run_and_spool_bytes": maximum,
        "filesystem_free_bytes": free,
        "minimum_filesystem_free_bytes": minimum_free,
        "status": "PASS" if total <= maximum and storage["status"] == "PASS" else "STORAGE_GUARD_STOP",
        "target_filesystem": storage,
    }
    atomic_json(run / "runtime" / "storage_admission.json", record)
    if record["status"] != "PASS":
        raise RuntimeError("STORAGE_GUARD_STOP")


def _goal_realization(
    option: Any, local: MeasuredTrace, fixed: MeasuredTrace, goal_index: int | None
) -> dict[str, Any]:
    binding = option.goal_binding
    kind = binding.get("kind", "source_task_goal")
    is_joint = kind == "articulation_local_subgoal"
    target_phase = (
        "joint_subgoal"
        if is_joint
        else "final_placement"
        if kind
        in {
            "fixture_region_object_goal",
            "coffee_receptacle_object_goal",
            "fixture_relative_object_goal",
        }
        else option.end_event
    )
    if is_joint:
        name = binding["joint"]
        if name not in local.fixture_joint_names:
            return {"valid": False, "reason": "named_joint_absent", "target_phase": target_phase}
        index = local.fixture_joint_names.index(name)
        target = float(binding["target_q"])
        unit = binding["joint_unit"]
        local_values = np.asarray(local.fixture_qpos)[:, index]
        fixed_values = np.asarray(fixed.fixture_qpos)[:, index]
        errors = np.abs(np.concatenate([local_values, fixed_values[1:]]) - target)
        target_entity, target_frame = name, "named_joint"
    else:
        target_pose = (
            np.asarray(binding["target_pose_world"], dtype=float)
            if "target_pose_world" in binding
            else np.asarray(option.intended_object_path[goal_index], dtype=float)
            if option.intended_object_path is not None and goal_index is not None
            else None
        )
        if target_pose is None:
            return {
                "valid": False,
                "reason": "target_pose_unavailable",
                "target_phase": target_phase,
            }
        positions = np.concatenate(
            [np.asarray(local.object_poses)[:, :3, 3], np.asarray(fixed.object_poses)[1:, :3, 3]]
        )
        errors = np.linalg.norm(positions - target_pose[:3, 3], axis=1)
        unit = "m"
        target = target_pose.round(9).tolist()
        target_entity, target_frame = local.object_ref, "world"
    times = np.concatenate([np.asarray(local.sim_time), np.asarray(fixed.sim_time)[1:]])
    nearest = int(np.argmin(errors))
    event_indices = np.flatnonzero(local.local_event_flags)
    event_error = (
        float(errors[event_indices[0] + 1])
        if len(event_indices) and target_phase != "final_placement"
        else None
    )
    event_reason = (
        None
        if event_error is not None
        else "target_phase_differs_from_local_event"
        if target_phase == "final_placement"
        else "local_event_not_observed"
    )
    return {
        "valid": True,
        "goal_binding": binding,
        "target_entity": target_entity,
        "target_frame": target_frame,
        "target_phase": target_phase,
        "target": target,
        "unit": unit,
        "nearest_arrival_error": float(errors[nearest]),
        "nearest_arrival_sim_time_s": float(times[nearest]),
        "target_event_error": event_error,
        "target_event_error_missing_reason": event_reason,
        "local_end_error": float(errors[len(local.sim_time) - 1]),
        "fixed_end_error": (float(errors[-1]) if len(fixed.commanded_actions) else None),
        "fixed_end_error_missing_reason": (
            None if len(fixed.commanded_actions) else "fixed_continuation_not_executed"
        ),
    }


def _outcome_labels(local: dict[str, Any], fixed_status: str) -> dict[str, bool]:
    success_observed = (
        bool(local["official_success_at_child"]) or fixed_status == "official_success"
    )
    return {
        "SUCCESS_WITHIN_HORIZON": success_observed,
        "OBSERVED_OPERATION_FAILURE": False,
        "HORIZON_CENSORED": fixed_status == "horizon_censored",
        "COLLECTION_BUDGET_CENSORED": False,
        "NO_SUPPORTED_CONTINUATION": fixed_status == "NO_SUPPORTED_CONTINUATION"
        and not success_observed,
        "PRE_EXECUTION_REJECTED": False,
        "EXECUTION_ABORTED_BY_DECLARED_GUARD": local["termination_reason"]
        == "EXECUTION_ABORTED_BY_DECLARED_GUARD"
        or fixed_status == "EXECUTION_ABORTED_BY_DECLARED_GUARD",
        "TECHNICAL_UNKNOWN": False,
    }


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def _module_provenance() -> list[dict[str, str]]:
    import cfact.generation.options as generation_module
    import cfact.execution.options as execution_module

    rows = []
    for name, path in (
        ("worker", Path(__file__)),
        (generation_module.__name__, Path(generation_module.__file__)),
        (execution_module.__name__, Path(execution_module.__file__)),
    ):
        rows.append(
            {
                "module": name,
                "file": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


def _case_seed(
    cfg: dict[str, Any], task: str, root_id: str, node_id: str, candidate_id: str, purpose: str
) -> int:
    seed = stable_seed(cfg["run_seed"], task, root_id, node_id, candidate_id, purpose)
    return seed


def _card(option: Any) -> dict[str, Any]:
    from cfact.generation.effect_intents import intent_for_option
    return {
        "effect_goal": option.candidate_edit.get("effect_goal"),
        "continuation_origin": option.candidate_edit.get("continuation_origin", "SOURCE_SUFFIX"),
        "effect_intent": intent_for_option(option, option.parent_context_id.split("__")[0]).record() if option.operation_context else None,
        "candidate_id": option.candidate_id,
        "generator_revision": option.generator_revision,
        "source_method": option.source_method,
        "subtype": option.subtype,
        "parent_context_id": option.parent_context_id,
        "operation_type": option.operation_type,
        "decision_group_id": option.decision_group_id,
        "source_episode": option.source_episode,
        "source_fragment": option.source_fragment,
        "mode_descriptor": option.mode_descriptor,
        "object_roles": option.object_roles,
        "reference_frame": option.reference_frame,
        "current_contact_mode": option.current_contact_mode,
        "intended_fixture_subgoal": option.intended_fixture_subgoal,
        "entry_constraints": option.entry_constraints,
        "support_constraints": option.support_constraints,
        "task_constraints": option.task_constraints,
        "gripper_event_policy": option.gripper_event_policy,
        "goal_binding": option.goal_binding,
        "motion_template": option.motion_template,
        "connection_plan": option.connection_plan,
        "control_limits": option.control_limits,
        "end_event": option.end_event,
        "local_timeout": option.local_timeout,
        "precondition_checks": option.precondition_checks,
        "provenance": option.provenance,
        "operation_context": option.operation_context.record()
        if option.operation_context
        else None,
        "candidate_plan": compile_candidate_plan(option, option.operation_context).record()
        if option.executable and option.operation_context
        else None,
        "rejection_reason": option.rejection_reason,
    }


def _restore(env: Any, initial: dict[str, Any], state: Any) -> None:
    started = time.monotonic()
    if not isinstance(state, dict):
        raise ValueError("new_collection_requires_complete_parent_snapshot")
    restore_receipt = restore_snapshot(env, state, initial=initial)
    if getattr(env,"_cfact_v2_task",None):
        from cfact.execution.snapshot import snapshot_differences
        actual=capture_snapshot(env)
        differences=snapshot_differences(state,actual)
        receipt={"status":"PASS" if not differences else "FAIL",
            "sim_time_s":actual["sim_time_s"],"state_sha256":array_digest(actual["states"]),
            "complete_value_state_differences":differences,"restore":restore_receipt}
        if ledger(env):
            with (ledger(env).directory / "complete_restore_receipts.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt,sort_keys=True)+"\n")
        if differences:raise ValueError("complete_parent_restore_value_state_mismatch")
    if ledger(env) is not None:
        ledger(env).restore_wall_seconds += time.monotonic() - started


def _save_actual_child(
    env: Any, edge_dir: Path, option: Any, local: dict[str, Any], trace: MeasuredTrace
) -> np.ndarray:
    """The only child constructor; called immediately after local, before tail."""
    snapshot = capture_snapshot(env)
    state = np.asarray(snapshot["states"]).copy()
    save_snapshot(edge_dir / "child_snapshot.json", snapshot)
    np.savez_compressed(edge_dir / "child_context.npz", sim_state=state)
    source_cursor = dict(local.get("source_cursor", {}))
    anchor = None
    if getattr(env, "_cfact_v2_task", None):
        from cfact.generation.task_adapters import TaskEffectAdapter
        adapter = TaskEffectAdapter(target_task(env._cfact_v2_task), option.operation_context.joint_binding)
        anchor = adapter.classify_stage(trace.arrays(), source_cursor=source_cursor["next_action_index"],
            local_event=local["local_event_reached"], release=option.subtype == "release_and_retreat", local_goal=option.goal_binding)
        atomic_json(edge_dir / "observed_effect.json", anchor.evidence)
        atomic_json(edge_dir / "event_anchor.json", anchor.record())
        commands=np.asarray(trace.commanded_actions)
        actual_eef=np.asarray(trace.arrays()["actual_eef_pose"])
        targets=np.asarray(trace.controller_target_poses)
        atomic_json(edge_dir / "effect_realization.json", {
            "intent_is_ground_truth":False,
            "requested_tool_translation_m":float(np.linalg.norm(option.poses[:,:3,3]-option.operation_context.current_eef_pose[:3,3],axis=1).max()),
            "issued_action_count":len(commands),"issued_action_sha256":array_digest(commands),
            "commanded_tool_translation_m":float(np.linalg.norm(targets[:,:3,3]-actual_eef[0,:3,3],axis=1).max()) if len(targets) else None,
            "achieved_tool_translation_m":float(np.linalg.norm(actual_eef[:,:3,3]-actual_eef[0,:3,3],axis=1).max()),
            "observed_task_effect":anchor.evidence})
    fixture_pose, object_pose, eef_pose = child_geometry(trace)
    context = dataclasses.replace(
        option.operation_context,
        current_fixture_pose=fixture_pose,
        current_object_pose=object_pose,
        current_eef_pose=eef_pose,
        source_cursor=int(source_cursor["next_action_index"]),
        stage=anchor.stage.value if anchor else option.operation_context.stage,
        holding_object=anchor.evidence["held_verified"] if anchor else option.operation_context.holding_object,
    )
    context_record = context.record()
    parent_record = json.loads((edge_dir.parent.parent / "context.json").read_text())
    parent_history = parent_record.get("history_refs", [])
    current_history = ledger(env).references(segment=str(edge_dir)) if ledger(env) else []
    history = parent_history + current_history[1:]
    atomic_json(
        edge_dir / "child_context.json",
        make_child_context(
            sim_state=state,
            operation_context=context_record,
            source_cursor=source_cursor,
            snapshot={
                "path": "child_snapshot.json",
                "schema": snapshot.get("schema"),
                "capabilities": snapshot.get("capabilities", {}),
            },
            history_refs=history,
            goal_binding=option.goal_binding,
            local_event_reached=local["local_event_reached"],
            history_complete=len(history) >= 3,
        ),
    )
    return state


def _propose_goals(args, cfg, context, node):
    if not cfg.get("method", {}).get("recursive_goals_v2"):
        return []
    from cfact.generation.stage_compiler import propose_goals
    goals = propose_goals(target_task(args.task), context,
        horizon=cfg.get("method", {}).get("primitive_horizon_steps", 40))
    atomic_json(node / "effect_goals.json", {"goals": [g.record() for g in goals],
        "frozen_before_compilation": True})
    return goals


def _minimum_event_steps(cfg, option):
    method=cfg.get("method",{})
    if option.candidate_edit.get("effect_goal"):
        return method.get("minimum_event_steps",0)
    if option.source_method=="Reference" and method.get("recursive_goals_v2"):
        return method.get("source_reference_minimum_event_steps",0)
    return 0


def _compile_goals(args, cfg, env, interface, context, goals, source, status, node, legacy):
    from cfact.generation.stage_compiler import compile_goal
    row = next(r for r in cfg["tasks"] if r["task"] == args.task)
    signals = interface.get_subtask_term_signals()
    contact = bool(signals.get(row["event_signal"], False))
    if context.holding_object:
        contact = True  # Verified held parent probe or actual child evidence.
    geometry = _articulation_geometry(env, context.joint_binding, context.task_goal) if context.joint_binding else None
    if target_task(args.task).primitive_family == "button":
        from cfact.generation.task_adapters import activation_state
        if activation_state(env).get("turned_on"):
            atomic_json(node / "goal_compile.json", {"status":"ALREADY_ACTIVATED", "source_suffix_status":status})
            return []
    generated=[]; records=[]
    grip = float(source.gripper_actions[0,0]) if source is not None else 1.
    parent_record = json.loads((node / "context.json").read_text())
    parent_edge = parent_record.get("parent_edge")
    if parent_edge:
        with np.load(args.run.resolve() / parent_edge / "execution.npz", allow_pickle=False) as trace:
            commands = trace["commanded_actions"]
            if len(commands):grip = float(commands[-1,commands.shape[1]-6])
    closing_prior = None
    if context.joint_binding and grip < 0 and status == "AVAILABLE":
        timeline = np.asarray(source.gripper_actions).reshape(-1)
        if np.any(timeline > 0):closing_prior = timeline
    for goal in goals:
        try:
            option = compile_goal(goal, context, geometry=geometry, source=source,
                source_status=status, gripper=grip, gripper_timeline=closing_prior, contact_verified=contact,
                sim_time=float(env.env.sim.data.time))
            generated.append(option)
            records.append({"goal":goal.record(), "candidate_id":option.candidate_id, "status":"COMPILED"})
        except ValueError as exc:
            records.append({"goal":goal.record(), "status":"UNSUPPORTED", "reason":str(exc)})
    atomic_json(node / "goal_compile.json", {"records":records, "source_suffix_status":status,
        "current_geometry":context.record(), "contact_verified":contact})
    return legacy + generated


def _select_pool(cfg, options, *, max_executed, max_g0):
    if cfg.get("method", {}).get("recursive_goals_v2"):
        from cfact.generation.stage_compiler import select_goal_pool
        return select_goal_pool(options, maximum=max_executed)
    return select_option_pool(options, max_executed=max_executed, max_g0=max_g0)


def _engineering_select(selected, rejected, cfg):
    profile = cfg.get("engineering_profile") or {}
    maximum = profile.get("max_candidates_per_family")
    if maximum is None:
        return selected
    if cfg.get("method", {}).get("recursive_goals_v2"):
        return selected
    counts = {}
    result = []
    for option in selected:
        count = counts.get(option.source_method, 0)
        if count >= int(maximum):
            rejected.append(
                {
                    "candidate_id": option.candidate_id,
                    "reason": "engineering_budget",
                    "disposition": "not_executed",
                    "profile": profile.get("name"),
                }
            )
        else:
            result.append(option)
            counts[option.source_method] = count + 1
    return result


def _operation_context(
    interface,
    info,
    *,
    parent_id,
    object_ref,
    fixture_ref,
    contract,
    source,
    task_goal,
    joint_binding,
    stage,
    source_cursor=None,
):
    return OperationContext(
        parent_context_id=parent_id,
        object_roles={
            "manipulated": object_ref,
            "held": object_ref if contract.holding_object else None,
            "reference": fixture_ref,
        },
        task_goal=task_goal,
        stage=stage,
        source_cursor=int(
            source_cursor["next_action_index"] if source_cursor else source.event_first_step
        ),
        controller_semantics=interface.action_codec.semantics(),
        current_fixture_pose=np.asarray(info.object_poses[fixture_ref]),
        current_object_pose=np.asarray(info.object_poses[object_ref]),
        current_eef_pose=np.asarray(info.eef_pose),
        joint_binding=joint_binding,
        holding_object=contract.holding_object,
    )


def _probe_held_parent(
    env: Any,
    interface: Any,
    initial: dict[str, Any],
    actions: np.ndarray,
    contact_signal: str,
    object_ref: str,
    minimum_lift_m: float = 0.03,
) -> dict[str, Any]:
    """Use the first measured stable held state after the source contact entry."""
    env.reset_to(initial)
    contact_seen = False
    first_object_xyz = None
    prior_relative = None
    stable = 0
    pose_dim = len(
        interface.target_pose_to_action(target_pose=interface.get_datagen_info().eef_pose)
    )
    for step, action in enumerate(actions):
        env.step(action)
        info = current_info(env, interface)
        if bool(info.subtask_term_signals.get(contact_signal, False)):
            contact_seen = True
        if not contact_seen:
            continue
        obj = np.asarray(info.object_poses[object_ref])
        eef = np.asarray(info.eef_pose)
        relative = np.linalg.inv(eef) @ obj
        if first_object_xyz is None:
            first_object_xyz = obj[:3, 3].copy()
        step_slip = (
            np.inf
            if prior_relative is None
            else float(np.linalg.norm(relative[:3, 3] - prior_relative[:3, 3]))
        )
        stable = stable + 1 if step_slip <= 0.003 else 0
        prior_relative = relative
        displacement = float(np.linalg.norm(obj[:3, 3] - first_object_xyz))
        lift = float(obj[2, 3] - first_object_xyz[2])
        gripper_closing = len(action) == pose_dim + 6 and action[pose_dim] > 0
        moved = lift >= minimum_lift_m if minimum_lift_m > 0 else displacement >= 0.04
        if (
            stable >= 5
            and moved
            and gripper_closing
            and bool(info.subtask_term_signals.get(contact_signal, False))
        ):
            return {
                "event": {
                    "event_name": "grasp_established",
                    "stable_steps": stable,
                    "object_motion_m": displacement,
                    "object_lift_m": lift,
                    "gripper_command": float(action[pose_dim]),
                    "gripper_action_channel": pose_dim,
                    "per_step_relative_translation_threshold_m": 0.003,
                },
                "parent_action_count": step + 1,
                "parent_state": np.asarray(env.env.sim.get_state().flatten()).copy(),
                "parent_signals": {str(k): bool(v) for k, v in info.subtask_term_signals.items()},
            }
    raise RuntimeError("source replay never established measured held-object parent")


def _probe_articulation_parent(env, interface, initial, actions, row, stable_steps=4):
    """First real stable operation contact, before any candidate is proposed.

    An early contact may precede the recorded closing event. Continue the paid
    source replay to the first measured stable bound-body contact. Selection
    uses contact and grasp evidence, never a candidate's future effect.
    """
    first=_probe_parent(env,interface,initial,actions,row["event_signal"],row["event_minimum_duration"])
    fallback=capture_snapshot(env)
    try:
        role=resolve_role_binding(interface,row["event_object_ref"])
        declared=declared_operation_joint(interface,row.get("operation_joint_role"))
        binding=bind_joint(env.env.sim.model,role,explicit_joint=declared or row.get("operation_joint"))
    except (ValueError,KeyError,AttributeError):
        first["event"]["contact_parent_rule"]="first_source_contact_unsupported_topology"
        return first
    start=int(first["parent_action_count"])
    # A discrete closing command is used only if this source actually has one.
    needs_closed=bool(np.any(actions[start:,actions.shape[1]-6]>0))
    target_body=int(binding["joint_body_id"])
    model=env.env.sim.model
    def bound_body(body):
        while body>0:
            if body==target_body:return True
            body=int(model.body_parentid[body])
        return False
    previous=None;stable=0
    for step in range(start,min(len(actions),start+80)):
        if bool(env.is_success()["task"]):break
        env.step(actions[step])
        if bool(env.is_success()["task"]):break
        info=current_info(env,interface)
        relative=np.linalg.inv(info.eef_pose)@info.object_poses[row["event_object_ref"]]
        _,contacts=observed_gripper_target_pairs(env)
        contact=any(bound_body(int(b)) for _,b in contacts["observed_gripper_target_body_contacts"])
        closed=not needs_closed or actions[step,actions.shape[1]-6]>0
        drift=np.inf if previous is None else float(np.linalg.norm(relative[:3,3]-previous[:3,3]))
        rotation=np.inf if previous is None else float(np.linalg.norm(relative[:3,:3]-previous[:3,:3]))
        stable=stable+1 if contact and closed and drift<=.003 and rotation<=.03 else 0
        previous=relative
        if stable>=stable_steps:
            return {"event":{"event_name":row["event_signal"],"rule":"first_stable_bound_body_contact",
                    "stable_steps":stable,"required_closed_source_gripper":needs_closed,
                    "bound_joint":binding["joint_name"],"per_step_relative_translation_m":.003,
                    "per_step_relative_rotation_frobenius":.03,"original_contact_action_count":start},
                    "parent_action_count":step+1,"parent_state":np.asarray(env.env.sim.get_state().flatten()).copy(),
                    "parent_signals":{str(k):bool(v) for k,v in info.subtask_term_signals.items()},
                    "first_success_step_before_parent":first.get("first_success_step_before_parent")}
    _restore(env,initial,fallback)
    first["event"]["contact_parent_rule"]="first_contact_fallback_no_stable_preterminal_contact"
    first["event"]["additional_source_probe_steps_paid"]=max(0,step-start+1) if start<len(actions) else 0
    return first


def _probe_button_parent(env, interface, initial, actions, event_signal, lookback=5):
    """Pick a saved real pre-activation state by a fixed source-event rule.

    The detecting action is paid and archived, but is outside canonical ancestry.
    No candidate outcome is inspected and no fixture state is changed.
    """
    from collections import deque
    from cfact.generation.task_adapters import activation_state
    env.reset_to(initial)
    ring = deque(maxlen=lookback)
    for step, action in enumerate(actions):
        prior = activation_state(env)
        if prior.get("status") != "MEASURED":
            raise RuntimeError("ACTIVATION_STATE_UNMEASURED")
        if prior["turned_on"]:
            raise RuntimeError("SOURCE_INITIAL_BUTTON_ALREADY_ACTIVATED")
        ring.append((step, capture_snapshot(env)))
        env.step(action)
        observed = activation_state(env)
        if observed.get("turned_on"):
            count, snapshot = ring[0]
            _restore(env, initial, snapshot)
            return {"event": {"event_name": event_signal, "rule": "fixed_5_action_pre_activation",
                              "activation_action_index": step, "probe_control_steps": step+1},
                    "parent_action_count": count,
                    "first_success_step_before_parent": None,
                    "parent_state": np.asarray(snapshot["states"]).copy(),
                    "parent_signals": interface.get_subtask_term_signals(),
                    "activation_before_parent": activation_state(env)}
    raise RuntimeError("SOURCE_ACTIVATION_EVENT_NOT_OBSERVED")


def _support_geometry(env: Any, task_name: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    if task_name == "CoffeeServeMug":
        fixture = env.env.counter
    else:
        fixture = getattr(
            env.env,
            {
                "PickPlaceCounterToCabinet": "cab",
                "PickPlaceCounterToSink": "sink",
                "CoffeeSetupMug": "coffee_machine",
            }[task_name],
        )
    regions = fixture.get_reset_regions(env=env.env)
    angle = float(fixture.rot)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    entries = []
    for name, info in sorted(regions.items()):
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = np.asarray(fixture.pos) + rotation @ np.asarray(info["offset"])
        entries.append(
            {
                "name": name,
                "pose": pose.tolist(),
                "size_m": list(info["size"]),
                "height_m": info.get("height"),
            }
        )
    obj = env.env.objects["obj"]
    return entries, np.asarray(obj.get_bbox_points(), dtype=float)


def _fixture_joint_spec(env: Any, binding: dict[str, Any] | None) -> dict[str, Any]:
    if binding:
        validate_joint_binding(env.env.sim.model, binding)
    return trace_joint_spec(binding)


def _articulation_geometry(
    env: Any, binding: dict[str, Any], task_goal: dict[str, Any]
) -> dict[str, Any]:
    if not binding:
        raise ValueError("operation_has_no_articulation_binding")
    sim = env.env.sim
    validate_joint_binding(sim.model, binding)
    if not binding["joint_limited"]:
        raise ValueError("bound_joint_is_unlimited_not_a_bounded_subgoal")
    if task_goal.get("joint_name") != binding["joint_name"]:
        raise ValueError("task_goal_joint_binding_mismatch")
    jid = int(binding["joint_id"])
    return {
        "joint_name": binding["joint_name"],
        "joint_type": binding["joint_type"],
        "current_q": float(sim.data.qpos[binding["qpos_address"]]),
        "joint_range": tuple(binding["joint_range"]),
        "target_q": float(task_goal["target_q"]),
        "axis_world": np.asarray(sim.data.xaxis[jid]).copy(),
        "anchor_world": np.asarray(sim.data.xanchor[jid]).copy(),
    }


def _g0_amplitude(cfg, env, args, source, info, contract, binding, task_goal, node):
    if cfg.get("method",{}).get("g0_scale_policy") != "source_geometry_v1":
        return 0.003
    from cfact.generation.g0_scale import source_geometry_scale
    allowance=0.03
    evidence={"source":"current_parent_geometry","collision_safety":"NOT_GLOBALLY_CERTIFIED"}
    if binding:
        geometry=_articulation_geometry(env,binding,task_goal)
        remaining=abs(geometry["target_q"]-geometry["current_q"])
        if geometry["joint_type"]=="slide":allowance=min(allowance,.1*remaining)
        else:
            offset=np.asarray(info.eef_pose)[:3,3]-geometry["anchor_world"]
            axis=geometry["axis_world"];radius=float(np.linalg.norm(offset-axis*np.dot(offset,axis)))
            allowance=min(allowance,.1*radius*remaining)
            evidence["measured_joint_tool_radius_m"]=radius
        evidence.update(joint_name=geometry["joint_name"],joint_type=geometry["joint_type"],remaining_q=remaining)
    elif contract.holding_object:
        try:
            _,bbox=_support_geometry(env,args.task)
            size=np.ptp(bbox,axis=0)
            allowance=min(allowance,.5*float(min(size[:2])))
            evidence["actual_object_footprint_local_m"]=size[:2].tolist()
        except (AttributeError,KeyError,ValueError) as error:
            evidence["object_footprint_unavailable"]=str(error)
            allowance=min(allowance,0.003)
    record=source_geometry_scale(source.source_poses,geometry_allowance_m=allowance)
    atomic_json(node / "g0_scale.json", {**record,"geometry_evidence":evidence})
    return record["requested_amplitude_m"]


def _option_has_event_geometry(
    option: Any, primitive_family: str, minimum_translation_m: float
) -> bool:
    if option.candidate_edit.get("kind") == "effect_goal_compilation":
        return option.executable
    path = option.intended_object_path
    if path is None or len(path) < 2:
        return False
    fixture_goal = option.intended_fixture_subgoal
    if "joint" in fixture_goal and "desired_q" in fixture_goal:
        if float(fixture_goal["desired_q"]) != float(
            fixture_goal.get("current_q", fixture_goal["desired_q"])
        ):
            return True
    positions = np.asarray(path[:, :3, 3], dtype=float)
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    if float(displacement.max()) >= minimum_translation_m:
        return True
    rotations = np.asarray(path[:, :3, :3], dtype=float)
    relative = np.einsum("ij,njk->nik", rotations[0].T, rotations)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    if float(np.arccos(cosine).max()) > 1e-6:
        return True
    if primitive_family == "button":
        tool = np.asarray(option.poses, dtype=float)
        tool_pos = np.linalg.norm(np.diff(tool[:, :3, 3], axis=0), axis=1)
        tool_rot = np.einsum("nij,njk->nik", tool[:-1, :3, :3].transpose(0, 2, 1), tool[1:, :3, :3])
        angle = np.arccos(np.clip((np.trace(tool_rot, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
        return bool(np.any(tool_pos > 1e-6) or np.any(angle > 1e-6))
    return False


def _execution_trajectory(
    WaypointSequence: Any,
    WaypointTrajectory: Any,
    actual_eef_pose: np.ndarray,
    option: Any,
    cfg: dict[str, Any],
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
    plan = compile_candidate_plan(option, option.operation_context)
    poses, grips, audit = connect_and_retime_poses(
        actual_eef_pose, plan.poses, plan.gripper_actions, maximum_waypoints=2000
    )
    audit["candidate_plan"] = plan.record()
    audit["source_action_index"] = [
        int(plan.source_action_indices[max(0, index)]) for index in audit["source_waypoint_index"]
    ]
    trajectory = _build_trajectory(
        WaypointSequence, WaypointTrajectory, actual_eef_pose, poses[1:], grips[1:], 0
    )
    return trajectory, audit, poses, grips


class MotionPlanRejected(ValueError):
    """A candidate did not pass the configured pre-execution geometry gate."""


def _audit_motion_plan(
    env: Any,
    poses: np.ndarray,
    grips: np.ndarray,
    cfg: dict[str, Any],
    output: Path | None,
    *,
    holding_object: bool,
) -> dict[str, Any]:
    policy = cfg.get("motion_validation", {})
    mode = policy.get("mode", "off")
    if mode == "off":
        return {"solver_status": "NOT_CHECKED", "reason": "validation_mode_off"}
    if mode not in {"audit", "audit_connector", "enforce"}:
        raise ValueError(f"unsupported_motion_validation_mode:{mode}")
    began = time.monotonic()
    try:
        allowed, contact_policy = observed_gripper_target_pairs(env)
        held_body = None
        if holding_object:
            name = env.env.objects["obj"].root_body
            held_body = int(env.env.sim.model.body_name2id(name))
        validator = KinematicPathValidator(
            env, allowed_body_pairs=allowed, held_object_body_id=held_body
        )
        joint_path, report = validator.validate(poses, grips)
    except (ValueError, KeyError, AttributeError) as error:
        unavailable = {
            "backend": "mujoco_kinematic_copy",
            "solver_status": "VALIDATION_UNAVAILABLE",
            "reason": f"{type(error).__name__}:{error}",
            "mode": mode,
            "holding_object_at_parent": holding_object,
            "complete_geometric_scope": False,
        }
        if output is not None:
            atomic_json(output / "motion_validation.json", unavailable)
        if mode == "enforce":
            raise MotionPlanRejected(unavailable["reason"]) from error
        return unavailable
    report.update(
        {
            "mode": mode,
            "world_snapshot_id": array_digest(validator.snapshot_qpos),
            "world_sim_time_s": validator.snapshot_time,
            "planning_wall_seconds": time.monotonic() - began,
            "contact_policy": contact_policy,
            "holding_object_at_parent": holding_object,
            "final_timed_plan_kind": "20 Hz Cartesian OSC targets",
            "reference_plan_kind": "original option poses saved in plan.npz",
            "joint_solution_branch_file": "kinematic_plan.npz" if output is not None else None,
            "repair_applied": False,
            "revalidation_result": "same final Cartesian target array checked after retiming",
            "complete_geometric_scope": False,
            "planning_scope": (
                "kinematic copy; held object follows initial tool transform until open command"
                if holding_object
                else "kinematic copy; moving articulation held at parent snapshot"
            ),
        }
    )
    if (
        mode == "audit_connector"
        and report["solver_status"] in {"ROBOT_CONTACT_NOT_ALLOWED", "JOINT_STEP_LIMIT"}
        and report["failures"]
        and report["failures"][0].get("waypoint") == 1
        and len(poses) > 1
    ):
        connector_cfg = policy.get("connector", {})
        budget = ConnectorBudget(**connector_cfg)
        # The validator above may have reached a later release state. Take a
        # fresh snapshot of the same physical parent for this join query.
        join_validator = KinematicPathValidator(
            env,
            allowed_body_pairs=allowed,
            held_object_body_id=held_body,
            allow_inherited_held_contacts=holding_object,
        )
        try:
            path, connector_report = join_validator.diagnose_first_join(
                poses[1], budget=budget, seed=int(array_digest(poses)[:8], 16)
            )
        except (ValueError, KeyError, AttributeError) as error:
            path = None
            connector_report = {
                "backend": "bounded_bidirectional_joint_rrt_connect",
                "status": "DIAGNOSTIC_ERROR",
                "reason": f"{type(error).__name__}:{error}",
                "executed_joint_path_equivalent": False,
                "continuous_collision": "NOT_CHECKED",
            }
        report["first_join_connector"] = connector_report
        if path is not None and output is not None:
            np.savez_compressed(output / "first_join_connector.npz", arm_qpos=path)
        report["repair_applied"] = False
        report["connector_execution_status"] = "NOT_EXECUTED_AUDIT_ONLY"
    if output is not None:
        np.savez_compressed(output / "kinematic_plan.npz", arm_qpos=joint_path)
        atomic_json(output / "motion_validation.json", report)
    if mode == "enforce" and (report["solver_status"] != "GEOMETRIC_PASS" or holding_object):
        raise MotionPlanRejected(
            report["solver_status"]
            if not holding_object
            else "HELD_OBJECT_ATTACHMENT_NOT_VALIDATED"
        )
    return report


def _measured_arm_guard(env: Any, cfg: dict[str, Any]) -> MeasuredArmContactGuard | None:
    policy = cfg.get("contact_guard", {})
    mode = policy.get("mode", "off")
    if mode == "off":
        return None
    if mode != "arm_scene_abort":
        raise ValueError(f"unsupported_contact_guard_mode:{mode}")
    return MeasuredArmContactGuard(env, penetration_limit_m=float(policy["penetration_limit_m"]))


def _save_tracking_audit(edge_dir: Path, trace: MeasuredTrace) -> None:
    plan_file = edge_dir / "kinematic_plan.npz"
    report_file = edge_dir / "motion_validation.json"
    if not plan_file.is_file() or not report_file.is_file():
        return
    with np.load(plan_file) as source:
        planned = source["arm_qpos"]
    validation = json.loads(report_file.read_text())
    actual = np.asarray(trace.robot_qpos)
    matched = min(len(planned), len(actual))
    joint_error = np.linalg.norm(actual[:matched] - planned[:matched], axis=1)
    target = np.asarray(trace.controller_target_poses)
    eef = np.asarray(trace.actual_eef_pose)
    count = min(len(target), len(eef) - 1)
    position_error = np.linalg.norm(eef[1 : count + 1, :3, 3] - target[:count, :3, 3], axis=1)
    rotation_error = []
    for actual_pose, target_pose in zip(eef[1 : count + 1], target[:count]):
        relative = target_pose[:3, :3] @ actual_pose[:3, :3].T
        rotation_error.append(float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    allowed = {tuple(pair) for pair in validation["allowed_body_pairs"]}
    forbidden = []
    checked = 0
    for step, payload in enumerate(trace.robot_contacts_json):
        contacts = json.loads(payload)
        if contacts == "NOT_CHECKED":
            continue
        checked += 1
        for contact in contacts:
            if contact["distance_m"] <= 0 and tuple(sorted(contact["body_ids"])) not in allowed:
                forbidden.append({"step": step, **contact})

    def summary(values: np.ndarray) -> dict[str, float | None]:
        return {
            "median": float(np.median(values)) if len(values) else None,
            "maximum": float(np.max(values)) if len(values) else None,
        }

    atomic_json(
        edge_dir / "tracking_audit.json",
        {
            "planned_ik_branch_complete": len(planned) == validation["total_waypoints"],
            "planned_frames": len(planned),
            "measured_frames": len(actual),
            "matched_prefix_frames": matched,
            "planned_ik_vs_actual_arm_l2_rad": summary(joint_error),
            "commanded_target_vs_actual_eef_position_m": summary(position_error),
            "commanded_target_vs_actual_eef_rotation_rad": summary(np.asarray(rotation_error)),
            "observed_contact_frames_checked": checked,
            "observed_unallowed_contact_count": len(forbidden),
            "observed_unallowed_contacts_first_eight": forbidden[:8],
            "interpretation": "discrete physical control frames; controller targets do not command IK qpos",
            "continuous_between_frames": "NOT_CHECKED",
            "substep_contacts": "NOT_CHECKED",
        },
    )


def _save_command_differences(node: Path) -> None:
    records = []
    for edge in sorted((node / "edges").glob("*")):
        if all(
            (edge / name).is_file()
            for name in ("option.json", "execution.npz", "executable_plan.npz")
        ):
            records.append((edge, json.loads((edge / "option.json").read_text())))
    reference = next((edge for edge, row in records if row["source_method"] == "Reference"), None)

    def measured(edge):
        with np.load(edge / "execution.npz") as data:
            commands = data["commanded_actions"]
        with np.load(edge / "executable_plan.npz") as data:
            indices = data["source_action_index"][1 : len(commands) + 1]
        occurrences = {}
        result = {}
        for index, command in zip(indices, commands):
            index = int(index)
            if index < 0:
                continue
            occurrence = occurrences.get(index, 0)
            result[(index, occurrence)] = command
            occurrences[index] = occurrence + 1
        return result, len(commands)

    for edge, record in records:
        result = {
            "schema": "cfact.actual-command-difference.v1",
            "reference_edge": None,
            "comparison": "same_source_action_index_and_retiming_occurrence",
            "plan_difference_is_separate": True,
        }
        if reference is None:
            result["status"] = "reference_execution_unavailable"
        else:
            baseline, ref_count = measured(reference)
            actions, action_count = measured(edge)
            common = sorted(baseline.keys() & actions.keys())
            differences = np.asarray([actions[index] - baseline[index] for index in common])
            result.update(
                {
                    "status": "measured" if common else "no_common_issued_source_indices",
                    "reference_edge": reference.name,
                    "matched_control_ticks": len(common),
                    "candidate_control_ticks": action_count,
                    "reference_control_ticks": ref_count,
                    "maximum_absolute_command_difference": float(np.max(np.abs(differences)))
                    if common
                    else None,
                    "mean_absolute_command_difference": float(np.mean(np.abs(differences)))
                    if common
                    else None,
                }
            )
        atomic_json(edge / "command_difference.json", result)


def _tail(
    env: Any,
    interface: Any,
    option: Any,
    trajectory: Any,
    consumed_waypoints: int,
    *,
    steps: int,
    object_ref: str,
    fixture_ref: str,
    joint_spec: dict[str, Any],
    quad: QuadReviewWriter,
    local_trace: MeasuredTrace,
    holding_object: bool,
    motion_validation_cfg: dict[str, Any] | None = None,
    local_termination_reason: str | None = None,
) -> tuple[MeasuredTrace, str, dict[str, Any]]:
    trace = MeasuredTrace(object_ref, fixture_ref, **joint_spec)
    trace.capture(env, interface)
    status = "horizon_censored"
    remaining = [waypoint for seq in trajectory.waypoint_sequences for waypoint in seq][
        consumed_waypoints:
    ]
    plan: dict[str, Any] = {
        "continuation_revision": "measured_child_connected_v4",
        "consumed_waypoints": consumed_waypoints,
        "remaining_waypoints": len(remaining),
        "full_task_goal": option.operation_context.task_goal,
        "source_cursor": {"issued_waypoints": consumed_waypoints},
        "physical_progress": {"joint_qpos": trace.fixture_qpos[0].tolist()},
    }
    if ledger(env):
        ledger(env).phase = "fixed_continuation"
    if local_termination_reason == "EXECUTION_ABORTED_BY_DECLARED_GUARD":
        plan["stop_reason"] = "local_declared_guard"
        return trace, "EXECUTION_ABORTED_BY_DECLARED_GUARD", plan
    if trace.task_success[-1]:
        plan["stop_reason"] = "official_success_at_child"
        return trace, "official_success", plan
    commands = np.asarray(local_trace.commanded_actions)
    gripper_index = commands.shape[1] - 6 if commands.ndim == 2 and len(commands) else None
    opening_seen = bool(gripper_index is not None and np.any(commands[:, gripper_index] < 0))
    if holding_object and not opening_seen:
        initial = local_trace.object_to_gripper_transform[0]
        latest = local_trace.object_to_gripper_transform[-1]
        slip = float(np.linalg.norm(latest[:3, 3] - initial[:3, 3]))
        plan["measured_grasp_translation_change_m"] = slip
        if slip > 0.03:
            plan["stop_reason"] = "lost_confirmed_grasp_relation"
            return trace, "NO_SUPPORTED_CONTINUATION", plan
    if option.goal_binding.get("kind") == "articulation_local_subgoal":
        spec = option.intended_fixture_subgoal
        names = list(joint_spec["fixture_joint_names"])
        joint = spec.get("joint")
        if joint not in names:
            plan["stop_reason"] = "named_joint_missing_at_child"
            return trace, "NO_SUPPORTED_CONTINUATION", plan
        contact_shift = float(
            np.linalg.norm(
                local_trace.object_to_gripper_transform[-1][:3, 3]
                - local_trace.object_to_gripper_transform[0][:3, 3]
            )
        )
        plan["measured_handle_tool_relation_change_m"] = contact_shift
        if contact_shift > 0.04:
            plan["stop_reason"] = "handle_tool_contact_relation_lost"
            return trace, "NO_SUPPORTED_CONTINUATION", plan
        try:
            raw_poses = articulation_followthrough(
                joint_name=joint,
                target_q=float(option.operation_context.task_goal["target_q"]),
                joint_unit=option.goal_binding["joint_unit"],
                stage=option.operation_context.stage,
                actual_child={"sim_time_s": trace.sim_time[0]},
                joint_type="slide" if option.goal_binding["joint_unit"] == "m" else "hinge",
                current_q=float(trace.fixture_qpos[0][names.index(joint)]),
                joint_range=tuple(spec["range"]),
                axis_world=np.asarray(
                    env.env.sim.data.xaxis[env.env.sim.model.joint_name2id(joint)]
                ),
                anchor_world=np.asarray(
                    env.env.sim.data.xanchor[env.env.sim.model.joint_name2id(joint)]
                ),
                current_handle_pose=trace.object_poses[0],
                current_eef_pose=trace.actual_eef_pose[0],
            )
        except (ValueError, KeyError, np.linalg.LinAlgError) as error:
            plan["stop_reason"] = f"articulation_followthrough:{error}"
            return trace, "NO_SUPPORTED_CONTINUATION", plan
        raw_grips = np.repeat(option.gripper_actions[-1:], len(raw_poses), axis=0)
        plan["transition_reason"] = "measured_joint_contact_followthrough"
    elif not remaining:
        plan["stop_reason"] = "no_remaining_reference_skill"
        return (
            trace,
            ("official_success" if trace.task_success[-1] else "NO_SUPPORTED_CONTINUATION"),
            plan,
        )
    else:
        raw_poses = np.asarray([waypoint.pose for waypoint in remaining])
        raw_grips = np.asarray([waypoint.gripper_action for waypoint in remaining])
    if opening_seen:
        # The object is no longer assumed attached. Align only the retreat to
        # the measured tool pose, keeping its relative shape and open command.
        alignment = trace.actual_eef_pose[0] @ np.linalg.inv(raw_poses[0])
        raw_poses = np.einsum("ij,njk->nik", alignment, raw_poses)
        raw_grips[:] = np.minimum(raw_grips, -1.0)
        plan["transition_reason"] = "release_seen_reanchor_tool_retreat"
    elif "transition_reason" not in plan:
        plan["transition_reason"] = "measured_child_connect_to_remaining_skill"
    try:
        executable, executable_grips, audit = connect_and_retime_poses(
            trace.actual_eef_pose[0], raw_poses, raw_grips, maximum_waypoints=2000
        )
    except ValueError as error:
        plan["stop_reason"] = str(error)
        return trace, "NO_SUPPORTED_CONTINUATION", plan
    plan.update(audit)
    plan["raw_planned_poses"] = raw_poses.round(9).tolist()
    plan["executable_poses"] = executable.round(9).tolist()
    plan["executable_gripper_actions"] = executable_grips.round(9).tolist()
    try:
        plan["motion_validation"] = _audit_motion_plan(
            env,
            executable,
            executable_grips,
            motion_validation_cfg or {},
            None,
            holding_object=holding_object and not opening_seen,
        )
    except MotionPlanRejected as error:
        plan["stop_reason"] = f"motion_plan_rejected:{error}"
        return trace, "NO_SUPPORTED_CONTINUATION", plan
    contact_guard = _measured_arm_guard(env, motion_validation_cfg or {})
    for step in range(steps):
        if trace.task_success[-1]:
            status = "official_success"
            break
        quad.append(env, trace.sim_time[-1], "fixed_continuation")
        if step + 1 < len(executable):
            target_pose, grip = executable[step + 1], executable_grips[step + 1]
        else:
            target_pose, grip = executable[-1], executable_grips[-1]
        action = interface.action_codec.command(target_pose, grip)
        env.step(action)
        trace.after_step(
            action,
            env,
            interface,
            False,
            target_pose=np.asarray(target_pose),
            raw_planned_action=action,
            transition_reason=plan["transition_reason"],
        )
        guard_measurement = contact_guard.check(trace) if contact_guard is not None else None
        if guard_measurement is not None:
            status = "EXECUTION_ABORTED_BY_DECLARED_GUARD"
            plan["guard_measurement"] = guard_measurement
            break
    if trace.task_success[-1] and status != "EXECUTION_ABORTED_BY_DECLARED_GUARD":
        status = "official_success"
    quad.append(env, trace.sim_time[-1], "ended")
    return trace, status, plan


def _save_clean_observation(env: Any, trace: MeasuredTrace, node: Path) -> None:
    from cfact.storage.schema import OBSERVATION_FIELDS

    arrays = trace.arrays()
    data = {key: arrays[key] for key in OBSERVATION_FIELDS if key in arrays}
    camera_rows = []
    rgb_frames = []
    depth_frames = []
    depth_missing = []
    observation = observe(env)
    sim = observation.sim
    for camera in QuadReviewWriter.CAMERAS:
        step_ledger = ledger(env)
        width, height = step_ledger.width, step_ledger.height
        rgb_frames.append(
            np.asarray(
                observation.render(height=height, width=width, camera_name=camera)[::-1], dtype=np.uint8
            )
        )
        try:
            rendered = observation.render(height=height, width=width, camera_name=camera, depth=True)
            depth = np.asarray(rendered[1] if isinstance(rendered, tuple) else rendered)
            if depth.shape != (height, width):
                raise ValueError(f"depth_shape_{depth.shape}")
            depth_frames.append(depth[::-1].astype(np.float32))
        except (TypeError, ValueError, RuntimeError) as error:
            depth_missing.append({"camera": camera, "reason": str(error)})
        camera_id = sim.model.camera_name2id(camera)
        fovy = float(sim.model.cam_fovy[camera_id])
        focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        camera_rows.append(
            {
                "name": camera,
                "width": width,
                "height": height,
                "intrinsics_px": [
                    [focal, 0.0, width / 2],
                    [0.0, focal, height / 2],
                    [0.0, 0.0, 1.0],
                ],
                "world_position_m": np.asarray(sim.data.cam_xpos[camera_id]).tolist(),
                "world_rotation_row_major": np.asarray(sim.data.cam_xmat[camera_id])
                .reshape(3, 3)
                .tolist(),
                "sim_time_s": trace.sim_time[0],
            }
        )
    data["rgb"] = np.stack(rgb_frames)
    data.update(observation.archive_values())
    if not depth_missing:
        data["depth"] = np.stack(depth_frames)
    np.savez_compressed(node / "observation.npz", **data)
    atomic_json(
        node / "cameras.json",
        {
            "observation_capture": observation.metadata,
            "cameras": camera_rows,
            "depth_missing": depth_missing,
            "rgb_color_order": "RGB",
            "depth_kind": "raw_MuJoCo_render_depth_buffer_not_metric",
            "clean_model_observation": True,
            "presentation_panel_excluded": True,
            "field_allowlist": list(OBSERVATION_FIELDS),
        },
    )


def _slice_source(segment: Any, cut: int) -> Any:
    motion = segment.motion_object_poses
    if motion is None or len(motion) < 5:
        raise ValueError("source_remaining_motion_too_short")
    cut = int(cut)
    if cut < 0 or cut >= len(segment.source_poses) - 3:
        raise ValueError("source_cursor_has_no_supported_remaining_segment")
    indices = segment.source_action_indices
    first_action = (
        int(indices[cut])
        if indices is not None and indices[cut] >= 0
        else segment.event_first_step + max(0, cut - 1)
    )
    return dataclasses.replace(
        segment,
        event_first_step=first_action,
        source_poses=segment.source_poses[cut:].copy(),
        gripper_actions=segment.gripper_actions[cut:].copy(),
        eef_pose_at_entry=segment.source_poses[cut].copy(),
        motion_object_poses=motion[cut:].copy(),
        source_action_indices=None if indices is None else indices[cut:].copy(),
    )


def _expand_child_pool(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    env: Any,
    interface: Any,
    root: dict[str, Any],
    root_segment: Any,
    donors: list[Any],
    ancestor_option: Any,
    ancestor_edge: Path,
    root_dir: Path,
    root_key: str,
    object_ref: str,
    fixture_ref: str,
    contract: Any,
    joint_spec: dict[str, Any],
    charged_steps: int,
    source_transform: Any,
) -> tuple[dict[str, Any], int, list[Any]]:
    from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory

    run = args.run.resolve()
    child_snapshot = load_snapshot(ancestor_edge / "child_snapshot.json")
    child_state = np.asarray(child_snapshot["states"])
    _restore(env, root["initial"], child_snapshot)
    info = current_info(env, interface)
    if bool(env.is_success()["task"]):
        raise ValueError("successful_child_not_eligible")
    node_id = (
        "n_" + hashlib.sha256(str(ancestor_edge.relative_to(root_dir)).encode()).hexdigest()[:12]
    )
    node = root_dir / "nodes" / node_id
    node.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(node / "context.npz", sim_state=child_state)
    parent_trace = MeasuredTrace(object_ref, fixture_ref, **joint_spec)
    parent_trace.capture(env, interface)
    _save_clean_observation(env, parent_trace, node)
    saved_child_context = json.loads((ancestor_edge / "child_context.json").read_text())
    inherited = saved_child_context["goal_binding"]
    inherited_context = saved_child_context["operation_context"]
    if cfg.get("method", {}).get("recursive_goals_v2"):
        anchor_record = json.loads((ancestor_edge / "event_anchor.json").read_text())
        contract = dataclasses.replace(contract, holding_object=anchor_record["evidence"]["held_verified"])

    atomic_json(
        node / "context.json",
        {
            "node_id": node_id,
            "parent_edge": _relative(ancestor_edge, run),
            "sim_state_sha256": array_digest(child_state),
            "sim_time_s": parent_trace.sim_time[0],
            "goal_binding": inherited,
            "operation_stage": "placement_approach"
            if contract.holding_object
            else "articulation_followthrough",
            "fixture_joint_names": joint_spec["fixture_joint_names"],
            "fixture_joint_types": joint_spec["fixture_joint_types"],
            "fixture_joint_units": joint_spec["fixture_joint_units"],
            "fixture_joint_qvel_units": joint_spec["fixture_joint_qvel_units"],
        },
    )
    source = next(
        (d for d in donors if d.episode_id == ancestor_option.source_episode), root_segment
    )
    ancestor_result = json.loads((ancestor_edge / "local_result.json").read_text())
    cursor = ancestor_result["source_cursor"]
    next_action = int(cursor["next_action_index"])
    source_indices = source.source_action_indices
    remaining = (
        np.flatnonzero(np.asarray(source_indices) >= next_action)
        if source_indices is not None
        else np.asarray([next_action - source.event_first_step + 1])
    )
    from cfact.generation.stage_compiler import source_suffix
    suffix_status, cut = source_suffix(source, next_action)
    raw_suffix_status=suffix_status
    if suffix_status=="AVAILABLE" and inherited_context.get("joint_binding") and cfg.get("method",{}).get("validate_source_suffix_entry"):
        from cfact.generation.stage_compiler import contact_suffix_compatibility
        limits=next(row.get("reference_neighborhood",{}) for row in cfg["tasks"] if row["task"]==args.task)
        suffix_status,evidence=contact_suffix_compatibility(source,cut,np.asarray(info.object_poses[object_ref]),np.asarray(info.eef_pose),limits)
        atomic_json(node/"source_suffix_compatibility.json",{**evidence,"raw_suffix_status":raw_suffix_status,
            "source_suffix_status":suffix_status,"source_cursor":next_action,"source_episode":source.episode_id,
            "classification_before_candidate_execution":True})
    progress = (cut or 0) / max(1, len(source.source_poses) - 1)
    source_prior = source
    if suffix_status == "AVAILABLE":
        try:
            source = _slice_source(source, cut)
        except ValueError:
            suffix_status = "INCOMPATIBLE"
    if suffix_status != "AVAILABLE" and not cfg.get("method", {}).get("recursive_goals_v2"):
        atomic_json(node / "candidate_pool.json", {"selected_ids": [],
            "reason": "SOURCE_SUFFIX_" + suffix_status, "frozen_before_execution": True})
        return {"node_id": node_id, "status": "NO_SUPPORTED_CONTINUATION", "edges": []}, charged_steps, []
    suffix_donors = []
    for donor in donors:
        try:
            suffix_donors.append(
                _slice_source(donor, int(round(progress * (len(donor.source_poses) - 1))))
            )
        except ValueError:
            pass
    operation_context = _operation_context(
        interface,
        info,
        parent_id=f"{root_key}/{node_id}",
        object_ref=object_ref,
        fixture_ref=fixture_ref,
        contract=contract,
        source=source,
        task_goal=(
            {
                **inherited_context["task_goal"],
                **inherited,
                "source_task_intent": inherited_context["task_goal"],
            }
            if contract.holding_object and "target_pose_world" in inherited
            else inherited_context["task_goal"]
        ),
        joint_binding=inherited_context["joint_binding"],
        stage=inherited_context["stage"] if cfg.get("method", {}).get("recursive_goals_v2") else ("placement_approach" if contract.holding_object else "articulation_followthrough"),
        source_cursor=cursor,
    )
    child_context_record = json.loads((ancestor_edge / "child_context.json").read_text())
    node_context = json.loads((node / "context.json").read_text())
    node_context.update(
        {
            "operation_context": operation_context.record(),
            "history_refs": child_context_record["history_refs"],
            "snapshot": str(ancestor_edge / "child_snapshot.json"),
        }
    )
    atomic_json(node / "context.json", node_context)
    goals = _propose_goals(args, cfg, operation_context, node)
    raw, donor_audit = build_object_effect_options(
        g0_amplitude_m=_g0_amplitude(cfg,env,args,source,info,contract,operation_context.joint_binding,operation_context.task_goal,node),
        root=source,
        donors=suffix_donors,
        operation_context=operation_context,
        current_fixture_pose=np.asarray(info.object_poses[fixture_ref]),
        current_object_pose=np.asarray(info.object_poses[object_ref]),
        current_eef_pose=np.asarray(info.eef_pose),
        contract=contract,
        official_transform=source_transform,
        parent_context_id=f"{root_key}/{node_id}",
        end_event="placement_ready" if contract.holding_object else "articulation_subgoal_reached",
        local_timeout=cfg["execution"]["local_timeout_steps"],
        max_raw_per_generator=cfg["candidate_pool"]["max_raw_proposals_per_generator"],
        compatibility_limits=next(
            row["reference_neighborhood"] for row in cfg["tasks"] if row["task"] == args.task
        ),
    )
    if contract.holding_object and inherited.get("kind") in {
        "fixture_region_object_goal",
        "coffee_receptacle_object_goal",
    }:
        raw = bind_inherited_placement_goal(raw, operation_context)
    elif contract.holding_object:
        try:
            if args.task == "CoffeeSetupMug":
                machine = env.env.coffee_machine
                site_id = env.env.sim.model.site_name2id(
                    f"{machine.naming_prefix}receptacle_place_site"
                )
                raw, _ = solve_g2_coffee_receptacle_goals(
                    raw,
                    pour_site_world_m=np.asarray(env.env.sim.data.site_xpos[site_id]),
                    current_object_pose=np.asarray(info.object_poses[object_ref]),
                    current_eef_pose=np.asarray(info.eef_pose),
                )
            else:
                regions, bbox = _support_geometry(env, args.task)
                raw, _ = solve_g2_support_goals(
                    raw,
                    support_regions=regions,
                    object_bbox_local=bbox,
                    current_object_pose=np.asarray(info.object_poses[object_ref]),
                    current_eef_pose=np.asarray(info.eef_pose),
                )
        except (AttributeError, KeyError, ValueError):
            pass
    else:
        try:
            articulation = _articulation_geometry(
                env, operation_context.joint_binding, operation_context.task_goal
            )
            raw, _ = solve_g2_articulation_subgoals(
                raw,
                **{
                    key: value
                    for key, value in articulation.items()
                    if key != "joint_to_handle_distance_m"
                },
                current_handle_pose=np.asarray(info.object_poses[object_ref]),
                current_eef_pose=np.asarray(info.eef_pose),
            )
        except (AttributeError, KeyError, ValueError):
            pass
    if cfg.get("method", {}).get("recursive_goals_v2"):
        raw = _compile_goals(args, cfg, env, interface, operation_context, goals,
            source if suffix_status == "AVAILABLE" else source_prior, suffix_status, node,
            raw if suffix_status == "AVAILABLE" else [])
    selected, rejected = _select_pool(cfg,
        raw,
        max_executed=cfg["candidate_pool"]["max_executed_options_per_node"],
        max_g0=cfg["candidate_pool"]["max_selected_g0"],
    )
    executable_selected = []
    for option in selected:
        if option.intended_object_path is None:
            rejected.append(
                {
                    "candidate_id": option.candidate_id,
                    "reason": "PRE_EXECUTION_REJECTED:object_path_missing",
                }
            )
            continue
        if not _option_has_event_geometry(option, contract.primitive_family, 0.02):
            rejected.append(
                {
                    "candidate_id": option.candidate_id,
                    "reason": "PRE_EXECUTION_REJECTED:planned_child_motion_below_event_guard",
                }
            )
            continue
        executable_selected.append(option)
    selected = executable_selected
    with (node / "proposals.jsonl").open("w") as stream:
        for option in raw:
            stream.write(json.dumps(_card(option), sort_keys=True) + "\n")
    selected = _engineering_select(selected, rejected, cfg)
    atomic_json(
        node / "candidate_pool.json",
        {
            "selected_ids": [o.candidate_id for o in selected],
            "rejected": rejected,
            "donor_audit": donor_audit,
            "frozen_before_execution": True,
            "parent_edge": _relative(ancestor_edge, run),
        },
    )
    if len(selected) < 2:
        return (
            {"node_id": node_id, "status": "NO_BRANCH_INSUFFICIENT_DISTINCT_OPTIONS", "edges": []},
            charged_steps,
            [],
        )
    edges = []
    for option in selected:
        charged_steps = ledger(env).control_steps
        _check_storage(run, cfg)
        edge_dir = node / "edges" / option.candidate_id
        edge_dir.mkdir(parents=True, exist_ok=True)
        physical_seed = stable_seed(
            cfg["run_seed"],
            args.task,
            args.root_id,
            node_id,
            option.candidate_id,
            "physical_execution",
        )
        atomic_json(
            edge_dir / "option.json",
            {
                **_card(option),
                "candidate_rng_seed": physical_seed,
                "physics_rng_policy": "restored_from_parent_snapshot",
            },
        )
        np.savez_compressed(
            edge_dir / "plan.npz",
            intended_eef_poses=option.poses,
            intended_object_poses=option.intended_object_path
            if option.intended_object_path is not None
            else np.empty((0, 4, 4)),
            gripper_actions=option.gripper_actions,
        )
        if (
            charged_steps + option.local_timeout + cfg["execution"]["continuation_steps"]
            > cfg["execution"]["max_env_steps_per_root"]
        ):
            edges.append({"candidate_id": option.candidate_id, "status": "BUDGET_CENSORED"})
            continue
        quad = None
        started = time.monotonic()
        try:
            _case_seed(
                cfg, args.task, args.root_id, node_id, option.candidate_id, "physical_execution"
            )
            _restore(env, root["initial"], child_snapshot)
            before = current_info(env, interface)
            trajectory, connection, executable_poses, executable_grips = _execution_trajectory(
                WaypointSequence, WaypointTrajectory, np.asarray(before.eef_pose), option, cfg
            )
            np.savez_compressed(
                edge_dir / "executable_plan.npz",
                intended_eef_poses=executable_poses,
                gripper_actions=executable_grips,
                source_waypoint_index=np.asarray(connection["source_waypoint_index"]),
                source_action_index=np.asarray(connection["source_action_index"]),
            )
            atomic_json(
                edge_dir / "connection.json",
                {k: v for k, v in connection.items() if k != "source_waypoint_index"},
            )
            _audit_motion_plan(
                env,
                executable_poses,
                executable_grips,
                cfg,
                edge_dir,
                holding_object=contract.holding_object,
            )
            path = (
                run
                / "videos"
                / ("reference" if option.source_method == "Reference" else option.source_method)
                / args.split
                / args.task
                / root_key
                / f"{node_id}__{option.candidate_id}__quad.mp4"
            )
            quad = QuadReviewWriter(
                path,
                task=args.task,
                root=args.root_id,
                node=node_id,
                candidate=option.candidate_id,
                method=option.source_method,
                intent=option.subtype,
                fps=cfg["video"]["fps"],
                cell_width=cfg["video"]["cell_width"],
                cell_height=cfg["video"]["cell_height"],
                feature_scale=cfg["video"].get("feature_scale", 1.0),
            )
            positions = option.intended_object_path[:, :3, 3]
            gripper_values = option.gripper_actions[:, 0]
            release = np.where((gripper_values[:-1] > 0) & (gripper_values[1:] < 0))[0]
            goal_index = (
                max(1, int(release[0]) - 3)
                if contract.holding_object and len(release)
                else min(len(positions) - 1, max(1, int(len(positions) * 0.8)))
            )
            if option.subtype == "articulation_subgoal":
                goal_index = len(positions) - 1
            detector = LocalEventDetector(
                goal_xyz=positions[goal_index],
                initial_xyz=np.asarray(before.object_poses[object_ref])[:3, 3],
                held=contract.holding_object,
                minimum_motion_m=0.015,
                joint_goal=option.intended_fixture_subgoal.get("desired_q"),
                joint_index=(
                    joint_spec["fixture_joint_names"].index(
                        option.intended_fixture_subgoal["joint"]
                    )
                    if "joint" in option.intended_fixture_subgoal
                    else None
                ),
            )
            trace = MeasuredTrace(object_ref, fixture_ref, **joint_spec)
            if ledger(env):
                ledger(env).begin(str(edge_dir), "local_operation")
            local = execute_option(
                env,
                interface,
                trajectory,
                trace,
                detector,
                max_steps=option.local_timeout,
                minimum_event_steps=_minimum_event_steps(cfg,option),
                frame_callback=quad.append,
                contact_guard=_measured_arm_guard(env, cfg),
                source_action_indices=connection["source_action_index"][1:],
                source_episode=option.source_episode,
                initial_source_cursor=(
                    option.source_fragment[0]
                    if option.source_fragment is not None
                    else option.operation_context.source_cursor
                ),
            )
            new_child = _save_actual_child(env, edge_dir, option, local, trace)
            np.savez_compressed(edge_dir / "execution.npz", **trace.arrays())
            _save_tracking_audit(edge_dir, trace)
            tail, tail_status, tail_plan = _tail(
                env,
                interface,
                option,
                trajectory,
                local["consumed_waypoints"],
                steps=cfg["execution"]["continuation_steps"],
                object_ref=object_ref,
                fixture_ref=fixture_ref,
                joint_spec=joint_spec,
                quad=quad,
                local_trace=trace,
                holding_object=contract.holding_object,
                motion_validation_cfg=cfg,
                local_termination_reason=local["termination_reason"],
            )
            charged_steps += len(trace.commanded_actions) + len(tail.commanded_actions)
            cont = edge_dir / "continuation"
            cont.mkdir(exist_ok=True)
            np.savez_compressed(cont / "execution.npz", **tail.arrays())
            atomic_json(cont / "plan.json", tail_plan)
            atomic_json(
                cont / "result.json",
                {
                    "continuation_revision": "measured_child_connected_v4",
                    "task_horizon_control_steps": cfg["execution"]["continuation_steps"],
                    "goal_binding": option.goal_binding,
                    "status": tail_status,
                    "official_success_end": tail.task_success[-1],
                },
            )
            atomic_json(
                edge_dir / "goal_realization.json",
                _goal_realization(option, trace, tail, goal_index),
            )
            media = quad.close()
            quad = None
            media["video"] = _relative(path, run)
            atomic_json(edge_dir / "media.json", media)
            atomic_json(
                edge_dir / "local_result.json",
                {
                    **local,
                    "child_state_sha256": array_digest(new_child),
                    "planned_subgoal_index": goal_index,
                    "planned_path_length": len(positions),
                    "Y_fixed": {
                        "status": tail_status,
                        "official_success_end": tail.task_success[-1],
                    },
                    "outcome_labels": _outcome_labels(local, tail_status),
                },
            )
            atomic_json(
                edge_dir / "costs.json",
                {
                    "control_steps_local": len(trace.commanded_actions),
                    "control_steps_fixed": len(tail.commanded_actions),
                    "simulation_seconds_local": trace.sim_time[-1] - trace.sim_time[0],
                    "simulation_seconds_fixed": tail.sim_time[-1] - tail.sim_time[0],
                    "restore_wall_seconds_total": ledger(env).restore_wall_seconds,
                    "wall_seconds": time.monotonic() - started,
                },
            )
            edges.append(
                {
                    "candidate_id": option.candidate_id,
                    "status": "PHYSICS_COMPLETE",
                    "local_event": local["termination_reason"],
                    "Y_fixed": tail_status,
                    "edge_path": _relative(edge_dir, run),
                }
            )
        except MotionPlanRejected as error:
            if quad is not None:
                try:
                    quad.close()
                except Exception:
                    pass
            atomic_json(
                edge_dir / "pre_execution_rejection.json",
                {
                    "status": "PRE_EXECUTION_REJECTED",
                    "reason": str(error),
                    "motion_validation": "motion_validation.json",
                },
            )
            edges.append(
                {
                    "candidate_id": option.candidate_id,
                    "status": "PRE_EXECUTION_REJECTED",
                    "reason": str(error),
                    "edge_path": _relative(edge_dir, run),
                }
            )
        except Exception as error:
            if quad is not None:
                try:
                    quad.close()
                except Exception:
                    pass
            atomic_json(
                edge_dir / "technical_error.json",
                {
                    "status": "TECHNICAL_ERROR",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
            edges.append(
                {
                    "candidate_id": option.candidate_id,
                    "status": "TECHNICAL_ERROR",
                    "error": str(error),
                    "edge_path": _relative(edge_dir, run),
                }
            )
    _save_command_differences(node)
    charged_steps = ledger(env).control_steps
    return (
        {
            "node_id": node_id,
            "parent_edge": _relative(ancestor_edge, run),
            "status": "PHYSICS_COMPLETE"
            if all(
                e["status"] in {"PHYSICS_COMPLETE", "BUDGET_CENSORED", "PRE_EXECUTION_REJECTED"}
                for e in edges
            )
            else "PARTIAL",
            "edges": edges,
        },
        charged_steps,
        selected,
    )


def _expand_release_pool(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    env: Any,
    interface: Any,
    root: dict[str, Any],
    ancestor_option: Any,
    ancestor_edge: Path,
    root_dir: Path,
    root_key: str,
    object_ref: str,
    fixture_ref: str,
    contract: Any,
    joint_spec: dict[str, Any],
    charged_steps: int,
) -> tuple[dict[str, Any], int]:
    """Third semantic stage: release and retreat from a measured placement-ready child."""
    from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory
    from cfact.generation.options import _stable_option_id

    run = args.run.resolve()
    parent_snapshot = load_snapshot(ancestor_edge / "child_snapshot.json")
    state = np.asarray(parent_snapshot["states"])
    _restore(env, root["initial"], parent_snapshot)
    info = current_info(env, interface)
    if bool(env.is_success()["task"]):
        raise ValueError("successful_child_not_eligible")
    ancestor_result = json.loads((ancestor_edge / "local_result.json").read_text())
    with np.load(ancestor_edge / "executable_plan.npz") as saved_plan:
        waypoint_map = saved_plan["source_waypoint_index"]
    issued = min(int(ancestor_result["consumed_waypoints"]), len(waypoint_map) - 1)
    start = max(0, int(waypoint_map[issued]) + 1)
    source_poses = ancestor_option.poses[start:].copy()
    source_grip = ancestor_option.gripper_actions[start:].copy()
    node_id = (
        "n_" + hashlib.sha256(str(ancestor_edge.relative_to(root_dir)).encode()).hexdigest()[:12]
    )
    node = root_dir / "nodes" / node_id
    node.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(node / "context.npz", sim_state=state)
    parent_trace = MeasuredTrace(object_ref, fixture_ref, **joint_spec)
    parent_trace.capture(env, interface)
    _save_clean_observation(env, parent_trace, node)
    binding = ancestor_option.goal_binding
    atomic_json(
        node / "context.json",
        {
            "node_id": node_id,
            "parent_edge": _relative(ancestor_edge, run),
            "sim_state_sha256": array_digest(state),
            "sim_time_s": parent_trace.sim_time[0],
            "goal_binding": binding,
            "operation_stage": "release_and_retreat",
            "fixture_joint_names": joint_spec["fixture_joint_names"],
            "fixture_joint_types": joint_spec["fixture_joint_types"],
            "fixture_joint_units": joint_spec["fixture_joint_units"],
            "fixture_joint_qvel_units": joint_spec["fixture_joint_qvel_units"],
        },
    )
    reason = None
    if len(source_poses) < 5 or source_grip[0, 0] <= 0 or not np.any(source_grip[:, 0] < 0):
        reason = "NO_RECORDED_RELEASE_SUFFIX"
    target_from_binding = binding.get("kind") in {
        "fixture_region_object_goal",
        "coffee_receptacle_object_goal",
    }
    if not target_from_binding and binding.get("kind") != "source_task_goal":
        reason = "NO_INHERITED_PLACEMENT_GOAL"
    if reason is not None:
        atomic_json(
            node / "candidate_pool.json",
            {"selected_ids": [], "reason": reason, "frozen_before_execution": True},
        )
        return {"node_id": node_id, "status": reason, "edges": []}, charged_steps
    current_eef = np.asarray(info.eef_pose)
    align = current_eef @ np.linalg.inv(source_poses[0])
    base_poses = np.einsum("ij,njk->nik", align, source_poses)
    current_object = np.asarray(info.object_poses[object_ref])
    goal_xyz = (
        np.asarray(binding["target_pose_world"])[:3, 3]
        if target_from_binding
        else np.asarray(ancestor_option.intended_object_path)[-1, :3, 3].copy()
    )
    parent_context = f"{root_key}/{node_id}"
    release_context = dataclasses.replace(
        ancestor_option.operation_context,
        parent_context_id=parent_context,
        stage="release_and_retreat",
        current_fixture_pose=np.asarray(info.object_poses[fixture_ref]),
        current_object_pose=current_object,
        current_eef_pose=current_eef,
        source_cursor=int(ancestor_result["source_cursor"]["next_action_index"]),
    )
    selected = []
    for ordinal, (method, dwell) in enumerate((("Reference", 0), ("G0", 5))):
        poses = (
            base_poses
            if dwell == 0
            else np.concatenate([np.repeat(base_poses[:1], dwell, axis=0), base_poses])
        )
        grip = (
            source_grip
            if dwell == 0
            else np.concatenate([np.repeat(source_grip[:1], dwell, axis=0), source_grip])
        )
        if not np.any(grip[:, 0] < 0):
            continue
        intended_object = np.repeat(current_object[None], len(poses), axis=0)
        option = dataclasses.replace(
            ancestor_option,
            candidate_id=_stable_option_id(
                parent_context, method, "release_and_retreat", poses, binding, ordinal
            ),
            parent_context_id=parent_context,
            operation_context=release_context,
            source_action_indices=np.concatenate(
                [
                    np.repeat(ancestor_option.source_action_indices[start : start + 1], dwell),
                    ancestor_option.source_action_indices[start:],
                ]
            )
            if ancestor_option.source_action_indices is not None
            else None,
            candidate_plan=None,
            source_method=method,
            subtype="release_and_retreat",
            decision_group_id=parent_context,
            intended_object_path=intended_object,
            poses=poses,
            gripper_actions=grip,
            end_event="release_complete",
            local_timeout=cfg["execution"]["local_timeout_steps"],
            motion_template="recorded_release_suffix_with_current_child_alignment",
            gripper_event_policy={"kind": "release_window", "extra_closed_dwell_steps": dwell},
            mode_descriptor={"stage": "release_and_retreat", "closed_dwell_steps": dwell},
            provenance=(ancestor_option.candidate_id,),
        )
        selected.append(option)
    with (node / "proposals.jsonl").open("w") as stream:
        for option in selected:
            stream.write(json.dumps(_card(option), sort_keys=True) + "\n")
    atomic_json(
        node / "candidate_pool.json",
        {
            "selected_ids": [x.candidate_id for x in selected],
            "frozen_before_execution": True,
            "parent_edge": _relative(ancestor_edge, run),
            "stage": "release_and_retreat",
        },
    )
    if len(selected) < 2:
        return {
            "node_id": node_id,
            "status": "NO_BRANCH_INSUFFICIENT_DISTINCT_OPTIONS",
            "edges": [],
        }, charged_steps
    edges = []
    for option in selected:
        charged_steps = ledger(env).control_steps
        _check_storage(run, cfg)
        edge_dir = node / "edges" / option.candidate_id
        edge_dir.mkdir(parents=True, exist_ok=True)
        physical_seed = stable_seed(
            cfg["run_seed"],
            args.task,
            args.root_id,
            node_id,
            option.candidate_id,
            "physical_execution",
        )
        atomic_json(
            edge_dir / "option.json",
            {
                **_card(option),
                "candidate_rng_seed": physical_seed,
                "physics_rng_policy": "restored_from_parent_snapshot",
            },
        )
        np.savez_compressed(
            edge_dir / "plan.npz",
            intended_eef_poses=option.poses,
            intended_object_poses=option.intended_object_path,
            gripper_actions=option.gripper_actions,
        )
        if (
            charged_steps + option.local_timeout + cfg["execution"]["continuation_steps"]
            > cfg["execution"]["max_env_steps_per_root"]
        ):
            atomic_json(
                edge_dir / "budget_censor.json",
                {"status": "BUDGET_CENSORED", "charged_steps": charged_steps},
            )
            edges.append(
                {
                    "candidate_id": option.candidate_id,
                    "status": "BUDGET_CENSORED",
                    "edge_path": _relative(edge_dir, run),
                }
            )
            continue
        quad = None
        started = time.monotonic()
        try:
            _case_seed(
                cfg, args.task, args.root_id, node_id, option.candidate_id, "physical_execution"
            )
            _restore(env, root["initial"], parent_snapshot)
            before = current_info(env, interface)
            trajectory, connection, executable_poses, executable_grips = _execution_trajectory(
                WaypointSequence, WaypointTrajectory, np.asarray(before.eef_pose), option, cfg
            )
            np.savez_compressed(
                edge_dir / "executable_plan.npz",
                intended_eef_poses=executable_poses,
                gripper_actions=executable_grips,
                source_waypoint_index=np.asarray(connection["source_waypoint_index"]),
                source_action_index=np.asarray(connection["source_action_index"]),
            )
            atomic_json(
                edge_dir / "connection.json",
                {k: v for k, v in connection.items() if k != "source_waypoint_index"},
            )
            _audit_motion_plan(
                env,
                executable_poses,
                executable_grips,
                cfg,
                edge_dir,
                holding_object=contract.holding_object,
            )
            path = (
                run
                / "videos"
                / ("reference" if option.source_method == "Reference" else option.source_method)
                / args.split
                / args.task
                / root_key
                / f"{node_id}__{option.candidate_id}__quad.mp4"
            )
            quad = QuadReviewWriter(
                path,
                task=args.task,
                root=args.root_id,
                node=node_id,
                candidate=option.candidate_id,
                method=option.source_method,
                intent=option.subtype,
                fps=cfg["video"]["fps"],
                cell_width=cfg["video"]["cell_width"],
                cell_height=cfg["video"]["cell_height"],
                feature_scale=cfg["video"].get("feature_scale", 1.0),
            )
            gripper_index = len(
                interface.target_pose_to_action(target_pose=np.asarray(before.eef_pose))
            )
            detector = ReleaseEventDetector(target_xyz=goal_xyz, gripper_action_index=gripper_index)
            trace = MeasuredTrace(object_ref, fixture_ref, **joint_spec)
            if ledger(env):
                ledger(env).begin(str(edge_dir), "local_operation")
            local = execute_option(
                env,
                interface,
                trajectory,
                trace,
                detector,
                max_steps=option.local_timeout,
                minimum_event_steps=_minimum_event_steps(cfg,option),
                frame_callback=quad.append,
                contact_guard=_measured_arm_guard(env, cfg),
                source_action_indices=connection["source_action_index"][1:],
                source_episode=option.source_episode,
                initial_source_cursor=(
                    option.source_fragment[0]
                    if option.source_fragment is not None
                    else option.operation_context.source_cursor
                ),
            )
            child_state = _save_actual_child(env, edge_dir, option, local, trace)
            np.savez_compressed(edge_dir / "execution.npz", **trace.arrays())
            _save_tracking_audit(edge_dir, trace)
            tail, tail_status, tail_plan = _tail(
                env,
                interface,
                option,
                trajectory,
                local["consumed_waypoints"],
                steps=cfg["execution"]["continuation_steps"],
                object_ref=object_ref,
                fixture_ref=fixture_ref,
                joint_spec=joint_spec,
                quad=quad,
                local_trace=trace,
                holding_object=True,
                motion_validation_cfg=cfg,
                local_termination_reason=local["termination_reason"],
            )
            charged_steps += len(trace.commanded_actions) + len(tail.commanded_actions)
            cont = edge_dir / "continuation"
            cont.mkdir(exist_ok=True)
            np.savez_compressed(cont / "execution.npz", **tail.arrays())
            atomic_json(cont / "plan.json", tail_plan)
            atomic_json(
                cont / "result.json",
                {
                    "continuation_revision": "measured_child_connected_v4",
                    "task_horizon_control_steps": cfg["execution"]["continuation_steps"],
                    "goal_binding": binding,
                    "status": tail_status,
                    "official_success_end": tail.task_success[-1],
                },
            )
            atomic_json(
                edge_dir / "goal_realization.json",
                _goal_realization(option, trace, tail, len(option.intended_object_path) - 1),
            )
            media = quad.close()
            quad = None
            media["video"] = _relative(path, run)
            atomic_json(edge_dir / "media.json", media)
            atomic_json(
                edge_dir / "local_result.json",
                {
                    **local,
                    "child_state_sha256": array_digest(child_state),
                    "Y_fixed": {
                        "status": tail_status,
                        "official_success_end": tail.task_success[-1],
                    },
                    "outcome_labels": _outcome_labels(local, tail_status),
                },
            )
            atomic_json(
                edge_dir / "costs.json",
                {
                    "control_steps_local": len(trace.commanded_actions),
                    "control_steps_fixed": len(tail.commanded_actions),
                    "simulation_seconds_local": trace.sim_time[-1] - trace.sim_time[0],
                    "simulation_seconds_fixed": tail.sim_time[-1] - tail.sim_time[0],
                    "restore_wall_seconds_total": ledger(env).restore_wall_seconds,
                    "wall_seconds": time.monotonic() - started,
                },
            )
            edges.append(
                {
                    "candidate_id": option.candidate_id,
                    "status": "PHYSICS_COMPLETE",
                    "local_event": local["termination_reason"],
                    "Y_fixed": tail_status,
                    "edge_path": _relative(edge_dir, run),
                }
            )
        except MotionPlanRejected as error:
            if quad is not None:
                try:
                    quad.close()
                except Exception:
                    pass
            atomic_json(
                edge_dir / "pre_execution_rejection.json",
                {
                    "status": "PRE_EXECUTION_REJECTED",
                    "reason": str(error),
                    "motion_validation": "motion_validation.json",
                },
            )
            edges.append(
                {
                    "candidate_id": option.candidate_id,
                    "status": "PRE_EXECUTION_REJECTED",
                    "reason": str(error),
                    "edge_path": _relative(edge_dir, run),
                }
            )
        except Exception as error:
            if quad is not None:
                try:
                    quad.close()
                except Exception:
                    pass
            atomic_json(
                edge_dir / "technical_error.json",
                {
                    "status": "TECHNICAL_ERROR",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
            edges.append(
                {
                    "candidate_id": option.candidate_id,
                    "status": "TECHNICAL_ERROR",
                    "error": str(error),
                    "edge_path": _relative(edge_dir, run),
                }
            )
    return {
        "node_id": node_id,
        "parent_edge": _relative(ancestor_edge, run),
        "status": "PHYSICS_COMPLETE"
        if all(
            e["status"] in {"PHYSICS_COMPLETE", "BUDGET_CENSORED", "PRE_EXECUTION_REJECTED"}
            for e in edges
        )
        else "PARTIAL",
        "edges": edges,
    }, charged_steps


def _source_attempt_result(route: dict[str, Any] | None) -> dict[str, Any] | None:
    if not route or route.get("source_class") != "historical_synthetic_source_attempt":
        return None
    package_value = route.get("source_identity", {}).get("package_path")
    if not package_value:
        return None
    package = Path(package_value)
    if not package.is_absolute():
        package = PROJECT / package
    result_path = package.resolve().parent / "result.json"
    if not result_path.is_file():
        return None
    raw = result_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return {"path": str(result_path), "sha256": hashlib.sha256(raw).hexdigest(), "payload": payload}


def _record_blocked_source_replay(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    root: dict[str, Any],
    route: dict[str, Any],
    dataset: Path,
    source_group: str,
    error: Exception,
    attempt_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Record a source attempt whose prior failure is corroborated by failed physical replay."""
    run = args.run.resolve()
    state = np.asarray(root["initial"]["states"])
    root_key = f"{args.task}__{args.root_id}__{array_digest(state)[:12]}"
    root_dir = run / "data" / args.split / args.task / root_key
    root_dir.mkdir(parents=True, exist_ok=True)
    identity = route.get("source_identity", {})
    donor_routes = [
        {
            key: donor.get(key)
            for key in (
                "source_id",
                "prepared_hdf5",
                "prepared_group",
                "prepared_sha256",
                "source_identity_sha256",
            )
        }
        for donor in route.get("donors", [])
    ]
    root_manifest = {
        "root_id": root_key,
        "task": args.task,
        "split": args.split,
        "source_episode": args.root_id,
        "source_group": source_group,
        "source_id": route.get("source_id"),
        "source_class": route.get("source_class"),
        "source_identity": identity,
        "source_identity_sha256": route.get("source_identity_sha256"),
        "role_in_release": route.get("role_in_release"),
        "prepared_hdf5": str(dataset),
        "prepared_sha256": route.get("prepared_sha256"),
        "donor_source_routes": donor_routes,
        "generator_revision": cfg["generator_revision"],
        "seed": stable_seed(cfg["run_seed"], args.task, args.root_id, "root_initialization"),
        "worker_pid": os.getpid(),
        "physical_gpu_index": args.gpu,
        "physical_gpu_uuid": os.environ.get("CFACT_GPU_UUID", "UNKNOWN_STANDALONE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "source_replay_history": root.get("source_replay_history"),
        "source_replay_control_steps": int(root.get("actual_replay_steps", 0)),
        "source_replay_block": {
            "status": "BLOCKED_SOURCE_REPLAY",
            "reason": str(error),
            "prior_source_outcome": attempt_evidence,
        },
    }
    result = {
        "task": args.task,
        "root_id": root_key,
        "split": args.split,
        "status": "BLOCKED_SOURCE_REPLAY",
        "nodes": [],
        "edges": [],
        "charged_control_steps": int(root.get("actual_replay_steps", 0)),
        "source_replay_history": root.get("source_replay_history"),
        "official_task_failure_asserted": False,
        "requested_max_decision_depth": cfg["tree_limits"]["maximum_decision_depth"],
        "observed_max_decision_depth": 0,
        "source_replay_block": root_manifest["source_replay_block"],
        "limitations": [
            "The source replay did not reach the required measured operation precondition; no physical task failure is inferred from missing supported continuation."
        ],
    }
    atomic_json(root_dir / "root_manifest.json", root_manifest)
    atomic_json(
        root_dir / "tree.json",
        {
            "root": None,
            "edges": [],
            "status": "BLOCKED_SOURCE_REPLAY",
            "source_replay_block": root_manifest["source_replay_block"],
        },
    )
    atomic_json(root_dir / "root_result.json", result)
    atomic_json(
        root_dir / "completed.json",
        {
            "status": result["status"],
            "generator_revision": cfg["generator_revision"],
            "root_result": _relative(root_dir / "root_result.json", run),
        },
    )
    return result


def run_root(args: argparse.Namespace) -> dict[str, Any]:
    import robocasa  # noqa: F401
    import mimicgen.env_interfaces.robocasa  # noqa: F401
    import cfact.adapters.interfaces  # noqa: F401
    import mimicgen.utils.pose_utils as pose_utils
    from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory
    from mimicgen.env_interfaces.base import make_interface

    cfg = json.loads(args.config.read_text())
    if cfg.get("storage", {}).get("allowed_write_root"):
        from cfact.storage.policy import contained_path
        contained_path(args.run, Path(cfg["storage"]["allowed_write_root"]))
    row = next(x for x in cfg["tasks"] if x["task"] == args.task)
    split = json.loads(args.source_manifest.read_text())
    source_row = next(x for x in split["tasks"] if x["task"] == args.task)
    if source_row["status"] != "READY_FOR_DEVELOPMENT":
        raise RuntimeError(source_row["status"])
    route = None
    if "requests_by_id" in split:
        route = split["requests_by_id"].get(args.root_id)
        if route is None or route["task"] != args.task:
            raise ValueError("root request is not in the frozen source identity manifest")
        if route.get("request_status") not in {
            "REQUESTED_NEW",
            "REQUESTED_NEW_SYNTHETIC_SOURCE",
            "REQUESTED_NEW_TREE_ADAPTER_SOURCE_PRIOR",
        }:
            raise ValueError("root request is not authorized for new execution")
        dataset = Path(route["prepared_hdf5"])
        source_group = route["prepared_group"]
    else:
        source_split = "development" if args.split == "adapter" else args.split
        if args.root_id not in {x["episode_id"] for x in source_row[source_split]}:
            raise ValueError("root not in frozen source split")
        dataset = Path(row["prepared_hdf5"])
        source_group = args.root_id
    if not dataset.is_absolute():
        dataset = PROJECT / dataset
    if route is not None:
        identity = route.get("source_identity", {})
        identity_digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            route.get("source_id") != identity.get("source_id")
            or route.get("original_episode_id") != identity.get("original_episode_id")
            or route.get("prepared_group") != source_group
            or Path(route.get("prepared_hdf5", "")).resolve() != dataset.resolve()
            or identity_digest != route.get("source_identity_sha256")
        ):
            raise ValueError("frozen source identity and selected prepared HDF5 group disagree")
        if not route.get("prepared_sha256"):
            raise ValueError("frozen source route is missing its prepared HDF5 digest")
    validate_prepared_identity(dataset, required_episodes=[source_group])
    root = _load_root(dataset, source_group)
    if (
        root["actions"].ndim != 2
        or len(root["actions"]) < 1
        or not np.isfinite(root["actions"]).all()
        or np.asarray(root["initial"]["states"]).ndim != 1
        or not np.isfinite(root["initial"]["states"]).all()
    ):
        raise ValueError(
            "selected prepared source group has invalid recorded actions or initial state"
        )
    completed_roots = list(
        (args.run.resolve() / "data" / args.split / args.task).glob(
            f"{args.task}__{args.root_id}__*/root_result.json"
        )
    )
    if len(completed_roots) > 1:
        raise ValueError("ambiguous_existing_root_results")
    if completed_roots:
        if cfg.get("method", {}).get("recursive_goals_v2"):
            from cfact.collection.snapshot import implementation_digest
            identity = json.loads(completed_roots[0].with_name("root_manifest.json").read_text())
            if identity.get("execution_code_sha256") != implementation_digest():
                raise ValueError("completed_root_code_changed_create_a_new_run")
            if identity.get("execution_config_sha256") != hashlib.sha256(args.config.read_bytes()).hexdigest():
                raise ValueError("completed_root_config_changed_create_a_new_run")
        return json.loads(completed_roots[0].read_text())
    task = target_task(args.task)
    seed = stable_seed(cfg["run_seed"], args.task, args.root_id, "root_initialization")
    env = _create_env(root["env_meta"], seed, "robot0_agentview_left", args.gpu,
                      width=cfg["video"]["cell_width"], height=cfg["video"]["cell_height"])
    root_lock = None
    alive = (
        args.run.resolve()
        / "runtime"
        / "scene_alive"
        / f"{args.split}__{args.task}__{args.root_id}.json"
    )
    alive.parent.mkdir(parents=True, exist_ok=True)
    lifecycle = (
        args.run.resolve()
        / "runtime"
        / "scene_lifecycle"
        / f"{args.split}__{args.task}__{args.root_id}.json"
    )
    scene_ready_at = datetime.now(timezone.utc).isoformat()
    alive_record = {
        "pid": os.getpid(),
        "task": args.task,
        "root_id": args.root_id,
        "split": args.split,
        "physical_gpu_index": args.gpu,
        "physical_gpu_uuid": os.environ.get("CFACT_GPU_UUID", "UNKNOWN_STANDALONE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "scene_ready_at": scene_ready_at,
        "heartbeat_at": scene_ready_at,
    }
    atomic_json(alive, alive_record)
    atomic_json(lifecycle, alive_record)
    try:
        interface = make_interface(
            task.adapter_interface or task.mimicgen_interface, "robosuite", env.env
        )
        install_action_codec(interface)
        env._cfact_v2_task = args.task
        env._cfact_step_ledger = ControlStepLedger(
            env,
            args.run.resolve() / "history" / args.split / args.task / args.root_id,
            interface=interface,
            width=cfg["video"]["cell_width"],
            height=cfg["video"]["cell_height"],
        )
        try:
            if task.primitive_family == "placement":
                try:
                    probe = _probe_held_parent(
                        env,
                        interface,
                        root["initial"],
                        root["actions"],
                        row["event_signal"],
                        row["event_object_ref"],
                        minimum_lift_m=0.0 if args.task == "CoffeeServeMug" else 0.03,
                    )
                except RuntimeError as error:
                    attempt = _source_attempt_result(route)
                    if (
                        str(error) == "source replay never established measured held-object parent"
                        and attempt is not None
                        and attempt["payload"].get("upstream_success_observed") is False
                    ):
                        root["actual_replay_steps"] = ledger(env).control_steps
                        root["source_replay_history"] = str(ledger(env).directory)
                        return _record_blocked_source_replay(
                            args, cfg, root, route, dataset, source_group, error, attempt
                        )
                    raise
            elif task.primitive_family == "button" and cfg.get("method", {}).get("recursive_goals_v2"):
                probe = _probe_button_parent(env, interface, root["initial"], root["actions"], row["event_signal"])
            elif cfg.get("method", {}).get("stable_articulation_parent"):
                probe = _probe_articulation_parent(env,interface,root["initial"],root["actions"],row,
                    stable_steps=cfg["method"].get("stable_contact_steps",4))
            else:
                probe = _probe_parent(
                    env,
                    interface,
                    root["initial"],
                    root["actions"],
                    row["event_signal"],
                    row["event_minimum_duration"],
                )
        except RuntimeError as error:
            root["actual_replay_steps"] = ledger(env).control_steps
            root["source_replay_history"] = str(ledger(env).directory)
            return _record_blocked_source_replay(
                args,
                cfg,
                root,
                route or {},
                dataset,
                source_group,
                error,
                {
                    "status": "measured_precondition_not_observed",
                    "official_task_failure_asserted": False,
                },
            )
        state = probe.pop("parent_state")
        parent_snapshot = capture_snapshot(env)
        frozen_parent = cfg.get("source_parent_snapshot")
        if frozen_parent:
            # Explicit matched-baseline input; the Source prefix is still paid.
            from cfact.execution.snapshot import snapshot_differences
            path = Path(frozen_parent["path"]).resolve()
            if not path.is_relative_to(Path(cfg["storage"]["allowed_write_root"]).resolve()):
                raise ValueError("frozen_parent_snapshot_outside_allowed_root")
            if frozen_parent["source_id"] != route["source_id"]:
                raise ValueError("frozen_parent_source_identity_mismatch")
            if int(frozen_parent["source_action_prefix"]) != int(probe["parent_action_count"]):
                raise ValueError("frozen_parent_source_cursor_mismatch")
            if hashlib.sha256(path.read_bytes()).hexdigest() != frozen_parent["sha256"]:
                raise ValueError("frozen_parent_snapshot_hash_changed")
            matched_parent = load_snapshot(path)
            if not np.array_equal(parent_snapshot["states"], matched_parent["states"]):
                raise ValueError("frozen_parent_replayed_physics_state_mismatch")
            before = snapshot_differences(matched_parent, parent_snapshot)
            _restore(env, root["initial"], matched_parent)
            parent_snapshot = capture_snapshot(env)
            after = snapshot_differences(matched_parent, parent_snapshot)
            atomic_json(args.run / "FROZEN_PARENT_REUSE_RECEIPT.json", {"status":"PASS" if not after else "FAIL", "source_id":route["source_id"], "source_prefix_control_steps":int(probe["parent_action_count"]), "snapshot_sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "pre_restore_rng_difference_count":len([d for d in before if "/rng/" in d["path"]]), "pre_restore_representation_difference_count":len([d for d in before if "/rng/" not in d["path"]]), "post_restore_differences":after})
            if after:
                raise ValueError("frozen_parent_complete_restore_mismatch")
        _restore(env, root["initial"], parent_snapshot)
        parent = current_info(env, interface)
        object_ref = row["event_object_ref"]
        fixture_ref = row["operation_object_ref"]
        joint_binding = {}
        task_goal = {
            "kind": "source_task_goal",
            "origin": "prepared_source_segment",
            "episode": source_group,
        }
        goal_qualification = (
            "source_terminal_official_success"
            if root["source_terminal_official_success"] is True
            else "source_endpoint_only"
        )
        task_goal.update(
            {
                "goal_qualification": goal_qualification,
                "source_provenance": root["source_provenance"],
            }
        )
        binding_error = None
        if task.primitive_family not in {"button", "placement"}:
            try:
                role_binding = resolve_role_binding(interface, object_ref)
                declared_joint = declared_operation_joint(interface, row.get("operation_joint_role"))
                explicit_joint = row.get("operation_joint")
                if declared_joint is not None and explicit_joint not in (None, declared_joint):
                    raise ValueError("operation_joint_role_conflicts_with_explicit_joint")
                joint_binding = bind_joint(
                    env.env.sim.model, role_binding, explicit_joint=declared_joint or explicit_joint
                )
                if declared_joint is not None:
                    joint_binding["joint_selection_evidence"] = {
                        "origin": "official_fixture_joint_property",
                        "declaration": dict(row["operation_joint_role"]),
                        "resolved_joint_name": declared_joint,
                        "bound_role_ancestor_verified": True,
                    }
                final = np.asarray(root["terminal_recorded_state"])
                raw_source_q = float(final[1 + joint_binding["qpos_address"]])
                task_goal = source_joint_task_goal(
                    joint_binding,
                    raw_source_q,
                    episode=source_group,
                    state_index=root["terminal_recorded_state_index"],
                    source_provenance=root["source_provenance"],
                    source_terminal_success=root["source_terminal_official_success"],
                )
            except (ValueError, KeyError, AttributeError) as error:
                binding_error = str(error)
                joint_binding = {}
        joint_spec = _fixture_joint_spec(env, joint_binding)
        if object_ref not in parent.object_poses or fixture_ref not in parent.object_poses:
            raise KeyError("required measured object/fixture role missing")
        segment = load_source_segment(
            dataset,
            source_group,
            event_signal=row["event_signal"],
            event_minimum_duration=row["event_minimum_duration"],
            object_ref=fixture_ref,
            motion_object_ref=object_ref,
            start_step=int(probe["parent_action_count"]),
            operation_type=task.operation_family,
            stage="operation_entry",
            object_roles={"manipulated": object_ref, "reference": fixture_ref},
        )
        segment = dataclasses.replace(segment, episode_id=args.root_id)
        donors = []
        donor_entry_audit = []
        donor_rows = source_row["donors"]
        if route is not None and route.get("donor_source_ids"):
            donor_map = {str(item["source_id"]): item for item in donor_rows}
            donor_rows = [
                donor_map[identity]
                for identity in route["donor_source_ids"]
                if identity in donor_map
            ]
        for donor_row in donor_rows:
            donor_id = str(donor_row.get("source_id", donor_row.get("episode_id")))
            donor_dataset = Path(donor_row.get("prepared_hdf5", dataset))
            if not donor_dataset.is_absolute():
                donor_dataset = PROJECT / donor_dataset
            donor_group = str(donor_row.get("prepared_group", donor_row.get("episode_id")))
            try:
                validate_prepared_identity(donor_dataset, required_episodes=[donor_group])
                donor_start = (
                    recorded_held_entry_step(
                        donor_dataset,
                        donor_group,
                        contact_signal=row["event_signal"],
                        object_ref=object_ref,
                        minimum_lift_m=0.0 if args.task == "CoffeeServeMug" else 0.03,
                    )
                    if task.primitive_family == "placement"
                    else None
                )
                donor_segment = load_source_segment(
                    donor_dataset,
                    donor_group,
                    event_signal=row["event_signal"],
                    event_minimum_duration=row["event_minimum_duration"],
                    object_ref=fixture_ref,
                    motion_object_ref=object_ref,
                    start_step=donor_start,
                    operation_type=task.operation_family,
                    stage="operation_entry",
                    object_roles={"manipulated": object_ref, "reference": fixture_ref},
                )
                donors.append(dataclasses.replace(donor_segment, episode_id=donor_id))
                donor_entry_audit.append(
                    {
                        "donor_id": donor_id,
                        "donor_group": donor_group,
                        "status": "READY",
                        "entry_step": donor_start,
                    }
                )
            except ValueError as error:
                donor_entry_audit.append(
                    {"donor_id": donor_id, "status": "REJECT", "reason": str(error)}
                )
        contract = operation_contract(
            operation_type=task.operation_family,
            primitive_family=task.primitive_family,
            event_object_ref=object_ref,
            operation_object_ref=fixture_ref,
            holding_object=task.primitive_family == "placement",
        )
        run = args.run.resolve()
        root_key = f"{args.task}__{args.root_id}__{array_digest(state)[:12]}"
        root_dir = run / "data" / args.split / args.task / root_key
        root_dir.mkdir(parents=True, exist_ok=True)
        if (root_dir / "root_result.json").is_file():
            return json.loads((root_dir / "root_result.json").read_text())
        root_lock = RunLock(root_dir / "run.lock")
        root_lock.acquire()
        atomic_json(
            root_dir / "root_manifest.json",
            {
                "execution_code_sha256": __import__("cfact.collection.snapshot", fromlist=["implementation_digest"]).implementation_digest(),
                "execution_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "render_backend": os.environ.get("MUJOCO_GL"),
                "root_id": root_key,
                "task": args.task,
                "split": args.split,
                "source_episode": args.root_id,
                "source_group": source_group,
                "source_id": None if route is None else route.get("source_id"),
                "source_class": None if route is None else route.get("source_class"),
                "source_identity": None if route is None else route.get("source_identity"),
                "source_identity_sha256": None
                if route is None
                else route.get("source_identity_sha256"),
                "role_in_release": None if route is None else route.get("role_in_release"),
                "prepared_hdf5": str(dataset),
                "prepared_sha256": None if route is None else route.get("prepared_sha256"),
                "donor_source_routes": [
                    {
                        key: donor.get(key)
                        for key in (
                            "source_id",
                            "prepared_hdf5",
                            "prepared_group",
                            "prepared_sha256",
                            "source_identity_sha256",
                        )
                    }
                    for donor in donor_rows
                ],
                "generator_revision": cfg["generator_revision"],
                "seed": seed,
                "worker_pid": os.getpid(),
                "physical_gpu_index": args.gpu,
                "physical_gpu_uuid": os.environ.get("CFACT_GPU_UUID", "UNKNOWN_STANDALONE"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                "original_module_provenance": _module_provenance(),
            },
        )
        original_dir = run / "videos" / "original" / args.task / args.root_id
        original_dir.mkdir(parents=True, exist_ok=True)
        video_sources = None if route is None else route.get("source_video_paths")
        copied = []
        missing = []
        if isinstance(video_sources, dict):
            for camera, source_value in sorted(video_sources.items()):
                if not source_value:
                    missing.append(camera)
                    continue
                source_path = Path(source_value)
                if not source_path.is_absolute():
                    source_path = PROJECT / source_path
                if not source_path.is_file():
                    missing.append(camera)
                    continue
                original_copy = original_dir / f"source_original__{camera}.mp4"
                if not original_copy.exists():
                    try:
                        os.link(source_path, original_copy)
                    except OSError:
                        shutil.copy2(source_path, original_copy)
                copied.append(
                    {
                        "camera": camera,
                        "source_path": str(source_path),
                        "video_path": _relative(original_copy, run),
                        "sha256": hashlib.sha256(original_copy.read_bytes()).hexdigest(),
                    }
                )
        original_status = {
            "status": "SOURCE_ORIGINAL_COPIED" if copied else "SOURCE_VIDEO_NOT_AVAILABLE",
            "source_videos": copied,
            "missing_cameras": missing,
            "source_video_origin": "official_lerobot_original"
            if copied
            else "not_recorded_in_source_package",
        }
        atomic_json(original_dir / "index.json", original_status)
        node_id = "n_root"
        node = root_dir / "nodes" / node_id
        node.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(node / "context.npz", sim_state=state)
        trace_parent = MeasuredTrace(object_ref, fixture_ref, **joint_spec)
        trace_parent.capture(env, interface)
        _save_clean_observation(env, trace_parent, node)
        atomic_json(
            node / "context.json",
            {
                "node_id": node_id,
                "parent_context_id": root_key,
                "task": args.task,
                "source_episode": args.root_id,
                "source_action_prefix": probe["parent_action_count"],
                "sim_state_sha256": array_digest(state),
                "sim_time_s": trace_parent.sim_time[0],
                "object_ref": object_ref,
                "fixture_ref": fixture_ref,
                "fixture_joint_names": joint_spec["fixture_joint_names"],
                "fixture_joint_types": joint_spec["fixture_joint_types"],
                "fixture_joint_units": joint_spec["fixture_joint_units"],
                "fixture_joint_qvel_units": joint_spec["fixture_joint_qvel_units"],
                "goal_binding": {"kind": "source_task_goal"},
            },
        )
        operation_context = _operation_context(
            interface,
            parent,
            parent_id=f"{root_key}/{node_id}",
            object_ref=object_ref,
            fixture_ref=fixture_ref,
            contract=contract,
            source=segment,
            task_goal=task_goal,
            joint_binding=joint_binding,
            stage="held_transport" if contract.holding_object else "articulation",
        )
        save_snapshot(node / "parent_snapshot.json", parent_snapshot)
        root_context = json.loads((node / "context.json").read_text())
        root_context.update(
            {
                "operation_context": operation_context.record(),
                "snapshot": "parent_snapshot.json",
                "history_refs": [r for r in ledger(env).references(segment="source_prefix") if r["sim_time_s"] <= parent_snapshot["sim_time_s"] + 1e-9],
                "history_complete": len(ledger(env).references(segment="source_prefix"))
                >= 3,
                "source_replay_history": str(ledger(env).directory),
                "source_video_identity": original_status,
            }
        )
        atomic_json(node / "context.json", root_context)
        goals = _propose_goals(args, cfg, operation_context, node)
        options, donor_audit = build_object_effect_options(
            g0_amplitude_m=_g0_amplitude(cfg,env,args,segment,parent,contract,joint_binding,task_goal,node),
            root=segment,
            operation_context=operation_context,
            donors=donors,
            current_fixture_pose=np.asarray(parent.object_poses[fixture_ref]),
            current_object_pose=np.asarray(parent.object_poses[object_ref]),
            current_eef_pose=np.asarray(parent.eef_pose),
            contract=contract,
            official_transform=pose_utils.transform_source_data_segment_using_object_pose,
            parent_context_id=f"{root_key}/{node_id}",
            end_event="entry_reached_while_held"
            if contract.holding_object
            else "articulation_subgoal_reached",
            local_timeout=cfg["execution"]["local_timeout_steps"],
            max_raw_per_generator=cfg["candidate_pool"]["max_raw_proposals_per_generator"],
            compatibility_limits=row["reference_neighborhood"],
        )
        if contract.holding_object:
            try:
                if args.task == "CoffeeSetupMug":
                    machine = env.env.coffee_machine
                    site_id = env.env.sim.model.site_name2id(
                        f"{machine.naming_prefix}receptacle_place_site"
                    )
                    options, geometry_audit = solve_g2_coffee_receptacle_goals(
                        options,
                        pour_site_world_m=np.asarray(env.env.sim.data.site_xpos[site_id]),
                        current_object_pose=np.asarray(parent.object_poses[object_ref]),
                        current_eef_pose=np.asarray(parent.eef_pose),
                    )
                else:
                    regions, bbox = _support_geometry(env, args.task)
                    options, geometry_audit = solve_g2_support_goals(
                        options,
                        support_regions=regions,
                        object_bbox_local=bbox,
                        current_object_pose=np.asarray(parent.object_poses[object_ref]),
                        current_eef_pose=np.asarray(parent.eef_pose),
                    )
            except (AttributeError, KeyError, ValueError) as error:
                geometry_audit = [
                    {"status": "UNSUPPORTED", "reason": f"support_geometry_adapter:{error}"}
                ]
        else:
            try:
                articulation = _articulation_geometry(
                    env, operation_context.joint_binding, operation_context.task_goal
                )
                options, geometry_audit = solve_g2_articulation_subgoals(
                    options,
                    **{
                        key: value
                        for key, value in articulation.items()
                        if key != "joint_to_handle_distance_m"
                    },
                    current_handle_pose=np.asarray(parent.object_poses[object_ref]),
                    current_eef_pose=np.asarray(parent.eef_pose),
                )
            except (AttributeError, KeyError, ValueError) as error:
                geometry_audit = [
                    {"status": "UNSUPPORTED", "reason": f"articulation_geometry_adapter:{error}"}
                ]
        if binding_error is not None:
            options = [
                dataclasses.replace(option, rejection_reason="operation_binding:" + binding_error)
                for option in options
            ]
        if cfg.get("method", {}).get("recursive_goals_v2"):
            options = _compile_goals(args, cfg, env, interface, operation_context, goals,
                segment, "AVAILABLE", node, options)
        selected, rejected = _select_pool(cfg,
            options,
            max_executed=cfg["candidate_pool"]["max_executed_options_per_node"],
            max_g0=cfg["candidate_pool"]["max_selected_g0"],
        )
        root_selected = []
        for option in selected:
            if option.intended_object_path is None:
                rejected.append(
                    {
                        "candidate_id": option.candidate_id,
                        "reason": "PRE_EXECUTION_REJECTED:object_path_missing",
                    }
                )
                continue
            if not _option_has_event_geometry(option, task.primitive_family, 0.03):
                rejected.append(
                    {
                        "candidate_id": option.candidate_id,
                        "reason": "PRE_EXECUTION_REJECTED:planned_motion_below_event_guard",
                    }
                )
                continue
            root_selected.append(option)
        selected = root_selected
        with (node / "proposals.jsonl").open("w") as stream:
            for option in options:
                stream.write(json.dumps(_card(option), sort_keys=True) + "\n")
        selected = _engineering_select(selected, rejected, cfg)
        atomic_json(
            node / "candidate_pool.json",
            {
                "selected_ids": [x.candidate_id for x in selected],
                "rejected": rejected,
                "donor_audit": donor_audit,
                "donor_entry_audit": donor_entry_audit,
                "geometry_audit": geometry_audit,
                "frozen_before_execution": True,
            },
        )
        edges = []
        charged_steps = int(probe["parent_action_count"])
        root_step_limit = int(cfg["execution"]["max_env_steps_per_root"])
        for option in selected:
            charged_steps = ledger(env).control_steps
            _check_storage(run, cfg)
            edge_dir = node / "edges" / option.candidate_id
            edge_dir.mkdir(parents=True, exist_ok=True)
            physical_seed = stable_seed(
                cfg["run_seed"],
                args.task,
                args.root_id,
                node_id,
                option.candidate_id,
                "physical_execution",
            )
            atomic_json(
                edge_dir / "option.json",
                {
                    **_card(option),
                    "candidate_rng_seed": physical_seed,
                    "physics_rng_policy": "restored_from_parent_snapshot",
                },
            )
            np.savez_compressed(
                edge_dir / "plan.npz",
                intended_eef_poses=option.poses,
                intended_object_poses=option.intended_object_path
                if option.intended_object_path is not None
                else np.empty((0, 4, 4)),
                gripper_actions=option.gripper_actions,
            )
            if (
                charged_steps + option.local_timeout + cfg["execution"]["continuation_steps"]
                > root_step_limit
            ):
                atomic_json(
                    edge_dir / "budget_censor.json",
                    {
                        "status": "BUDGET_CENSORED",
                        "charged_steps": charged_steps,
                        "root_step_limit": root_step_limit,
                    },
                )
                edges.append(
                    {
                        "candidate_id": option.candidate_id,
                        "status": "BUDGET_CENSORED",
                        "edge_path": _relative(edge_dir, run),
                    }
                )
                continue
            start = time.monotonic()
            quad = None
            try:
                _case_seed(
                    cfg, args.task, args.root_id, node_id, option.candidate_id, "physical_execution"
                )
                _restore(env, root["initial"], parent_snapshot)
                before = current_info(env, interface)
                trajectory, connection, executable_poses, executable_grips = _execution_trajectory(
                    WaypointSequence, WaypointTrajectory, np.asarray(before.eef_pose), option, cfg
                )
                np.savez_compressed(
                    edge_dir / "executable_plan.npz",
                    intended_eef_poses=executable_poses,
                    gripper_actions=executable_grips,
                    source_waypoint_index=np.asarray(connection["source_waypoint_index"]),
                    source_action_index=np.asarray(connection["source_action_index"]),
                )
                atomic_json(
                    edge_dir / "connection.json",
                    {k: v for k, v in connection.items() if k != "source_waypoint_index"},
                )
                _audit_motion_plan(
                    env,
                    executable_poses,
                    executable_grips,
                    cfg,
                    edge_dir,
                    holding_object=contract.holding_object,
                )
                path = (
                    run
                    / "videos"
                    / ("reference" if option.source_method == "Reference" else option.source_method)
                    / args.split
                    / args.task
                    / root_key
                    / f"{node_id}__{option.candidate_id}__quad.mp4"
                )
                quad = QuadReviewWriter(
                    path,
                    task=args.task,
                    root=args.root_id,
                    node=node_id,
                    candidate=option.candidate_id,
                    method=option.source_method,
                    intent=option.subtype,
                    fps=cfg["video"]["fps"],
                    cell_width=cfg["video"]["cell_width"],
                    cell_height=cfg["video"]["cell_height"],
                feature_scale=cfg["video"].get("feature_scale", 1.0),
                )
                object_path = option.intended_object_path
                if object_path is None or len(object_path) < 2:
                    raise RuntimeError("local event requires planned object trajectory")
                positions = object_path[:, :3, 3]
                displacements = np.linalg.norm(positions - positions[0], axis=1)
                maximum = float(displacements.max())
                goal_index = int(np.argmax(displacements >= maximum * 0.55))
                if option.subtype == "articulation_subgoal":
                    goal_index = len(positions) - 1
                detector = LocalEventDetector(
                    goal_xyz=positions[goal_index],
                    initial_xyz=np.asarray(before.object_poses[object_ref])[:3, 3],
                    held=contract.holding_object,
                    joint_goal=option.intended_fixture_subgoal.get("desired_q"),
                    joint_index=(
                        joint_spec["fixture_joint_names"].index(
                            option.intended_fixture_subgoal["joint"]
                        )
                        if "joint" in option.intended_fixture_subgoal
                        else None
                    ),
                )
                trace = MeasuredTrace(object_ref, fixture_ref, **joint_spec)
                if ledger(env):
                    ledger(env).begin(str(edge_dir), "local_operation")
                local = execute_option(
                    env,
                    interface,
                    trajectory,
                    trace,
                    detector,
                    max_steps=option.local_timeout,
                    minimum_event_steps=_minimum_event_steps(cfg,option),
                    frame_callback=quad.append,
                    contact_guard=_measured_arm_guard(env, cfg),
                    source_action_indices=connection["source_action_index"][1:],
                    source_episode=option.source_episode,
                    initial_source_cursor=(
                        option.source_fragment[0]
                        if option.source_fragment is not None
                        else option.operation_context.source_cursor
                    ),
                )
                child_state = _save_actual_child(env, edge_dir, option, local, trace)
                np.savez_compressed(edge_dir / "execution.npz", **trace.arrays())
                _save_tracking_audit(edge_dir, trace)
                tail, tail_status, tail_plan = _tail(
                    env,
                    interface,
                    option,
                    trajectory,
                    local["consumed_waypoints"],
                    steps=cfg["execution"]["continuation_steps"],
                    object_ref=object_ref,
                    fixture_ref=fixture_ref,
                    joint_spec=joint_spec,
                    quad=quad,
                    local_trace=trace,
                    holding_object=contract.holding_object,
                    motion_validation_cfg=cfg,
                    local_termination_reason=local["termination_reason"],
                )
                charged_steps += len(trace.commanded_actions) + len(tail.commanded_actions)
                cont = edge_dir / "continuation"
                cont.mkdir(exist_ok=True)
                np.savez_compressed(cont / "execution.npz", **tail.arrays())
                atomic_json(cont / "plan.json", tail_plan)
                atomic_json(
                    cont / "result.json",
                    {
                        "continuation_revision": "measured_child_connected_v4",
                        "task_horizon_control_steps": cfg["execution"]["continuation_steps"],
                        "goal_binding": option.goal_binding,
                        "status": tail_status,
                        "official_success_end": tail.task_success[-1],
                    },
                )
                atomic_json(
                    edge_dir / "goal_realization.json",
                    _goal_realization(option, trace, tail, goal_index),
                )
                media = quad.close()
                quad = None
                media["video"] = _relative(path, run)
                atomic_json(edge_dir / "media.json", media)
                atomic_json(
                    edge_dir / "local_result.json",
                    {
                        **local,
                        "child_state_sha256": array_digest(child_state),
                        "planned_subgoal_index": goal_index,
                        "planned_path_length": len(positions),
                        "Y_fixed": {
                            "official_success_end": tail.task_success[-1],
                            "status": tail_status,
                        },
                        "source_method": option.source_method,
                        "candidate_id": option.candidate_id,
                        "outcome_labels": _outcome_labels(local, tail_status),
                    },
                )
                atomic_json(
                    edge_dir / "costs.json",
                    {
                        "control_steps_local": len(trace.commanded_actions),
                        "control_steps_fixed": len(tail.commanded_actions),
                        "simulation_seconds_local": trace.sim_time[-1] - trace.sim_time[0],
                        "simulation_seconds_fixed": tail.sim_time[-1] - tail.sim_time[0],
                        "restore_wall_seconds_total": ledger(env).restore_wall_seconds,
                        "wall_seconds": time.monotonic() - start,
                    },
                )
                edges.append(
                    {
                        "candidate_id": option.candidate_id,
                        "status": "PHYSICS_COMPLETE",
                        "local_event": local["termination_reason"],
                        "Y_fixed": tail_status,
                        "edge_path": _relative(edge_dir, run),
                    }
                )
            except MotionPlanRejected as error:
                if quad is not None:
                    try:
                        quad.close()
                    except Exception:
                        pass
                atomic_json(
                    edge_dir / "pre_execution_rejection.json",
                    {
                        "status": "PRE_EXECUTION_REJECTED",
                        "reason": str(error),
                        "motion_validation": "motion_validation.json",
                    },
                )
                edges.append(
                    {
                        "candidate_id": option.candidate_id,
                        "status": "PRE_EXECUTION_REJECTED",
                        "reason": str(error),
                        "edge_path": _relative(edge_dir, run),
                    }
                )
            except Exception as error:
                if quad is not None:
                    try:
                        quad.close()
                    except Exception:
                        pass
                atomic_json(
                    edge_dir / "technical_error.json",
                    {
                        "status": "TECHNICAL_ERROR",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    },
                )
                edges.append(
                    {
                        "candidate_id": option.candidate_id,
                        "status": "TECHNICAL_ERROR",
                        "error": str(error),
                        "edge_path": _relative(edge_dir, run),
                    }
                )
        _save_command_differences(node)
        child_pools = []
        third_candidates = []
        frontier_priority = cfg["tree_limits"].get(
            "frontier_family_priority", ["G2", "G1", "Reference", "G0"]
        )
        child_frontier = []
        third_frontier = []
        frontier_decisions = []
        if cfg.get("method", {}).get("recursive_goals_v2"):
            from cfact.generation.frontier import FrontierChild, expand_frontier
            from cfact.generation.event_anchor import EventAnchor, OperationStage
            from cfact.generation.task_adapters import TaskEffectAdapter
            adapter = TaskEffectAdapter(task, joint_binding)
            def children_for(options, edges, depth):
                children = []
                for option in options:
                    for edge in edges:
                        if edge.get("status") != "PHYSICS_COMPLETE" or edge["candidate_id"] != option.candidate_id:
                            continue
                        rec = json.loads((run / edge["edge_path"] / "event_anchor.json").read_text())
                        rec.pop("label_origin", None)
                        rec["stage"] = OperationStage(rec["stage"])
                        children.append(FrontierChild(option, edge, EventAnchor(**rec), depth))
                return children
            def expand(child):
                nonlocal charged_steps
                pool, charged_steps, options = _expand_child_pool(
                    args=args, cfg=cfg, env=env, interface=interface, root=root,
                    root_segment=segment, donors=donors, ancestor_option=child.option,
                    ancestor_edge=run / child.edge["edge_path"], root_dir=root_dir,
                    root_key=root_key, object_ref=object_ref, fixture_ref=fixture_ref,
                    contract=contract, joint_spec=joint_spec, charged_steps=charged_steps,
                    source_transform=pose_utils.transform_source_data_segment_using_object_pose)
                return pool, children_for(options, pool["edges"], child.decision_depth+1)
            child_pools, frontier_decisions = expand_frontier(children_for(selected, edges, 1),
                adapter=adapter, expand=expand,
                maximum_depth=cfg["tree_limits"]["maximum_decision_depth"],
                maximum_pools=cfg["tree_limits"]["candidate_pools_per_root"]-1,
                parents_per_level=cfg["tree_limits"]["second_depth_pools"],
                remaining_budget=lambda: cfg["execution"]["max_env_steps_per_root"]-ledger(env).control_steps,
                minimum_pool_steps=2*cfg["execution"]["local_timeout_steps"],
                maximum_attempted_pools=cfg.get("method",{}).get("frontier_attempt_budget"),
                source_first=cfg.get("method",{}).get("frontier_source_first",False))
            third_pools = []
            atomic_json(root_dir / "frontier_decisions.json", frontier_decisions)
        else:
            if (
                cfg["tree_limits"]["maximum_decision_depth"] >= 2
                and int(cfg.get("engineering_profile", {}).get("max_decision_parents_per_root", 100000))
                > 1
            ):
                # Round-robin by selected option order; no terminal outcome is used
                # to select which genuine event children receive another pool.
                eligible = [
                    (option, edge)
                    for option in selected
                    for edge in edges
                    if option.candidate_id == edge["candidate_id"]
                    and edge.get("local_event") == "local_event_reached"
                    and not json.loads((run / edge["edge_path"] / "local_result.json").read_text())[
                        "official_success_at_child"
                    ]
                ]
                child_frontier = select_child_frontier(
                    eligible,
                    family_priority=frontier_priority,
                    maximum=min(
                        cfg["tree_limits"]["second_depth_pools"],
                        int(cfg.get("engineering_profile", {}).get("max_children_per_parent", 100000)),
                    ),
                )
                for ancestor_option, edge in child_frontier:
                    child_pool, charged_steps, child_options = _expand_child_pool(
                        args=args,
                        cfg=cfg,
                        env=env,
                        interface=interface,
                        root=root,
                        root_segment=segment,
                        donors=donors,
                        ancestor_option=ancestor_option,
                        ancestor_edge=run / edge["edge_path"],
                        root_dir=root_dir,
                        root_key=root_key,
                        object_ref=object_ref,
                        fixture_ref=fixture_ref,
                        contract=contract,
                        joint_spec=joint_spec,
                        charged_steps=charged_steps,
                        source_transform=pose_utils.transform_source_data_segment_using_object_pose,
                    )
                    child_pools.append(child_pool)
                    if contract.holding_object:
                        third_candidates.extend(
                            (option, candidate_edge)
                            for option in child_options
                            for candidate_edge in child_pool["edges"]
                            if option.candidate_id == candidate_edge["candidate_id"]
                            and candidate_edge.get("local_event") == "local_event_reached"
                            and not json.loads(
                                (run / candidate_edge["edge_path"] / "local_result.json").read_text()
                            )["official_success_at_child"]
                        )
            third_pools = []
            if (
                cfg["tree_limits"]["maximum_decision_depth"] >= 3
                and int(cfg.get("engineering_profile", {}).get("max_decision_parents_per_root", 100000))
                > 2
            ):
                third_frontier = select_child_frontier(
                    third_candidates,
                    family_priority=frontier_priority,
                    maximum=cfg["tree_limits"]["third_depth_pools"],
                )
                for ancestor_option, edge in third_frontier:
                    third_pool, charged_steps = _expand_release_pool(
                        args=args,
                        cfg=cfg,
                        env=env,
                        interface=interface,
                        root=root,
                        ancestor_option=ancestor_option,
                        ancestor_edge=run / edge["edge_path"],
                        root_dir=root_dir,
                        root_key=root_key,
                        object_ref=object_ref,
                        fixture_ref=fixture_ref,
                        contract=contract,
                        joint_spec=joint_spec,
                        charged_steps=charged_steps,
                    )
                    third_pools.append(third_pool)
        all_edges = (
            edges
            + [edge for pool in child_pools for edge in pool["edges"]]
            + [edge for pool in third_pools for edge in pool["edges"]]
        )
        status = (
            "PHYSICS_COMPLETE"
            if all(
                x["status"] in {"PHYSICS_COMPLETE", "BUDGET_CENSORED", "PRE_EXECUTION_REJECTED"}
                for x in all_edges
            )
            else "PARTIAL"
        )
        result = {
            "task": args.task,
            "root_id": root_key,
            "split": args.split,
            "status": status if all_edges else "NO_SUPPORTED_CONTINUATION",
            "binding_diagnostic": binding_error,
            "nodes": [node_id] + [x["node_id"] for x in child_pools + third_pools],
            "edges": all_edges,
            "charged_control_steps": ledger(env).control_steps,
            "restore_wall_seconds": ledger(env).restore_wall_seconds,
            "requested_max_decision_depth": cfg["tree_limits"]["maximum_decision_depth"],
            "frontier_selection": {
                "family_priority": frontier_priority,
                "second_depth_parent_edges": [e["edge_path"] for _, e in child_frontier],
                "third_depth_parent_edges": [e["edge_path"] for _, e in third_frontier],
                "ranking_uses_terminal_outcomes": False,
            },
            "generic_frontier_decisions": frontier_decisions,
            "observed_max_decision_depth": max([1]+[p.get("decision_depth",2) for p in child_pools if p["edges"]]) if cfg.get("method", {}).get("recursive_goals_v2") else (
                3
                if any(x["edges"] for x in third_pools)
                else (2 if any(x["edges"] for x in child_pools) else 1)
            ),
            "limitations": ["release_support_contact_inferred_from_stability_and_tool_separation"],
        }
        atomic_json(
            root_dir / "tree.json",
            {
                "root": node_id,
                "edges": edges,
                "child_pools": child_pools,
                "third_pools": third_pools,
            },
        )
        atomic_json(root_dir / "root_result.json", result)
        atomic_json(
            root_dir / "completed.json",
            {
                "status": result["status"],
                "generator_revision": cfg["generator_revision"],
                "root_result": _relative(root_dir / "root_result.json", run),
            },
        )
        return result
    finally:
        if ledger(env):
            ledger(env).finish()
        atomic_json(
            lifecycle, {**alive_record, "scene_closed_at": datetime.now(timezone.utc).isoformat()}
        )
        alive.unlink(missing_ok=True)
        if root_lock is not None:
            root_lock.release()
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--root-id", required=True)
    parser.add_argument(
        "--split",
        choices=("development", "adapter", "confirmation", "formal"),
        default="development",
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    result = run_root(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
