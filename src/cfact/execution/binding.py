"""Resolve operation roles through the existing interface and model topology.

The interface selects the object. Model ancestry only resolves that selected
object's joint; unrelated fixtures and world distances never enter selection.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def resolve_role_binding(interface: Any, role: str) -> dict[str, Any]:
    calls = []
    original = interface.get_object_pose

    def record(*args, **kwargs):
        pose = original(*args, **kwargs)
        name = kwargs.get("obj_name", args[0] if args else None)
        kind = kwargs.get("obj_type", args[1] if len(args) > 1 else "body")
        calls.append((pose, name, kind))
        return pose

    interface.get_object_pose = record
    try:
        poses = interface.get_object_poses()
    finally:
        interface.get_object_pose = original
    if role not in poses:
        raise ValueError(f"operation_role_missing:{role}")
    matches = [(name, kind) for pose, name, kind in calls if pose is poses[role]]
    if not matches:
        matches = [(name, kind) for pose, name, kind in calls if np.array_equal(pose, poses[role])]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError(f"operation_role_entity_ambiguous:{role}:{matches}")
    name, kind = matches[0]
    model = interface.env.sim.model
    entity_id = int(getattr(model, f"{kind}_name2id")(name))
    body_id = entity_id if kind == "body" else int(getattr(model, f"{kind}_bodyid")[entity_id])
    return {
        "role": role,
        "entity_name": str(name),
        "entity_type": kind,
        "entity_id": entity_id,
        "body_id": body_id,
        "binding_origin": "existing_interface_object_role",
    }


def declared_operation_joint(interface: Any, declaration: dict[str, str] | None) -> str | None:
    """Resolve an adapter-declared joint role through an official fixture property.

    The caller still runs bind_joint, which verifies model ancestry. There is
    no fallback to joint motion, fixture proximity, or a joint-name substring.
    """
    if declaration is None:
        return None
    if set(declaration) != {"fixture_attribute", "joint_property"}:
        raise ValueError("invalid_operation_joint_role_declaration")
    for name in declaration.values():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError("invalid_operation_joint_role_attribute")
    fixture = getattr(interface.env, declaration["fixture_attribute"])
    element = getattr(fixture, declaration["joint_property"])
    if element is None or not hasattr(element, "get"):
        raise ValueError("official_operation_joint_role_missing")
    name = element.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("official_operation_joint_name_missing")
    return name


def bind_joint(
    model: Any, role_binding: dict[str, Any], *, explicit_joint: str | None = None
) -> dict[str, Any]:
    body = int(role_binding["body_id"])
    ancestry = []
    while body > 0:
        ancestry.append(body)
        body = int(model.body_parentid[body])
    candidates = [
        i
        for i in range(int(model.njnt))
        if int(model.jnt_bodyid[i]) in ancestry and int(model.jnt_type[i]) in (2, 3)
    ]
    if explicit_joint is not None:
        selected = int(model.joint_name2id(explicit_joint))
        if selected not in candidates:
            raise ValueError("explicit_joint_not_ancestor_of_bound_role")
        candidates = [selected]
    if len(candidates) != 1:
        names = [str(model.joint_id2name(i)) for i in candidates]
        raise ValueError(f"bound_object_joint_ambiguous:{names}")
    jid = candidates[0]
    kind = int(model.jnt_type[jid])
    result = {
        **role_binding,
        "joint_name": str(model.joint_id2name(jid)),
        "joint_id": jid,
        "joint_body_id": int(model.jnt_bodyid[jid]),
        "joint_type": "slide" if kind == 2 else "hinge",
        "joint_unit": "m" if kind == 2 else "rad",
        "qpos_address": int(model.jnt_qposadr[jid]),
        "qvel_address": int(model.jnt_dofadr[jid]),
        "joint_limited": bool(model.jnt_limited[jid]),
        "joint_range": np.asarray(model.jnt_range[jid], dtype=float).tolist(),
    }
    return result


def validate_joint_binding(model: Any, binding: dict[str, Any]) -> None:
    entity_id = int(getattr(model, f"{binding['entity_type']}_name2id")(binding["entity_name"]))
    body_id = (
        entity_id
        if binding["entity_type"] == "body"
        else int(getattr(model, f"{binding['entity_type']}_bodyid")[entity_id])
    )
    if entity_id != binding["entity_id"] or body_id != binding["body_id"]:
        raise ValueError("joint_binding_model_entity_mismatch")
    actual = bind_joint(model, binding, explicit_joint=binding["joint_name"])
    for key in (
        "joint_id",
        "joint_body_id",
        "joint_type",
        "joint_unit",
        "qpos_address",
        "qvel_address",
    ):
        if actual[key] != binding[key]:
            raise ValueError(f"joint_binding_model_mismatch:{key}")


def trace_joint_spec(binding: dict[str, Any] | None) -> dict[str, tuple]:
    if not binding:
        return {
            key: ()
            for key in (
                "fixture_joint_names",
                "fixture_qpos_indices",
                "fixture_qvel_indices",
                "fixture_joint_types",
                "fixture_joint_units",
                "fixture_joint_qvel_units",
            )
        }
    return {
        "fixture_joint_names": (binding["joint_name"],),
        "fixture_qpos_indices": (binding["qpos_address"],),
        "fixture_qvel_indices": (binding["qvel_address"],),
        "fixture_joint_types": (binding["joint_type"],),
        "fixture_joint_units": (binding["joint_unit"],),
        "fixture_joint_qvel_units": (binding["joint_unit"] + "/s",),
    }


def source_joint_task_goal(
    binding: dict[str, Any],
    raw_source_q: float,
    *,
    episode: str,
    state_index: int,
    source_provenance: dict[str, Any],
    source_terminal_success: bool | None,
) -> dict[str, Any]:
    """Constrain source goal evidence to this explicitly bound joint's domain.

    MuJoCo's soft limits can leave a successful observed source endpoint beyond
    the nominal range. Project only the intended target; preserve the raw source
    evidence and the actual parent state independently. Unlimited joints retain
    their source endpoint and are never assigned an invented finite limit.
    """
    raw_q = float(raw_source_q)
    if not np.isfinite(raw_q) or binding["joint_type"] not in {"slide", "hinge"}:
        raise ValueError("invalid_source_single_joint_goal_evidence")
    limited = bool(binding["joint_limited"])
    limits = np.asarray(binding["joint_range"], dtype=float)
    if limited and (
        limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]
    ):
        raise ValueError("invalid_bound_joint_goal_domain")
    q_task = float(np.clip(raw_q, limits[0], limits[1])) if limited else raw_q
    projected = q_task != raw_q
    qualification = (
        "source_terminal_official_success"
        if source_terminal_success is True
        else "source_endpoint_only"
    )
    return {
        "kind": "named_joint_task_goal",
        "joint_name": binding["joint_name"],
        "joint": binding["joint_name"],
        "target_q": q_task,
        "q_task": q_task,
        "raw_source_q": raw_q,
        "unit": binding["joint_unit"],
        "origin": "prepared_source_last_observed_state",
        "episode": episode,
        "state_index": int(state_index),
        "goal_qualification": qualification,
        "source_provenance": dict(source_provenance),
        "source_goal_evidence": {
            "raw_source_q": raw_q,
            "episode": episode,
            "state_index": int(state_index),
            "source_terminal_official_success": source_terminal_success,
            "goal_qualification": qualification,
        },
        "goal_constraint": {
            "kind": "bound_joint_range_projection" if limited else "unlimited_source_endpoint",
            "applied": projected,
            "reason": (
                "source_endpoint_outside_bound_joint_range"
                if projected
                else "source_endpoint_within_bound_joint_range"
                if limited
                else "joint_has_no_finite_goal_domain"
            ),
            "joint_range": limits.tolist() if limited else None,
            "input_q": raw_q,
            "output_q": q_task,
            "projection_delta": q_task - raw_q,
            "unit": binding["joint_unit"],
        },
    }
