"""Shared measured-child and training-observation contracts.

Time is evidence, never a default. Legacy readers may recover it only from the
official flattened simulation state and/or the measured terminal trace.
"""

from __future__ import annotations

import copy
import numpy as np

from cfact.io.serialization import array_digest

CHILD_SCHEMA = "cfact.actual-child.v2"
OPERATION_SCHEMA = "cfact.operation_context.v1"
OBSERVATION_FIELDS = (
    "sim_time", "robot_qpos", "robot_qvel", "actual_eef_pose",
    "object_poses", "fixture_poses", "fixture_joint_qpos", "fixture_joint_qvel",
    "relevant_fixture_qpos", "relevant_fixture_qvel", "object_velocities",
    "object_to_gripper_transform",
)


def validate_child_context(context: dict, sim_state: np.ndarray, *, terminal_time=None) -> dict:
    """Validate new children, or annotate verifiable legacy time without rewriting raw."""
    result = copy.deepcopy(context)
    state = np.asarray(sim_state)
    if state.ndim != 1 or not len(state) or not np.isfinite(state).all():
        raise ValueError("invalid_actual_child_state")
    digest = array_digest(state)
    expected = result.get("state_sha256") or result.get("sim_state_sha256")
    if expected and expected != digest:
        raise ValueError("actual_child_state_hash_mismatch")
    # MuJoCo's official flattened MjSimState starts with time.
    state_time = float(state[0])
    if not np.isfinite(state_time) or state_time < 0:
        raise ValueError("actual_child_invalid_state_time")
    if terminal_time is not None and not np.isclose(
        state_time, float(terminal_time), rtol=0, atol=1e-9
    ):
        raise ValueError("actual_child_state_trace_time_mismatch")
    if result.get("sim_time_s") is None:
        result["sim_time_s"] = state_time
        result["time_origin"] = (
            "legacy_official_flat_state_verified_by_local_trace"
            if terminal_time is not None else "legacy_official_flat_state"
        )
    elif not np.isclose(float(result["sim_time_s"]), state_time, rtol=0, atol=1e-9):
        raise ValueError("actual_child_context_time_mismatch")
    if result.get("schema") == CHILD_SCHEMA:
        for field in ("operation_context", "source_cursor", "snapshot", "history_refs"):
            if field not in result:
                raise ValueError(f"actual_child_missing_{field}")
        if not isinstance(result["source_cursor"], dict):
            raise ValueError("actual_child_invalid_source_cursor")
    result["state_sha256"] = digest
    return result


def make_child_context(*, sim_state, operation_context, source_cursor, snapshot,
                       history_refs, **metadata) -> dict:
    """The single constructor for new saved local children, before tail execution."""
    state = np.asarray(sim_state)
    result = {
        **copy.deepcopy(metadata),
        "schema": CHILD_SCHEMA,
        "sim_time_s": float(state[0]),
        "time_origin": "measured_local_child_before_fixed_continuation",
        "operation_context": copy.deepcopy(operation_context),
        "source_cursor": copy.deepcopy(source_cursor),
        "snapshot": copy.deepcopy(snapshot),
        "history_refs": copy.deepcopy(history_refs),
    }
    return validate_child_context(result, state)


def training_observation(arrays: dict, index: int) -> dict:
    """An explicit allowlist prevents future task labels entering observations."""
    return {key: arrays[key][index] for key in OBSERVATION_FIELDS if key in arrays}
