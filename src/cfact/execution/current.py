"""Current physical geometry for proposal construction and inherited child state."""
from __future__ import annotations
import copy
import numpy as np
from cfact.execution.observation import observation_poses


def current_info(env, interface):
    """Refresh archived/proposal poses while preserving native controller semantics."""
    native = interface.get_datagen_info()
    observed = observation_poses(env, interface)
    if observed is None:
        return native  # Explicit non-MuJoCo test double; no physical claim.
    poses, eef, metadata = observed
    result = copy.copy(native)
    result.object_poses = {name: np.array(pose, copy=True) for name, pose in poses.items()}
    result.eef_pose = np.array(eef, copy=True)
    result.current_geometry_receipt = metadata
    return result


def child_geometry(trace):
    """Require refreshed poses when capture declares native physical observations."""
    metadata = getattr(trace, "observation_metadata", None)
    if metadata:
        n = len(trace.sim_time)
        names = ("observation_fixture_poses", "observation_object_poses", "observation_eef_poses")
        if any(len(getattr(trace, name)) != n for name in names):
            raise ValueError("incomplete_current_child_geometry")
        return tuple(np.array(getattr(trace, name)[-1], copy=True) for name in names)
    if hasattr(trace, "observation_metadata") and getattr(trace, "_capture_failed", False):
        raise ValueError("failed_current_child_geometry")
    return tuple(np.array(getattr(trace, name)[-1], copy=True) for name in
                 ("fixture_poses", "object_poses", "actual_eef_pose"))
