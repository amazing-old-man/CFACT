from types import SimpleNamespace
import numpy as np
import pytest
from cfact.execution.current import current_info, child_geometry


def test_current_proposal_view_preserves_native_controller_target_and_cache(monkeypatch):
    old = np.eye(4)
    fresh = old.copy()
    fresh[0, 3] = .02
    native = SimpleNamespace(object_poses={"obj": old}, eef_pose=old, target_pose=old,
                             subtask_term_signals={"success": False})
    interface = SimpleNamespace(get_datagen_info=lambda: native)
    monkeypatch.setattr("cfact.execution.current.observation_poses",
                        lambda *_: ({"obj": fresh}, fresh, {"capture_integration_steps": 0}))
    observed = current_info(None, interface)
    assert observed is not native
    assert observed.target_pose is native.target_pose
    assert observed.subtask_term_signals is native.subtask_term_signals
    np.testing.assert_array_equal(observed.object_poses["obj"], fresh)
    np.testing.assert_array_equal(native.object_poses["obj"], old)


def test_child_geometry_uses_actual_post_integration_poses_and_rejects_partial_capture():
    stale = np.eye(4)
    fresh = stale.copy()
    fresh[0, 3] = .02
    trace = SimpleNamespace(sim_time=[1., 1.05], observation_metadata=[{}, {}],
        observation_fixture_poses=[fresh, fresh], observation_object_poses=[fresh, fresh],
        observation_eef_poses=[fresh, fresh], fixture_poses=[stale, stale],
        object_poses=[stale, stale], actual_eef_pose=[stale, stale])
    for value in child_geometry(trace):
        np.testing.assert_array_equal(value, fresh)
    trace.observation_object_poses.pop()
    with pytest.raises(ValueError, match="incomplete"):
        child_geometry(trace)
