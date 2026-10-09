"""Independent oracle: the installed robosuite OSC set_goal implementation."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cfact.adapters.action_codec import ActionCodec, install_action_codec


def interface_for_controller(input_type="delta", frame="base", mode="achieved"):
    pytest.importorskip("robosuite")
    from robosuite.controllers.parts.arm.osc import OperationalSpaceController
    # Exercise the real set_goal, scale_action and goal composition methods;
    # only simulator-derived kinematics are supplied as an independent fixture.
    c = OperationalSpaceController.__new__(OperationalSpaceController)
    c.input_type, c.input_ref_frame = input_type, frame
    c.impedance_mode, c.use_ori, c._goal_update_mode = "fixed", True, mode
    c.input_min = np.array([-1., -0.8, -1., -1., -1., -1.])
    c.input_max = np.array([1., 1.2, 1., 1., 1., 1.])
    c.output_min = np.array([-.05, -.04, -.03, -.4, -.3, -.2])
    c.output_max = np.array([.05, .06, .03, .4, .3, .2])
    c.action_scale = None
    c.position_limits = c.orientation_limits = None
    c.interpolator_pos = c.interpolator_ori = None
    c.origin_pos = np.array([.3, -.6, .2])
    c.origin_ori = Rotation.from_euler("xyz", [.2, -.3, 1.1]).as_matrix()
    c.ref_pos = np.array([.6, -.2, .9])
    c.ref_ori_mat = Rotation.from_euler("xyz", [-.2, .5, -.6]).as_matrix()
    c.goal_pos = np.array([.2, .3, .5])
    c.goal_ori = Rotation.from_euler("xyz", [.7, -.1, .3]).as_matrix()
    c.update = lambda: None
    composite = SimpleNamespace(
        name="HYBRID_MOBILE_BASE",
        _action_split_indexes={"right": (0, 6), "right_gripper": (6, 7), "base": (7, 10), "torso": (10, 11)},
        action_limits=(np.full(12, -1.), np.ones(12)),
        get_controller=lambda arm: c,
    )
    pose = np.eye(4)
    pose[:3, 3], pose[:3, :3] = c.ref_pos, c.ref_ori_mat
    interface = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(composite_controller=composite)]),
                                get_robot_eef_pose=lambda: pose.copy())
    return interface, c


@pytest.mark.parametrize("input_type", ["delta", "absolute"])
@pytest.mark.parametrize("frame", ["base", "world"])
@pytest.mark.parametrize("mode", ["achieved", "desired"])
def test_codec_matches_actual_controller_goal(input_type, frame, mode):
    interface, c = interface_for_controller(input_type, frame, mode)
    # Exceeds input bounds on purpose; scaling is asymmetric in channel 1.
    action = np.r_[[1.8, -.9, .6, -.4, 1.4, -.8], [.5, .1, .2, .3, .4], 1 if mode == "desired" else -1]
    saved = action.copy()
    predicted = ActionCodec(interface).decode(action)
    c.set_goal(action[:6])
    expected = np.eye(4)
    expected[:3, 3], expected[:3, :3] = c.goal_pos, c.goal_ori
    if frame == "base":
        origin = np.eye(4)
        origin[:3, 3], origin[:3, :3] = c.origin_pos, c.origin_ori
        expected = origin @ expected
    # The declared rotation-matrix acceptance is <= 1e-8. Keep the existing
    # stricter translation regression at 1e-10 m (declared limit: 1e-8 m).
    np.testing.assert_allclose(predicted[:3, 3], expected[:3, 3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(predicted[:3, :3], expected[:3, :3], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(saved, action)


@pytest.mark.parametrize("input_type", ["delta", "absolute"])
def test_encoding_uses_controller_coordinates_and_clips_after_transform(input_type):
    interface, c = interface_for_controller(input_type)
    codec = ActionCodec(interface)
    target = interface.get_robot_eef_pose()
    target[:3, 3] += [.01, -.012, .009]
    target[:3, :3] = Rotation.from_rotvec([.03, -.02, .04]).as_matrix() @ target[:3, :3]
    command = codec.encode(target)
    c.set_goal(command)
    expected = np.eye(4)
    expected[:3, 3] = c.origin_pos + c.origin_ori @ c.goal_pos
    expected[:3, :3] = c.origin_ori @ c.goal_ori
    np.testing.assert_allclose(expected[:3, 3], target[:3, 3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(expected[:3, :3], target[:3, :3], atol=1e-8, rtol=0)


def test_installed_interface_uses_full_action_mode_and_keeps_source():
    interface, _ = interface_for_controller()
    install_action_codec(interface)
    action = np.arange(12, dtype=float) / 12
    original = action.copy()
    np.testing.assert_array_equal(interface.action_to_gripper_action(action), action[6:7])
    interface.action_to_target_pose(action)
    np.testing.assert_array_equal(action, original)


def test_absolute_goal_limits_match_backend_bypass_and_delta_reports_unsupported():
    interface, c = interface_for_controller("absolute")
    c.position_limits = np.array([[-.01, -.01, -.01], [.01, .01, .01]])
    action = np.r_[[.4, -.2, .3, .1, -.1, .2], [0., 0., 0., 0., 0., -1.]]
    expected = ActionCodec(interface).decode(action)
    c.set_goal(action[:6])
    np.testing.assert_allclose(expected[:3, 3], c.origin_pos + c.origin_ori @ c.goal_pos, atol=1e-10)
    c.input_type = "delta"
    with pytest.raises(ValueError, match="goal_limits_not_implemented"):
        ActionCodec(interface).decode(action)
    with pytest.raises(NotImplementedError):
        c.set_goal(action[:6])
