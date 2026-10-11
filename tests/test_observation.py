from types import SimpleNamespace

import mujoco
import numpy as np

from cfact.execution.observation import observe, observation_poses, PHASE


def scene():
    from robosuite.utils.binding_utils import MjSim

    xml = '''<mujoco><worldbody>
      <camera name="robot0_agentview_left" pos="0 -2 1"/>
      <camera name="robot0_agentview_right" pos="0 2 1"/>
      <camera name="robot0_eye_in_hand" pos="1 0 1"/>
      <body name="object"><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>'''
    sim = MjSim(mujoco.MjModel.from_xml_string(xml))
    sim.forward()
    sim.data.qpos[0] = .03  # Deliberately stale derived cache; no integration.
    return SimpleNamespace(env=SimpleNamespace(sim=sim))


def test_shadow_refreshes_current_pose_without_touching_any_live_cache():
    env = scene()
    live = env.env.sim.data._data
    before = {k: np.array(getattr(live, k), copy=True)
              for k in ("qpos", "qvel", "xpos", "qacc", "qacc_warmstart", "ctrl")}
    captured = observe(env)
    assert captured.metadata["observation_phase"] == PHASE
    assert captured.metadata["capture_integration_steps"] == 0
    assert captured.sim.data.body_xpos[1, 0] == .03
    assert live.xpos[1, 0] == 0
    for k, expected in before.items():
        np.testing.assert_array_equal(getattr(live, k), expected)


def test_archived_values_do_not_alias_later_state_or_visual_changes():
    env = scene()
    captured = observe(env)
    stored = captured.archive_values()
    env.env.sim.model.site_rgba[:] = .5
    env.env.sim.model.geom_rgba[:] = .2
    env.env.sim.data.qpos[0] = .5
    observe(env)
    assert captured.state[1] == .03
    assert stored["model_geom_rgba"][0, 0] != .2
    assert stored["integration_state"][1] == .03


def test_pose_read_uses_private_view_and_preserves_original_interface():
    env = scene()

    class Interface:
        def __init__(self, base):
            self.env = base

        def get_object_poses(self):
            result = np.eye(4)
            result[:3, 3] = self.env.sim.data.body_xpos[1]
            return {"obj": result}

        def get_robot_eef_pose(self):
            return self.get_object_poses()["obj"]

    interface = Interface(env.env)
    poses, eef, metadata = observation_poses(env, interface)
    assert poses["obj"][0, 3] == eef[0, 3] == .03
    assert interface.env.sim is env.env.sim
    assert interface.get_object_poses()["obj"][0, 3] == 0
    assert metadata["pose_source"] == "shadow_mj_forward"
