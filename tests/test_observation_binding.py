"""Real MimicGen instance binding and native kinematic boundaries."""
import copy
from functools import partial
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from cfact.execution.binding import resolve_role_binding
from cfact.execution.observation import observation_poses


def interface_scene(task):
    from robosuite.utils.binding_utils import MjSim
    from mimicgen.env_interfaces.robocasa.single_stage.mg_drawer import MG_OpenDrawer
    from mimicgen.env_interfaces.robocasa.single_stage.mg_coffee import MG_CoffeeServeMug

    xml = '''<mujoco><worldbody>
    <camera name="robot0_agentview_left"/><camera name="robot0_agentview_right"/>
    <camera name="robot0_eye_in_hand"/>
    <body name="cup"><freejoint/><geom type="sphere" size=".03"/></body>
    <body name="drawer" pos=".4 .5 .6" quat=".9238795325 0 0 .3826834324">
      <joint name="drawer_slide" type="slide" axis="1 0 0"/>
      <geom name="chosen_drawer_door_handle_reg_main" pos=".1 .2 .3"
        type="sphere" size=".02"/>
    </body>
    <body name="robot"><joint name="robot_slide" type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".01"/><site name="eef" pos=".1 .1 .1"/>
    </body>
    <site name="machine_receptacle_place_site" pos="1 2 3"/>
    </worldbody></mujoco>'''
    sim = MjSim(mujoco.MjModel.from_xml_string(xml))
    sim.forward()
    base = SimpleNamespace(sim=sim, drawer=SimpleNamespace(name='chosen_drawer'),
                           objects={'obj': SimpleNamespace(root_body='cup')},
                           coffee_machine=SimpleNamespace(name='machine'),
                           robots=[SimpleNamespace(composite_controller=SimpleNamespace(
                               get_controller=lambda arm: SimpleNamespace(ref_name='eef')))])
    cls = MG_OpenDrawer if task == 'drawer' else MG_CoffeeServeMug
    interface = cls(base)
    sim.data.qpos[:7] = [.03, .02, .01, np.cos(.2), 0, 0, np.sin(.2)]
    sim.data.qpos[7:] = [.037, .028]
    return SimpleNamespace(env=base), interface


def independent(env, kind, name):
    m = env.env.sim.model._model
    d = mujoco.MjData(m)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, env.env.sim.data._data, values, spec)
    mujoco.mj_setState(m, d, values, spec)
    mujoco.mj_forward(m, d)
    idx = mujoco.mj_name2id(m, getattr(mujoco.mjtObj, 'mjOBJ_' + kind.upper()), name)
    prefix = {'body': '', 'geom': 'geom_', 'site': 'site_'}[kind]
    pose = np.eye(4)
    pose[:3, 3] = getattr(d, prefix + 'xpos')[idx]
    pose[:3, :3] = getattr(d, prefix + 'xmat')[idx].reshape(3, 3)
    return pose


@pytest.mark.parametrize('task', ['drawer', 'coffee'])
def test_actual_resolver_instance_binding_reads_shadow(task):
    env, interface = interface_scene(task)
    role = 'handle' if task == 'drawer' else 'coffee_machine'
    resolve_role_binding(interface, role)
    assert interface.get_object_pose.__self__ is interface
    stale_view = copy.copy(interface)
    stale_view.env = copy.copy(interface.env)
    assert stale_view.get_object_pose.__self__ is interface
    native = env.env.sim.data._data
    before = {name: np.array(getattr(native, name), copy=True) for name in
              ('qpos', 'qvel', 'xpos', 'geom_xpos', 'site_xpos', 'qacc', 'qacc_warmstart', 'ctrl')}
    poses, eef, metadata = observation_poses(env, interface)
    if task == 'drawer':
        expected = independent(env, 'geom', 'chosen_drawer_door_handle_reg_main')
        np.testing.assert_array_equal(poses['handle'], expected)
        assert not np.array_equal(expected, interface.get_object_poses()['handle'])
    else:
        np.testing.assert_array_equal(poses['obj'], independent(env, 'body', 'cup'))
        np.testing.assert_array_equal(poses['coffee_machine'], independent(
            env, 'site', 'machine_receptacle_place_site'))
    np.testing.assert_array_equal(eef, independent(env, 'site', 'eef'))
    assert metadata['pose_source'] == 'shadow_mj_forward'
    assert interface.env is env.env
    assert interface.get_object_pose.__self__ is interface
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(native, name), value)


@pytest.mark.parametrize('kind', ['partial', 'closure', 'foreign_method', 'missing'])
def test_unsupported_pose_binding_fails_closed(kind):
    env, interface = interface_scene('drawer')
    original = interface.get_object_pose
    if kind == 'partial':
        interface.get_object_pose = partial(original)
    elif kind == 'closure':
        interface.get_object_pose = lambda *a, **k: original(*a, **k)
    elif kind == 'foreign_method':
        _, other = interface_scene('drawer')
        interface.get_object_pose = other.get_object_pose
    else:
        interface.get_object_pose = None
    with pytest.raises((TypeError, ValueError), match='observation_.*binding'):
        observation_poses(env, interface)


def test_private_view_rebinds_all_three_pose_entrypoints():
    from cfact.execution.observation import _shadow_interface_view, observe
    env, interface = interface_scene('drawer')
    for name in ('get_object_pose', 'get_object_poses', 'get_robot_eef_pose'):
        setattr(interface, name, getattr(interface, name))
    captured = observe(env)
    view = _shadow_interface_view(interface, captured.sim)
    for name in ('get_object_pose', 'get_object_poses', 'get_robot_eef_pose'):
        assert getattr(view, name).__self__ is view
    assert view.env.sim is captured.sim
    assert view.env is not interface.env


def trace_scene():
    from cfact.execution.options import MeasuredTrace
    env, source = interface_scene('drawer')

    class TraceInterface(type(source)):
        def get_datagen_info(self):
            return SimpleNamespace(object_poses=self.get_object_poses(),
                                   eef_pose=self.get_robot_eef_pose(), subtask_term_signals={})

    interface = TraceInterface(source.env)
    m = env.env.sim.model._model
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'robot_slide')
    robot = env.env.robots[0]
    robot._ref_joint_pos_indexes = [int(m.jnt_qposadr[jid])]
    robot._ref_joint_vel_indexes = [int(m.jnt_dofadr[jid])]
    robot._ref_joint_indexes = [jid]
    robot.robot_joints = ['robot_slide']
    robot.eef_site_id = {'right': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'eef')}
    env.is_success = lambda: {'task': False}
    return env, interface, MeasuredTrace('handle', 'handle')


def test_failed_capture_cannot_serialize_or_resume_mixed_trace(monkeypatch):
    import cfact.execution.observation as module
    env, interface, trace = trace_scene()
    trace.capture(env, interface)
    def failed(env, interface):
        raise ValueError('missing_current_role')
    monkeypatch.setattr(module, 'observation_poses', failed)
    with pytest.raises(ValueError, match='missing_current_role'):
        trace.capture(env, interface)
    with pytest.raises(ValueError, match='failed_observation_capture'):
        trace.arrays()
    with pytest.raises(ValueError, match='failed_observation_capture'):
        trace.capture(env, interface)


@pytest.mark.parametrize('corruption', ['missing_role', 'nonfinite', 'incomplete_labels'])
def test_current_labels_fail_closed(corruption, monkeypatch):
    import cfact.execution.observation as module
    env, interface, trace = trace_scene()
    if corruption == 'incomplete_labels':
        trace.capture(env, interface)
        trace.observation_fixture_poses.clear()
        with pytest.raises(ValueError, match='incomplete_current_observation_labels'):
            trace.arrays()
        return
    original = module.observation_poses
    def invalid(env, interface):
        poses, eef, metadata = original(env, interface)
        if corruption == 'missing_role':
            poses.pop('handle')
        else:
            eef[0, 3] = np.nan
        return poses, eef, metadata
    monkeypatch.setattr(module, 'observation_poses', invalid)
    with pytest.raises((KeyError, ValueError)):
        trace.capture(env, interface)
    with pytest.raises(ValueError, match='failed_observation_capture'):
        trace.arrays()


def test_native_capture_only_forwards_private_data(monkeypatch):
    env, interface = interface_scene('drawer')
    resolve_role_binding(interface, 'handle')
    forward = mujoco.mj_forward
    calls = []
    def checked_forward(model, data):
        assert data is not env.env.sim.data._data
        calls.append(id(data))
        return forward(model, data)
    def forbidden(*args, **kwargs):
        raise AssertionError('observation integrated physics')
    monkeypatch.setattr(mujoco, 'mj_forward', checked_forward)
    for name in ('mj_step', 'mj_step1', 'mj_step2'):
        monkeypatch.setattr(mujoco, name, forbidden)
    observation_poses(env, interface)
    assert calls == [id(env.env.sim._cfact_observation_sim.data._data)]


def test_nonfinite_native_state_and_mismatched_sim_are_rejected():
    env, interface = interface_scene('drawer')
    other_env, _ = interface_scene('drawer')
    with pytest.raises(ValueError, match='sim_binding_mismatch'):
        observation_poses(other_env, interface)
    env.env.sim.data.qpos[0] = np.nan
    with pytest.raises(ValueError, match='nonfinite_integration_state'):
        observation_poses(env, interface)


@pytest.mark.parametrize('alias', ['interface', 'environment'])
def test_rejected_copy_alias_preserves_original_pointers(alias, monkeypatch):
    import cfact.execution.observation as module
    env, interface = interface_scene('drawer')
    original_env, original_sim = interface.env, interface.env.sim
    original_copy = module.copy.copy
    def alias_copy(value):
        if value is (interface if alias == 'interface' else original_env):
            return value
        return original_copy(value)
    monkeypatch.setattr(module.copy, 'copy', alias_copy)
    with pytest.raises(TypeError, match='unsupported_binding_copy'):
        observation_poses(env, interface)
    assert interface.env is original_env
    assert env.env is original_env
    assert original_env.sim is original_sim
