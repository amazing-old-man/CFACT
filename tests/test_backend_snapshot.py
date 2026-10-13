from types import SimpleNamespace
import copy
import random
import numpy as np
import pytest

from cfact.execution.snapshot import capture_snapshot, restore_snapshot, save_snapshot, load_snapshot


def fake_env():
    class Sim:
        def __init__(self):
            self.state = np.array([2.5, .3, -.2])
            self.data = SimpleNamespace(time=2.5, ctrl=np.array([.6]), qacc_warmstart=np.array([.7]))
            self.model = SimpleNamespace(body_pos=np.array([[1., 2., 3.]]))
        def get_state(self):
            return self.state.copy()
        def set_state_from_flattened(self, value):
            self.state = value.copy()
            self.data.time = value[0]
        def forward(self):
            pass
    controller = SimpleNamespace(goal_pos=np.array([1., 2., 3.]),
                                 interpolator_pos=SimpleNamespace(goal=np.array([.1, .2, .3]), step=4))
    gripper = SimpleNamespace(current_action=np.array([.8]))
    composite = SimpleNamespace(part_controllers={"right": controller})
    robot = SimpleNamespace(composite_controller=composite, gripper={"right": gripper})
    base = SimpleNamespace(sim=Sim(), robots=[robot], timestep=50, rng=np.random.default_rng(44),
                           fixtures={"coffee": SimpleNamespace(_turned_on=True)})
    class Env:
        def __init__(self):
            self.env = base
        def get_state(self):
            return {"states": base.sim.state.copy(), "model": "fixed-model"}
        def reset_to(self, initial):
            gripper.current_action[:] = 0
            controller.goal_pos[:] = 0
            base.sim.set_state_from_flattened(initial["states"])
    return Env()


def test_snapshot_restores_gripper_after_reset_and_all_rng_without_time_step(tmp_path):
    env = fake_env()
    snapshot = capture_snapshot(env)
    save_snapshot(tmp_path / "parent.json", snapshot)
    expected = (env.env.rng.random(), np.random.random(), random.random())
    for marker in [.1, .9, .9, .1]:
        env.env.robots[0].gripper["right"].current_action[:] = marker
        env.env.fixtures["coffee"]._turned_on = False
        env.env.sim.data.ctrl[:] = marker
        report = restore_snapshot(env, load_snapshot(tmp_path / "parent.json"), initial={"states": np.zeros(3)})
        assert report["complete"] and report["physics_steps"] == 0
        assert env.env.sim.data.time == 2.5
        assert env.env.robots[0].gripper["right"].current_action[0] == .8
        assert env.env.fixtures["coffee"]._turned_on
        np.testing.assert_array_equal(env.env.sim.data.ctrl, [.6])
        assert (env.env.rng.random(), np.random.random(), random.random()) == expected
    np.testing.assert_array_equal(snapshot["robots"][0]["controllers"]["right"]["values"]["goal_pos"], [1., 2., 3.])


def test_legacy_snapshot_requires_explicit_limited_capability():
    env = fake_env()
    old = {"states": np.array([4., .2, .1])}
    with pytest.raises(ValueError, match="legacy_snapshot"):
        restore_snapshot(env, old)
    assert restore_snapshot(env, old, allow_legacy=True)["complete"] is False


def test_snapshot_keeps_interpolator_state_and_rejects_incomplete():
    env = fake_env()
    snapshot = capture_snapshot(env)
    env.env.robots[0].composite_controller.part_controllers["right"].interpolator_pos.step = 99
    restore_snapshot(env, snapshot)
    assert env.env.robots[0].composite_controller.part_controllers["right"].interpolator_pos.step == 4
    incomplete = copy.deepcopy(snapshot)
    del incomplete["rng"]
    with pytest.raises(ValueError, match="incomplete"):
        restore_snapshot(env, incomplete)


def test_snapshot_rejects_different_model_without_physics_step():
    env = fake_env()
    snapshot = capture_snapshot(env)
    snapshot["model_sha256"] = "another-model"
    with pytest.raises(ValueError, match="snapshot_model_identity_mismatch"):
        restore_snapshot(env, snapshot)


def test_snapshot_preserves_real_nullable_torque_buffer_through_reset(tmp_path):
    # Hybrid robot torque buffers can contain None in object arrays. Raw object
    # array bytes are process addresses; JSON values must preserve the entries.
    from cfact.execution.snapshot import _is_value
    env = fake_env()
    env.env.robots[0].recent_torques = SimpleNamespace(
        dim=3, last=np.array([None, None, None], dtype=object),
        current=np.array([None, 0.2, None], dtype=object),
    )
    before = capture_snapshot(env)
    assert _is_value(env.env.robots[0].recent_torques.current)
    save_snapshot(tmp_path / "nullable.json", before)
    loaded = load_snapshot(tmp_path / "nullable.json")
    buffer = env.env.robots[0].recent_torques
    buffer.last = np.zeros(3)
    buffer.current = np.zeros(3)
    restore_snapshot(env, loaded)
    after = capture_snapshot(env)
    for name in ("last", "current"):
        actual = after["robots"][0]["buffers"]["recent_torques"][name]
        expected = before["robots"][0]["buffers"]["recent_torques"][name]
        assert actual.dtype == np.dtype(object)
        np.testing.assert_array_equal(actual, expected)
    assert not _is_value(np.array([SimpleNamespace(value=1)], dtype=object))
