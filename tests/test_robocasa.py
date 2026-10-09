from __future__ import annotations

import pytest
import numpy as np

from cfact.adapters.robocasa import (
    _list_of_flat_dict_to_dict_of_list,
    synchronize_fixed_fixture_poses,
)
from cfact.adapters.events import (
    coffee_machine_button_signals,
    microwave_button_signals,
    select_coffee_machine_button,
)
from cfact.adapters.frames import rotate_planar_base_delta_to_world


def test_prepare_dict_transpose_preserves_key_order_and_values() -> None:
    result = _list_of_flat_dict_to_dict_of_list(
        [{"pose": 1, "signal": 0}, {"pose": 2, "signal": 1}]
    )
    assert result == {"pose": [1, 2], "signal": [0, 1]}


def test_prepare_dict_transpose_rejects_inconsistent_schema() -> None:
    with pytest.raises(ValueError, match="identical keys"):
        _list_of_flat_dict_to_dict_of_list([{"pose": 1}, {"signal": 1}])


def test_prepare_dict_transpose_accepts_empty_input() -> None:
    assert _list_of_flat_dict_to_dict_of_list([]) == {}


def test_mobile_base_planar_delta_rotates_back_to_world_axes() -> None:
    source = np.asarray([0.25, -0.5])
    rotated = rotate_planar_base_delta_to_world(source, np.pi / 2)
    assert np.allclose(rotated, [0.5, 0.25])
    assert np.array_equal(source, [0.25, -0.5])


def test_source_model_pose_sync_repairs_fixed_fixture_reference() -> None:
    import pytest

    pytest.importorskip("robosuite", reason="requires pinned simulator transform utilities")

    class Fixture:
        root_body = "blender_main"
        joints = []
        lid_closed_pos = np.ones(3)

        def set_pos(self, value):
            self.pos = np.asarray(value)

        def set_euler(self, value):
            self.euler = np.asarray(value)

    fixture = Fixture()

    class Model:
        body_jntnum = np.array([0])
        body_pos = np.array([[0.1, 0.2, 0.3]])
        body_quat = np.array([[1.0, 0.0, 0.0, 0.0]])

        @staticmethod
        def body_name2id(name):
            assert name == "blender_main"
            return 0

    class Data:
        body_xpos = np.array([[1.0, 2.0, 3.0]])
        body_xquat = np.array([[1.0, 0.0, 0.0, 0.0]])

    class Sim:
        model = Model()
        data = Data()

    class Env:
        fixtures = {"blender": fixture}
        sim = Sim()
        updated = False

        def update_state(self):
            self.updated = True

    env = Env()
    synchronize_fixed_fixture_poses(env)
    assert fixture.pos.tolist() == [0.1, 0.2, 0.3]
    assert np.allclose(fixture.euler, 0.0)
    assert fixture.lid_closed_pos is None
    assert env.updated is True


def test_modern_microwave_interface_separates_contact_from_success() -> None:
    class Microwave:
        name = "microwave"

    class Env:
        microwave = Microwave()
        robots = [{"unused": True}]

        def __init__(self):
            self.robots = [type("Robot", (), {"gripper": {"right": "gripper"}})()]

        @staticmethod
        def check_contact(gripper, geom):
            assert gripper == "gripper"
            assert geom == "microwave_start_button"
            return True

        @staticmethod
        def _check_success():
            return False

    signals = microwave_button_signals(Env())
    assert signals == {"stage_contact_button": 1, "success": 0}


def test_modern_coffee_button_is_deterministic_and_contact_aware() -> None:
    class Machine:
        naming_prefix = "coffee_"
        _start_button_names = ["right", "left"]

    class Model:
        @staticmethod
        def geom_name2id(name):
            return {"coffee_left": 0, "coffee_right": 1}[name]

    class Data:
        geom_xpos = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    class Env:
        coffee_machine = Machine()
        robots = [type("Robot", (), {"gripper": {"right": "gripper"}})()]
        sim = type("Sim", (), {"model": Model(), "data": Data()})()
        contacted = set()

        def check_contact(self, gripper, geom):
            assert gripper == "gripper"
            return geom in self.contacted

        @staticmethod
        def _check_success():
            return False

    env = Env()
    assert select_coffee_machine_button(env, np.asarray([0.08, 0.0, 0.0])) == "coffee_right"
    env.contacted = {"coffee_left"}
    assert select_coffee_machine_button(env, np.asarray([0.08, 0.0, 0.0])) == "coffee_left"
    assert coffee_machine_button_signals(env) == {
        "stage_contact_button": 1,
        "success": 0,
    }
