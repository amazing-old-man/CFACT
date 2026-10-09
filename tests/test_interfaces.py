from __future__ import annotations

import pytest

pytest.importorskip("robosuite", reason="requires the pinned simulator checkout")
pytest.importorskip("mimicgen", reason="requires the pinned MimicGen checkout")
import numpy as np

from cfact.adapters.interfaces import MG_ModernPnPCounterToSink


class _Sink:
    name = "sink_fixture"

    @staticmethod
    def get_reset_regions():
        # Deliberately return right before left: the adapter must not depend on
        # dictionary insertion order or consume environment randomness.
        return {"basin_right": {}, "basin_left": {}}


def test_double_basin_reference_is_stable_symmetric_centroid() -> None:
    interface = MG_ModernPnPCounterToSink.__new__(MG_ModernPnPCounterToSink)
    interface.env = type("Env", (), {"sink": _Sink()})()
    calls: list[str] = []

    def get_object_pose(*, obj_name, obj_type, **_kwargs):
        assert obj_type == "geom"
        calls.append(obj_name)
        pose = np.eye(4)
        pose[:3, 3] = [-0.16, 0.25, 0.80] if obj_name.endswith("basin_left") else [0.16, 0.25, 0.80]
        return pose

    interface.get_object_pose = get_object_pose

    first = interface._sink_reference_pose()
    second = interface._sink_reference_pose()

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[:3, 3], [0.0, 0.25, 0.80])
    assert calls == [
        "sink_fixture_reg_basin_left",
        "sink_fixture_reg_basin_right",
        "sink_fixture_reg_basin_left",
        "sink_fixture_reg_basin_right",
    ]


def test_sink_reference_rejects_inconsistent_region_rotations() -> None:
    interface = MG_ModernPnPCounterToSink.__new__(MG_ModernPnPCounterToSink)
    interface.env = type("Env", (), {"sink": _Sink()})()

    def get_object_pose(*, obj_name, **_kwargs):
        pose = np.eye(4)
        if obj_name.endswith("basin_right"):
            pose[0, 0] = -1.0
        return pose

    interface.get_object_pose = get_object_pose

    with np.testing.assert_raises_regex(ValueError, "share one orientation"):
        interface._sink_reference_pose()
