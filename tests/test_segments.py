from __future__ import annotations

import numpy as np
import h5py

from cfact.generation.segments import (
    SourceSegment,
    deform_candidate_poses,
    first_sustained_true,
    load_source_segment,
    relative_entry_descriptor,
    select_compatible_donor,
)


def segment(name: str, x: float, gripper: float = 1.0) -> SourceSegment:
    pose = np.eye(4)
    pose[0, 3] = x
    return SourceSegment(
        episode_id=name,
        event_first_step=1,
        end_step=4,
        eef_pose_at_entry=pose,
        object_pose_at_entry=np.eye(4),
        source_poses=np.repeat(pose[None], 4, axis=0),
        gripper_actions=np.full((4, 1), gripper),
        base_action_max_abs=(0.0, 0.0, 0.0, 0.0),
        mode_values=(-1.0,),
    )


def test_first_sustained_true_ignores_short_dropout_contact() -> None:
    signal = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    assert first_sustained_true(signal, 3) == 4
    assert first_sustained_true(signal, 4) is None


def test_donor_selection_is_geometry_based_and_rejects_gripper_mismatch() -> None:
    root = segment("root", 0.0)
    chosen, rows = select_compatible_donor(
        root,
        [segment("far", 0.4), segment("near", 0.1), segment("wrong", 0.01, -1.0)],
    )
    assert chosen.episode_id == "near"
    assert next(row for row in rows if row["donor_id"] == "wrong")["compatible"] is False


def test_donor_selection_can_report_no_legal_candidate() -> None:
    root = segment("root", 0.0, 1.0)
    chosen, rows = select_compatible_donor(
        root,
        [segment("wrong", 0.01, -1.0)],
        allow_none=True,
    )
    assert chosen is None
    assert rows[0]["compatible"] is False


def test_singular_donor_is_rejected_without_blocking_valid_donor() -> None:
    root = segment("root", 0.0)
    singular = segment("singular", 0.01)
    singular_pose = np.array(singular.object_pose_at_entry, copy=True)
    singular_pose[3] = 0.0
    singular = SourceSegment(**{**singular.__dict__, "object_pose_at_entry": singular_pose})
    chosen, rows = select_compatible_donor(
        root,
        [singular, segment("valid", 0.2)],
        allow_none=True,
    )
    assert chosen is not None
    assert chosen.episode_id == "valid"
    rejected = next(row for row in rows if row["donor_id"] == "singular")
    assert rejected["compatible"] is False
    assert rejected["geometry_distance_available"] is False
    assert "combined" not in rejected
    assert any(
        reason.startswith("invalid_donor_entry_geometry:")
        for reason in rejected["rejection_reasons"]
    )


def test_invalid_root_returns_no_donor_and_never_reports_distances() -> None:
    root = segment("root", 0.0)
    invalid_pose = np.array(root.object_pose_at_entry, copy=True)
    invalid_pose[0, 0] = np.nan
    root = SourceSegment(**{**root.__dict__, "object_pose_at_entry": invalid_pose})
    chosen, rows = select_compatible_donor(
        root,
        [segment("valid_a", 0.1), segment("valid_b", 0.2)],
        allow_none=True,
    )
    assert chosen is None
    assert len(rows) == 2
    assert all(row["compatible"] is False for row in rows)
    assert all(row["geometry_distance_available"] is False for row in rows)
    assert all("combined" not in row for row in rows)
    assert all(
        row["rejection_reasons"][0].startswith("invalid_root_entry_geometry:") for row in rows
    )


def test_relative_descriptor_rejects_singular_pose_with_value_error() -> None:
    invalid = segment("invalid", 0.0)
    singular_pose = np.zeros((4, 4), dtype=float)
    invalid = SourceSegment(**{**invalid.__dict__, "object_pose_at_entry": singular_pose})
    with np.testing.assert_raises_regex(ValueError, "object_pose_at_entry_singular"):
        relative_entry_descriptor(invalid)


def test_g0_is_smooth_and_returns_to_reference_endpoint() -> None:
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.06, 7)
    changed, params = deform_candidate_poses(
        poses, method="G0", primitive_family="button", object_pose=np.eye(4)
    )
    assert params["excursion_m"] == 0.003
    assert np.allclose(changed[[0, -1]], poses[[0, -1]])
    assert not np.allclose(changed[1:-1], poses[1:-1])


def test_g2_button_changes_press_geometry_but_preserves_endpoint() -> None:
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    poses[:, 2, 3] = [0.0, 0.01, 0.02, 0.03, 0.02, 0.01, 0.0]
    changed, params = deform_candidate_poses(
        poses, method="G2", primitive_family="button", object_pose=np.eye(4)
    )
    assert params["press_depth_delta_m"] == 0.004
    assert np.allclose(changed[[0, -1]], poses[[0, -1]])
    assert np.max(np.abs(changed[:, 2, 3])) > np.max(np.abs(poses[:, 2, 3]))


def test_g2_placement_shapes_target_local_approach_and_preserves_endpoints() -> None:
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.12, 7)
    changed, params = deform_candidate_poses(
        poses, method="G2", primitive_family="placement", object_pose=np.eye(4)
    )
    assert params["kind"] == "insertion_approach_geometry"
    assert params["approach_clearance_m"] == 0.008
    assert np.array_equal(changed[0], poses[0])
    assert np.array_equal(changed[-1], poses[-1])
    assert np.max(changed[1:-1, 2, 3]) == 0.008


def test_g2_lever_adds_object_local_angular_sweep_and_preserves_endpoints() -> None:
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    angles = np.linspace(0.0, 0.3, len(poses))
    for index, angle in enumerate(angles):
        cosine, sine = np.cos(angle), np.sin(angle)
        poses[index, :3, :3] = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        poses[index, 0, 3] = 0.05
    changed, params = deform_candidate_poses(
        poses, method="G2", primitive_family="lever", object_pose=np.eye(4)
    )
    assert params["angle_delta_rad"] == 0.08
    assert params["dominant_rotation_axis"] == 2
    assert params["endpoint_angle_offset_rad"] == 0.0
    assert np.array_equal(changed[0], poses[0])
    assert np.array_equal(changed[-1], poses[-1])
    assert not np.allclose(changed[1:-1], poses[1:-1])


def test_physical_start_override_controls_source_suffix(tmp_path) -> None:
    dataset = tmp_path / "prepared.hdf5"
    with h5py.File(dataset, "w") as target:
        group = target.create_group("data/demo_4")
        group.create_dataset("actions", data=np.zeros((6, 12)))
        info = group.create_group("datagen_info")
        signals = info.create_group("subtask_term_signals")
        signals.create_dataset("stage_contact", data=[0, 0, 0, 0, 1, 1])
        poses = np.repeat(np.eye(4)[None], 6, axis=0)
        poses[:, 0, 3] = np.arange(6)
        info.create_dataset("target_pose", data=poses)
        info.create_dataset("eef_pose", data=poses)
        info.create_dataset("gripper_action", data=np.ones((6, 1)))
        objects = info.create_group("object_poses")
        objects.create_dataset("rack", data=np.repeat(np.eye(4)[None], 6, axis=0))

    annotated = load_source_segment(
        dataset,
        "demo_4",
        event_signal="stage_contact",
        event_minimum_duration=1,
        object_ref="rack",
    )
    physical = load_source_segment(
        dataset,
        "demo_4",
        event_signal="stage_contact",
        event_minimum_duration=1,
        object_ref="rack",
        start_step=2,
    )
    assert annotated.event_first_step == 4
    assert physical.event_first_step == 2
    assert physical.source_poses[0, 0, 3] == 2
    assert physical.source_poses[1, 0, 3] == 2


def test_segment_entry_holds_prior_command_before_opening_action(tmp_path) -> None:
    dataset = tmp_path / "discrete_event.hdf5"
    with h5py.File(dataset, "w") as source:
        group = source.create_group("data/demo")
        group.create_dataset("actions", data=np.zeros((4, 12)))
        info = group.create_group("datagen_info")
        info.create_dataset("subtask_term_signals/contact", data=[0, 1, 1, 1])
        info.create_dataset("target_pose", data=np.repeat(np.eye(4)[None], 4, axis=0))
        info.create_dataset("eef_pose", data=np.repeat(np.eye(4)[None], 4, axis=0))
        info.create_dataset("object_poses/obj", data=np.repeat(np.eye(4)[None], 4, axis=0))
        info.create_dataset("gripper_action", data=np.array([[1.], [1.], [-1.], [-1.]]))
    result = load_source_segment(
        dataset, "demo", event_signal="contact", event_minimum_duration=1,
        object_ref="obj", motion_object_ref="obj", start_step=2,
    )
    np.testing.assert_array_equal(result.gripper_actions[:, 0], [1., -1., -1.])
    np.testing.assert_array_equal(result.source_action_indices, [-1, 2, 3])
    assert len(result.motion_object_poses) == len(result.source_poses) == 3
