from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cfact.generation.context import CANDIDATE_PLAN_SCHEMA, OperationContext
from cfact.generation.options import (
    _discrete_time_indices,
    bind_inherited_placement_goal,
    build_object_effect_options,
    operation_contract,
    solve_g2_articulation_subgoals,
)
from cfact.generation.segments import SourceSegment, select_compatible_donor


def source(name="root", *, opening=6, stage="held_transport"):
    poses = np.repeat(np.eye(4)[None], 9, axis=0)
    poses[:, 0, 3] = np.linspace(0, 0.16, len(poses))
    grips = np.ones((len(poses), 1))
    grips[opening:] = -1
    obj = poses.copy()
    obj[opening:, 0, 3] = obj[opening, 0, 3]
    poses[opening:, 2, 3] = np.linspace(0, 0.04, len(poses) - opening)
    return SourceSegment(
        episode_id=name, event_first_step=10, end_step=18,
        eef_pose_at_entry=poses[0], object_pose_at_entry=np.eye(4),
        source_poses=poses, gripper_actions=grips,
        base_action_max_abs=(0., 0., 0., 0.), mode_values=(-1.,),
        motion_object_ref="obj", motion_object_poses=obj,
        operation_type="pick_place", stage=stage,
        object_roles={"held": "obj", "container": "sink"},
        source_action_indices=np.array([-1, 10, 11, 12, 13, 14, 15, 16, 17]),
    )


def context(*, target_x=0.30, held=True):
    target = np.eye(4)
    target[0, 3] = target_x
    return OperationContext(
        parent_context_id="root/child", object_roles={"held": "obj", "container": "sink"},
        task_goal={"kind": "fixture_region_object_goal", "target_pose_world": target.tolist()},
        stage="held_transport" if held else "constrained_contact", source_cursor=10,
        controller_semantics={"gripper": "negative_open_positive_close"},
        current_fixture_pose=np.eye(4), current_object_pose=np.eye(4),
        current_eef_pose=np.eye(4), holding_object=held,
    )


def build(root=None, donors=(), ctx=None, amplitude=0.003, held=True):
    ctx = ctx or context(held=held)
    contract = operation_contract(
        operation_type="pick_place" if held else "open_door",
        primitive_family="placement" if held else "linear_joint",
        event_object_ref="obj", operation_object_ref="sink", holding_object=held,
    )
    return build_object_effect_options(
        root=root or source(), donors=donors, current_fixture_pose=np.eye(4),
        current_object_pose=np.eye(4), current_eef_pose=np.eye(4), contract=contract,
        official_transform=lambda obj_pose, src_eef_poses, src_obj_pose:
            np.einsum("ij,njk->nik", obj_pose @ np.linalg.inv(src_obj_pose), src_eef_poses),
        parent_context_id=ctx.parent_context_id, end_event="placement_ready",
        operation_context=ctx, g0_amplitude_m=amplitude,
    )


def test_zero_edit_is_identical_to_reference_after_complete_goal_binding():
    options, _ = build(amplitude=0.0)
    ref = options[0]
    for option in options:
        if option.source_method == "G0":
            np.testing.assert_array_equal(option.poses, ref.poses)
            np.testing.assert_array_equal(option.gripper_actions, ref.gripper_actions)
            np.testing.assert_array_equal(option.intended_object_path, ref.intended_object_path)
            np.testing.assert_array_equal(option.candidate_plan.source_action_indices,
                                          ref.candidate_plan.source_action_indices)
    assert ref.poses[6, 0, 3] == pytest.approx(0.30)


def test_inherited_placement_constraint_does_not_erase_g0_transport_edit():
    options, _ = build(ctx=context(target_x=0.30))
    rebound = bind_inherited_placement_goal(options, context(target_x=0.40))
    ref = rebound[0]
    for option in rebound:
        if option.source_method != "G0":
            continue
        delta = option.poses[:, :3, 3] - ref.poses[:, :3, 3]
        assert np.max(np.linalg.norm(delta[:6], axis=1)) == pytest.approx(0.003)
        np.testing.assert_allclose(delta[[0, 6, 7, 8]], 0, atol=1e-15)
        assert option.intended_object_path[6, 0, 3] == pytest.approx(0.40)
        assert option.candidate_plan.plan_difference["retreat_maximum_translation_m"] == 0
        assert option.candidate_plan.edit["applied_after"] == "complete_task_goal_constraint"


def test_donor_motion_gripper_and_retreat_use_the_donor_time_map():
    donor = source("donor", opening=3)
    donor.source_poses[4:, 2, 3] = np.linspace(0.10, 0.50, 5)
    options, _ = build(donors=[donor])
    candidate = next(option for option in options if option.source_method == "G1")
    np.testing.assert_array_equal(candidate.gripper_actions, donor.gripper_actions)
    np.testing.assert_array_equal(candidate.candidate_plan.source_action_indices,
                                  donor.source_action_indices)
    assert candidate.candidate_plan.events[0]["candidate_waypoint_index"] == 3
    assert candidate.candidate_plan.events[0]["source_action_index"] == 12
    assert candidate.poses[-1, 2, 3] == pytest.approx(0.50)
    assert candidate.poses[3, 0, 3] == pytest.approx(0.30)


def test_donor_stage_and_object_roles_are_checked_before_selection():
    root = source()
    donor = replace(source("wrong_stage"), stage="released")
    chosen, audit = select_compatible_donor(root, [donor], allow_none=True)
    assert chosen is None
    assert "operation_stage_mismatch" in audit[0]["rejection_reasons"]
    donor = replace(source("wrong_object"), motion_object_ref="other_object")
    chosen, audit = select_compatible_donor(root, [donor], allow_none=True)
    assert chosen is None
    assert "motion_object_role_mismatch" in audit[0]["rejection_reasons"]


@pytest.mark.parametrize("q_parent,q_task,q_range", [
    (-1.1, -1.8, (-2., -1.)), (1.8, 1.1, (1., 2.)),
    (-1.8, -1.1, (-2., -1.)), (1.1, 1.8, (1., 2.)),
])
def test_g2_subgoal_retains_explicit_task_goal_for_both_coordinate_signs(
    q_parent, q_task, q_range
):
    ctx = replace(context(held=False), task_goal={"joint": "drawer", "q_task": q_task},
                  joint_binding={"joint_name": "drawer"})
    options, _ = build(ctx=ctx, held=False)
    solved, audit = solve_g2_articulation_subgoals(
        options, joint_name="drawer", joint_type="slide", current_q=q_parent,
        joint_range=q_range, target_q=q_task, axis_world=np.array([1., 0., 0.]),
        anchor_world=np.zeros(3), current_handle_pose=np.eye(4), current_eef_pose=np.eye(4),
    )
    g2 = [option for option in solved if option.source_method == "G2"]
    assert len(g2) == 3 and all(row["status"] == "PASS" for row in audit)
    for option, fraction in zip(g2, (0.25, 0.45, 0.65)):
        expected_subgoal = q_parent + fraction * (q_task - q_parent)
        assert option.goal_binding["q_sub"] == pytest.approx(expected_subgoal)
        assert option.candidate_plan.task_goal["q_task"] == q_task
        assert option.poses[-1, 0, 3] == pytest.approx(expected_subgoal - q_parent)
        assert option.candidate_plan.schema == CANDIDATE_PLAN_SCHEMA
        expected_source_end = max(1, round(fraction * 8))
        assert option.candidate_plan.source_action_indices[-1] == 9 + expected_source_end
        assert np.all(option.gripper_actions > 0)
        assert option.candidate_plan.source_episode == "root"


def test_g2_never_guesses_task_goal_from_joint_limit_magnitude():
    options, _ = build(held=False)
    solved, audit = solve_g2_articulation_subgoals(
        options, joint_name="drawer", joint_type="slide", current_q=0.,
        joint_range=(-0.6, 0.), axis_world=np.array([1., 0., 0.]),
        anchor_world=np.zeros(3), current_handle_pose=np.eye(4), current_eef_pose=np.eye(4),
    )
    assert audit == [{"status": "REJECT", "reason": "complete_task_joint_goal_missing"}]
    assert not any(option.source_method == "G2" for option in solved)


def test_discrete_retiming_retains_every_command_change_without_interpolation():
    commands = np.array([[1.], [1.], [-1.], [1.], [1.], [-1.], [-1.]])
    for count in (4, 7, 20):
        indices = _discrete_time_indices(commands, count)
        mapped = commands[indices, 0]
        assert set(mapped) == {-1., 1.}
        changes = np.concatenate(([0], np.flatnonzero(mapped[1:] != mapped[:-1]) + 1))
        np.testing.assert_array_equal(mapped[changes], [1., -1., 1., -1.])
        assert indices[-1] == len(commands) - 1
    with pytest.raises(ValueError, match="preserve_all_gripper_events"):
        _discrete_time_indices(commands, 3)
