from __future__ import annotations

import numpy as np

from cfact.generation.options import (
    _object_motion_to_eef,
    build_object_effect_options,
    operation_contract,
    select_option_pool,
    select_child_frontier,
    compile_held_path_with_release,
    placement_shift_weights,
)
from cfact.generation.segments import SourceSegment
from cfact.execution.options import (
    LocalEventDetector,
    ReleaseEventDetector,
    MeasuredTrace,
    execute_option,
    connect_and_retime_poses,
    articulation_followthrough,
)


def segment(name: str, lateral: float) -> SourceSegment:
    poses = np.repeat(np.eye(4)[None], 8, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.15, len(poses))
    motion = poses[1:].copy()
    motion[:, 1, 3] = lateral * np.linspace(0, 1, len(motion))
    return SourceSegment(
        name,
        3,
        10,
        np.eye(4),
        np.eye(4),
        poses,
        np.ones((8, 1)),
        (0.0, 0.0, 0.0, 0.0),
        (-1.0,),
        "obj",
        motion,
    )


def transform(*, obj_pose, src_eef_poses, src_obj_pose):
    return np.einsum("ij,njk->nik", obj_pose @ np.linalg.inv(src_obj_pose), src_eef_poses)


def test_current_grasp_transform_drives_object_motion() -> None:
    path = segment("d", 0.02).motion_object_poses
    obj = np.eye(4)
    obj[:3, 3] = [1, 2, 3]
    eef = obj.copy()
    eef[2, 3] += 0.1
    desired, tools = _object_motion_to_eef(path, np.eye(4), np.eye(4), obj, eef, held=True)
    np.testing.assert_allclose(desired[0], obj)
    np.testing.assert_allclose(tools[0], eef)
    for o, tool in zip(desired, tools):
        np.testing.assert_allclose(np.linalg.inv(tool) @ o, np.linalg.inv(eef) @ obj)


def test_release_suffix_retracts_tool_from_settled_object() -> None:
    objects = np.repeat(np.eye(4)[None], 6, axis=0)
    objects[:4, 0, 3] = np.linspace(0, 0.12, 4)
    objects[4:, 0, 3] = 0.12
    source_tools = objects.copy()
    source_tools[4, 2, 3] = 0.10
    source_tools[5, 2, 3] = 0.20
    grip = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]])
    tools = compile_held_path_with_release(objects, source_tools, grip, np.eye(4), np.eye(4))
    np.testing.assert_allclose(tools[:3], objects[:3])
    assert np.linalg.norm(tools[-1, :3, 3] - objects[-1, :3, 3]) > 0.19
    np.testing.assert_allclose(placement_shift_weights(grip, 6)[3:], 1.0)


def test_variable_pool_and_unsupported_geometry() -> None:
    root = segment("root", 0)
    donors = [segment(f"d{i}", i * 0.015) for i in range(1, 5)]
    contract = operation_contract(
        operation_type="pick_place",
        primitive_family="placement",
        event_object_ref="obj",
        operation_object_ref="sink",
        holding_object=True,
    )
    options, _ = build_object_effect_options(
        root=root,
        donors=donors,
        current_fixture_pose=np.eye(4),
        current_object_pose=np.eye(4),
        current_eef_pose=np.eye(4),
        contract=contract,
        official_transform=transform,
        parent_context_id="task/root/n_root",
        end_event="entry_reached_while_held",
    )
    assert sum(o.source_method == "G1" for o in options) == len(donors)
    assert all(
        o.precondition_checks["support_and_clearance"] == "UNSUPPORTED"
        for o in options
        if o.source_method == "G2"
    )
    pool, rejected = select_option_pool(options, max_executed=5)
    assert len(pool) <= 5
    assert len(set(x.candidate_id for x in pool)) == len(pool)
    assert all(x.source_method != "G2" for x in pool)
    assert rejected


def test_event_requires_actual_new_object_motion_and_dwell() -> None:
    trace = MeasuredTrace("obj", "sink")
    trace.object_poses = [np.eye(4)]
    trace.object_to_gripper_transform = [np.eye(4)]
    detector = LocalEventDetector(np.array([0.1, 0, 0]), np.zeros(3), held=True, stable_steps=2)
    assert not detector.update(trace)[0]
    moved = np.eye(4)
    moved[0, 3] = 0.1
    trace.object_poses.append(moved)
    trace.object_to_gripper_transform.append(np.eye(4))
    assert not detector.update(trace)[0]
    assert not detector.update(trace)[0]
    assert detector.update(trace)[0]


def test_release_event_needs_opening_separation_and_stable_object() -> None:
    trace = MeasuredTrace("obj", "sink")
    trace.object_poses = [np.eye(4)]
    trace.object_to_gripper_transform = [np.eye(4)]
    trace.sim_time = [0.0]
    detector = ReleaseEventDetector(np.zeros(3), gripper_action_index=1, stable_steps=2)
    for command, shift in ((1, 0.04), (-1, 0.04), (-1, 0.04)):
        relative = np.eye(4)
        relative[0, 3] = shift
        trace.object_poses.append(np.eye(4))
        trace.object_to_gripper_transform.append(relative)
        trace.sim_time.append(trace.sim_time[-1] + 0.05)
        trace.commanded_actions.append(np.array([0.0, command]))
        reached, measurement = detector.update(trace)
    assert reached
    assert measurement["opening_command_seen"]
    assert not measurement["support_contact_directly_observed"]


def test_articulation_plan_follows_measured_joint_axis() -> None:
    from cfact.generation.options import solve_g2_articulation_subgoals

    root = segment("handle", 0.0)
    contract = operation_contract(
        operation_type="open_door",
        primitive_family="linear_joint",
        event_object_ref="handle",
        operation_object_ref="handle",
        holding_object=False,
    )
    options, _ = build_object_effect_options(
        root=root,
        donors=[],
        current_fixture_pose=np.eye(4),
        current_object_pose=np.eye(4),
        current_eef_pose=np.eye(4),
        contract=contract,
        official_transform=transform,
        parent_context_id="drawer/root",
        end_event="articulation_subgoal_reached",
    )
    solved, audit = solve_g2_articulation_subgoals(
        options,
        joint_name="drawer_joint",
        joint_type="slide",
        current_q=0.0,
        joint_range=(-0.6, 0.0),
        target_q=-0.6,
        axis_world=np.array([1.0, 0.0, 0.0]),
        anchor_world=np.zeros(3),
        current_handle_pose=np.eye(4),
        current_eef_pose=np.eye(4),
    )
    g2 = [x for x in solved if x.source_method == "G2"]
    assert len(g2) == 3
    assert all(x.executable for x in g2)
    assert all(-0.6 <= x.goal_binding["target_q"] <= 0.0 for x in g2)
    for option in g2:
        np.testing.assert_allclose(option.poses[-1, 0, 3], option.goal_binding["target_q"])
    assert all(x["status"] == "PASS" for x in audit)


def test_support_solver_checks_full_footprint() -> None:
    from cfact.generation.options import solve_g2_support_goals

    root = segment("obj", 0.0)
    contract = operation_contract(
        operation_type="pick_place",
        primitive_family="placement",
        event_object_ref="obj",
        operation_object_ref="sink",
        holding_object=True,
    )
    options, _ = build_object_effect_options(
        root=root,
        donors=[],
        current_fixture_pose=np.eye(4),
        current_object_pose=np.eye(4),
        current_eef_pose=np.eye(4),
        contract=contract,
        official_transform=transform,
        parent_context_id="sink/root",
        end_event="entry_reached_while_held",
    )
    bbox = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (0.0, 0.05)])
    tiny = [{"name": "tiny", "pose": np.eye(4).tolist(), "size_m": [0.15, 0.15]}]
    solved, audit = solve_g2_support_goals(
        options,
        support_regions=tiny,
        object_bbox_local=bbox,
        current_object_pose=np.eye(4),
        current_eef_pose=np.eye(4),
    )
    assert not any(x.source_method == "G2" for x in solved)
    assert audit[0]["reason"] == "object_footprint_exceeds_support_region"


def test_connection_retimes_translation_rotation_and_preserves_opening() -> None:
    start = np.eye(4)
    targets = np.repeat(np.eye(4)[None], 2, axis=0)
    targets[0, 0, 3] = 0.03
    targets[1, 0, 3] = 0.06
    targets[1, :3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    poses, grip, audit = connect_and_retime_poses(
        start,
        targets,
        np.array([[1.0], [-1.0]]),
        maximum_translation_m=0.01,
        maximum_rotation_rad=0.1,
    )
    assert np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).max() <= 0.0100001
    assert np.all(grip[:-1] > 0)
    assert grip[-1, 0] < 0
    assert audit["retiming_added_control_steps"] > 0


def test_articulation_followthrough_starts_at_measured_tool_pose() -> None:
    handle = np.eye(4)
    handle[0, 3] = 0.2
    eef = handle.copy()
    eef[2, 3] = 0.05
    path = articulation_followthrough(
        joint_type="slide",
        joint_name="drawer_joint",
        target_q=-0.5,
        joint_unit="m",
        stage="constrained_contact",
        actual_child={"sim_time_s": 0.5},
        current_q=-0.1,
        joint_range=(-0.5, 0.0),
        axis_world=np.array([1.0, 0.0, 0.0]),
        anchor_world=np.zeros(3),
        current_handle_pose=handle,
        current_eef_pose=eef,
        samples=5,
    )
    np.testing.assert_allclose(path[0], eef)
    np.testing.assert_allclose(path[-1, 0, 3], eef[0, 3] - 0.4)


def test_official_success_takes_priority_at_same_step_as_local_event() -> None:
    from types import SimpleNamespace

    class FakeTrace:
        def __init__(self) -> None:
            self.sim_time = []
            self.commanded_actions = []
            self.local_event_flags = []
            self.task_success = []
            self.fixture_qpos = [np.zeros(0)]
            self.object_poses = [np.eye(4)]

        def capture(self, env, interface) -> None:
            self.sim_time.append(0.0)
            self.task_success.append(False)

        def after_step(self, action, env, interface, event, **kwargs) -> None:
            self.commanded_actions.append(action)
            self.sim_time.append(0.05)
            self.task_success.append(True)
            self.local_event_flags.append(event)

    class FakeDetector:
        def update(self, trace):
            return True, {"contact_relation_valid": True}

    trace = FakeTrace()
    waypoint = SimpleNamespace(pose=np.eye(4), gripper_action=np.ones(1), noise=None)
    trajectory = SimpleNamespace(waypoint_sequences=[[waypoint]])
    interface = SimpleNamespace(action_codec=SimpleNamespace(
        command=lambda target_pose, gripper_action: np.zeros(12)
    ))
    env = SimpleNamespace(step=lambda action: None)
    result = execute_option(env, interface, trajectory, trace, FakeDetector(), max_steps=2)
    assert result["termination_reason"] == "official_task_success"
    assert result["official_success_at_child"] is True
    assert result["local_event_reached"] is False
    assert trace.local_event_flags == [True]


def test_frontier_uses_fixed_method_coverage_without_tail_outcomes() -> None:
    from types import SimpleNamespace

    rows = [
        (SimpleNamespace(candidate_id=f"c{i}", source_method=method), {"Y_fixed": outcome})
        for i, (method, outcome) in enumerate(
            (("Reference", "success"), ("G0", "failure"), ("G1", "failure"), ("G2", "success"))
        )
    ]
    chosen = select_child_frontier(rows, family_priority=("G2", "G1", "Reference", "G0"), maximum=2)
    assert [option.source_method for option, _ in chosen] == ["G2", "G1"]
    flipped = [
        (option, {"Y_fixed": "failure" if edge["Y_fixed"] == "success" else "success"})
        for option, edge in rows
    ]
    assert [
        option.candidate_id
        for option, _ in select_child_frontier(
            flipped, family_priority=("G2", "G1", "Reference", "G0"), maximum=2
        )
    ] == ["c3", "c2"]
    assert select_child_frontier(rows, family_priority=("G2",), maximum=0) == []


def test_official_success_at_child_skips_unsupported_skill_checks() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    from cfact.collection import worker as module

    class SuccessfulChildTrace:
        def __init__(self, *args, **kwargs) -> None:
            self.task_success = []
            self.fixture_qpos = [np.zeros(0)]

        def capture(self, env, interface) -> None:
            self.task_success = [True]

    moved = np.eye(4)
    moved[0, 3] = 0.25
    local_trace = SimpleNamespace(
        commanded_actions=[np.ones(13)],
        object_to_gripper_transform=[np.eye(4), moved],
    )
    trajectory = SimpleNamespace(
        waypoint_sequences=[[SimpleNamespace(pose=np.eye(4), gripper_action=np.ones(1))]]
    )
    option = SimpleNamespace(
        goal_binding={"kind": "fixture_region_object_goal"},
        operation_context=SimpleNamespace(task_goal={"kind": "fixture_region_object_goal"}),
    )
    with patch.object(module, "MeasuredTrace", SuccessfulChildTrace):
        _, status, plan = module._tail(
            None,
            None,
            option,
            trajectory,
            0,
            steps=150,
            object_ref="obj",
            fixture_ref="fixture",
            joint_spec={},
            quad=None,
            local_trace=local_trace,
            holding_object=True,
            local_termination_reason="official_task_success",
        )
    assert status == "official_success"
    assert plan["stop_reason"] == "official_success_at_child"
    labels = module._outcome_labels(
        {"official_success_at_child": True, "termination_reason": "official_task_success"},
        "NO_SUPPORTED_CONTINUATION",
    )
    assert labels["SUCCESS_WITHIN_HORIZON"] is True
    assert labels["NO_SUPPORTED_CONTINUATION"] is False
