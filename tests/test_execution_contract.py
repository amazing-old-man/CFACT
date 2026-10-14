from types import SimpleNamespace

import numpy as np
import pytest

from cfact.execution.binding import bind_joint, resolve_role_binding, validate_joint_binding
from cfact.execution.options import articulation_followthrough, execute_option
from cfact.collection.worker import _engineering_select, _goal_realization


class Model:
    def __init__(self, unrelated_first=False):
        self.body_parentid = np.array([0, 0, 1, 0])
        self.geom_bodyid = np.array([2, 3])
        self.njnt = 2
        self.names = ["cabinet_slide", "unrelated_hinge"]
        self.jnt_bodyid = np.array([1, 3])
        self.jnt_type = np.array([2, 3])
        self.jnt_qposadr = np.array([0, 1])
        self.jnt_dofadr = np.array([0, 1])
        self.jnt_limited = np.array([True, True])
        self.jnt_range = np.array([[-0.4, 0], [-2, 2]])
        if unrelated_first:
            self.names.reverse()
            for name in (
                "jnt_bodyid",
                "jnt_type",
                "jnt_qposadr",
                "jnt_dofadr",
                "jnt_limited",
                "jnt_range",
            ):
                setattr(self, name, getattr(self, name)[::-1])

    def joint_name2id(self, name):
        return self.names.index(name)

    def joint_id2name(self, index):
        return self.names[index]

    def geom_name2id(self, name):
        return ["selected_handle", "irrelevant_handle"].index(name)


def test_binding_uses_interface_role_and_ancestry_not_enumeration_or_distance():
    for reordered in (False, True):
        model = Model(reordered)

        class Interface:
            env = SimpleNamespace(sim=SimpleNamespace(model=model))

            def get_object_pose(self, obj_name, obj_type):
                return np.eye(4)

            def get_object_poses(self):
                # Equal poses deliberately make a nearest-geometry choice ambiguous.
                return {
                    "handle": self.get_object_pose("selected_handle", "geom"),
                    "other": self.get_object_pose("irrelevant_handle", "geom"),
                }

        role = resolve_role_binding(Interface(), "handle")
        binding = bind_joint(model, role)
        assert binding["joint_name"] == "cabinet_slide"
        assert binding["joint_unit"] == "m"
        validate_joint_binding(model, binding)


def test_multi_joint_ancestry_requires_explicit_semantics():
    model = Model()
    model.jnt_bodyid[:] = 1
    with pytest.raises(ValueError, match="bound_object_joint_ambiguous"):
        bind_joint(model, {"body_id": 2})
    assert (
        bind_joint(model, {"body_id": 2}, explicit_joint="cabinet_slide")["joint_name"]
        == "cabinet_slide"
    )
    with pytest.raises(ValueError, match="not_ancestor"):
        bind_joint(Model(), {"body_id": 2}, explicit_joint="unrelated_hinge")


@pytest.mark.parametrize(
    "q_parent,q_task,limits", [(0.0, -0.35, (-0.4, 0.0)), (0.3, 0.0, (0.0, 0.4))]
)
def test_followthrough_uses_explicit_complete_goal_with_signed_coordinates(
    q_parent, q_task, limits
):
    alpha = 0.45
    q_sub = q_parent + alpha * (q_task - q_parent)
    current = np.eye(4)
    current[0, 3] = q_sub
    path = articulation_followthrough(
        joint_name="bound_slide",
        joint_type="slide",
        joint_unit="m",
        current_q=q_sub,
        target_q=q_task,
        joint_range=limits,
        axis_world=np.array([1.0, 0, 0]),
        anchor_world=np.zeros(3),
        current_handle_pose=current,
        current_eef_pose=current,
        samples=3,
        stage="articulation_followthrough",
        actual_child={"sim_time_s": 1.25},
    )
    assert path[0, 0, 3] == pytest.approx(q_sub)
    assert path[-1, 0, 3] == pytest.approx(q_task)


class Trace:
    def __init__(self):
        self.sim_time = []
        self.commanded_actions = []
        self.local_event_flags = []
        self.task_success = []
        self.object_poses = []
        self.fixture_qpos = []

    def capture(self, env, interface):
        self.sim_time.append(env.t)
        self.task_success.append(False)
        pose = np.eye(4)
        pose[0, 3] = env.t
        self.object_poses.append(pose)
        self.fixture_qpos.append(np.array([env.t / 2]))

    def after_step(self, action, env, interface, event, **kwargs):
        self.commanded_actions.append(action.copy())
        self.capture(env, interface)
        self.local_event_flags.append(event)


def test_events_use_post_action_state_and_cursor_is_not_physical_progress():
    env = SimpleNamespace(t=0.0)
    env.step = lambda action: setattr(env, "t", env.t + 0.05)
    codec = SimpleNamespace(command=lambda pose, grip: np.r_[pose[:3, 3], grip])
    interface = SimpleNamespace(action_codec=codec)
    waypoint = SimpleNamespace(pose=np.eye(4), gripper_action=np.ones(1), noise=None)
    trajectory = SimpleNamespace(waypoint_sequences=[[waypoint, waypoint, waypoint]])
    trace = Trace()

    class Detector:
        def update(self, measured):
            assert measured.sim_time[-1] == 0.05
            return True, {"contact_relation_valid": True}

    result = execute_option(
        env,
        interface,
        trajectory,
        trace,
        Detector(),
        max_steps=3,
        source_action_indices=[41, 42, 43],
        source_episode="demo",
    )
    assert result["event_action_index"] == 0
    assert result["event_state_index"] == 1
    assert result["source_cursor"]["next_action_index"] == 42
    assert result["physical_progress"]["joint_qpos_final"] == [0.025]
    assert len(trace.sim_time) == len(trace.commanded_actions) + 1


def test_goal_diagnostic_uses_state_after_event_action():
    option = SimpleNamespace(
        goal_binding={
            "kind": "articulation_local_subgoal",
            "joint": "bound",
            "joint_unit": "m",
            "target_q": 0.5,
        },
        end_event="joint_subgoal",
    )
    local = SimpleNamespace(
        fixture_joint_names=("bound",),
        fixture_qpos=[[0.0], [0.5]],
        sim_time=[1.0, 1.05],
        local_event_flags=[True],
    )
    fixed = SimpleNamespace(fixture_qpos=[[0.5]], sim_time=[1.05], commanded_actions=[])
    result = _goal_realization(option, local, fixed, 0)
    assert result["target_event_error"] == 0


def test_engineering_budget_keeps_original_order_and_declares_nonexecution():
    options = [
        SimpleNamespace(candidate_id=str(i), source_method=family)
        for i, family in enumerate(["Reference", "G0", "G0", "G1", "G2", "G2"])
    ]
    rejected = []
    selected = _engineering_select(
        options,
        rejected,
        {"engineering_profile": {"name": "coverage", "max_candidates_per_family": 1}},
    )
    assert [option.candidate_id for option in selected] == ["0", "1", "3", "4"]
    assert [row["candidate_id"] for row in rejected] == ["2", "5"]
    assert {row["reason"] for row in rejected} == {"engineering_budget"}


def test_history_records_blocked_prefix_and_isolates_sibling_segments(tmp_path):
    from cfact.execution.history import ControlStepLedger

    class Sim:
        data = SimpleNamespace(time=0.0)

        def get_state(self):
            return np.array([self.data.time, 2.0])

        def render(self, height, width, camera_name):
            return np.full((height, width, 3), round(self.data.time * 20), dtype=np.uint8)

    sim = Sim()
    env = SimpleNamespace(env=SimpleNamespace(sim=sim))

    def actual_step(action):
        sim.data.time += 0.05

    env.step = actual_step
    history = ControlStepLedger(env, tmp_path, width=4, height=3)
    env.step(np.array([1.0]))
    env.step(np.array([2.0]))
    assert history.control_steps == 2
    assert len(history.references(segment="source_prefix")) == 3
    history.begin("edge_A", "local_operation")
    env.step(np.array([3.0]))
    sim.data.time = 0.10  # a restore has no env.step and does not consume a control tick
    history.begin("edge_B", "local_operation")
    env.step(np.array([4.0]))
    assert history.control_steps == 4
    first = history.references(segment="edge_A")
    second = history.references(segment="edge_B")
    assert len(first) == len(second) == 2
    assert {r["frame"] for r in first}.isdisjoint({r["frame"] for r in second})
    with np.load(tmp_path / second[-1]["frame"]) as values:
        np.testing.assert_array_equal(values["action_from_previous"], [4.0])
        assert values["rgb"].shape == (3, 3, 4, 3)
        assert values["sim_time_s"] == pytest.approx(0.15)
    history.finish()
    import json

    assert json.loads((tmp_path / "index.json").read_text())["control_steps"] == 4


def test_zero_step_keeps_source_cursor_and_has_one_real_terminal_state():
    env = SimpleNamespace(t=2.0)
    trace = Trace()
    trajectory = SimpleNamespace(waypoint_sequences=[])
    result = execute_option(
        env,
        SimpleNamespace(),
        trajectory,
        trace,
        SimpleNamespace(),
        max_steps=0,
        initial_source_cursor=37,
    )
    assert result["control_steps"] == 0
    assert result["source_cursor"]["next_action_index"] == 37
    assert len(trace.sim_time) == 1
    assert trace.commanded_actions == []


def test_saved_child_is_independent_before_tail_and_refreshes_context(tmp_path, monkeypatch):
    import json
    from cfact.collection import worker
    from cfact.generation.context import OperationContext
    from cfact.execution.snapshot import load_snapshot

    pose = np.eye(4)
    context = OperationContext(
        "parent",
        {"manipulated": "handle"},
        {"joint": "bound", "q_task": -0.4, "target_q": -0.4},
        "articulation",
        10,
        {},
        pose,
        pose,
        pose,
        {"joint_name": "bound"},
        False,
    )
    node = tmp_path / "nodes" / "n_parent"
    edge = node / "edges" / "candidate"
    edge.mkdir(parents=True)
    parent_refs = [{"frame": str(i), "sim_time_s": i / 20} for i in (1, 2, 3)]
    (node / "context.json").write_text(json.dumps({"history_refs": parent_refs}))
    local_refs = [
        {"frame": "local_initial", "sim_time_s": 0.15},
        {"frame": "local_after", "sim_time_s": 0.20},
    ]
    env = SimpleNamespace(
        state=np.array([0.20, -0.1]),
        _cfact_step_ledger=SimpleNamespace(references=lambda **kwargs: local_refs),
    )
    monkeypatch.setattr(
        worker,
        "capture_snapshot",
        lambda current: {"states": current.state.copy(), "schema": "test", "sim_time_s": 0.2},
    )
    moved = pose.copy()
    moved[0, 3] = -0.1
    trace = SimpleNamespace(fixture_poses=[pose], object_poses=[moved], actual_eef_pose=[moved])
    option = SimpleNamespace(operation_context=context, goal_binding={"q_task": -0.4})
    local = {"source_cursor": {"next_action_index": 11}, "local_event_reached": True}
    saved_state = worker._save_actual_child(env, edge, option, local, trace)
    env.state[:] = [0.8, -0.4]  # a later continuation must not change the saved local child
    np.testing.assert_array_equal(
        load_snapshot(edge / "child_snapshot.json")["states"], [0.2, -0.1]
    )
    np.testing.assert_array_equal(saved_state, [0.2, -0.1])
    record = json.loads((edge / "child_context.json").read_text())
    assert record["sim_time_s"] == 0.2
    assert record["operation_context"]["task_goal"]["q_task"] == -0.4
    assert record["operation_context"]["current_object_pose"][0][3] == -0.1
    assert record["operation_context"]["source_cursor"] == 11
    assert [r["frame"] for r in record["history_refs"]] == ["1", "2", "3", "local_after"]
    assert record["history_complete"]


def test_connector_only_action_does_not_advance_source_cursor():
    env = SimpleNamespace(t=0.0)
    env.step = lambda action: setattr(env, "t", env.t + 0.05)
    interface = SimpleNamespace(action_codec=SimpleNamespace(command=lambda pose, grip: np.ones(2)))
    waypoint = SimpleNamespace(pose=np.eye(4), gripper_action=np.ones(1), noise=None)
    trajectory = SimpleNamespace(waypoint_sequences=[[waypoint]])
    detector = SimpleNamespace(update=lambda trace: (False, {"contact_relation_valid": True}))
    result = execute_option(
        env,
        interface,
        trajectory,
        Trace(),
        detector,
        max_steps=1,
        source_action_indices=[-1],
        initial_source_cursor=37,
    )
    assert result["source_cursor"]["next_action_index"] == 37
    assert result["source_cursor"]["last_issued_action_index"] is None


def test_soft_joint_limit_parent_is_measured_not_clamped_and_targets_stay_bounded():
    from cfact.generation.options import (
        build_object_effect_options,
        operation_contract,
        solve_g2_articulation_subgoals,
    )
    from cfact.generation.segments import SourceSegment

    poses = np.repeat(np.eye(4)[None], 8, axis=0)
    poses[:, 0, 3] = np.linspace(0, -0.2, len(poses))
    source = SourceSegment(
        "recorded",
        10,
        17,
        np.eye(4),
        np.eye(4),
        poses,
        np.ones((8, 1)),
        (0.0, 0.0, 0.0, 0.0),
        (-1.0,),
        "handle",
        poses,
    )
    contract = operation_contract(
        operation_type="open_door",
        primitive_family="linear_joint",
        event_object_ref="handle",
        operation_object_ref="handle",
        holding_object=False,
    )
    options, _ = build_object_effect_options(
        root=source,
        donors=[],
        current_fixture_pose=np.eye(4),
        current_object_pose=np.eye(4),
        current_eef_pose=np.eye(4),
        contract=contract,
        official_transform=lambda **kwargs: kwargs["src_eef_poses"],
        parent_context_id="parent",
        end_event="articulation_subgoal_reached",
    )
    actual_q = 6.751064938265821e-05
    q_task = -0.4483991074942055
    solved, audit = solve_g2_articulation_subgoals(
        options,
        joint_name="bound",
        joint_type="slide",
        current_q=actual_q,
        joint_range=(-0.6, 0.0),
        axis_world=np.array([1.0, 0.0, 0.0]),
        anchor_world=np.zeros(3),
        current_handle_pose=np.eye(4),
        current_eef_pose=np.eye(4),
        target_q=q_task,
    )
    g2 = [option for option in solved if option.source_method == "G2"]
    assert len(g2) == 3
    assert all(row["status"] == "PASS" for row in audit)
    expected = [-0.11204914388651438, -0.201742467515232, -0.2914357911439497]
    np.testing.assert_allclose([o.goal_binding["q_sub"] for o in g2], expected, atol=1e-12, rtol=0)
    assert all(o.goal_binding["q_parent"] == actual_q for o in g2)
    assert all(o.goal_binding["q_task"] == q_task for o in g2)
    child_pose = np.eye(4)
    child_pose[0, 3] = actual_q
    tail = articulation_followthrough(
        joint_name="bound",
        joint_type="slide",
        joint_unit="m",
        current_q=actual_q,
        target_q=q_task,
        joint_range=(-0.6, 0.0),
        axis_world=np.array([1.0, 0.0, 0.0]),
        anchor_world=np.zeros(3),
        current_handle_pose=child_pose,
        current_eef_pose=child_pose,
        stage="articulation",
        actual_child={"sim_time_s": 7.15},
        samples=3,
    )
    assert tail[0, 0, 3] == actual_q
    assert tail[-1, 0, 3] == pytest.approx(q_task)
    _, invalid = solve_g2_articulation_subgoals(
        options,
        joint_name="bound",
        joint_type="slide",
        current_q=actual_q,
        joint_range=(-0.6, 0.0),
        axis_world=np.array([1.0, 0.0, 0.0]),
        anchor_world=np.zeros(3),
        current_handle_pose=np.eye(4),
        current_eef_pose=np.eye(4),
        target_q=0.1,
    )
    assert invalid == [{"status": "REJECT", "reason": "task_target_outside_joint_range"}]


def test_blocked_prefix_keeps_real_history_without_inventing_parent(tmp_path):
    import json
    from cfact.collection.worker import _record_blocked_source_replay

    args = SimpleNamespace(
        run=tmp_path, task="example_operation", root_id="source_demo", split="formal", gpu=0
    )
    cfg = {
        "generator_revision": "test",
        "run_seed": 1,
        "tree_limits": {"maximum_decision_depth": 3},
    }
    history = tmp_path / "history" / "actual_replay"
    root = {
        "initial": {"states": np.array([0.5, 1.0])},
        "actual_replay_steps": 19,
        "source_replay_history": str(history),
    }
    result = _record_blocked_source_replay(
        args,
        cfg,
        root,
        {},
        tmp_path / "source.hdf5",
        "demo",
        RuntimeError("precondition_not_observed"),
        {"status": "measured_only"},
    )
    assert result["nodes"] == [] and result["edges"] == []
    assert result["charged_control_steps"] == 19
    assert result["source_replay_history"] == str(history)
    assert result["official_task_failure_asserted"] is False
    manifest_path = next((tmp_path / "data").glob("*/*/*/root_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_replay_history"] == str(history)
    assert manifest["source_replay_control_steps"] == 19
    assert not (manifest_path.parent / "nodes").exists()


@pytest.mark.parametrize(
    "limits,raw_q,expected",
    [
        ((-0.6, 0.0), -0.4483991074942055, -0.4483991074942055),
        ((-0.6, 0.0), 6.751064938265821e-05, 0.0),
        ((-0.6, 0.0), -0.600031, -0.6),
        ((0.0, 0.4), 0.27, 0.27),
        ((0.0, 0.4), -0.000042, 0.0),
        ((0.0, 0.4), 0.400018, 0.4),
        ((-0.4, 0.2), 0.13, 0.13),
    ],
)
def test_source_goal_projects_only_evidence_outside_explicit_bound_range(limits, raw_q, expected):
    from cfact.execution.binding import source_joint_task_goal

    binding = {
        "joint_name": "bound_slide",
        "joint_type": "slide",
        "joint_unit": "m",
        "joint_limited": True,
        "joint_range": limits,
    }
    original = dict(binding)
    goal = source_joint_task_goal(
        binding,
        raw_q,
        episode="demo",
        state_index=177,
        source_provenance={"raw_sha256": "recorded_digest"},
        source_terminal_success=True,
    )
    assert goal["q_task"] == goal["target_q"] == expected
    assert goal["raw_source_q"] == goal["source_goal_evidence"]["raw_source_q"] == raw_q
    assert goal["goal_qualification"] == "source_terminal_official_success"
    assert goal["goal_constraint"]["applied"] == (raw_q != expected)
    assert goal["goal_constraint"]["input_q"] == raw_q
    assert goal["goal_constraint"]["output_q"] == expected
    assert goal["goal_constraint"]["projection_delta"] == expected - raw_q
    assert goal["goal_constraint"]["joint_range"] == list(limits)
    assert binding == original  # no state or binding is adjusted to fit the target


def test_unlimited_source_goal_does_not_invent_bounds_or_promote_source_qualification():
    from cfact.execution.binding import source_joint_task_goal

    binding = {
        "joint_name": "bound_hinge",
        "joint_type": "hinge",
        "joint_unit": "rad",
        "joint_limited": False,
        "joint_range": [0.0, 0.0],
    }
    goal = source_joint_task_goal(
        binding,
        7.2,
        episode="demo",
        state_index=90,
        source_provenance={},
        source_terminal_success=None,
    )
    assert goal["q_task"] == goal["raw_source_q"] == 7.2
    assert goal["goal_constraint"]["applied"] is False
    assert goal["goal_constraint"]["joint_range"] is None
    assert goal["goal_qualification"] == "source_endpoint_only"
    assert goal["source_goal_evidence"]["source_terminal_official_success"] is None
