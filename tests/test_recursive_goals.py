from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from cfact.generation.context import OperationContext
from cfact.generation.stage_compiler import propose_goals, compile_goal, source_suffix
from cfact.generation.options import compile_candidate_plan
from cfact.tasks import target_task

def context():
    binding={'joint_type':'slide'}
    return OperationContext('OpenDrawer/root', {'manipulated':'handle'},
        {'target_q':.3}, 'CONTACT_ESTABLISHED', 15, {}, np.eye(4), np.eye(4), np.eye(4), binding)

def geom():
    return dict(joint_name='drawer',joint_type='slide',current_q=.1,target_q=.3,
        joint_range=(0.,.4),axis_world=np.array([1.,0.,0.]),anchor_world=np.zeros(3))

def test_goals_compile_different_commands_preserve_task_and_current_geometry():
    ctx=context(); goals=propose_goals(target_task('OpenDrawer'),ctx)
    options=[compile_goal(g,ctx,geometry=geom(),contact_verified=True) for g in goals]
    assert len({compile_candidate_plan(o).poses.tobytes() for o in options})==3
    assert all(o.operation_context.task_goal==ctx.task_goal for o in options)
    assert all(np.all(o.source_action_indices==-1) for o in options)
    child_pose=np.eye(4);child_pose[0,3]=.05
    child=replace(ctx,current_eef_pose=child_pose,current_object_pose=child_pose)
    child_geometry={**geom(),'current_q':.15}
    updated=compile_goal(goals[0],child,geometry=child_geometry,contact_verified=True)
    assert not np.array_equal(updated.poses,options[0].poses)
    assert np.array_equal(updated.poses[0],child_pose)
    assert updated.candidate_edit['continuation_origin']=='CURRENT_STATE_PRIMITIVE'

def test_exhausted_incompatible_and_available_remain_distinct():
    source=SimpleNamespace(source_action_indices=np.arange(10),source_poses=np.repeat(np.eye(4)[None],10,axis=0),motion_object_poses=np.ones((10,4,4)))
    assert source_suffix(source,2)==('AVAILABLE',2)
    assert source_suffix(source,10)==('EXHAUSTED',None)
    source.source_action_indices=None
    assert source_suffix(source,2)==('INCOMPATIBLE',None)
    ctx=context();goals=propose_goals(target_task('OpenDrawer'),ctx)
    option=compile_goal(goals[0],ctx,geometry=geom(),source_status='EXHAUSTED',contact_verified=True)
    assert option.executable and option.motion_template=='CURRENT_STATE_PRIMITIVE'
    with pytest.raises(ValueError,match='NO_VERIFIED_CONTACT'):
        compile_goal(goals[0],ctx,geometry=geom())

def test_free_body_cannot_gain_fake_joint_or_continuation():
    assert propose_goals(target_task('OpenBlenderLid'),context())==[]
    assert propose_goals(target_task('OpenDrawer'),replace(context(),stage='TERMINAL_SUCCESS'))==[]

def test_compiled_joint_goal_matches_existing_measured_outcome_contract():
    from cfact.collection.worker import _goal_realization
    ctx=context();goal=propose_goals(target_task('OpenDrawer'),ctx)[0]
    option=compile_goal(goal,ctx,geometry=geom(),contact_verified=True)
    trace=SimpleNamespace(fixture_joint_names=['drawer'],fixture_qpos=[[.1],[.14]],sim_time=[0.,.05],local_event_flags=[False],commanded_actions=np.zeros((1,12)))
    fixed=SimpleNamespace(fixture_qpos=[[.14]],sim_time=[.05],commanded_actions=[])
    result=_goal_realization(option,trace,fixed,1)
    assert result['valid'] and result['target']==pytest.approx(.16)
    assert result['local_end_error']==pytest.approx(.02)

def test_source_prior_closing_event_is_preserved_in_recompiled_goal():
    ctx=context();goal=propose_goals(target_task('OpenDrawer'),ctx)[0]
    timeline=np.r_[np.full(9,-1.),np.ones(30)]
    option=compile_goal(goal,ctx,geometry=geom(),contact_verified=True,
        gripper=-1.,gripper_timeline=timeline,source_status='AVAILABLE')
    assert np.all(option.gripper_actions[:9,0]==-1)
    assert np.all(option.gripper_actions[9:,0]==1)
    assert option.motion_template=='SOURCE_PRIOR_RECOMPILED'
    assert option.candidate_edit['continuation_origin']=='SOURCE_PRIOR_RECOMPILED'


def test_source_entry_compatibility_preserves_existing_source_and_missing_limits():
    from cfact.generation.stage_compiler import contact_suffix_compatibility
    poses=np.repeat(np.eye(4)[None],8,axis=0);poses[3:,0,3]=.15
    source=SimpleNamespace(source_poses=poses,motion_object_poses=np.repeat(np.eye(4)[None],8,axis=0))
    saved=source.source_poses.copy();limits={'entry_translation_m':.06,'entry_rotation_rad':.6}
    status,record=contact_suffix_compatibility(source,3,np.eye(4),np.eye(4),limits)
    assert status=='INCOMPATIBLE' and record['remaining_source_waypoints']==5
    assert record['source_preserved'] and np.array_equal(source.source_poses,saved)
    assert contact_suffix_compatibility(source,3,np.eye(4),np.eye(4),{})[0]=='AVAILABLE'
    assert contact_suffix_compatibility(source,0,np.eye(4),np.eye(4),limits)[0]=='AVAILABLE'


def test_reference_minimum_requires_explicit_new_protocol_and_keeps_legacy_zero():
    from cfact.collection.worker import _minimum_event_steps
    option=SimpleNamespace(candidate_edit={},source_method='Reference')
    cfg={'method':{'recursive_goals_v2':True,'source_reference_minimum_event_steps':20,'minimum_event_steps':20}}
    assert _minimum_event_steps(cfg,option)==20
    assert _minimum_event_steps({'method':{'recursive_goals_v2':True}},option)==0
    assert _minimum_event_steps({'method':{**cfg['method'],'recursive_goals_v2':False}},option)==0
    option.candidate_edit={'effect_goal':{'mode':'hold'}}
    assert _minimum_event_steps(cfg,option)==20
