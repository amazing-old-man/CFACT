import json
from types import SimpleNamespace
import numpy as np
import pytest
from cfact.tasks import target_task
from cfact.generation.task_adapters import TaskEffectAdapter, activation_state
from cfact.generation.event_anchor import OperationStage, classify_stage
from cfact.generation.frontier import FrontierChild, expand_frontier, select_effect_frontier

def physical_arrays(kind='slide', progress=.01):
    poses=np.repeat(np.eye(4)[None,None],5,axis=0)
    actions=np.zeros((4,12));actions[:,6]=1
    return dict(object_poses=poses, actual_eef_pose=poses[:,0],sim_time=np.arange(5)*.05,
                task_success=np.zeros(5,dtype=bool), commanded_actions=actions,
                fixture_joint_types=np.array([kind]), relevant_fixture_qpos=np.linspace(0,progress,5)[:,None],
                subtask_signals_json=np.array([json.dumps({'stage_contact_handle':True})]*5))

def test_actual_topology_controls_effect_units_and_distance():
    adapter=TaskEffectAdapter(target_task('OpenCabinet'))
    a=physical_arrays(kind='hinge',progress=.1)
    x=adapter.observe_effect(a)
    assert x['units']=='rad' and x['physical_progress_above_floor']
    b={**a,'relevant_fixture_qpos':a['relevant_fixture_qpos']+.05}
    assert adapter.task_effect_distance(x,adapter.observe_effect(b))==pytest.approx(1)

def test_activation_requires_official_measured_fixture_value():
    adapter=TaskEffectAdapter(target_task('TurnOnMicrowave'))
    a=physical_arrays();a['task_success'][-1]=True
    assert not adapter.observe_effect(a)['supported']
    fixture=SimpleNamespace(get_state=lambda:dict(turned_on=True))
    assert activation_state(SimpleNamespace(env=SimpleNamespace(microwave=fixture)))['turned_on']
    a['fixture_activation_json']=np.array([json.dumps({'status':'MEASURED','turned_on':False})]*5)
    x=adapter.observe_effect(a)
    assert x['activation'] is False and x['official_success'] is True

def test_free_body_lid_is_not_inferred_as_hinge_from_registry():
    adapter=TaskEffectAdapter(target_task('OpenBlenderLid'))
    assert not adapter.observe_effect(physical_arrays())['supported']

def child(value, depth, path, *, supported=True):
    effect=dict(family='articulated_contact',units='m',value=[value],supported=supported,
                capability_reason=None if supported else 'NO_VERIFIED_BINDING',sim_time_s=1.,
                official_success=False,activation=None,physical_progress_above_floor=True,
                held_verified=False,contact=True)
    anchor=classify_stage(effect,source_cursor=depth)
    return FrontierChild(SimpleNamespace(source_method='G0'),dict(edge_path=path,Y_fixed='future_success_do_not_read'),anchor,depth)

def test_measured_frontier_uses_diversity_and_ignores_future_tail_and_method():
    adapter=TaskEffectAdapter(target_task('OpenDrawer'))
    children=[child(v,1,str(i)) for i,v in enumerate([.02,.021,.08])]
    selected=select_effect_frontier(children,adapter,2)
    assert [c.edge['edge_path'] for c in selected]==['0','2']
    children[0].option.source_method='G2'
    children[1].edge['Y_fixed']='official_success'
    assert [c.edge['edge_path'] for c in select_effect_frontier(children,adapter,2)]==['0','2']

def test_generic_frontier_expands_depth_four_with_finite_budget_and_no_unsupported():
    calls=[];adapter=TaskEffectAdapter(target_task('OpenDrawer'))
    def expand(c):
        calls.append(c.decision_depth)
        return {'node_id':'n'+str(c.decision_depth),'edges':[]},[child(.02,c.decision_depth+1,'e'+str(c.decision_depth))]
    pools,decisions=expand_frontier([child(.01,1,'root'),child(.05,1,'unsupported',supported=False)],
        adapter=adapter,expand=expand,maximum_depth=4,maximum_pools=10,parents_per_level=2,
        remaining_budget=lambda:1000,minimum_pool_steps=100)
    assert calls==[1,2,3]
    assert [p['decision_depth'] for p in pools]==[2,3,4]
    assert any(d.get('reason')=='NO_VERIFIED_BINDING' for d in decisions)
    assert any(d.get('reason')=='DEPTH_BUDGET_EXHAUSTED' for d in decisions)

def test_no_verified_held_child_is_never_called_recoverable():
    a=physical_arrays();a['subtask_signals_json']=np.array(['{}']*5)
    adapter=TaskEffectAdapter(target_task('CoffeeServeMug'))
    anchor=adapter.classify_stage(a,source_cursor=40)
    assert anchor.stage==OperationStage.UNSUPPORTED
    assert anchor.allowed_continuation is None


def test_proposed_release_or_unbound_local_goal_does_not_create_observed_stage():
    a=physical_arrays();adapter=TaskEffectAdapter(target_task('CoffeeServeMug'))
    anchor=adapter.classify_stage(a,source_cursor=40,local_event=True,release=True)
    assert anchor.stage==OperationStage.HELD_TRANSPORT
    assert not anchor.evidence['actual_opening_command_seen']
    assert not anchor.evidence['place_ready_checks']['verified']
