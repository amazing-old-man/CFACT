"""RoboCasa OPENWAM-inspired control interventions, using actual env.step only."""
import argparse,dataclasses,hashlib,json,os,time
from pathlib import Path
import numpy as np
from cfact.collection.runtime import _load_root,_create_env
from cfact.execution.snapshot import capture_snapshot,restore_snapshot,save_snapshot,load_snapshot,snapshot_differences
from cfact.execution.options import MeasuredTrace
from cfact.execution.history import ControlStepLedger
from cfact.execution.binding import trace_joint_spec
from cfact.execution.current import current_info
from cfact.adapters.action_codec import install_action_codec
from cfact.generation.context import OperationContext
from cfact.generation.task_adapters import TaskEffectAdapter
from cfact.storage.schema import make_child_context
from cfact.io.review_video import QuadReviewWriter
from cfact.io.serialization import atomic_json,array_digest
from cfact.tasks import target_task
import robocasa,mimicgen.env_interfaces.robocasa,cfact.adapters.interfaces
from mimicgen.env_interfaces.base import make_interface
FAMILIES=['stop','scale_half','scale_1_5','reverse','axis_swap','translation_orientation_noise','gripper_open','delay_10']
def controls(source,cursor,count,family,seed):
    """Continuation after Source ends is genuinely stepped stop control, labelled -1."""
    if source.shape[1]!=12 or np.any(source[:,-1]>=0):raise ValueError('UNSUPPORTED_GENERIC_NONDELTA_CONTROL')
    stop=np.zeros(12);stop[6]=source[min(cursor+count-1,len(source)-1),6];stop[-1]=-1
    commands=np.repeat(stop[None],count,axis=0);mapping=np.full(count,-1,dtype=int)
    available=max(0,min(count,len(source)-cursor))
    if available:commands[:available]=source[cursor:cursor+available];mapping[:available]=np.arange(cursor,cursor+available)
    if family=='stop':commands[:,:6]=0
    elif family=='scale_half':commands[:,:6]*=.5
    elif family=='scale_1_5':commands[:,:6]*=1.5
    elif family=='reverse':commands[:,:6]*=-1
    elif family=='axis_swap':commands[:,[0,1]]=commands[:,[1,0]]
    elif family=='translation_orientation_noise':
        scale=np.maximum(np.std(source[:,:6],axis=0)*.2,.01)
        commands[:,:6]+=np.random.default_rng(seed).normal(0,scale,(count,6))
    elif family=='gripper_open':commands[:,6]=-1
    elif family=='delay_10':
        commands=np.concatenate([np.repeat(stop[None],10,axis=0),commands])[:count]
        mapping=np.concatenate([np.full(10,-1),mapping])[:count]
    else:raise ValueError(family)
    commands[:,:7]=np.clip(commands[:,:7],-1,1)
    return commands,mapping
p=argparse.ArgumentParser();p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scheme',choices=['GENERIC_CF_SINGLE','GENERIC_CF_OFFDEMO'],required=True);p.add_argument('--budget',type=int,default=1500);args=p.parse_args()
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research');run=args.output.resolve();assert run.is_relative_to(ROOT);run.mkdir(parents=True,exist_ok=True);started=time.monotonic();rp=next(args.reference_run.glob('data/*/*/*/root_result.json'));manifest=json.loads(rp.with_name('root_manifest.json').read_text());task=target_task(manifest['task']);root=_load_root(Path(manifest['prepared_hdf5']),manifest['source_group']);original_node=json.loads((rp.parent/'nodes/n_root/context.json').read_text());parent=load_snapshot(rp.parent/'nodes/n_root/parent_snapshot.json');cursor=original_node['source_action_prefix'];row=json.loads((ROOT/'repo/src/cfact/resources/tasks.json').read_text())[task.name];spec=trace_joint_spec(original_node['operation_context']['joint_binding']);root_key=rp.parent.name+'__'+args.scheme;rootdir=run/'data/development'/task.name/root_key;node=rootdir/'nodes/n_root';node.mkdir(parents=True,exist_ok=True);rootmanifest={**manifest,'source_replay_history':None,'scheme':args.scheme};atomic_json(rootdir/'root_manifest.json',rootmanifest)
seed=int(manifest['seed']);env=_create_env(root['env_meta'],seed,'robot0_agentview_left',0 if os.environ['MUJOCO_GL']=='egl' else -1);ledger=None;edges=[];parents={};restore_receipts=[]
def restore(snapshot):
    t=time.monotonic();receipt=restore_snapshot(env,snapshot,initial=root['initial']);ledger.restore_wall_seconds+=time.monotonic()-t
    diff=snapshot_differences(snapshot,capture_snapshot(env));restore_receipts.append(dict(receipt=receipt,differences=diff,status='PASS' if not diff else 'FAIL'))
    if diff:raise ValueError('GENERIC_COMPLETE_PARENT_RESTORE_MISMATCH')
def save_node(nodepath,snapshot,record):
    nodepath.mkdir(parents=True,exist_ok=True);save_snapshot(nodepath/'parent_snapshot.json',snapshot);np.savez_compressed(nodepath/'context.npz',sim_state=snapshot['states']);atomic_json(nodepath/'context.json',record)
try:
    interface=make_interface(task.adapter_interface or task.mimicgen_interface,'robosuite',env.env);install_action_codec(interface);ledger=ControlStepLedger(env,run/'history/development'/task.name/manifest['source_episode'],width=224,height=224);env._cfact_step_ledger=ledger
    env.reset_to(root['initial'])
    for action in root['actions'][:cursor]:env.step(action)
    replay_refs=ledger.references(segment='source_prefix',last=3);prefix_state=capture_snapshot(env)['states'];atomic_json(run/'SOURCE_PREFIX_PARENT_RECEIPT.json',dict(simulator_state_equal=bool(np.array_equal(prefix_state,parent['states'])),prefix_control_steps=cursor,parent_state_sha256=array_digest(parent['states']),replayed_state_sha256=array_digest(prefix_state)));
    if not np.array_equal(prefix_state,parent['states']):raise ValueError('SOURCE_PREFIX_PARENT_STATE_MISMATCH')
    restore(parent)
    ctx={**original_node,'history_refs':replay_refs,'source_replay_history':str(ledger.directory),'snapshot':'parent_snapshot.json'};save_node(node,parent,ctx);rootmanifest['source_replay_history']=str(ledger.directory);atomic_json(rootdir/'root_manifest.json',rootmanifest);parents['n_root']=(parent,cursor,ctx)
    plan={'scheme':args.scheme,'families':FAMILIES,'candidate_ceiling':8,'control_step_ceiling':args.budget,'local_horizon_steps':80,'fixed_horizon_steps':80,'offdemo_ratio':.25 if args.scheme.endswith('OFFDEMO') else 0,'source_parent_snapshot_sha256':hashlib.sha256((rp.parent/'nodes/n_root/parent_snapshot.json').read_bytes()).hexdigest(),'seed':seed,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'frozen_before_outcomes':True,'no_outcome_filter':True,'extension':'actual delta stop commands with source index -1; no static state/image padding'};atomic_json(rootdir/'BASELINE_PROTOCOL.json',plan)
    if args.scheme=='GENERIC_CF_OFFDEMO':
        prefix=node/'edges/offdemo_prefix';prefix.mkdir(parents=True,exist_ok=True);restore(parent);prefix_quad=QuadReviewWriter(run/'videos/offdemo_prefix.mp4',task=task.name,root=manifest['source_episode'],node='n_root',candidate='offdemo_prefix',method=args.scheme,intent='mandatory_prefix',fps=20,cell_width=224,cell_height=224,feature_scale=1.);ledger.begin(str(prefix),'offdemo_start_prefix');trace=MeasuredTrace(row['event_object_ref'],row['operation_object_ref'],**spec);trace.capture(env,interface);prefix_quad.append(env,trace.sim_time[-1],'local_operation');issued,mapping=controls(root['actions'],cursor,10,'translation_orientation_noise',seed+101)
        for action in issued:env.step(action);trace.after_step(action,env,interface,False);prefix_quad.append(env,trace.sim_time[-1],'local_operation')
        snapshot=capture_snapshot(env);save_snapshot(prefix/'child_snapshot.json',snapshot);np.savez_compressed(prefix/'execution.npz',**trace.arrays());np.savez_compressed(prefix/'child_context.npz',sim_state=snapshot['states']);current=current_info(env,interface);op={**original_node['operation_context'],'source_cursor':min(len(root['actions']),cursor+10),'current_fixture_pose':current.object_poses[row['operation_object_ref']].tolist(),'current_object_pose':current.object_poses[row['event_object_ref']].tolist(),'current_eef_pose':current.eef_pose.tolist()};atomic_json(prefix/'child_context.json',make_child_context(sim_state=snapshot['states'],operation_context=op,source_cursor={'next_action_index':min(len(root['actions']),cursor+10)},snapshot={'path':'child_snapshot.json'},history_refs=ledger.references(segment=str(prefix),last=3),goal_binding=op['task_goal']))
        (prefix/'continuation').mkdir(exist_ok=True);zero=MeasuredTrace(row['event_object_ref'],row['operation_object_ref'],**spec);zero.capture(env,interface);np.savez_compressed(prefix/'continuation/execution.npz',**zero.arrays());atomic_json(prefix/'continuation/result.json',dict(status='NOT_APPLICABLE'));atomic_json(prefix/'local_result.json',dict(termination_reason='PERTURBED_START_PREFIX_RETAINED',local_event_reached=False,official_success_at_child=trace.task_success[-1],source_cursor={'next_action_index':min(len(root['actions']),cursor+10)},Y_fixed={'status':'NOT_APPLICABLE'}));atomic_json(prefix/'costs.json',dict(control_steps_local=10,control_steps_fixed=0));media=prefix_quad.close();media['video']='videos/offdemo_prefix.mp4';atomic_json(prefix/'media.json',media)
        record={**ctx,'node_id':'n_offdemo','parent_edge':str(prefix.relative_to(run)),'sim_time_s':snapshot['sim_time_s'],'operation_context':op,'sim_state_sha256':array_digest(snapshot['states']),'history_refs':ledger.references(segment=str(prefix),last=3),'snapshot':'parent_snapshot.json'};save_node(rootdir/'nodes/n_offdemo',snapshot,record);parents['n_offdemo']=(snapshot,min(len(root['actions']),cursor+10),record);edges.append(dict(candidate_id='offdemo_prefix',status='PHYSICS_COMPLETE',edge_path=str(prefix.relative_to(run)),role='MANDATORY_START_PREFIX',local_event='PERTURBED_START_PREFIX_RETAINED',Y_fixed='NOT_APPLICABLE'))
    for i,family in enumerate(FAMILIES):
        parent_name='n_offdemo' if args.scheme=='GENERIC_CF_OFFDEMO' and i>=6 else 'n_root';snapshot,start_index,record=parents[parent_name];ep=rootdir/'nodes'/parent_name/'edges'/f'g_{i:02d}_{family}';ep.mkdir(parents=True,exist_ok=True)
        if ledger.control_steps+160>args.budget:
            edges.append(dict(candidate_id=ep.name,status='BUDGET_CENSORED',edge_path=str(ep.relative_to(run))));continue
        commands,mapping=controls(root['actions'],start_index,80,family,seed+i);next_index=max(start_index,int(mapping[mapping>=0].max())+1) if np.any(mapping>=0) else start_index;atomic_json(ep/'option.json',dict(candidate_id=ep.name,source_method='Generic',subtype=family,parent_context_id=parent_name,source_episode=manifest['source_episode'],effect_intent_is_ground_truth=False,source_mapping=mapping.tolist(),seed=seed+i));restore(snapshot);quad=QuadReviewWriter(run/'videos'/f'{ep.name}.mp4',task=task.name,root=manifest['source_episode'],node=parent_name,candidate=ep.name,method=args.scheme,intent=family,fps=20,cell_width=224,cell_height=224,feature_scale=1.)
        ledger.begin(str(ep),'local_operation');trace=MeasuredTrace(row['event_object_ref'],row['operation_object_ref'],**spec);trace.capture(env,interface);quad.append(env,trace.sim_time[-1],'local_operation')
        for action in commands:env.step(action);trace.after_step(action,env,interface,False);quad.append(env,trace.sim_time[-1],'local_operation')
        actual=capture_snapshot(env);save_snapshot(ep/'child_snapshot.json',actual);np.savez_compressed(ep/'child_context.npz',sim_state=actual['states']);np.savez_compressed(ep/'execution.npz',**trace.arrays());effect=TaskEffectAdapter(task).observe_effect(trace.arrays(),event_signal=row['event_signal']);atomic_json(ep/'observed_effect.json',effect);current=current_info(env,interface);op={**record['operation_context'],'source_cursor':next_index,'current_fixture_pose':current.object_poses[row['operation_object_ref']].tolist(),'current_object_pose':current.object_poses[row['event_object_ref']].tolist(),'current_eef_pose':current.eef_pose.tolist()};atomic_json(ep/'child_context.json',make_child_context(sim_state=actual['states'],operation_context=op,source_cursor={'next_action_index':next_index},snapshot={'path':'child_snapshot.json'},history_refs=ledger.references(segment=str(ep),last=3),goal_binding=op['task_goal']))
        (ep/'continuation').mkdir(exist_ok=True);ledger.begin(str(ep),'fixed_continuation');tail=MeasuredTrace(row['event_object_ref'],row['operation_object_ref'],**spec);tail.capture(env,interface);fixed,fixed_map=controls(root['actions'],next_index,80,'stop',seed+999)
        # Fixed continuation is source control, followed by actual stop commands.
        available=max(0,min(80,len(root['actions'])-start_index-80))
        if available:fixed[:available]=root['actions'][next_index:next_index+available]
        for action in fixed:env.step(action);tail.after_step(action,env,interface,False);quad.append(env,tail.sim_time[-1],'fixed_continuation')
        np.savez_compressed(ep/'continuation/execution.npz',**tail.arrays());status='official_success' if tail.task_success[-1] else 'horizon_censored';atomic_json(ep/'continuation/result.json',dict(status=status,official_success_end=tail.task_success[-1]));media=quad.close();media['video']=str((run/'videos'/f'{ep.name}.mp4').relative_to(run));atomic_json(ep/'media.json',media);atomic_json(ep/'local_result.json',dict(termination_reason='fixed_horizon_censored',local_event_reached=False,official_success_at_child=trace.task_success[-1],source_cursor={'next_action_index':next_index},Y_fixed={'status':status},actual_horizon_s=4.0));atomic_json(ep/'costs.json',dict(control_steps_local=80,control_steps_fixed=80));edges.append(dict(candidate_id=ep.name,status='PHYSICS_COMPLETE',edge_path=str(ep.relative_to(run)),local_event='fixed_horizon_censored',Y_fixed=status))
    pools=[dict(node_id='n_offdemo',parent_edge=next(e['edge_path'] for e in edges if e.get('role')=='MANDATORY_START_PREFIX'),edges=[e for e in edges if '/n_offdemo/' in e['edge_path']],decision_depth=2)] if 'n_offdemo' in parents else []
    rootedges=[e for e in edges if '/n_root/' in e['edge_path']];atomic_json(rootdir/'tree.json',dict(root='n_root',edges=rootedges,child_pools=pools,third_pools=[]));result=dict(task=task.name,root_id=root_key,split='development',status='PHYSICS_COMPLETE',scheme=args.scheme,edges=edges,charged_control_steps=ledger.control_steps,restore_wall_seconds=ledger.restore_wall_seconds,observed_max_decision_depth=2 if 'n_offdemo' in parents else 1,recursive_policy_expansions=0,offdemo_prefix_is_not_recursive_method_credit=True,wall_seconds=time.monotonic()-started);atomic_json(rootdir/'root_result.json',result);atomic_json(run/'COMPLETE_RESTORE_RECEIPTS.json',dict(status='PASS',receipts=restore_receipts));print(json.dumps(result,indent=2))
finally:
    if ledger:ledger.finish()
    if hasattr(env,'close'):env.close()
