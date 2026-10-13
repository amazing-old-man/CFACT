"""Family noise calibration from already saved, actual development parents."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
from cfact.collection.runtime import _load_root, _create_env
from cfact.execution.snapshot import capture_snapshot,restore_snapshot,load_snapshot,save_snapshot,snapshot_differences
from cfact.execution.history import ControlStepLedger
from cfact.execution.options import MeasuredTrace
from cfact.execution.binding import trace_joint_spec
from cfact.adapters.action_codec import install_action_codec
from cfact.generation.task_adapters import FLOORS,family_for_task
from cfact.tasks import target_task
import robocasa,mimicgen.env_interfaces.robocasa,cfact.adapters.interfaces
from mimicgen.env_interfaces.base import make_interface
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',required=True);args=p.parse_args()
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research');OUT=args.output.resolve();assert OUT.is_relative_to(ROOT);OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
rootpath=next(args.run.glob('data/*/*/*/root_result.json'));manifest=json.loads(rootpath.with_name('root_manifest.json').read_text());root=_load_root(Path(manifest['prepared_hdf5']),manifest['source_group']);context=json.loads(rootpath.parent.joinpath('nodes/n_root/context.json').read_text());snapshot=load_snapshot(rootpath.parent/'nodes/n_root/parent_snapshot.json');task=target_task(args.task)
# Deterministic Reference choice, no outcome ranking. Actual issued commands only.
reference=next(x for x in sorted((rootpath.parent/'nodes/n_root/edges').glob('*/candidate.json')) if json.loads(x.read_text())['source_method']=='Reference') if any((rootpath.parent/'nodes/n_root/edges').glob('*/candidate.json')) else None
if reference is None:
 for card in sorted((rootpath.parent/'nodes/n_root/edges').glob('*/option.json')):
  if json.loads(card.read_text())['source_method']=='Reference':reference=card;break
if reference is None:
 pool=json.loads((rootpath.parent/'nodes/n_root/candidate_pool.json').read_text())
 # Pool contains cards in raw/selected according to the pinned schema.
 cid=next(x['candidate_id'] for k in ['raw_proposals','options','selected'] for x in pool.get(k,[]) if isinstance(x,dict) and x.get('source_method')=='Reference')
 edge=rootpath.parent/'nodes/n_root/edges'/cid
else:edge=reference.parent
with np.load(edge/'execution.npz',allow_pickle=False) as z:commands=z['commanded_actions'].copy()
if len(commands)<20:
 with np.load(edge/'continuation/execution.npz',allow_pickle=False) as z:commands=np.concatenate([commands,z['commanded_actions'].reshape(-1,commands.shape[1])])
commands=commands[:20];assert len(commands)>0
b=commands.copy();b[:,:3]*=-1
row=json.loads((ROOT/'repo/src/cfact/resources/tasks.json').read_text())[args.task];binding=context['operation_context'].get('joint_binding',{});spec=trace_joint_spec(binding) if binding else dict(fixture_qpos_indices=(),fixture_qvel_indices=(),fixture_joint_names=(),fixture_joint_types=(),fixture_joint_units=(),fixture_joint_qvel_units=())
env=_create_env(root['env_meta'],manifest['seed'],'robot0_agentview_left',0 if os.environ['MUJOCO_GL']=='egl' else -1);ledger=None
try:
 interface=make_interface(task.adapter_interface or task.mimicgen_interface,'robosuite',env.env);install_action_codec(interface);ledger=ControlStepLedger(env,OUT/'history',width=224,height=224);env._cfact_step_ledger=ledger
 alltraces=[];roundtrips=[]
 for label,issued in [('A1',commands),('B',b),('A2',commands)]:
  t=time.monotonic();receipt=restore_snapshot(env,snapshot,initial=root['initial']);ledger.restore_wall_seconds+=time.monotonic()-t;actual=capture_snapshot(env);diff=snapshot_differences(snapshot,actual);roundtrips.append(dict(label=label,differences=diff,receipt=receipt));save_snapshot(OUT/(label+'_start.json'),actual)
  ledger.begin(label,'aba_validation');trace=MeasuredTrace(row['event_object_ref'],row['operation_object_ref'],**spec);trace.capture(env,interface)
  for action in issued:
   env.step(action);trace.after_step(action,env,interface,False)
  arrays=trace.arrays();np.savez_compressed(OUT/(label+'.npz'),**arrays);alltraces.append(arrays);save_snapshot(OUT/(label+'_end.json'),capture_snapshot(env))
 a,c=alltraces[0],alltraces[2];noise={k:float(np.max(np.abs(a[k].astype(float)-c[k].astype(float)))) if a[k].size else 0. for k in ['sim_time','robot_qpos','robot_qvel','actual_eef_pose','object_poses','relevant_fixture_qpos','relevant_fixture_qvel','task_success','commanded_actions']}
 aba_pass=not any(r['differences'] for r in roundtrips) and all(v==0 for v in noise.values());family=family_for_task(task);unit=str(a['fixture_joint_types'][0]) if len(a['fixture_joint_types'])==1 else 'object';floor=FLOORS['activation'] if family=='button_activation' else FLOORS.get(unit,FLOORS['object']);noise_key='relevant_fixture_qpos' if unit in ['slide','hinge'] else 'object_poses';threshold=max(floor,5*noise[noise_key]);record=dict(status='PASS' if aba_pass else 'FAIL',task=args.task,family=family,source_identity=manifest.get('source_identity'),source_sha256=manifest['prepared_sha256'],seed=manifest['seed'],actual_total_control_steps=ledger.control_steps,wall_seconds=time.monotonic()-started,roundtrips=roundtrips,ABA_max_absolute_noise=noise,minimum_physical_effect_floor=floor,frozen_threshold=threshold,threshold_unit=unit,calibration_scope='DEVELOPMENT_ONLY',executed_horizon_s=len(commands)/20,noise_factor=5,render_integration_steps=0,reference_commands_sha256=hashlib.sha256(commands.tobytes()).hexdigest())
 (OUT/'SAME_PARENT_ABA.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
 if not aba_pass:raise SystemExit(2)
finally:
 if ledger:ledger.finish()
 if hasattr(env,'close'):env.close()
