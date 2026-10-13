"""MuJoCo full snapshot A/B/A and synchronized RGB audit on prechosen dev root."""
import copy,hashlib,json,time,os
from pathlib import Path
import numpy as np
from cfact.collection.runtime import _load_root,_create_env,_probe_parent
from cfact.collection.worker import _probe_held_parent,stable_seed
from cfact.execution.snapshot import capture_snapshot,restore_snapshot,save_snapshot,_pack
from cfact.execution.history import ControlStepLedger
from cfact.execution.observation import observe,observation_poses
from cfact.adapters.action_codec import install_action_codec
from cfact.tasks import target_task
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research');OUT=Path(os.environ.get('CFACT_ABA_OUT',str(ROOT/'01_development/ABA_BASE'))).resolve();assert OUT.is_relative_to(ROOT);OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
import robocasa,mimicgen.env_interfaces.robocasa,cfact.adapters.interfaces
from mimicgen.env_interfaces.base import make_interface
TASK='OpenDrawer';DEMO='demo_0';dataset=Path('/mnt/ssd2/yanghc/CFACT/source/PhysicalAI-Robotics-Manipulation-Kitchen-Demos__522e4ffa__train40_v1/prepared')/(TASK+'.hdf5');root=_load_root(dataset,DEMO);task=target_task(TASK);row=json.loads((ROOT/'repo/src/cfact/resources/tasks.json').read_text())[TASK]
env=_create_env(root['env_meta'],stable_seed(20261014,TASK,DEMO,'aba'),'robot0_agentview_left',0)
ledger=None
try:
 interface=make_interface(task.adapter_interface or task.mimicgen_interface,'robosuite',env.env);install_action_codec(interface)
 ledger=ControlStepLedger(env,OUT/'history',width=224,height=224);env._cfact_step_ledger=ledger
 probe=_probe_parent(env,interface,root['initial'],root['actions'],row['event_signal'],row['event_minimum_duration']);snapshot=capture_snapshot(env);
 buffers={k:{n:{'type':type(x).__name__,'dtype':str(getattr(x,'dtype',None)),'shape':list(getattr(x,'shape',[])),'values':str(x)[:300]} for n,x in vars(v).items()} for k,v in vars(env.env.robots[0]).items() if k.startswith('recent_') and hasattr(v,'__dict__')};(OUT/'LIVE_BUFFER_INSPECTION.json').write_text(json.dumps(buffers,indent=2));save_snapshot(OUT/'parent_snapshot.json',snapshot)
 cursor=probe['parent_action_count'];a=root['actions'][cursor:cursor+10].copy();b=a.copy();b[:,:6]*=-1
 def diffs(x,y,path=''):
  result=[]
  if isinstance(x,dict):
   if set(x)!=set(y):result.append({'path':path,'reason':'KEY_SET_CHANGED'})
   for k in set(x)&set(y):result+=diffs(x[k],y[k],path+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   if len(x)!=len(y):return [{'path':path,'reason':'LENGTH_CHANGED'}]
   for i,(u,v) in enumerate(zip(x,y)):result+=diffs(u,v,path+'/'+str(i))
  elif isinstance(x,np.ndarray):
   if not np.array_equal(x,y):result.append({'path':path,'shape':list(x.shape),'max_abs_diff':float(np.max(np.abs(x-y))) if x.size and x.dtype.kind in 'biufc' and x.shape==y.shape else None})
  elif x!=y:result.append({'path':path,'expected':str(x)[:80],'actual':str(y)[:80]})
  return result
 starts=[];traces=[];roundtrip=[]
 for label,commands in [('A1',a),('B',b),('A2',a)]:
  t=time.monotonic();receipt=restore_snapshot(env,snapshot,initial=root['initial']);ledger.restore_wall_seconds+=time.monotonic()-t;restored=capture_snapshot(env);roundtrip.append({'run':label,'differences':diffs(snapshot,restored),'restore_receipt':receipt});save_snapshot(OUT/(label+'_start.json'),restored)
  ledger.begin(label,'aba_validation');states=[];effects=[];eef=[];times=[]
  for action in [None,*commands]:
   if action is not None:env.step(action)
   cap=observe(env);poses=observation_poses(env,interface);states.append(cap.state);eef.append(poses[1]);effects.append(poses[0][row['event_object_ref']]);times.append(float(env.env.sim.data.time))
  values={'states':np.asarray(states),'effect_pose':np.asarray(effects),'eef_pose':np.asarray(eef),'sim_time':np.asarray(times),'commanded_actions':commands};np.savez_compressed(OUT/(label+'.npz'),**values);traces.append(values)
  save_snapshot(OUT/(label+'_end.json'),capture_snapshot(env));starts.append(restored)
 x,z=traces[0],traces[2];noise={k:float(np.max(np.abs(x[k]-z[k]))) for k in ['states','effect_pose','eef_pose','sim_time','commanded_actions']}
 record={'status':'PASS' if not any(v['differences'] for v in roundtrip) and all(v==0 for v in noise.values()) else 'FAIL','task':TASK,'root':DEMO,'base_commit':'5d5edccc7d2167f5aacc08dd9c9f53b031a5f605','snapshot_code_sha256':hashlib.sha256((ROOT/'repo/src/cfact/execution/snapshot.py').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'seed':20261014,'parent_replay_steps':cursor,'branch_steps':len(a)*3,'actual_total_control_steps':ledger.control_steps,'roundtrip':roundtrip,'ABA_max_absolute_noise':noise,'state_alignment':'K_actions_Kplus1_states','render_integration_steps':0,'wall_seconds':time.monotonic()-started,'result_scope':'development_only'}
 (OUT/'SAME_PARENT_ABA.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
finally:
 if ledger:ledger.finish()
 if hasattr(env,'close'):env.close()
