"""Independent readback of all development physical siblings and ancestry."""
import argparse,csv,hashlib,itertools,json,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd
from cfact.execution.snapshot import load_snapshot
from cfact.tasks import target_task
from cfact.generation.task_adapters import TaskEffectAdapter,FLOORS,family_for_task
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.input=args.input.resolve();ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research');OUT=args.output.resolve();assert OUT.is_relative_to(ROOT);OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();H=[.25,.5,1.,2.,4.]
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def rms(a):return float(np.sqrt(np.mean(a*a))) if a.size else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def join(a,b):
 result={}
 for k in a:
  if k in b and len(a[k].shape) and len(a[k])==len(a['sim_time']) and len(b[k])==len(b['sim_time']):result[k]=np.concatenate([a[k],b[k][1:]])
 result['commanded_actions']=np.concatenate([a['commanded_actions'].reshape(-1,12),b['commanded_actions'].reshape(-1,12)])
 for k in ['fixture_joint_types','fixture_joint_units']:result[k]=a[k]
 return result
def contact_onset(a):
 for i,row in enumerate(a['subtask_signals_json']):
  if any(v for k,v in json.loads(str(row)).items() if k.startswith('stage_contact')):return i
 return None
def distance(a,b,i,j,task):
 family=family_for_task(task)
 if family=='articulated_contact':
  types=a['fixture_joint_types'].astype(str)
  if len(types)!=1 or types[0] not in ['slide','hinge']:return None,None,None
  value=float(np.linalg.norm(a['relevant_fixture_qpos'][i]-b['relevant_fixture_qpos'][j]));return value,('m' if types[0]=='slide' else 'rad'),FLOORS[types[0]]
 if family=='button_activation':
  if 'fixture_activation_json' not in a or 'fixture_activation_json' not in b:return None,'ACTIVATION_UNMEASURED',None
  x=json.loads(str(a['fixture_activation_json'][i]));y=json.loads(str(b['fixture_activation_json'][j]))
  if x.get('status')!='MEASURED' or y.get('status')!='MEASURED':return None,'ACTIVATION_UNMEASURED',None
  return int(x['turned_on']!=y['turned_on']),'boolean',1
 if family=='free_body_lid':return None,'UNSUPPORTED_FREE_BODY_LID',None
 return float(np.linalg.norm(a['object_poses'][i,0,:3,3]-b['object_poses'][j,0,:3,3])),'m_object_translation',FLOORS['object']
calibrations={}
for pth in (ROOT/'01_development').glob('FAMILY_ABA/*/SAME_PARENT_ABA.json'):
 c=json.loads(pth.read_text());calibrations[c['family']+(':'+c['threshold_unit'] if c['family']=='articulated_contact' else '')]=c
old=ROOT/'01_development/SNAPSHOT_FIX_01/SAME_PARENT_ABA.json'
if old.exists():calibrations['slide_OpenDrawer']=json.loads(old.read_text())
rows=[];actionrows=[];lineages=[];failures=[];rootmetrics=[];total_steps=0;total_edges=0;physicaledges=0;invariant_failures=[];rgbrows=[]
for rp in sorted(args.input.rglob('root_result.json')):
 if 'data' not in rp.parts:continue
 result=json.loads(rp.read_text());task=target_task(result['task']);taskname=task.name;run=rp.parents[4];root=rp.parent;tree=json.loads((root/'tree.json').read_text());parentmap={}
 for pool in tree.get('child_pools',[])+tree.get('third_pools',[]):parentmap[pool['node_id']]=pool.get('parent_edge')
 histories=list((run/'history').glob('*/*/*/frames.jsonl'));hist=[];hdir=None
 if histories:
  # Direct task runs have one root; select by the root manifest request ID.
  rootmanifest=json.loads((root/'root_manifest.json').read_text());history=next((q for q in histories if q.parent.name==str(rootmanifest.get('source_episode'))),histories[0]);hdir=history.parent;hist=[json.loads(x) for x in history.read_text().splitlines()]
  groups=defaultdict(list)
  for row in hist:groups[(row['segment'],row['phase'])].append(row)
 else:groups={}
 byparent=defaultdict(list);loaded={};bins=set();nonrootbins=set();root_pairs=0;child_pairs=0;calibrated_pairs=0;lineage_pass=True
 total_steps+=result['charged_control_steps'];total_edges+=len(result['edges'])
 for edge in result['edges']:
  ep=run/edge['edge_path'] if edge.get('edge_path') else None
  if edge['status']!='PHYSICS_COMPLETE':failures.append(dict(task=taskname,root=root.name,edge=edge['candidate_id'],status=edge['status'],reason=edge.get('reason',edge.get('error'))));continue
  physicaledges+=1
  try:
   a=load(ep/'execution.npz');b=load(ep/'continuation/execution.npz');assert len(a['sim_time'])==len(a['commanded_actions'])+1;assert len(b['sim_time'])==len(b['commanded_actions'])+1
   for z in [a,b]:assert np.allclose(np.diff(z['sim_time']),.05,atol=1e-9,rtol=0)
   for k in ['sim_time','object_poses','actual_eef_pose','robot_qpos','relevant_fixture_qpos']:assert np.array_equal(a[k][-1],b[k][0]),'LOCAL_FIXED_'+k
   snapshot=load_snapshot(ep/'child_snapshot.json');child=load(ep/'child_context.npz')['sim_state'];ctx=json.loads((ep/'child_context.json').read_text());assert np.allclose(np.asarray(ctx['operation_context']['current_eef_pose']),a['actual_eef_pose'][-1],atol=1e-10,rtol=0),'SAVED_CHILD_EEF_GEOMETRY_MISMATCH';assert np.array_equal(snapshot['states'],child);assert float(snapshot['sim_time_s'])==a['sim_time'][-1]
   local_frames=groups.get((str(ep),'local_operation'),[])
   if local_frames:
    assert len(local_frames)==len(a['sim_time']);last=load(hdir/local_frames[-1]['frame']);assert np.array_equal(last['sim_state'],child)
   parent=ep.parent.parent.name
   if parent!='n_root':
    ancestor=run/parentmap[parent];expected=load_snapshot(ancestor/'child_snapshot.json');actual=load(ep.parent.parent/'context.npz')['sim_state'];assert np.array_equal(actual,expected['states']);assert local_frames and np.array_equal(load(hdir/local_frames[0]['frame'])['sim_state'],expected['states'])
   trace=join(a,b);loaded[edge['candidate_id']]=(edge,a,trace,local_frames)
   if edge.get('role')!='MANDATORY_START_PREFIX' and len(a['commanded_actions'])>0:byparent[parent].append(edge['candidate_id'])
   effectpath=ep/'observed_effect.json'
   if effectpath.exists():effect=json.loads(effectpath.read_text())
   else:effect=TaskEffectAdapter(task).observe_effect(a)
   if effect['supported']:
    scale=FLOORS['activation'] if effect['units']=='boolean' else FLOORS['slide'] if effect['units']=='m' else FLOORS['hinge'] if effect['units']=='rad' else FLOORS['object'];key=(effect['units'],*np.rint(np.asarray(effect['value'])/scale).astype(int).tolist());bins.add(key)
    if parent!='n_root':nonrootbins.add(key)
   lineages.append(dict(task=taskname,root=root.name,edge=edge['candidate_id'],parent=parent,ancestry_parent_edge=parentmap.get(parent),local_steps=len(a['commanded_actions']),fixed_steps=len(b['commanded_actions']),status='PASS',trace_sha256=sha(ep/'execution.npz')))
  except Exception as exc:
   lineage_pass=False;invariant_failures.append(dict(task=taskname,root=root.name,edge=edge['candidate_id'],status='FAIL',reason=repr(exc)+' '+traceback.format_exc().splitlines()[-3]))
 for parent,ids in byparent.items():
  for ei,ej in itertools.combinations(ids,2):
   e,a,ta,fa=loaded[ei];f,b,tb,fb=loaded[ej];common=min(len(a['commanded_actions']),len(b['commanded_actions']));initial_equal=all(np.array_equal(a[k][0],b[k][0]) for k in ['sim_time','object_poses','actual_eef_pose','robot_qpos','relevant_fixture_qpos']);base=dict(task=taskname,root=root.name,parent=parent,edge_a=ei,edge_b=ej,initial_measured_equal=initial_equal)
   actionrows.append({**base,'common_local_steps':common,'issued_action_rms':rms(a['commanded_actions'][:common]-b['commanded_actions'][:common]),'different_command_lengths':len(a['commanded_actions'])!=len(b['commanded_actions'])})
   calkey=family_for_task(task)+(':'+str(a['fixture_joint_types'][0]) if family_for_task(task)=='articulated_contact' and len(a['fixture_joint_types'])==1 else '');cal=calibrations.get(calkey,calibrations.get('slide_OpenDrawer') if taskname=='OpenDrawer' else None);cal_ok=bool(cal and cal['status']=='PASS');effective=False
   for scope,pa,pb in [('local',a,b),('local_plus_fixed',ta,tb)]:
    for alignment in ['parent_time','first_contact']:
     oa=0 if alignment=='parent_time' else contact_onset(pa);ob=0 if alignment=='parent_time' else contact_onset(pb)
     for h in H:
      i=int(round(h*20))+oa if oa is not None else None;j=int(round(h*20))+ob if ob is not None else None
      row={**base,'scope':scope,'alignment':alignment,'horizon_s':h,'status':'MEASURED','calibrated_family':cal_ok}
      if i is None or j is None:row['status']='EVENT_UNOBSERVED'
      elif i>=len(pa['sim_time']) or j>=len(pb['sim_time']):row['status']='CENSORED_OR_STOPPED'
      else:
       d,unit,floor=distance(pa,pb,i,j,task);row.update(D_task_effect=d,effect_unit=unit,frozen_floor=floor,D_robot_realized_m=float(np.linalg.norm(pa['actual_eef_pose'][i,:3,3]-pb['actual_eef_pose'][j,:3,3])),official_success_a=bool(pa['task_success'][i]),official_success_b=bool(pb['task_success'][j]),D_action_issued_rms=rms(pa['commanded_actions'][oa:i]-pb['commanded_actions'][ob:j]))
       if d is not None:row['above_physical_floor']=d>=floor;effective=effective or (scope=='local' and d>=floor)
       else:row['status']='UNSUPPORTED_EFFECT_MEASUREMENT'
      rows.append(row)
   # Endpoints are separately labelled, never substituted for a fixed horizon.
   d,unit,floor=distance(a,b,-1,-1,task)
   if d is not None:effective=effective or d>=floor
   rows.append({**base,'scope':'local','alignment':'local_endpoint_variable_duration','horizon_s':None,'status':'MEASURED' if d is not None else 'UNSUPPORTED_EFFECT_MEASUREMENT','D_task_effect':d,'effect_unit':unit,'frozen_floor':floor,'above_physical_floor':d>=floor if d is not None else False,'calibrated_family':cal_ok})
   if effective and len(a['commanded_actions'])>0 and len(b['commanded_actions'])>0:
    if parent=='n_root':root_pairs+=1
    else:child_pairs+=1
    if cal_ok:calibrated_pairs+=1
   # Full RGB is available from genuine ledger frames; ROI is explicitly unknown.
   for h in H:
    index=int(round(h*20))
    if index<len(fa) and index<len(fb):
     ia=load(hdir/fa[index]['frame'])['rgb'].astype(float)/255;ib=load(hdir/fb[index]['frame'])['rgb'].astype(float)/255
     rgbrows.append({**base,'horizon_s':h,'full_image_mse_by_camera':np.mean((ia-ib)**2,axis=(1,2,3)).tolist(),'ROI':'ROI_UNAVAILABLE'})
 rootmetrics.append(dict(task=taskname,root=root.name,source_id=json.loads((root/'root_manifest.json').read_text()).get('source_id'),status=result['status'],charged_control_steps=result['charged_control_steps'],physical_edges=sum(e['status']=='PHYSICS_COMPLETE' for e in result['edges']),depth=result.get('observed_max_decision_depth',0),root_effective_pairs=root_pairs,nonroot_effective_pairs=child_pairs,calibrated_effective_pairs=calibrated_pairs,effect_bins=len(bins),nonroot_effect_bins=len(nonrootbins),nonroot_bins_per_charged_step=len(nonrootbins)/max(1,result['charged_control_steps']),lineage_pass=lineage_pass,has_two_root_edges=len(byparent.get('n_root',[]))>=2,has_recursive_parent_with_two_edges=any(k!='n_root' and len(v)>=2 for k,v in byparent.items())))
for name,data in [('AUDIT_SIBLING_EFFECTS',rows),('AUDIT_ACTION_DIVERSITY',actionrows)]:pd.DataFrame(data).to_parquet(OUT/(name+'.parquet'),index=False)
pd.DataFrame(failures+invariant_failures,columns=['task','root','edge','status','reason']).to_csv(OUT/'AUDIT_FAILURE_REASONS.csv',index=False);pd.DataFrame(rootmetrics).to_csv(OUT/'ROOT_METRICS.csv',index=False)
coverage=[]
for task in __import__('cfact.tasks',fromlist=['TARGET_TASKS']).TARGET_TASKS:
 group=[r for r in rootmetrics if r['task']==task.name];eligible=[r for r in group if r['has_two_root_edges'] and r['has_recursive_parent_with_two_edges'] and r['calibrated_effective_pairs']>0 and r['lineage_pass']];coverage.append(dict(task=task.name,roots=len(group),independent_source_ids=len({r['source_id'] for r in group if r['source_id']}),eligible_roots=len(eligible),NONTRIVIAL_RECURSIVE_TREE_SUPPORTED=len({r['source_id'] for r in eligible if r['source_id']})>=2,reason='PASS' if len(eligible)>=2 else 'INSUFFICIENT_TWO_INDEPENDENT_CALIBRATED_ROOTS'))
pd.DataFrame(coverage).to_csv(OUT/'AUDIT_TASK_COVERAGE.csv',index=False)
reports={'AUDIT_TREE_LINEAGE.json':{'edges':lineages,'invariant_failures':invariant_failures},'AUDIT_PARENT_ABA.json':{'calibrations':calibrations},'AUDIT_EFFECT_THRESHOLDS.json':{'floors_registered_before_method_physics':FLOORS,'family_calibrations':calibrations,'uncalibrated_tasks_cannot_count':True},'AUDIT_VISUAL_OBSERVABILITY.json':{'full_image_measurements':rgbrows,'ROI':'ROI_UNAVAILABLE','SSIM':'NOT_TESTED','interpretation':'Full-image difference is not proof of object observability'},'AUDIT_EVENT_ALIGNMENT.json':{'event':'first real stage_contact signal','windows':'all pairs at 0.25/0.5/1/2/4s; no padding; absent events censored separately'},'RECEIPT.json':{'input':str(args.input),'wall_seconds':time.monotonic()-started,'root_count':len(rootmetrics),'requested_edges':total_edges,'physical_edges':physicaledges,'charged_control_steps':total_steps,'calibrated_supported_tasks':sum(r['NONTRIVIAL_RECURSIVE_TREE_SUPPORTED'] for r in coverage),'statistical_unit':'independent_source_root','invariant_failures':invariant_failures,'all_pairs_measured':True,'root_metrics':rootmetrics}}
for name,obj in reports.items():(OUT/name).write_text(json.dumps(obj,indent=2))
(OUT/'SUMMARY_ZH.md').write_text(f"# 真实开发 sibling 诊断\n\n审计 {len(rootmetrics)} 个 root、{physicaledges} 条实际边、{total_steps} 个计费控制步。完整性失败 {len(invariant_failures)}。满足两个独立、噪声校准 root 的任务数 {sum(r['NONTRIVIAL_RECURSIVE_TREE_SUPPORTED'] for r in coverage)}/20。\n\n动作、机器人运动、任务效果分别计量，所有 pair 与不足时距均保留；首次真实接触窗口单列。没有可靠 ROI，不能由全图差异判断目标视觉可观测性。没有训练新模型，无法判定模型是否学会。端点的变时长差异不替代定时 horizon。详见逐对 Parquet、ROOT_METRICS.csv 与 invariant receipts。\n")
print(json.dumps(reports['RECEIPT.json'],indent=2))
