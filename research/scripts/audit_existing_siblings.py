"""Measure every existing training sibling pair; no fitting and no holdout outcomes."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research'); OUT=ROOT/'00_audit/existing_siblings'; OUT.mkdir(parents=True,exist_ok=True)
BASE=Path('/mnt/ssd2/yanghc/CFACT/output/cf_v073_train__20261011T125605Z__0199eac8/collection')
HORIZONS=[.25,.5,1.,2.,4.]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def rms(a):return float(np.sqrt(np.mean(np.square(a)))) if a.size else None
def join(a,b):
 result={}
 for k in ['sim_time','actual_eef_pose','object_poses','relevant_fixture_qpos','task_success','robot_qpos','robot_contacts_json','observation_capture_json']:
  result[k]=np.concatenate([a[k],b[k][1:]])
 result['commanded_actions']=np.concatenate([a['commanded_actions'].reshape(-1,12),b['commanded_actions'].reshape(-1,12)])
 return result
rows=[];actions=[];failures=[];lineage=[];identity=[];started=time.monotonic(); total_edges=0
for task in ['OpenDrawer','CoffeeServeMug']:
 for rp in sorted((BASE/'trees'/task).iterdir()):
  tree=json.loads((rp/'tree.json').read_text());identity.append({'path':str(rp/'tree.json'),'sha256':digest(rp/'tree.json'),'task':task,'source_id':tree['source_id']})
  parents={}; loaded={}
  for e in tree['edges']:
   ep=BASE/e['path'];total_edges+=1
   if e['status']!='PHYSICS_COMPLETE':failures.append({'task':task,'root':rp.name,'edge':e['edge_id'],'status':e['status']});continue
   try:
    a=load(ep/'local/execution.npz');b=load(ep/'fixed_continuation/execution.npz');local_K=len(a['commanded_actions']);fixed_K=len(b['commanded_actions'])
    assert len(a['sim_time'])==local_K+1 and len(b['sim_time'])==fixed_K+1
    assert np.array_equal(a['sim_time'][-1:],b['sim_time'][:1])
    for k in ['object_poses','actual_eef_pose','relevant_fixture_qpos']:assert np.array_equal(a[k][-1],b[k][0])
    assert np.allclose(np.diff(a['sim_time']),.05,atol=1e-9)
    trace=join(a,b);loaded[e['edge_id']]=(e,a,trace)
    parents.setdefault(e['parent_node'],[]).append(e['edge_id'])
    lineage.append({'task':task,'root':rp.name,'edge':e['edge_id'],'parent':e['parent_node'],'child':e['actual_child'],'local_steps':local_K,'fixed_steps':fixed_K,'K_actions_Kplus1_states':'PASS','local_fixed_boundary':'PASS','trace_sha256':digest(ep/'local/execution.npz')})
   except Exception as exc:failures.append({'task':task,'root':rp.name,'edge':e['edge_id'],'status':'READBACK_FAIL','reason':str(exc)})
  for parent,edges in parents.items():
   for ei,ej in itertools.combinations(edges,2):
    e,a,ta=loaded[ei];f,b,tb=loaded[ej]; common=min(len(a['commanded_actions']),len(b['commanded_actions']))
    initial_equal=all(np.array_equal(a[k][0],b[k][0]) for k in ['sim_time','actual_eef_pose','object_poses','relevant_fixture_qpos','robot_qpos'])
    initial_a=json.loads(str(a['observation_capture_json'][0]));initial_b=json.loads(str(b['observation_capture_json'][0])); integration_equal=initial_a.get('integration_state_sha256')==initial_b.get('integration_state_sha256') and initial_a.get('integration_state_sha256') is not None
    base={'task':task,'root':rp.name,'source_id':tree['source_id'],'parent':parent,'edge_a':ei,'edge_b':ej,'generator_a':e['generator'],'generator_b':f['generator'],'initial_measured_fields_equal':initial_equal,'initial_integration_hash_equal':integration_equal,'complete_controller_rng_verified':False}
    actions.append({**base,'common_local_steps':common,'D_action_issued_rms':rms(a['commanded_actions'][:common]-b['commanded_actions'][:common]),'different_command_lengths':len(a['commanded_actions'])!=len(b['commanded_actions'])})
    for scope,pa,pb in [('local',a,b),('local_plus_fixed',ta,tb)]:
     for hz in HORIZONS:
      i=int(round(hz*20));row={**base,'scope':scope,'horizon_s':hz,'record_status':'CENSORED_OR_STOPPED' if min(len(pa['sim_time']),len(pb['sim_time']))<=i else 'MEASURED','full_RGB_ROI_status':'NOT_MEASURED'}
      if row['record_status']=='MEASURED':
       row['D_action_issued_rms']=rms(pa['commanded_actions'][:i]-pb['commanded_actions'][:i]);row['D_robot_realized_m']=float(np.linalg.norm(pa['actual_eef_pose'][i,:3,3]-pb['actual_eef_pose'][i,:3,3]));row['official_success_a']=bool(pa['task_success'][i]);row['official_success_b']=bool(pb['task_success'][i])
       if task=='OpenDrawer':row['D_task_effect']=float(np.linalg.norm(pa['relevant_fixture_qpos'][i]-pb['relevant_fixture_qpos'][i]));row['effect_unit']='m_joint'
       else:row['D_task_effect']=float(np.linalg.norm(pa['object_poses'][i,0,:3,3]-pb['object_poses'][i,0,:3,3]));row['effect_unit']='m_object_translation'
      rows.append(row)
  print('AUDITED',task,rp.name,'parents',len(parents),'edges',len(loaded),flush=True)
df=pd.DataFrame(rows);df.to_parquet(OUT/'AUDIT_SIBLING_EFFECTS.parquet',index=False);pd.DataFrame(actions).to_parquet(OUT/'AUDIT_ACTION_DIVERSITY.parquet',index=False);pd.DataFrame(failures,columns=['task','root','edge','status','reason']).to_csv(OUT/'AUDIT_FAILURE_REASONS.csv',index=False)
summary={}
for task in ['OpenDrawer','CoffeeServeMug']:
 x=df[(df.task==task)&(df.record_status=='MEASURED')];aa=pd.DataFrame(actions);aa=aa[aa.task==task]
 summary[task]={'independent_source_ids':len({r['source_id'] for r in identity if r['task']==task}),'all_sibling_pairs':len(aa),'initial_fields_equal_pairs':int(aa.initial_measured_fields_equal.sum()),'initial_integration_equal_pairs':int(aa.initial_integration_hash_equal.sum()),'issued_action_rms_quantiles':aa.D_action_issued_rms.quantile([0,.25,.5,.75,1]).to_dict(),'effects_by_scope_horizon':[]}
 for (scope,h),g in x.groupby(['scope','horizon_s']):summary[task]['effects_by_scope_horizon'].append({'scope':scope,'horizon_s':h,'measured_pairs':len(g),'physical_effect_quantiles_m':g.D_task_effect.quantile([0,.25,.5,.75,1]).to_dict(),'numerically_nonzero_gt_1e_minus_9':int((g.D_task_effect>1e-9).sum()),'physical_acceptance_threshold':'NOT_CALIBRATED_NO_VALIDITY_CLAIM'})
receipt={'old_dataset':str(BASE),'scope':'EXISTING_TRAIN_ONLY','new_test_outcomes_accessed':False,'all_tree_identities':identity,'results':summary,'total_edges_examined':total_edges,'failures':failures,'new_env_steps':0,'wall_seconds':time.monotonic()-started,'source_code_base_commit':'5d5edccc7d2167f5aacc08dd9c9f53b031a5f605','statistical_unit':'source_root_not_pair','no_static_padding':True}
(OUT/'AUDIT_TREE_LINEAGE.json').write_text(json.dumps({'edges':lineage},indent=2));(OUT/'AUDIT_PARENT_ABA.json').write_text(json.dumps({'status':'NOT_TESTED','reason':'Archived siblings verify measured/integration initial equality; full controller/gripper/RNG A/B/A needs new physical replay.'},indent=2));(OUT/'AUDIT_EVENT_ALIGNMENT.json').write_text(json.dumps({'status':'NOT_TESTED','reason':'Need causal first-contact detector calibration before event-aligned comparison.'},indent=2));(OUT/'AUDIT_EFFECT_THRESHOLDS.json').write_text(json.dumps({'status':'NOT_CALIBRATED','diagnostic_zero_tolerance':1e-9,'acceptance_thresholds':None},indent=2));(OUT/'AUDIT_VISUAL_OBSERVABILITY.json').write_text(json.dumps({'status':'NOT_TESTED','ROI':'ROI_UNAVAILABLE','reason':'No validated fixture/object segmentation. Physical effects do not prove visual readability.'},indent=2));(OUT/'RECEIPT.json').write_text(json.dumps(receipt,indent=2))
lines=['# 现有训练数据同父效果审计\n','本轮未训练模型、未执行新物理。40 个旧训练 Source root 的所有可读 sibling 均保留；不足时距单列截断，不静态填充。\n']
for task,s in summary.items():
 lines.append(f"{task}：{s['independent_source_ids']} 个 Source 身份，{s['all_sibling_pairs']} 个 sibling 对。命令 RMS 中位数 {s['issued_action_rms_quantiles'][0.5]:.6g}。\n")
 for v in s['effects_by_scope_horizon']:
  lines.append(f"- {v['scope']} / {v['horizon_s']} 秒：可测 {v['measured_pairs']} 对，任务效果距离中位数 {v['physical_effect_quantiles_m'][0.5]*1000:.3f} mm。\n")
lines.append('\n结论边界：实际命令与任务物理效果存在数值差异；需通过同动作 A/B/A 噪声校准才能判定有效效果。视觉、完整快照复现、跨场景独立性未验证，不能判定模型失败的唯一根因或 v2 覆盖验收。所有输入 SHA 和逐对测量见 RECEIPT.json/Parquet。\n')
(OUT/'SUMMARY_ZH.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2),flush=True)
