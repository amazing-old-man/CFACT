import json,os,signal,subprocess,sys,time
from pathlib import Path
from cfact.sources.manifest import build_plan
from cfact.collection.snapshot import implementation_digest
R=Path('/mnt/ssd1/yanghc/CFACT_Research');O=R/'02_baselines/MATCHED_PARENT_FIXED';O.mkdir(parents=True,exist_ok=True);pid=json.loads((R/'01_development/SCAN_DISPATCH_HOLD.json').read_text())['pid'];code=R/'runtime/baseline_parent_fixed'
def own_workers():
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\x00')
  except (OSError,PermissionError):continue
  if b'cfact.collection.worker' in args and b'--run' in args and args[args.index(b'--run')+1].startswith(str(R).encode()):result.append(int(p.name))
 return result
try:
 while own_workers():time.sleep(3)
 B=R/'01_development/SMOKE1_BASE_CURRENT_GPU';master_root=next((B/'runs/OpenDrawer/data').glob('*/*/*/root_result.json')).parent;master_meta=json.loads((master_root/'root_manifest.json').read_text());master_ctx=json.loads((master_root/'nodes/n_root/context.json').read_text());original=json.loads((R/'01_development/DEV_ITERATION_02/physics/plan/config.json').read_text());frozen=dict(path=str(master_root/'nodes/n_root/parent_snapshot.json'),source_id=master_meta['source_id'],source_action_prefix=master_ctx['source_action_prefix']);results=[]
 for scheme in ['CFACT_EFFECT_SINGLE','CFACT_EFFECT_RECURSIVE']:
  D=O/scheme;D.mkdir(exist_ok=True);c=json.loads(json.dumps(original));c['source_parent_snapshot']=frozen;c['method']={'task_effect_v2':False,'g0_scale_policy':'source_geometry_v1'}
  if scheme.endswith('SINGLE'):
   c['tree_limits'].update(maximum_decision_depth=1,second_depth_pools=0,third_depth_pools=0,candidate_pools_per_root=1);c['candidate_pool']['max_executed_options_per_node']=8;c['engineering_profile'].update(max_candidates_per_family=2,max_children_per_parent=1,max_decision_parents_per_root=1)
  spec=json.loads((B/'spec.json').read_text());spec['tasks']=[x for x in spec['tasks'] if x['task']=='OpenDrawer'];spec['output_root']=str(D/'run');(D/'spec.json').write_text(json.dumps(spec,indent=2));(D/'parameters.json').write_text(json.dumps(c,indent=2));(D/'CODE_FREEZE.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),implementation_sha256=implementation_digest(),scope='DEVELOPMENT_MATCHED_COMPLETE_PARENT',scheme=scheme,source_parent=frozen,method_hypothesis_unchanged=True),indent=2));build_plan(D/'spec.json',D/'plan',D/'parameters.json');m=json.loads((D/'plan/source_manifest.json').read_text());req=next(iter(m['requests_by_id']));cmd=[sys.executable,'-B','-m','cfact.collection.worker','--config',str(D/'plan/config.json'),'--source-manifest',str(D/'plan/source_manifest.json'),'--task','OpenDrawer','--root-id',req,'--split','development','--run',str(D/'run'),'--gpu','0'];s=time.monotonic()
  while time.monotonic()-s<60:
   probe=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);g=next(list(map(int,l.split(','))) for l in probe.splitlines() if int(l.split(',')[0])==0)
   if g[1]<80 and g[2]>=8192 and g[3]<=80:break
   time.sleep(3)
  print('MATCHED_PARENT_PHYSICS',scheme,flush=True);ret=subprocess.run([sys.executable,'-B',str(R/'scripts/research_run.py'),'--receipt-dir',str(D/'runner'),'--wall-seconds','300','--',*cmd]);results.append(dict(scheme=scheme,returncode=ret.returncode));(O/'RESULTS.json').write_text(json.dumps(results,indent=2))
finally:
 os.kill(pid,signal.SIGCONT);(R/'01_development/SCAN_DISPATCH_HOLD.json').write_text(json.dumps(dict(pid=pid,status='RESUMED_AFTER_BASELINE_REPAIR',active_goal_never_paused=True),indent=2));print('SOURCE20_DISPATCH_RESUMED',flush=True)
