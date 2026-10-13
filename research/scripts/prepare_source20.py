"""Fixed, outcome-blind development source preparation; no source replacement."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research');OUT=ROOT/'01_development/PREPARE_SOURCE20';OUT.mkdir(parents=True,exist_ok=True)
from cfact.tasks import TARGET_TASKS
from cfact.sources.prepare import validate_prepared_identity
BASE=Path('/mnt/ssd2/yanghc/CFACT/source/PhysicalAI-Robotics-Manipulation-Kitchen-Demos__522e4ffa__train40_v1/prepared');RAW=Path('/mnt/hdd/yanghc/CFACT/Pointworld/CREC/MimicGenv1/outputs/cf20_tree_lerobot/20260924_full_source_v1_1/source_stage/raw_official');CPU=ROOT/'prepared/cpu-dev-v1';NEW=ROOT/'prepared/source20-dev-v1';NEW.mkdir(parents=True,exist_ok=True)
old=json.loads((ROOT/'00_audit/raw/SOURCE_HDF5_INSPECTION.json').read_text());expected={r['task']:r['historical_claim']['raw_hdf5_sha256'] for r in old['sources']};rows=[]
for task in TARGET_TASKS:
 name=task.name;prepared=(BASE if name in ['OpenDrawer','CoffeeServeMug'] else CPU if name in ['OpenOven','TurnOnMicrowave','OpenBlenderLid'] else NEW)/(name+'.hdf5')
 raw=RAW/(name+'.hdf5');row=dict(task=name,prepared_hdf5=str(prepared),roots=['demo_0','demo_1'],donors=['demo_3','demo_4'],scope='DEVELOPMENT_ONLY',chosen_before_outcomes=True)
 if prepared.exists():
  validate_prepared_identity(prepared,required_episodes=['demo_0','demo_1','demo_3','demo_4']);row.update(status='CURRENT_PREPARED_INPUT_VERIFIED',prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest())
 else:
  sha=hashlib.sha256(raw.read_bytes()).hexdigest();row.update(raw_hdf5=str(raw),raw_sha256=sha,expected_raw_sha256=expected[name]);assert sha==expected[name],'Input SHA changed before prepare'
  row['status']='PENDING_PREPARATION'
 rows.append(row)
(ROOT/'manifests/SOURCE20_DEV_SOURCE_LOCK.json').write_text(json.dumps({'seed':20261014,'tasks':rows,'no_formal_or_test_outcomes':True},indent=2))
for row in rows:
 if row['status']!='PENDING_PREPARATION':continue
 name=row['task'];command=[sys.executable,'-B','-m','cfact.sources.prepare','--raw',row['raw_hdf5'],'--output',row['prepared_hdf5'],'--task',name,'--gpu','-1' if os.environ.get('MUJOCO_GL')=='osmesa' else '0','--worker']
 for demo in ['demo_0','demo_1','demo_3','demo_4']:command+=['--demo',demo]
 print('PREPARE',name,flush=True);t=time.monotonic();r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/research_run.py'),*(['--cpu-only'] if os.environ.get('MUJOCO_GL')=='osmesa' else []),'--receipt-dir',str(OUT/name),'--wall-seconds','300','--',*command]);row.update(returncode=r.returncode,wall_seconds=time.monotonic()-t,status='FAILED_OR_BLOCKED' if r.returncode else 'PREPARED_CURRENT')
 if r.returncode==0:
  validate_prepared_identity(Path(row['prepared_hdf5']),required_episodes=['demo_0','demo_1','demo_3','demo_4']);row['prepared_sha256']=hashlib.sha256(Path(row['prepared_hdf5']).read_bytes()).hexdigest()
 (OUT/'RESULTS.json').write_text(json.dumps(rows,indent=2))
 if r.returncode:
  receipt=json.loads((OUT/name/'runner_receipt.json').read_text())
  if receipt['status'].startswith('BLOCKED_RESOURCE'):print('STOP_STAGE',receipt['status'],flush=True);break
(ROOT/'manifests/SOURCE20_FROZEN_IDENTITY.json').write_text(json.dumps({'status':'DEVELOPMENT_SOURCES_ONLY_FORMAL_NOT_ADMITTED','tasks':rows,'source_revisions':json.loads((ROOT/'00_audit/REPOSITORY_IDENTITY.json').read_text())['runtime_repositories'],'fixed_task_denominator':20},indent=2))
