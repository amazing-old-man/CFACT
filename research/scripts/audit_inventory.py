"""Read-only inventory of existing inputs; output confined to this research workspace."""
import ast, csv, hashlib, importlib.metadata as md, json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import h5py
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research').resolve()
AUDIT=ROOT/'00_audit'
OLD=Path('/mnt/ssd2/yanghc/CFACT')
LATEST=OLD/'output/rgb_capture_sync_fix_20261011T092328Z_5c14ccb3/dev'
def write(name,value):
 p=(AUDIT/name).resolve(); assert p.is_relative_to(ROOT)
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n' if not isinstance(value,str) else value)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def cmd(argv):
 r=subprocess.run(argv,capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'},timeout=60)
 return {'command':argv,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
started=time.monotonic(); now=datetime.now(timezone.utc).isoformat()
raw={}
for key,argv in {'df':['df','-B1','/mnt/ssd1/yanghc','/mnt/ssd2/yanghc','/mnt/hdd/yanghc'], 'inodes':['df','-i','/mnt/ssd1/yanghc','/mnt/ssd2/yanghc','/mnt/hdd/yanghc'], 'mounts':['findmnt','-J','-T','/mnt/ssd1/yanghc'], 'ram':['free','-b'], 'gpu':['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv'], 'gpu_processes':['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'], 'quota':['quota','-s']}.items():
 raw[key]=cmd(argv)
write('raw/host_inventory.json',raw)
identity={'checked_at_utc':now,'original_path':str(OLD),'original_head':cmd(['git','-C',str(OLD),'rev-parse','HEAD'])['stdout'].strip(),'original_status':cmd(['git','-C',str(OLD),'status','--porcelain'])['stdout'],'latest_actual_worktree':str(LATEST),'latest_head':cmd(['git','-C',str(LATEST),'rev-parse','HEAD'])['stdout'].strip(),'latest_status':cmd(['git','-C',str(LATEST),'status','--porcelain'])['stdout'],'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']},'runtime_repositories':{}}
for repo in ['robocasa','robosuite','robomimic','mimicgen']:
 p=OLD/'.runtime/sources'/repo
 identity['runtime_repositories'][repo]={'path':str(p),'HEAD':cmd(['git','-C',str(p),'rev-parse','HEAD'])['stdout'].strip(),'status':cmd(['git','-C',str(p),'status','--porcelain'])['stdout']}
identity['latest_pyproject']=(LATEST/'pyproject.toml').read_text(); identity['latest_collection']=json.loads((LATEST/'src/cfact/resources/collection.json').read_text())
write('REPOSITORY_IDENTITY.json',identity)
checks={}
for base in [OLD,LATEST]:
 paths=cmd(['git','-C',str(base),'ls-files','-z'])['stdout'].split('\0')
 checks[str(base)]={p:sha(base/p) for p in paths if p and (base/p).is_file()}
write('raw/ORIGINAL_TRACKED_SHA_BEFORE.json',checks)
core=['tasks.py','generation/options.py','generation/segments.py','generation/context.py','execution/options.py','execution/snapshot.py','execution/observation.py','collection/worker.py','collection/coordinator.py','run.py','export/lerobot.py','data/reader.py']
index=[]
for rel in core:
 p=LATEST/'src/cfact'/rel
 if not p.exists():index.append({'file':rel,'status':'MISSING'});continue
 source=p.read_text(); tree=ast.parse(source)
 index.append({'file':rel,'lines':len(source.splitlines()),'sha256':sha(p),'symbols':[{'name':x.name,'line':x.lineno,'end_line':x.end_lineno} for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))]})
 write('raw/code/'+rel+'.txt',source)
write('raw/CORE_CODE_INDEX.json',index)
manifest=Path('/mnt/hdd/yanghc/MimicGenv1/cf20/source_runtime/derivation/prepared_source_manifest.json')
files=json.loads(manifest.read_text())['files']; records=[]
for x in files:
 if x['kind']!='official':continue
 p=Path(x['prepared_hdf5']).resolve(); entry={'task':x['task'],'prepared_path':str(p),'source_present':p.is_file(),'historical_claim':{k:v for k,v in x.items() if k not in ['groups','preparation_job_evidence']}}
 if p.is_file():
  entry.update(bytes=p.stat().st_size,sha256=sha(p),historical_sha_matches=None)
  entry['historical_sha_matches']=entry['sha256']==x['prepared_hdf5_sha256']
  with h5py.File(p,'r') as f:
   data=f['data']; groups=sorted(data); entry['episode_count']=len(groups)
   entry['data_attrs']={k:str(v)[:10000] for k,v in data.attrs.items()}
   episodes=[]
   for key in groups:
    g=data[key]; info={'demo':key,'keys':list(g),'attrs':{k:str(v)[:1200] for k,v in g.attrs.items()},'actions_shape':list(g['actions'].shape) if 'actions' in g else None,'states_shape':list(g['states'].shape) if 'states' in g else None,'model_file_present':bool(g.attrs.get('model_file')),'ep_meta_present':bool(g.attrs.get('ep_meta')),'datagen_info_keys':list(g['datagen_info']) if 'datagen_info' in g else []}
    if key=='demo_0':info['full_ep_meta']=str(g.attrs.get('ep_meta',''));info['model_file_prefix']=str(g.attrs.get('model_file',''))[:2000]
    episodes.append(info)
   entry['episodes']=episodes
 records.append(entry);print('SOURCE',entry['task'],entry.get('episode_count'),entry.get('historical_sha_matches'),flush=True)
write('raw/SOURCE_HDF5_INSPECTION.json',{'time_utc':now,'manifest_path':str(manifest),'manifest_sha256':sha(manifest),'sources':records})
with (AUDIT/'SOURCE20_MATRIX.csv').open('w',newline='') as f:
 names=['task','SOURCE_PRESENT','IMPORTABLE','REPLAYABLE','CANDIDATE_SUPPORTED','RECURSION_SUPPORTED','PHYSICS_TESTED','episodes','prepared_path','sha256','notes'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
 for x in records:w.writerow({'task':x['task'],'SOURCE_PRESENT':'PASS' if x['source_present'] else 'FAIL','IMPORTABLE':'PREPARED_HDF5_PRESENT_NOT_REVALIDATED','REPLAYABLE':'UNKNOWN','CANDIDATE_SUPPORTED':'UNKNOWN','RECURSION_SUPPORTED':'UNKNOWN','PHYSICS_TESTED':'NOT_TESTED','episodes':x.get('episode_count'),'prepared_path':x['prepared_path'],'sha256':x.get('sha256'),'notes':'Source existence and bytes verified; historical PASS is not current physics evidence'})
write('raw/INVENTORY_RECEIPT.json',{'started_utc':now,'wall_seconds':time.monotonic()-started,'source_tasks_present':sum(x['source_present'] for x in records),'actual_new_physics_steps':0,'actual_new_executed_edges':0})
print('AUDIT_INVENTORY_DONE',str(AUDIT),flush=True)
