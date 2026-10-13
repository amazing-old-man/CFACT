import argparse,hashlib,json,os,time
from pathlib import Path
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research');OUT=ROOT/'00_audit/original_protection';OUT.mkdir(parents=True,exist_ok=True);p=argparse.ArgumentParser();p.add_argument('--phase',required=True);args=p.parse_args();before=json.loads((ROOT/'00_audit/raw/ORIGINAL_TRACKED_SHA_BEFORE.json').read_text());diffs=[];checked=0
for base,files in before.items():
 for name,expected in files.items():
  f=Path(base)/name;actual=hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None;checked+=1
  if actual!=expected:diffs.append({'path':str(f),'expected':expected,'actual':actual})
# Bounded metadata scan of exact read-only inputs and representative historical outputs.
old=Path('/mnt/ssd2/yanghc/CFACT');scopes={old/'src':10,old/'tests':10,old/'.runtime/sources':3,old/'source/PhysicalAI-Robotics-Manipulation-Kitchen-Demos__522e4ffa__train40_v1/prepared':2,old/'output/cf_v073_train__20261011T125605Z__0199eac8/collection/trees':8,Path('/mnt/hdd/yanghc/CFACT/Pointworld/CREC/MimicGenv1/outputs/cf20_tree_lerobot/20260924_full_source_v1_1/source_stage'):2};metadata={};started_at=(ROOT/'00_audit/raw/INVENTORY_RECEIPT.json').stat().st_mtime
for base,depth in scopes.items():
 for directory,dirs,names in os.walk(base):
  rel=Path(directory).relative_to(base);dirs[:]=[d for d in dirs if d not in ['.git','assets'] and len(rel.parts)<depth]
  for name in names:
   f=Path(directory)/name
   if f.is_symlink():continue
   st=f.stat();metadata[str(f)]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
prior=OUT/'during_METADATA.json';metadata_differences=[]
if args.phase=='after' and prior.exists():
 previous=json.loads(prior.read_text())
 for path in set(previous)|set(metadata):
  if previous.get(path)!=metadata.get(path):metadata_differences.append({'path':path,'before':previous.get(path),'after':metadata.get(path)})
(OUT/(args.phase+'_METADATA.json')).write_text(json.dumps(metadata,indent=2));record={'phase':args.phase,'tracked_files_checked':checked,'tracked_content_differences':diffs,'metadata_files_checked':len(metadata),'metadata_differences_since_during':metadata_differences,'files_modified_after_initial_audit':[k for k,v in metadata.items() if v['mtime_ns']/1e9>started_at],'status':'PASS' if not diffs and not metadata_differences else 'FAIL','limitation':'Initial audit hashed original tracked code and all twenty prepared source files. Historical-output complete metadata baseline begins during; no claim of a full-disk before/after scan.'};(OUT/(args.phase+'_RECEIPT.json')).write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['files_modified_after_initial_audit']},indent=2));print('NEW_MTIME_COUNT',len(record['files_modified_after_initial_audit']))
