import json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research');sources=json.loads((ROOT/'00_audit/raw/SOURCE_HDF5_INSPECTION.json').read_text())['sources'];selected=['OpenOven','TurnOnMicrowave','OpenBlenderLid'];records=[]
lock={'selection_rule':'Fixed demo_0/demo_1 development roots, demo_3/demo_4 donors before any new outcomes; no reselection by success','tasks':selected,'seed':20261014,'scope':'DEVELOPMENT_ONLY','base_commit':'53fd8e3','selected_groups':['demo_0','demo_1','demo_3','demo_4']};(ROOT/'manifests/DEV_FAMILY_SOURCE_LOCK.json').write_text(json.dumps(lock,indent=2))
for task in selected:
 entry=next(s for s in sources if s['task']==task);raw=Path(entry['historical_claim']['raw_hdf5']).resolve();out=ROOT/'prepared/cpu-dev-v1'/f'{task}.hdf5';receipt=ROOT/'01_development/PREPARE_CPU_FAMILY_SLICES'/task
 command=[sys.executable,'-m','cfact.sources.prepare','--raw',str(raw),'--output',str(out),'--task',task,'--gpu','-1','--worker']
 for demo in lock['selected_groups']:command+=['--demo',demo]
 started=time.monotonic();print('PREPARE',task,flush=True)
 r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/research_run.py'),'--cpu-only','--receipt-dir',str(receipt),'--wall-seconds','900','--',*command]);records.append({'task':task,'returncode':r.returncode,'wall_seconds':time.monotonic()-started,'raw_source':str(raw),'raw_expected_sha256':entry['historical_claim']['raw_hdf5_sha256'],'output':str(out),'output_present':out.is_file(),'receipt':str(receipt)})
 (ROOT/'01_development/PREPARE_CPU_FAMILY_SLICES/RESULTS.json').write_text(json.dumps(records,indent=2))
 print('PREPARE_RESULT',task,r.returncode,flush=True)
 if r.returncode:break
