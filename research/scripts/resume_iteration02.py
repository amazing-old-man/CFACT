import json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/ssd1/yanghc/CFACT_Research');O=R/'01_development/DEV_ITERATION_02/physics';m=json.loads((O/'plan/source_manifest.json').read_text());rows=[]
for t in ['CoffeeServeMug','TurnOnMicrowave','OpenBlenderLid']:
 req=next(k for k,v in m['requests_by_id'].items() if v['task']==t);run=O/'runs'/t
 cmd=[sys.executable,'-B','-m','cfact.collection.worker','--config',str(O/'plan/config.json'),'--source-manifest',str(O/'plan/source_manifest.json'),'--task',t,'--root-id',req,'--split','development','--run',str(run),'--gpu','0']
 print('RESUME_PHYSICS',t,flush=True);s=time.monotonic();p=subprocess.run([sys.executable,'-B',str(R/'scripts/research_run.py'),'--receipt-dir',str(O/'runner_resume'/t),'--wall-seconds','900','--',*cmd]);rows.append({'task':t,'returncode':p.returncode,'wall_seconds':time.monotonic()-s});(O/'RESUME_RESULTS.json').write_text(json.dumps(rows,indent=2))
