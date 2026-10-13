import json,os,subprocess,sys
from pathlib import Path
R=Path('/mnt/ssd1/yanghc/CFACT_Research')
for phase,code,tasks in [('DEV_ITERATION_02','iteration_02_frozen',['OpenDrawer']),('DEV_ITERATION_03','iteration_03_final',['CoffeeServeMug','OpenBlenderLid'])]:
 O=R/'01_development'/phase/'physics';m=json.loads((O/'plan/source_manifest.json').read_text());env=os.environ.copy();env.update(CFACT_SOURCE_ROOT=str(R/'runtime'/code),PYTHONPATH=str(R/'runtime'/code/'src'),MUJOCO_GL='osmesa',CUDA_VISIBLE_DEVICES='')
 for t in tasks:
  req=next(k for k,v in m['requests_by_id'].items() if v['task']==t);run=O/'runs'/t if phase.endswith('02') else O/'runs'/t/'demo_0';cmd=[sys.executable,'-B','-m','cfact.collection.worker','--config',str(O/'plan/config.json'),'--source-manifest',str(O/'plan/source_manifest.json'),'--task',t,'--root-id',req,'--split','development','--run',str(run),'--gpu','-1'];print('CPU_FALLBACK',phase,t,flush=True);subprocess.run([sys.executable,'-B',str(R/'scripts/research_run.py'),'--cpu-only','--receipt-dir',str(O/'runner_cpu_final'/t),'--wall-seconds','900','--',*cmd],env=env)
