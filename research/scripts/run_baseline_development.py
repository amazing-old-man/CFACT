import json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/ssd1/yanghc/CFACT_Research');ref=R/'01_development/SMOKE1_BASE_CURRENT_GPU/runs/OpenDrawer';O=R/'02_baselines/MATCHED_DRAWER';O.mkdir(parents=True,exist_ok=True);records=[]
for scheme in ['GENERIC_CF_SINGLE','GENERIC_CF_OFFDEMO']:
 s=time.monotonic()
 while time.monotonic()-s<120:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);g=next(list(map(int,l.split(','))) for l in rows.splitlines() if int(l.split(',')[0])==0)
  if g[1]<80 and g[2]>=8192 and g[3]<=80:break
  time.sleep(3)
 cmd=[sys.executable,'-B',str(R/'scripts/generic_baseline.py'),'--reference-run',str(ref),'--output',str(O/'runs'/scheme),'--scheme',scheme,'--budget','1500'];ret=subprocess.run([sys.executable,'-B',str(R/'scripts/research_run.py'),'--receipt-dir',str(O/'runner'/scheme),'--wall-seconds','900','--',*cmd]);records.append(dict(scheme=scheme,returncode=ret.returncode));(O/'RESULTS.json').write_text(json.dumps(records,indent=2))
