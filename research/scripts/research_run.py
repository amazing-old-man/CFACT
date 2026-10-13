"""Bounded subprocess guard; only stop the process group created by this runner."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
def own_signal(signum, frame):
 raise InterruptedError(f'OWN_RUNNER_SIGNAL_{signum}')
signal.signal(signal.SIGTERM,own_signal)
signal.signal(signal.SIGINT,own_signal)
ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research').resolve()
p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true');p.add_argument('--receipt-dir',type=Path,required=True);p.add_argument('--wall-seconds',type=int,default=900);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args();out=args.receipt_dir.resolve();assert out.is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=True)
dispatch_guard_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
dispatch_code_root=os.environ.get('CFACT_SOURCE_ROOT',str(ROOT/'repo'))
dispatch_code_commit=subprocess.check_output(['git','-C',dispatch_code_root,'rev-parse','HEAD'],text=True).strip()
policy=json.loads((ROOT/'manifests/BUDGET_POLICY.json').read_text());start=time.monotonic();child=None;reason=None;last_inventory=0;cached_inventory={}
def inventory():
 logical=0;allocated=0;control_steps=0;completed_wall=0;gpu_wall=0
 for directory, dirs, names in os.walk(ROOT):
  dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
  for name in names:
   path=Path(directory)/name
   if path.is_symlink():continue
   st=path.stat();logical+=st.st_size;allocated+=st.st_blocks*512
   if name=='frames.jsonl' and path.relative_to(ROOT).parts[0] in {'01_development','02_baselines','03_formal_runs'} and 'history' in path.relative_to(ROOT).parts:
    with path.open('rb') as f:
     f.seek(max(0,st.st_size-65536));lines=f.read().splitlines()
    if lines:
     try:control_steps+=int(json.loads(lines[-1]).get('control_step_total',0))
     except (ValueError,KeyError):pass
   if name=='runner_receipt.json' and path!=out/'runner_receipt.json':
    rec=json.loads(path.read_text());completed_wall+=rec['wall_seconds']
    if rec.get('compute_backend')=='EGL_GPU0':gpu_wall+=rec['wall_seconds']
 return dict(workspace_logical_bytes=logical,workspace_allocated_bytes=allocated,
             total_control_steps=control_steps,completed_runner_wall_seconds=completed_wall,
             completed_gpu_wall_seconds=gpu_wall)
def guard():
 global last_inventory,cached_inventory
 st=os.statvfs(ROOT);free=st.f_bavail*st.f_frsize
 mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
 r=subprocess.CompletedProcess([],0,'','') if args.cpu_only else subprocess.run(['nvidia-smi','--query-gpu=index,temperature.gpu,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);row={'time_utc':datetime.now(timezone.utc).isoformat(),'disk_free_bytes':free,'free_inodes':st.f_favail,'available_ram_bytes':mem['MemAvailable']}
 failures=[]
 if time.monotonic()-last_inventory>15:
  cached_inventory=inventory();last_inventory=time.monotonic()
 row.update(cached_inventory)
 if max(cached_inventory['workspace_logical_bytes'],cached_inventory['workspace_allocated_bytes'])>=policy['maximum_new_bytes']:failures.append('WORKSPACE_QUOTA')
 if cached_inventory['total_control_steps']>=policy['development']['maximum_total_control_steps']:failures.append('DEVELOPMENT_STEP_BUDGET')
 if cached_inventory['completed_runner_wall_seconds']+time.monotonic()-start>=policy['development']['max_stage_wall_seconds']:failures.append('DEVELOPMENT_STAGE_WALL_BUDGET')
 if cached_inventory['completed_gpu_wall_seconds']+(0 if args.cpu_only else time.monotonic()-start)>=policy['development']['maximum_gpu_hours']*3600:failures.append('DEVELOPMENT_GPU_HOUR_BUDGET')
 if free<policy['minimum_free_bytes']:failures.append('DISK_RESERVE')
 if st.f_favail<policy['minimum_free_inodes']:failures.append('INODE_RESERVE')
 if mem['MemAvailable']<policy['minimum_available_ram_gib']*2**30:failures.append('RAM_RESERVE')
 if args.cpu_only:
  if os.environ.get('MUJOCO_GL')!='osmesa' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':failures.append('CPU_BACKEND_NOT_EXPLICIT')
  row['compute_backend']='OSMESA_CPU_NO_CUDA'
 elif r.returncode:failures.append('GPU_PROBE_FAILED')
 else:
  gpu=[list(map(int,l.split(','))) for l in r.stdout.splitlines() if int(l.split(',')[0])==0]
  if len(gpu)!=1:failures.append('GPU0_UNAVAILABLE')
  else:
   _,temp,free_mib,util=gpu[0];row.update(temperature_c=temp,gpu_free_mib=free_mib,gpu_utilization_pct=util)
   if temp>=policy['maximum_gpu_temperature_c']:failures.append('GPU_TEMPERATURE_LIMIT')
   if free_mib<policy['minimum_free_gpu_mib']:failures.append('VRAM_RESERVE')
   if child is None and util>policy['maximum_gpu_utilization_pct_before_dispatch']:failures.append('GPU_BUSY_BEFORE_DISPATCH')
 row['failures']=failures
 with (out/'guard.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 return failures
def stop():
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
try:
 bad=guard()
 if bad:reason='BLOCKED_RESOURCE:'+','.join(bad)
 else:
  command=args.command[1:] if args.command[:1]==['--'] else args.command
  with (out/'process.log').open('x') as f:
   child=subprocess.Popen(command,cwd=ROOT/'repo',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   print('STARTED',child.pid,command,flush=True)
   while child.poll() is None:
    if time.monotonic()-start>args.wall_seconds:reason='WALL_BUDGET_EXHAUSTED';break
    bad=guard()
    if bad:reason='BLOCKED_RESOURCE:'+','.join(bad);break
    time.sleep(3)
   if reason:stop()
   else:reason='PASS' if child.returncode==0 else 'FAILED_SUBPROCESS'
except InterruptedError as exc:
 reason=str(exc)
finally:
 stop();record={'status':reason,'pid':child.pid if child else None,'exit_code':child.returncode if child else None,'wall_seconds':time.monotonic()-start,'command':args.command,'outputs_root':str(out),'compute_backend':'OSMESA_CPU_NO_CUDA' if args.cpu_only else 'EGL_GPU0','executed_code_root':dispatch_code_root,'executed_code_commit':dispatch_code_commit,'guard_code_sha256':dispatch_guard_sha256,'global_cost_inventory':cached_inventory,'stopped_only_own_process_group':True};(out/'runner_receipt.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
raise SystemExit(0 if reason=='PASS' else 1)
