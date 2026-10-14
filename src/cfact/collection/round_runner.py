"""Bounded persistent orchestration of the existing physical CFACT worker.

Use ``cfact recursive``. Jobs, inputs and code are frozen before dispatch.
Resource deferral preserves request identity; partial attempts are never spliced.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from cfact.io.serialization import atomic_json, RunLock
from cfact.collection.snapshot import source_digest
from cfact.storage.policy import contained_path


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def gpu_inventory():
    fields = "index,uuid,name,memory.total,memory.free,temperature.gpu,utilization.gpu,compute_mode"
    cmd = ["nvidia-smi", "--query-gpu="+fields, "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode:
        return {"status": "GPU_QUERY_UNAVAILABLE", "error": result.stderr or result.stdout, "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        i, uuid, name, total, free, temp, util, mode = [x.strip() for x in line.split(",")]
        gpus.append(dict(index=int(i), uuid=uuid, name=name, total_mib=int(total),
                         free_mib=int(free), temperature_c=int(temp), utilization_pct=int(util),
                         compute_mode=mode))
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    return {"status": "PASS", "gpus": gpus, "compute_process_metadata": processes.stdout}


def preflight(root):
    root = root.resolve()
    fs = os.statvfs(root)
    memory = {x.split(":")[0]: int(x.split()[1])*1024
              for x in Path("/proc/meminfo").read_text().splitlines()}
    return dict(time_utc=utc(), gpu=gpu_inventory(), available_ram_bytes=memory["MemAvailable"],
                cpu_affinity=list(os.sched_getaffinity(0)), free_disk_bytes=fs.f_bavail*fs.f_frsize,
                total_disk_bytes=fs.f_blocks*fs.f_frsize, free_inodes=fs.f_favail,
                output_realpath=str(root), scheduler="SLURM" if os.environ.get("SLURM_JOB_ID") else
                ("SLURM_UNALLOCATED" if __import__("shutil").which("squeue") else "LOCAL_LEASES"))


def choose_gpu(inventory, verified, *, measured_peak_mib=2889):
    candidates = []
    for g in inventory["gpu"]["gpus"]:
        # User override: temperature is monitored during execution, not admission.
        reserve = max(4096, g["total_mib"]*.05) + measured_peak_mib*1.3
        if g["uuid"] not in verified or g["compute_mode"] != "Default":
            continue
        if g["utilization_pct"] > 80 or g["free_mib"] < reserve:
            continue
        candidates.append(g)
    return min(candidates, key=lambda g: (g["utilization_pct"], -g["free_mib"])) if candidates else None


def ledger_steps(run, task, request):
    path = run / "history/development" / task / request / "frames.jsonl"
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size-32768))
        lines = stream.read().splitlines()
    return int(json.loads(lines[-1])["control_step_total"]) if lines else 0


def directory_bytes(path):
    total = 0
    for directory, dirs, names in os.walk(path):
        dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in names:
            p = Path(directory)/name
            if not p.is_symlink():
                s = p.stat()
                total += max(s.st_size, s.st_blocks*512)
    return total


def verify_queue(root, queue):
    freeze = read(queue["code_freeze"])
    if source_digest(Path(freeze["frozen_path"])) != freeze["code_sha256"]:
        raise ValueError("FROZEN_WORKER_CODE_CHANGED")
    for key in ["config", "source_manifest"]:
        import hashlib
        path = Path(queue[key]); contained_path(path, root)
        if __import__("hashlib").sha256(path.read_bytes()).hexdigest() != queue[key+"_sha256"]:
            raise ValueError("FROZEN_"+key.upper()+"_CHANGED")
    for job in queue["jobs"]:
        contained_path(Path(job["run"]), root)
    from cfact.sources.manifest import sha256_file
    sources = read(queue["source_manifest"])
    for path, row in sources["prepared_files"].items():
        if sha256_file(Path(path)) != row["sha256"]:
            raise ValueError("FROZEN_PREPARED_SOURCE_CHANGED")
    return freeze


def costs(root):
    receipts = [read(p) for p in sorted((root/"logs/receipts").glob("*.json"))]
    # The separate initial smoke is counted exactly once, never its exported copies.
    if (root/"logs/SMOKE_RECEIPT.json").exists():
        receipts.append(read(root/"logs/SMOKE_RECEIPT.json"))
    return {"control_steps": sum(x.get("control_steps", 0) for x in receipts),
            "gpu_worker_seconds": sum(x.get("gpu_worker_seconds", 0.) for x in receipts),
            "cpu_worker_seconds": sum(x.get("cpu_worker_seconds", 0.) for x in receipts),
            "queue_wait_seconds": sum(x.get("queue_wait_seconds", 0.) for x in receipts),
            "receipts": len(receipts)}


def live_costs(root,exclude_job=None):
    total=costs(root)
    for path in sorted((root/"logs").glob("SCHEDULER_STATUS_lane*.json")):
        active=read(path).get("active")
        if not active or active.get("job_id")==exclude_job or not active.get("pid"):continue
        if (root/"logs/receipts"/(active["job_id"]+".json")).exists():continue
        try:
            command=Path(f'/proc/{active["pid"]}/cmdline').read_bytes()
            if str(active["run"]).encode() not in command:continue
        except FileNotFoundError:continue
        total["control_steps"]+=active.get("control_steps",0)
        key="gpu_worker_seconds" if active["backend"]=="egl" else "cpu_worker_seconds"
        total[key]+=active.get("wall_seconds",0.)
    return total


def run_fleet(root,queue_path,queue):
    """Two fixed device lanes; every lane runs the same immutable physics code."""
    workers=int(queue["fleet_workers"])
    if workers not in {1,2}:raise ValueError("ONLY_MEASURED_ONE_OR_TWO_WORKER_FLEET_SUPPORTED")
    verify_queue(root,queue)
    if len(queue["verified_gpu_uuids"])<workers:raise ValueError("FLEET_REQUIRES_VERIFIED_PHYSICAL_GPU_MAPPING")
    state_path=root/"logs/SCHEDULER_STATUS.json"
    with RunLock(root/"logs/scheduler.lock"):
        children=[]
        try:
            for lane in range(workers):
                child_queue={**queue,"fleet_workers":1,"jobs":queue["jobs"][lane::workers],
                    "verified_gpu_uuids":[queue["verified_gpu_uuids"][lane]]}
                path=root/"config"/(queue_path.stem+f"_lane{lane}.json")
                if path.exists() and read(path)!=child_queue:raise ValueError("FROZEN_LANE_QUEUE_CHANGED")
                if not path.exists():atomic_json(path,child_queue)
                env=dict(os.environ,CFACT_QUEUE_LANE=str(lane),CFACT_FLEET_WIDTH=str(workers))
                log=(root/"logs"/(queue_path.stem+f"_lane{lane}.log")).open("a")
                child=subprocess.Popen([sys.executable,"-B","-m","cfact.collection.round_runner","run",
                    "--round",str(root),"--queue",str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
                log.close();children.append(child)
            while True:
                lanes=[read(p) for p in sorted((root/"logs").glob("SCHEDULER_STATUS_lane*.json"))
                       if read(p).get("queue") in [str(root/"config"/(queue_path.stem+f"_lane{x}.json")) for x in range(workers)]]
                alive=any(p.poll() is None for p in children)
                status="RUNNING" if alive else "COMPLETE" if len(lanes)==workers and all(s["status"]=="COMPLETE" for s in lanes) else "PARTIAL_OR_STOPPED"
                atomic_json(state_path,dict(status=status,scheduler_pid=os.getpid(),queue=str(queue_path),
                    heartbeat_utc=utc(),code_freeze=read(queue["code_freeze"]),fleet_workers=workers,
                    lanes=lanes,active=[s["active"] for s in lanes if s.get("active")],
                    completed=[r for s in lanes for r in s["completed"]],costs=live_costs(root)))
                if not alive:break
                time.sleep(5)
            if any(p.returncode for p in children):
                state=read(state_path);state.update(status="TECHNICAL_SCHEDULER_FAILURE",lane_exit_codes=[p.returncode for p in children]);atomic_json(state_path,state)
            return read(state_path)
        finally:
            for p in children:
                if p.poll() is None:p.terminate()
            for p in children:
                if p.poll() is None:p.wait(timeout=1250)


def run_queue(root, queue_path):
    root = root.resolve(); queue_path=queue_path.resolve(); contained_path(queue_path, root)
    queue = read(queue_path); freeze = verify_queue(root, queue)
    lane=os.environ.get("CFACT_QUEUE_LANE")
    if int(queue.get("fleet_workers",1))>1 and lane is None:return run_fleet(root,queue_path,queue)
    config = read(queue["config"]); budget = read(root/"config/BUDGET.json")
    identity = read(root/"manifests/BASE_IDENTITY.json")
    deadline = datetime.fromisoformat(identity["started_utc"]).timestamp()+budget["calendar_hours"]*3600
    state_path = root/"logs"/(f"SCHEDULER_STATUS_lane{lane}.json" if lane is not None else "SCHEDULER_STATUS.json")
    (root/"logs/receipts").mkdir(exist_ok=True)
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    state = dict(status="RUNNING", scheduler_pid=os.getpid(), started_utc=utc(),
                 queue=str(queue_path), code_freeze=freeze, completed=[], active=None)
    atomic_json(state_path, state)
    lock = RunLock(root/"logs"/(f"scheduler_lane{lane}.lock" if lane is not None else "scheduler.lock")); lock.acquire()
    try:
        for job in queue["jobs"]:
            receipt_path = root/"logs/receipts"/(job["job_id"]+".json")
            if receipt_path.exists():
                prior = read(receipt_path)
                if prior.get("code_sha256") != freeze["code_sha256"]:
                    raise ValueError("RESUME_JOB_CODE_CHANGED")
                state["completed"].append(prior); continue
            if stop_requested or (root/"logs/PAUSE_REQUESTED").exists():
                state["status"]="PAUSED"; break
            total = live_costs(root)
            width=int(os.environ.get("CFACT_FLEET_WIDTH",1))
            if time.time() >= deadline or total["control_steps"]+width*config["execution"]["max_env_steps_per_root"] > budget["control_steps"]:
                state["status"]="BUDGET_EXHAUSTED"; break
            wait_start=time.monotonic(); backoff=15; device=None
            while True:
                inv=preflight(root); atomic_json(root/"logs/RESOURCE_CURRENT.json", inv)
                hard=[]
                if inv["free_disk_bytes"] < budget["reserve_bytes"]+queue.get("reserve_root_bytes", 2*2**30):hard.append("DISK_RESERVE")
                if inv["available_ram_bytes"] < max(24*2**30, queue.get("peak_rss_bytes",8*2**30)*1.3):hard.append("RAM_RESERVE")
                if inv["free_inodes"] < 1000000:hard.append("INODE_RESERVE")
                if inv["scheduler"] == "SLURM_UNALLOCATED":hard.append("SCHEDULER_ALLOCATION_REQUIRED")
                verified=queue.get("verified_gpu_uuids", [])
                first=choose_gpu(inv, verified)
                if first and not hard:
                    # Two separate checks; preserve task/root/seed while waiting.
                    time.sleep(15)
                    second=preflight(root); picked=choose_gpu(second, verified)
                    if picked and picked["uuid"] == first["uuid"]:device=picked;break
                if not hard and queue.get("cpu_fallback_validated") and total["cpu_worker_seconds"]+1200 < budget["cpu_worker_hours"]*3600:
                    break
                if time.monotonic()-wait_start >= 4*3600 or total["queue_wait_seconds"]+time.monotonic()-wait_start >= 12*3600 or time.time() >= deadline:
                    state["status"]="RESOURCE_DEFERRED";break
                state.update(active={**job,"status":"RESOURCE_DEFERRED","reasons":hard or ["GPU_ADMISSION"],"wait_seconds":time.monotonic()-wait_start},heartbeat_utc=utc())
                atomic_json(state_path,state)
                if stop_requested or (root/"logs/PAUSE_REQUESTED").exists():state["status"]="PAUSED";break
                time.sleep(min(backoff,60));backoff=min(backoff*2,300)
            if state["status"] != "RUNNING":break
            backend="egl" if device else "osmesa"
            lane="gpu_worker_seconds" if device else "cpu_worker_seconds"
            lane_limit=budget["gpu_worker_hours" if device else "cpu_worker_hours"]*3600
            if total[lane]+width*budget["root_timeout_seconds"] > lane_limit:
                state["status"]="BUDGET_EXHAUSTED";break
            run=Path(job["run"]);run.mkdir(parents=True,exist_ok=True)
            # Full worker inputs and effective configuration are archived at dispatch.
            for key,name in [("config","config_frozen.json"),("source_manifest","source_split_frozen.json")]:
                dest=run/name
                if dest.exists() and dest.read_bytes() != Path(queue[key]).read_bytes():raise ValueError("RUN_INPUT_CHANGED")
                if not dest.exists():dest.write_bytes(Path(queue[key]).read_bytes())
            atomic_json(run/"run_manifest.json",dict(job=job,code_freeze=freeze,backend=backend,device=device))
            env=dict(os.environ, PYTHONPATH=freeze["frozen_path"]+"/src",CFACT_SOURCE_ROOT=freeze["frozen_path"],
                     MUJOCO_GL=backend,PYOPENGL_PLATFORM=backend,
                     CUDA_VISIBLE_DEVICES=device["uuid"] if device else "",CFACT_GPU_UUID=device["uuid"] if device else "CPU_ONLY")
            if device:env["MUJOCO_EGL_DEVICE_ID"]=str(device["index"])
            else:env.pop("MUJOCO_EGL_DEVICE_ID",None)
            cmd=[sys.executable,"-B","-m","cfact.collection.worker","--config",queue["config"],
                 "--source-manifest",queue["source_manifest"],"--task",job["task"],"--root-id",job["request_id"],
                 "--split","development","--run",str(run),"--gpu",str(device["index"] if device else -1)]
            start=time.monotonic();peak=0;peak_gpu=0;reason=None;child=None;gpu_lock=None;last_disk=0;disk=0
            if device:
                gpu_lock=RunLock(root/"logs"/(device["uuid"]+".lease"));gpu_lock.acquire()
            try:
                with (root/"logs"/(job["job_id"]+".log")).open("x") as log:
                    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        elapsed=time.monotonic()-start
                        rss=0
                        try:
                            rows=Path(f"/proc/{child.pid}/status").read_text().splitlines()
                            rss=int(next(x for x in rows if x.startswith("VmRSS:")).split()[1])*1024
                        except (FileNotFoundError,StopIteration):pass
                        peak=max(peak,rss);steps=ledger_steps(run,job["task"],job["request_id"])
                        inv=preflight(root)
                        if time.monotonic()-last_disk>60:disk=directory_bytes(root);last_disk=time.monotonic()
                        state.update(active={**job,"pid":child.pid,"backend":backend,"device":device,"wall_seconds":elapsed,
                            "peak_rss_bytes":peak,"control_steps":steps,"disk_bytes":disk},heartbeat_utc=utc(),costs=total)
                        atomic_json(state_path,state)
                        if elapsed>=budget["root_timeout_seconds"]:reason="ROOT_TIMEOUT"
                        if time.time()>=deadline:reason="CALENDAR_BUDGET"
                        if inv["free_disk_bytes"]<budget["reserve_bytes"] or disk>budget["disk_bytes"]:reason="DISK_RESERVE"
                        if inv["available_ram_bytes"]<24*2**30:reason="RAM_RESERVE"
                        charged=live_costs(root,exclude_job=job["job_id"])
                        if charged["control_steps"]+steps>=budget["control_steps"]:reason="CONTROL_STEP_BUDGET"
                        if charged["gpu_worker_seconds" if device else "cpu_worker_seconds"]+elapsed>=lane_limit-10:reason="WORKER_TIME_BUDGET"
                        if device:
                            g=next((g for g in inv["gpu"]["gpus"] if g["uuid"]==device["uuid"]),None)
                            if g is None:reason="GPU_QUERY_UNAVAILABLE"
                            elif g["temperature_c"]>=budget.get("gpu_cutoff_c",92):reason="GPU_TEMPERATURE_CUTOFF"
                            elif g["free_mib"]<4096:reason="GPU_MEMORY_RESERVE"
                            # Query only this worker's process metadata, including EGL graphics contexts.
                            pmon=subprocess.run(["nvidia-smi","pmon","-c","1","-s","m"],capture_output=True,text=True,timeout=10)
                            for line in pmon.stdout.splitlines():
                                fields=line.split()
                                if len(fields)>3 and fields[1]==str(child.pid):
                                    if int(fields[0])!=device["index"]:reason="GPU_MAPPING_MISMATCH"
                                    for value in fields[3:5]:
                                        if value.isdigit():peak_gpu=max(peak_gpu,int(value))
                            if elapsed>90 and not (run/"runtime/scene_alive").exists():reason="SCENE_START_WATCHDOG"
                        if reason:
                            os.killpg(child.pid,signal.SIGTERM);break
                        time.sleep(5)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            finally:
                if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
                if gpu_lock:gpu_lock.release()
            elapsed=time.monotonic()-start
            results=list(run.glob("data/*/*/*/root_result.json"))
            audits=[]
            from cfact.collection.recursive_audit import audit_root
            for p in results:
                if (p.parent/"tree.json").exists():
                    audited=audit_root(p,verify_rgb=False)
                    atomic_json(root/"derived"/(job["job_id"]+"_integrity.json"),audited)
                    audits.append({k:audited[k] for k in ["integrity","complete_edges","nonroot_parents","observed_depth"]})
            if any(a["integrity"]!="PASS" for a in audits):reason="DATA_INTEGRITY_FAILURE"
            rec=dict(job_id=job["job_id"],task=job["task"],request_id=job["request_id"],run=str(run),
                     status=reason or ("COMPLETE" if child.returncode==0 else "TECHNICAL_ERROR"),
                     result_paths=[str(p) for p in results],exit_code=child.returncode,pid=child.pid,
                     wall_seconds=elapsed,gpu_worker_seconds=elapsed if device else 0.,
                     cpu_worker_seconds=0. if device else elapsed,peak_rss_bytes=peak,peak_gpu_mib=peak_gpu,
                     control_steps=ledger_steps(run,job["task"],job["request_id"]),backend=backend,device=device,
                     code_sha256=freeze["code_sha256"],code_commit=freeze["sha"],finished_utc=utc(),
                     queue_wait_seconds=start-wait_start,command=cmd,audits=audits)
            atomic_json(receipt_path,rec);state["completed"].append(rec);state["active"]=None
            atomic_json(state_path,state)
            if rec["status"]=="DATA_INTEGRITY_FAILURE":state["status"]="DATA_INTEGRITY_FAILURE";break
            if queue.get("halt_on_technical", True) and (rec["status"]=="TECHNICAL_ERROR" or any(
                    any(e.get("status")=="TECHNICAL_ERROR" for e in read(p).get("edges", [])) for p in results)):
                state["status"]="NEEDS_TECHNICAL_FIX";break
        else:
            state["status"]="COMPLETE"
        state.update(heartbeat_utc=utc(),costs=costs(root));atomic_json(state_path,state)
        return state
    finally:
        lock.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["preflight","run","generate","smoke","resume","status","pause","verify","export"])
    parser.add_argument("--round",type=Path,required=True)
    parser.add_argument("--queue",type=Path)
    parser.add_argument("--input",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--scope",default="DEVELOPMENT");parser.add_argument("--calibrations",type=Path)
    args=parser.parse_args();root=args.round.resolve()
    if args.action=="preflight":result=preflight(root);atomic_json(root/"logs/PREFLIGHT.json",result)
    elif args.action=="status":result=read(root/"logs/SCHEDULER_STATUS.json")
    elif args.action=="pause":(root/"logs/PAUSE_REQUESTED").touch();result={"status":"PAUSE_REQUESTED_AFTER_ACTIVE_ROOT"}
    elif args.action in {"verify","export"}:
        if not args.input or not args.output:parser.error("--input and --output are required")
        python=os.environ.get("CFACT_EXPORT_PYTHON",sys.executable) if args.action=="export" else sys.executable
        command=[python,"-B","-m","cfact","export-recursive" if args.action=="export" else "audit-recursive",
            "--input",str(args.input),"--output",str(args.output)]
        if args.action=="export":command.extend(["--scope",args.scope])
        elif args.calibrations:command.extend(["--calibrations",str(args.calibrations)])
        subprocess.run(command,check=True);result={"status":"COMMAND_COMPLETE","action":args.action,"output":str(args.output)}
    else:
        if not args.queue:parser.error("--queue is required")
        if args.action=="smoke" and len(read(args.queue)["jobs"])!=1:parser.error("smoke requires exactly one frozen root job")
        if args.action=="resume":(root/"logs/PAUSE_REQUESTED").unlink(missing_ok=True)
        result=run_queue(root,args.queue)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
