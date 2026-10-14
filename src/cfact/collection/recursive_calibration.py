"""Actual A/B/A calibration using a saved full parent and issued controller actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
from cfact.collection.runtime import _load_root, _create_env
from cfact.execution.snapshot import load_snapshot, capture_snapshot, restore_snapshot, snapshot_differences, save_snapshot
from cfact.execution.history import ControlStepLedger
from cfact.execution.options import MeasuredTrace
from cfact.execution.binding import trace_joint_spec
from cfact.adapters.action_codec import install_action_codec
from cfact.generation.task_adapters import FLOORS, family_for_task
from cfact.tasks import target_task
from cfact.io.serialization import atomic_json
from cfact.collection.snapshot import implementation_digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--round",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=-1);args=p.parse_args()
    out=args.output.resolve();r=args.round.resolve()
    from cfact.storage.policy import contained_path
    contained_path(out,r);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();ledger=None;env=None
    result=next(args.run.glob("data/*/*/*/root_result.json"));rp=result.parent
    manifest=json.loads((rp/"root_manifest.json").read_text());task=target_task(manifest["task"])
    root=_load_root(Path(manifest["prepared_hdf5"]),manifest["source_group"])
    context=json.loads((rp/"nodes/n_root/context.json").read_text())
    snapshot=load_snapshot(rp/"nodes/n_root/parent_snapshot.json")
    cards=sorted((rp/"nodes/n_root/edges").glob("*/option.json"))
    ref=next(c for c in cards if json.loads(c.read_text())["source_method"]=="Reference" and (c.parent/"execution.npz").exists())
    with np.load(ref.parent/"execution.npz",allow_pickle=False) as a:commands=a["commanded_actions"][:20].copy()
    if len(commands)<20:raise ValueError("CALIBRATION_REQUIRES_ONE_SECOND_ACTUAL_ACTIONS")
    b=commands.copy();b[:,:3]*=-1
    cfg=json.loads((args.run/"config_frozen.json").read_text())
    policy=next(row for row in cfg["tasks"] if row["task"]==task.name)
    env=_create_env(root["env_meta"],manifest["seed"],"robot0_agentview_left",args.gpu)
    status="TECHNICAL_ERROR"
    try:
        import robocasa, mimicgen.env_interfaces.robocasa, cfact.adapters.interfaces
        from mimicgen.env_interfaces.base import make_interface
        interface=make_interface(task.adapter_interface or task.mimicgen_interface,"robosuite",env.env)
        install_action_codec(interface)
        ledger=ControlStepLedger(env,out/"history",width=224,height=224);env._cfact_step_ledger=ledger
        spec=trace_joint_spec(context["operation_context"]["joint_binding"])
        traces=[];restores=[]
        for label,issued in [("A1",commands),("B",b),("A2",commands)]:
            t=time.monotonic();receipt=restore_snapshot(env,snapshot,initial=root["initial"])
            ledger.restore_wall_seconds+=time.monotonic()-t
            current=capture_snapshot(env);diff=snapshot_differences(snapshot,current)
            restores.append(dict(label=label,differences=diff,restore=receipt))
            save_snapshot(out/(label+"_start.json"),current)
            if diff:raise ValueError("ABA_COMPLETE_PARENT_RESTORE_MISMATCH")
            ledger.begin(label,"aba_validation")
            trace=MeasuredTrace(policy["event_object_ref"],policy["operation_object_ref"],**spec)
            trace.capture(env,interface)
            for action in issued:
                env.step(action);trace.after_step(action,env,interface,False)
            values=trace.arrays();np.savez_compressed(out/(label+".npz"),**values);traces.append(values)
            save_snapshot(out/(label+"_end.json"),capture_snapshot(env))
        a,c=traces[0],traces[2]
        noise={k:float(np.max(np.abs(a[k].astype(float)-c[k].astype(float)))) if a[k].size else 0.
               for k in ["sim_time","robot_qpos","robot_qvel","actual_eef_pose","object_poses","relevant_fixture_qpos","task_success","commanded_actions"]}
        family=family_for_task(task)
        unit=str(a["fixture_joint_types"][0]) if family=="articulated_contact" and len(a["fixture_joint_types"])==1 else "activation" if family=="button_activation" else "object"
        if unit in {"hinge","slide"}:
            effect_noise=noise["relevant_fixture_qpos"]
        elif unit=="object":
            effect_noise=float(np.max(np.linalg.norm(a["object_poses"][:,0,:3,3]-c["object_poses"][:,0,:3,3],axis=1)))
        else:
            activation_a=[json.loads(str(x)) for x in a["fixture_activation_json"]]
            activation_c=[json.loads(str(x)) for x in c["fixture_activation_json"]]
            if not all(x.get("status")=="MEASURED" for x in activation_a+activation_c):
                raise ValueError("ACTIVATION_CALIBRATION_UNMEASURED")
            effect_noise=float(any(x["turned_on"]!=y["turned_on"] for x,y in zip(activation_a,activation_c)))
        noise["task_effect_distance_in_protocol_units"]=effect_noise
        margin=5*effect_noise
        threshold=max(FLOORS[unit],margin)
        status="PASS" if noise["sim_time"]<1e-9 and noise["commanded_actions"]==0 else "FAIL"
        record=dict(status=status,task=task.name,family=family,threshold_unit=unit,
                    frozen_threshold=threshold,minimum_physical_effect_floor=FLOORS[unit],
                    repeat_noise_margin=margin,ABA_max_absolute_noise=noise,roundtrips=restores,
                    horizon_s=1.,actual_total_control_steps=ledger.control_steps,
                    source_id=manifest.get("source_id"),source_group=manifest["source_group"],
                    backend=os.environ.get("MUJOCO_GL"),code_sha256=implementation_digest(),
                    acquisition_code_freeze=json.loads((args.run/"run_manifest.json").read_text())["code_freeze"],
                    commands_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),
                    wall_seconds=time.monotonic()-started,calibration_scope="REPRESENTATIVE_FAMILY_DEVELOPMENT_ROOT")
        atomic_json(out/"SAME_PARENT_ABA.json",record);print(json.dumps(record),flush=True)
    finally:
        if ledger:ledger.finish()
        if env:env.close()
        elapsed=time.monotonic()-started;backend=os.environ.get("MUJOCO_GL")
        receipt=dict(status=status,task=task.name,job_id="aba_"+out.name,run=str(out),
                     control_steps=ledger.control_steps if ledger else 0,
                     gpu_worker_seconds=elapsed if backend=="egl" else 0.,cpu_worker_seconds=elapsed if backend!="egl" else 0.,
                     wall_seconds=elapsed,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     backend=backend,code_sha256=implementation_digest())
        atomic_json(r/"logs/receipts"/(receipt["job_id"]+".json"),receipt)


if __name__=="__main__":main()
