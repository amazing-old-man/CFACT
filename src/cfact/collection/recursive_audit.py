"""Read actual sibling effects at the same physical elapsed time, and ancestry."""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from cfact.execution.snapshot import load_snapshot
from cfact.generation.task_adapters import family_for_task, FLOORS
from cfact.tasks import target_task
from cfact.io.serialization import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def effect_value(a, index, family):
    if family=="articulated_contact":
        kinds=a["fixture_joint_types"].astype(str)
        if len(kinds)!=1 or kinds[0] not in {"slide","hinge"}:return None, None
        return a["relevant_fixture_qpos"][index], kinds[0]
    if family=="button_activation":
        d=json.loads(str(a["fixture_activation_json"][index]))
        return (np.array([int(d["turned_on"])]),"activation") if d.get("status")=="MEASURED" else (None,None)
    if family=="held_object_placement":
        return a["object_poses"][index,0,:3,3],"object"
    return None,None


def audit_root(result_path, *, horizon=None, calibrations=None, verify_rgb=True):
    calibrations=calibrations or {}
    rp=Path(result_path).resolve();root=rp.parent;run=rp.parents[4];result=read(rp)
    family=family_for_task(target_task(result["task"]));tree=read(root/"tree.json")
    if horizon is None:
        config=read(run/"config_frozen.json")
        horizon=config.get("method",{}).get("comparison_horizon_s",{}).get(family,1.)
    parentmap={p["node_id"]:run/p["parent_edge"] for p in tree.get("child_pools",[])+tree.get("third_pools",[]) if p.get("parent_edge")}
    manifest=read(root/"root_manifest.json")
    hist=run/"history"/result["split"]/result["task"]/manifest["source_episode"]
    groups=defaultdict(list)
    for line in (hist/"frames.jsonl").read_text().splitlines():
        row=json.loads(line);groups[(row["segment"],row["phase"])].append(row)
    byparent=defaultdict(list);failures=[];edges=[];primitive_examples=[];rgb_frames=0
    for edge in result["edges"]:
        if not edge.get("edge_path"):continue
        ep=run/edge["edge_path"]
        if edge["status"]!="PHYSICS_COMPLETE":
            failures.append({"edge":str(ep),"status":edge["status"],"reason":edge.get("error",edge.get("reason"))});continue
        try:
            a=arrays(ep/"execution.npz");k=len(a["commanded_actions"])
            assert len(a["sim_time"])==k+1, "K_KPLUS1"
            assert np.allclose(np.diff(a["sim_time"]),.05,atol=1e-9,rtol=0),"CONTROL_CLOCK"
            for key in ["robot_qpos","robot_qvel","object_poses","actual_eef_pose","task_success"]:
                assert len(a[key])==k+1 and np.isfinite(a[key]).all(),key
            child=load_snapshot(ep/"child_snapshot.json")
            assert float(child["sim_time_s"])==a["sim_time"][-1],"CHILD_TIME"
            assert np.array_equal(arrays(ep/"child_context.npz")["sim_state"],child["states"]),"CHILD_STATE"
            parent=ep.parent.parent.name
            state=arrays(ep.parent.parent/"context.npz")["sim_state"]
            if parent!="n_root":
                ancestor=load_snapshot(parentmap[parent]/"child_snapshot.json")
                assert np.array_equal(state,ancestor["states"]),"ANCESTRY_CHILD_PARENT_STATE"
                assert read(ep.parent.parent/"context.json")["history_refs"]==read(parentmap[parent]/"child_context.json")["history_refs"],"CANONICAL_HISTORY"
            frames=groups[(str(ep),"local_operation")]
            assert len(frames)==k+1,"RGB_KPLUS1"
            from cfact.execution.observation import array_sha
            for i,row in enumerate(frames):
                assert abs(row["sim_time_s"]-a["sim_time"][i])<1e-9,"RGB_CLOCK"
                if verify_rgb or i in {0,k}:
                    f=arrays(hist/row["frame"])
                    assert f["rgb"].shape==(3,224,224,3),"NATIVE_RGB_SHAPE"
                    assert [array_sha(im) for im in f["rgb"]]==row["image_sha256"],"RGB_HASH"
                    if i==0:assert np.array_equal(f["sim_state"],state),"PARENT_INITIAL_STATE"
                    if i==k:assert np.array_equal(f["sim_state"],child["states"]),"CHILD_LAST_STATE"
                    rgb_frames+=1
            ctx=read(ep/"child_context.json")
            assert np.allclose(ctx["operation_context"]["current_eef_pose"],a["actual_eef_pose"][-1],atol=1e-10,rtol=0),"FRESH_GEOMETRY"
            elapsed=a["sim_time"]-a["sim_time"][0]
            indices=np.flatnonzero(np.isclose(elapsed,horizon,atol=1e-8,rtol=0))
            card=read(ep/"option.json")
            if card.get("continuation_origin")=="CURRENT_STATE_PRIMITIVE":
                compilation=read(ep.parent.parent/"goal_compile.json")
                primitive_examples.append(dict(edge=str(ep),parent=parent,source_suffix_status=compilation["source_suffix_status"],
                    source_cursor=read(ep/"local_result.json")["source_cursor"],goal=card.get("effect_goal"),control_steps=k))
            edges.append(dict(edge=str(ep),parent=parent,steps=k,compare_index=int(indices[0]) if len(indices) else None))
            byparent[parent].append((ep,a,int(indices[0]) if len(indices) else None))
        except (AssertionError,ValueError,KeyError,OSError) as exc:
            failures.append(dict(edge=str(ep),status="INTEGRITY_ERROR",reason=str(exc)))
    pairs=[];effective_parents=set();calibrated_parents=set();bins=set();nonroot_bins=set()
    for parent,rows in byparent.items():
        for ep,a,i in rows:
            if i is None:continue
            value,unit=effect_value(a,i,family)
            if value is not None:
                key=(unit,*np.rint(value/FLOORS[unit]).astype(int).tolist());bins.add(key)
                if parent!="n_root":nonroot_bins.add(key)
        for (ep,a,i),(fp,b,j) in itertools.combinations(rows,2):
            rec=dict(parent=parent,edge_a=str(ep),edge_b=str(fp),horizon_s=horizon)
            if i is None or j is None:
                rec.update(status="CENSORED_INSUFFICIENT_REAL_DURATION",effective=False);pairs.append(rec);continue
            for key in ["sim_time","robot_qpos","robot_qvel","actual_eef_pose","object_poses","relevant_fixture_qpos"]:
                if not np.array_equal(a[key][0],b[key][0]):raise ValueError("SIBLING_INITIAL_STATE_MISMATCH:"+key)
            va,unit=effect_value(a,i,family);vb,other=effect_value(b,j,family)
            if va is None or unit!=other:
                rec.update(status="EFFECT_UNAVAILABLE",effective=False);pairs.append(rec);continue
            distance=float(np.linalg.norm(va-vb))
            cal=calibrations.get(family+":"+unit)
            threshold=max(FLOORS[unit],cal["frozen_threshold"]) if cal else FLOORS[unit]
            effective=distance>=threshold if unit=="activation" else distance>threshold
            rec.update(status="MEASURED",distance=distance,units=unit,threshold=threshold,effective=effective,
                       calibrated=bool(cal),calibration=cal.get("path") if cal else None)
            pairs.append(rec)
            if effective:effective_parents.add(parent)
            if effective and cal:calibrated_parents.add(parent)
    valid=not any(f["status"]=="INTEGRITY_ERROR" for f in failures)
    l2="n_root" in calibrated_parents and valid
    l3=l2 and any(p!="n_root" for p in calibrated_parents)
    return dict(task=result["task"],root=root.name,source_id=manifest.get("source_id"),source_group=manifest["source_group"],
                result_path=str(rp),level=3 if l3 else 2 if l2 else 1 if edges and valid else 0,
                integrity="PASS" if valid else "FAIL",complete_edges=len(edges),charged_control_steps=result["charged_control_steps"],
                raw_physical_trace_edges=sum((run/e["edge_path"]/"execution.npz").exists() for e in result["edges"] if e.get("edge_path")),
                requested_depth=result["requested_max_decision_depth"],observed_depth=result["observed_max_decision_depth"],
                comparable_parents=sum(sum(i is not None for _,_,i in rows)>=2 for rows in byparent.values()),
                nonroot_parents=sum(p!="n_root" for p in byparent),effective_parents=sorted(effective_parents),
                calibrated_effective_parents=sorted(calibrated_parents),effect_bins=len(bins),nonroot_effect_bins=len(nonroot_bins),
                terminal_paths=sum(not any(str(ep)==str(parentmap.get(p)) for p in parentmap) for ep,_,_ in itertools.chain.from_iterable(byparent.values())),
                rgb_frames_verified=rgb_frames,primitive_examples=primitive_examples,edges=edges,pairs=pairs,failures=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--calibrations",type=Path)
    p.add_argument("--horizon",type=float);p.add_argument("--quick",action="store_true")
    p.add_argument("--horizon-policy",type=Path,help="Frozen per-family comparison times; no endpoint padding")
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);cal={}
    policy=read(args.horizon_policy) if args.horizon_policy else {}
    times=policy.get("comparison_horizon_s",{})
    if args.horizon is not None and times:raise ValueError("Use either --horizon or --horizon-policy")
    for value in times.values():
        if not np.isfinite(value) or value<=0 or not np.isclose(value/.05,round(value/.05)):
            raise ValueError("Comparison horizon must be a positive physical control tick")
    if args.calibrations:
        for path in args.calibrations.rglob("SAME_PARENT_ABA.json"):
            c=read(path)
            if c["status"]=="PASS":cal[c["family"]+":"+c["threshold_unit"]]={**c,"path":str(path)}
    rows=[audit_root(p,horizon=times.get(family_for_task(target_task(read(p)["task"])),args.horizon),calibrations=cal,verify_rgb=not args.quick)
          for p in sorted(args.input.rglob("root_result.json")) if "data" in p.parts and (p.parent/"tree.json").exists()]
    import hashlib
    atomic_json(out/"ROOT_AUDIT.json",{"roots":rows,"horizon_s":args.horizon,"calibrations":cal,
        "horizon_policy":policy,"horizon_policy_path":str(args.horizon_policy) if args.horizon_policy else None,
        "horizon_policy_sha256":hashlib.sha256(args.horizon_policy.read_bytes()).hexdigest() if args.horizon_policy else None})
    compact=[{k:v for k,v in r.items() if k not in {"edges","pairs","failures","primitive_examples"}} for r in rows]
    if compact:
        with (out/"ROOT_METRICS.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=compact[0]);writer.writeheader();writer.writerows(compact)
    print(json.dumps({"roots":len(rows),"levels":[(r["task"],r["source_group"],r["level"]) for r in rows],"integrity_failures":sum(r["integrity"]!="PASS" for r in rows)}))
    if any(r["integrity"]!="PASS" for r in rows):raise SystemExit(2)


if __name__=="__main__":main()
