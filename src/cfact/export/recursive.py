"""Export frozen, measured recursive acquisitions with shared ancestor segments."""
import argparse
import hashlib
import json
from pathlib import Path
from cfact.io.serialization import RunLock, atomic_json
from cfact.collection.snapshot import implementation_digest
from cfact.storage.tree import publish_tree
from cfact.storage.integrity import input_fingerprint, release_inventory, verify_release
from cfact.export.dataset import write_dataset, verify_rows
from cfact.export.release import write_parquet
from cfact.export.verification import verify_run


def publish_acquisitions(acquisition, output, *, scope="DEVELOPMENT", root_names=None):
    acquisition=Path(acquisition).resolve();output=Path(output).resolve()
    roots=[p for p in sorted(acquisition.rglob("root_result.json")) if "data" in p.parts and (p.parent/"tree.json").exists()]
    if root_names is not None:roots=[p for p in roots if p.parent.name in root_names]
    roots=[p for p in roots if json.loads(p.read_text())["status"]=="PHYSICS_COMPLETE"]
    if not roots:raise ValueError("NO_COMPLETE_MEASURED_ROOTS")
    from cfact.storage.policy import contained_path
    first_run=roots[0].parents[4];config=json.loads((first_run/"config_frozen.json").read_text())
    contained_path(output,Path(config["storage"]["allowed_write_root"]))
    if output.is_relative_to(acquisition):raise ValueError("EXPORT_MUST_BE_SEPARATE_FROM_RAW_ACQUISITION")
    inventory=[];freeze=None
    for p in roots:
        run=p.parents[4];identity=json.loads((run/"run_manifest.json").read_text())["code_freeze"]
        if freeze is None:freeze=identity
        if identity["code_sha256"]!=freeze["code_sha256"]:raise ValueError("MIXED_COLLECTION_CODE_VERSIONS")
        if (run/"config_frozen.json").read_bytes()!=(first_run/"config_frozen.json").read_bytes():raise ValueError("MIXED_EFFECTIVE_CONFIGS")
        inventory.append({"root":p.parent.name,"fingerprint":input_fingerprint(run,[p.parent])})
    fingerprint=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    with RunLock(output/".publish.lock"):
        marker=output/"release.json"
        if marker.exists():
            old=json.loads(marker.read_text())
            if old["input_fingerprint"]!=fingerprint or old["publisher_sha256"]!=implementation_digest():
                raise ValueError("EXPORT_RESUME_CODE_OR_INPUT_CHANGED")
            if old["status"]=="COMPLETE":verify_release(output,old);return old
            raise ValueError("PARTIAL_EXPORT_REQUIRES_NEW_OUTPUT; RAW_INPUT_IS_UNCHANGED")
        record={"schema":"cfact.dual-view.v1","status":"BUILDING","scope":scope,
                "input_fingerprint":fingerprint,"input_inventory":inventory,"collection_code_freeze":freeze,
                "publisher_sha256":implementation_digest(),"full_media":False,"by_tree_exports":False,
                "episode_tree_map":"mappings/episode_tree_map.parquet",
                "terminal_observation_map":"mappings/terminal_observation_map.parquet",
                "lineage_paths":"mappings/lineage_paths.jsonl","file_integrity":"mappings/file_integrity.json",
                "lerobot_version":"0.4.0","training_started":False}
        atomic_json(marker,record)
        published=[]
        for path in roots:
            run=path.parents[4];scope_manifest=json.loads((run/"source_split_frozen.json").read_text())
            published.append(publish_tree(path.parent,run,output,config,scope_manifest))
        if len({p["tree"]["root_id"] for p in published})!=len(published):raise ValueError("DUPLICATE_SOURCE_ROOT_IDENTITY")
        segments=[s for p in published for s in p["segments"]];rows=[]
        for task in sorted({s["task"] for s in segments}):
            rows.extend(write_dataset(output,[s for s in segments if s["task"]==task],Path("lerobot/by_task")/task))
        write_parquet(output/record["episode_tree_map"],rows)
        by_segment={row["segment_id"]:row for row in rows};terminals=[t for p in published for t in p["terminals"]]
        for terminal in terminals:
            row=by_segment.get(terminal["segment_id"])
            if row:terminal.update({k:row[k] for k in ["compatible_schema","shard_id","shard_path","repo_id","episode_index"]})
            elif not terminal["zero_actions"]:raise ValueError("NONEMPTY_TERMINAL_HAS_NO_OFFICIAL_EPISODE")
        write_parquet(output/record["terminal_observation_map"],terminals)
        lineages=[l for p in published for l in p["lineages"]]
        (output/record["lineage_paths"]).write_text(''.join(json.dumps(l,sort_keys=True)+'\n' for l in lineages))
        edge_map={}
        for segment in segments:
            entry=edge_map.setdefault(segment["edge_id"],{"tree_id":segment["root_id"],"segments":[],
                "lineages":[l["lineage_id"] for l in lineages if segment["edge_id"] in l["edge_ids"]]})
            entry["segments"].append({"segment_id":segment["segment_id"],"kind":segment["segment"],"episode":by_segment.get(segment["segment_id"])})
        atomic_json(output/"mappings/edge_segment_episode_lineage.json",edge_map)
        for p in published:
            tree=p["tree"];tree_rows=[r for r in rows if r["root_id"]==tree["root_id"]]
            selected_lineages=[l for l in lineages if l["root_id"]==tree["root_id"]]
            index={"task":tree["task"],"tree_id":tree["root_id"],"tree":tree["tree_path"],
                   "source":tree["source_manifest"],"edges":tree["edges"],"episodes":tree_rows,
                   "is_independent_dataset":False,"lineages":selected_lineages}
            atomic_json(output/"lerobot/by_tree"/tree["task"]/tree["root_id"]/"index.json",index)
            atomic_json((output/tree["tree_path"]).with_name("lineage.json"),selected_lineages)
        official=verify_rows(output,rows);atomic_json(output/"reports/official_readback.json",official)
        record.update(status="COMPLETE",trees=[p["tree"]["tree_path"] for p in published],roots=len(roots),
                      edges=sum(len(p["tree"]["edges"]) for p in published),episodes=len(rows),
                      frames=sum(r["frame_count"] for r in rows),official_readback="reports/official_readback.json")
        atomic_json(output/record["file_integrity"],release_inventory(output));atomic_json(marker,record)
    result=verify_run(output);atomic_json(output/"FULL_READBACK.json",result)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--scope",default="DEVELOPMENT")
    p.add_argument("--root",action="append");args=p.parse_args()
    print(json.dumps(publish_acquisitions(args.input,args.output,scope=args.scope,root_names=args.root),indent=2))


if __name__=="__main__":main()
