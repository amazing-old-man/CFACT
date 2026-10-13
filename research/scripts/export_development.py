"""Materialize actual development trees and verify every official LeRobot row."""
import argparse,hashlib,json,time
from pathlib import Path
from cfact.storage.tree import publish_tree
from cfact.export.dataset import write_dataset,verify_rows
from cfact.export.release import write_parquet
from cfact.storage.integrity import release_inventory,verify_release
from cfact.collection.snapshot import implementation_digest
from cfact.io.serialization import atomic_json
from cfact.data.release import ReleaseDataset
p=argparse.ArgumentParser();p.add_argument('--acquisition',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();ROOT=Path('/mnt/ssd1/yanghc/CFACT_Research');out=args.output.resolve();assert out.is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();acquisition=args.acquisition.resolve();config=json.loads((acquisition/'plan/config.json').read_text());scope=json.loads((acquisition/'plan/source_manifest.json').read_text());roots=sorted(acquisition.glob('runs/*/data/*/*/*/root_result.json'));assert roots
published=[]
for rp in roots:
    assert json.loads(rp.read_text())['split']=='development'
    published.append(publish_tree(rp.parent,rp.parents[4],out,config,scope))
segments=[s for x in published for s in x['segments']];rows=[]
for task in sorted({s['task'] for s in segments}):
    rows.extend(write_dataset(out,[s for s in segments if s['task']==task],Path('lerobot/by_task')/task))
write_parquet(out/'mappings/episode_tree_map.parquet',rows);by_segment={r['segment_id']:r for r in rows};terminals=[t for x in published for t in x['terminals']]
for terminal in terminals:
    row=by_segment.get(terminal['segment_id'])
    if row:terminal.update({k:row[k] for k in ['compatible_schema','shard_id','shard_path','repo_id','episode_index']})
    elif not terminal['zero_actions']:raise ValueError('Nonempty terminal lacks official episode')
write_parquet(out/'mappings/terminal_observation_map.parquet',terminals);lineages=[l for x in published for l in x['lineages']];(out/'mappings/lineage_paths.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in lineages))
record={'schema':'cfact.dual-view.v1','status':'COMPLETE','research_admission':'DEVELOPMENT_ONLY_NOT_FORMAL_OR_TRAINING','trees':[x['tree']['tree_path'] for x in published],'episode_tree_map':'mappings/episode_tree_map.parquet','terminal_observation_map':'mappings/terminal_observation_map.parquet','lineage_paths':'mappings/lineage_paths.jsonl','file_integrity':'mappings/file_integrity.json','lerobot_version':'0.4.0','full_media':False,'by_tree_exports':False,'collection_code_freeze':json.loads((acquisition/'CODE_FREEZE.json').read_text()),'publisher_sha256':implementation_digest(),'roots':len(roots),'edges':sum(len(x['tree']['edges']) for x in published),'episodes':len(rows),'frames':sum(r['frame_count'] for r in rows),'input_fingerprint':hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in roots).encode()).hexdigest()}
official=verify_rows(out,rows);atomic_json(out/'reports/official_readback.json',official);atomic_json(out/'release.json',record);inventory=release_inventory(out);atomic_json(out/'mappings/file_integrity.json',inventory);integrity=verify_release(out,record)
reader=ReleaseDataset(out);counts={'roots':len(reader.roots())}
# Full official readback already validates every state/action and every image shape.
result={'status':'PASS','scope':'DEVELOPMENT_DIAGNOSTIC_RELEASE','official':official,'native':counts,'integrity':integrity,'wall_seconds':time.monotonic()-started,'all_rows_read':True,'training_admitted':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};atomic_json(out/'FULL_READBACK.json',result);print(json.dumps(result,indent=2))
