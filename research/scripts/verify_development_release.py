import argparse,hashlib,json,time
from pathlib import Path
from cfact.data.release import ReleaseDataset
from cfact.storage.integrity import verify_release
from cfact.export.dataset import verify_rows
from cfact.io.serialization import atomic_json
p=argparse.ArgumentParser();p.add_argument('release',type=Path);a=p.parse_args();out=a.release.resolve();assert out.is_relative_to(Path('/mnt/ssd1/yanghc/CFACT_Research'));s=time.monotonic();d=ReleaseDataset(out);roots=d.roots();edges=0;transitions=0;nodes=0
for r in roots:
 t=d.tree(r['root_id'])
 for n in t['nodes']:d.node(r['root_id'],n['node_id']);nodes+=1
 for e in t['edges']:
  if e['status']!='PHYSICS_COMPLETE':continue
  for phase in ['local','fixed_continuation']:
   x=d.trace(e['edge_id'],phase);transitions+=len(x['commanded_actions'])
  for row in d.episodes(e['edge_id']):assert d.resolve_episode(row['shard_id'],row['episode_index'])['segment_id']==row['segment_id']
  assert d.terminals(e['edge_id']);edges+=1
rows=d.episodes();official=verify_rows(out,rows);integrity=verify_release(out,d.manifest);result=dict(status='PASS',scope='DEVELOPMENT_DIAGNOSTIC_RELEASE',official=official,native=dict(roots=len(roots),nodes=nodes,physical_edges=edges,transitions=transitions),integrity=integrity,all_rows_read=True,training_admitted=False,wall_seconds=time.monotonic()-s,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());atomic_json(out/'FULL_READBACK.json',result);print(json.dumps(result,indent=2))
